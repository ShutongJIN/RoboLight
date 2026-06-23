
from isp_on_raw_img import *
from isp_on_raw_img import _BAYER_INDEX
import time
import cv2
import tqdm
import traceback
import argparse
import os

try:
    import cupy as cp
    import cupyx.scipy.ndimage as cp_ndimage
    # Test if GPU is actually available
    try:
        _ = cp.array([1, 2, 3])
        HAS_CUPY = True
    except Exception as e:
        HAS_CUPY = False
        print(f"[Warning] CuPy imported but GPU not available: {e}")
        print("Falling back to CPU processing.")
except ImportError:
    HAS_CUPY = False
    print("[Warning] CuPy not installed. Falling back to CPU processing.")
    print("To install CuPy:")
    print("  1. Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads")
    print("  2. pip install cupy-cuda11x (replace 11x with your CUDA version)")

def print_directory_tree(directory, prefix="", max_depth=None, current_depth=0):
    if max_depth is not None and current_depth > max_depth:
        return
    
    if not os.path.exists(directory):
        print(f"directory does not exist: {directory}")
        return
    
    items = sorted(os.listdir(directory))
    for i, item in enumerate(items):
        item_path = os.path.join(directory, item)
        is_last = i == len(items) - 1
        
        current_prefix = "└── " if is_last else "├── "
        print(f"{prefix}{current_prefix}{item}")
        
        if os.path.isdir(item_path):
            extension_prefix = "    " if is_last else "│   "
            print_directory_tree(item_path, prefix + extension_prefix, max_depth, current_depth + 1)

# ============= GPU-accelerated ISP functions =============

def split_bayer_channels_gpu(bayer16_gpu, bayer_pattern: str):
    """GPU version of split_bayer_channels"""
    if bayer_pattern not in _BAYER_INDEX:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    R_idx, G1_idx, G2_idx, B_idx = _BAYER_INDEX[bayer_pattern]

    R  = bayer16_gpu[R_idx[0]::2, R_idx[1]::2]
    G1 = bayer16_gpu[G1_idx[0]::2, G1_idx[1]::2]
    G2 = bayer16_gpu[G2_idx[0]::2, G2_idx[1]::2]
    B  = bayer16_gpu[B_idx[0]::2, B_idx[1]::2]

    return {"R":R, "G1":G1, "G2":G2, "B":B}

def merge_bayer_channels_gpu(shape_hw, channels: dict, bayer_pattern: str):
    """GPU version of merge_bayer_channels"""
    h, w = shape_hw
    bayer16_gpu = cp.zeros((h, w), dtype=channels["R"].dtype)
    if bayer_pattern not in _BAYER_INDEX:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    R_idx, G1_idx, G2_idx, B_idx = _BAYER_INDEX[bayer_pattern]
    bayer16_gpu[R_idx[0]::2, R_idx[1]::2] = channels["R"]
    bayer16_gpu[G1_idx[0]::2, G1_idx[1]::2] = channels["G1"]
    bayer16_gpu[G2_idx[0]::2, G2_idx[1]::2] = channels["G2"]
    bayer16_gpu[B_idx[0]::2, B_idx[1]::2] = channels["B"]

    return bayer16_gpu

def disk_mask_gpu(h, w, radius_ratio=0.12, center=None):
    """GPU version of disk_mask"""
    cy, cx = center if center is not None else (h//2, w//2)
    R = int(min(h, w) * radius_ratio)
    yy, xx = cp.ogrid[:h, :w]
    return (yy - cy)**2 + (xx - cx)**2 <= R**2

def estimate_chanel_vignette_gpu(chanel_data_gpu, sigma_ratio=0.125, center_radius_ratio=0.12):
    """GPU version of estimate_chanel_vignette"""
    h, w = chanel_data_gpu.shape
    sigma = max(1.0, min(h, w) * sigma_ratio)
    illuminate_map = cp_ndimage.gaussian_filter(chanel_data_gpu.astype(cp.float32), sigma=sigma)
    center_mask = disk_mask_gpu(h, w, radius_ratio=center_radius_ratio)
    center_mean = cp.mean(illuminate_map[center_mask])
    normalized_illuminate_map = illuminate_map / (center_mean + 1e-6)
    return normalized_illuminate_map

def estimate_bayer_gain_map_gpu(bayer16_gpu, pattern="GBRG",
                            black_levels=None,
                            sigma_ratio=0.125,
                            center_radius_ratio=0.12,
                            gain_clip=1.7):
    """GPU version of estimate_bayer_gain_map (simplified)"""
    channels = split_bayer_channels_gpu(bayer16_gpu, pattern)
    if black_levels is None:
        # Estimate on CPU for simplicity
        black_levels = estimate_black_levels(cp.asnumpy(bayer16_gpu), pattern, percentile=0.1)
    
    gain_channels = {}
    for channel_name, channel_data in channels.items():
        linear_data = cp.clip(channel_data.astype(cp.float32) - float(black_levels[channel_name]), 0, None)
        vignette_pattern = estimate_chanel_vignette_gpu(linear_data, sigma_ratio, center_radius_ratio)
        vignette_floor = 0.05
        gain = 1.0 / cp.clip(vignette_pattern, vignette_floor, None)
        
        # Smooth the gain map
        sigma = max(1.0, min(linear_data.shape) * sigma_ratio * 0.6)
        gain = cp_ndimage.gaussian_filter(gain, sigma=sigma)
        gain = cp.minimum(gain, gain_clip)
        gain_channels[channel_name] = gain

    gain_map = merge_bayer_channels_gpu(bayer16_gpu.shape, gain_channels, pattern)
    return gain_map.astype(cp.float32), black_levels

def bilateral_filter_gpu(image_gpu, sigma_color, sigma_space, radius=None):
    """
    GPU-accelerated bilateral filter using CuPy
    Approximation of bilateral filter for speed
    """
    if radius is None:
        radius = int(2 * sigma_space)
    
    h, w = image_gpu.shape
    img_f32 = image_gpu.astype(cp.float32)
    
    # Create spatial Gaussian kernel
    y, x = cp.ogrid[-radius:radius+1, -radius:radius+1]
    spatial_kernel = cp.exp(-(x*x + y*y) / (2.0 * sigma_space * sigma_space))
    
    # Pad image
    img_padded = cp.pad(img_f32, radius, mode='reflect')
    
    # Initialize output
    output = cp.zeros_like(img_f32)
    normalization = cp.zeros_like(img_f32)
    
    # Sliding window
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            # Get shifted image
            shifted = img_padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
            
            # Compute color distance
            color_diff = img_f32 - shifted
            color_kernel = cp.exp(-(color_diff * color_diff) / (2.0 * sigma_color * sigma_color))
            
            # Combine spatial and color kernels
            weight = spatial_kernel[dy+radius, dx+radius] * color_kernel
            
            output += shifted * weight
            normalization += weight
    
    # Normalize
    output = output / (normalization + 1e-10)
    return output

def raw16_bilateral_denoise_gpu(bayer16_gpu, pattern="GBRG", sigma_color=8.0, sigma_space=3.0, iterations=1, use_gpu=True):
    """GPU-accelerated bilateral denoise on raw16 Bayer image
    
    Args:
        use_gpu: If True, use GPU implementation; if False, fallback to CPU (OpenCV)
    """
    channels = split_bayer_channels_gpu(bayer16_gpu, pattern)
    out = {}
    
    if use_gpu and HAS_CUPY:
        # Pure GPU implementation
        for k, p in channels.items():
            f = p.astype(cp.float32)
            for _ in range(iterations):
                f = bilateral_filter_gpu(f, sigma_color, sigma_space)
            out[k] = cp.clip(f, 0, 65535).astype(bayer16_gpu.dtype)
    else:
        # CPU fallback (OpenCV is highly optimized)
        for k, p in channels.items():
            f = cp.asnumpy(p).astype(np.float32)
            for _ in range(iterations):
                f = cv2.bilateralFilter(f, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
            out[k] = cp.asarray(f.astype(bayer16_gpu.dtype))
    
    return merge_bayer_channels_gpu(bayer16_gpu.shape, out, pattern)

def apply_lsc_on_raw16_gpu(bayer16_gpu, gain_map_gpu, pattern="GBRG", 
                       black_levels=None, strength=1.0,
                       add_black_back=True):
    """GPU version of apply_lsc_on_raw16"""
    if black_levels is None:
        black_levels = estimate_black_levels(cp.asnumpy(bayer16_gpu), pattern)
    
    raw_channels = split_bayer_channels_gpu(bayer16_gpu, pattern)
    gain_channels = split_bayer_channels_gpu(gain_map_gpu, pattern)

    corrected_channels = {}
    for k in ("R","G1","G2","B"):
        raw = raw_channels[k].astype(cp.float32)
        bl  = float(black_levels[k])
        g   = cp.power(cp.maximum(gain_channels[k].astype(cp.float32), 1e-6), strength)
        lin = cp.clip(raw - bl, 0, None) * g
        out = lin + (bl if add_black_back else 0.0)
        corrected_channels[k] = cp.clip(out, 0, 65535).astype(cp.uint16)
    
    return merge_bayer_channels_gpu(bayer16_gpu.shape, corrected_channels, pattern)

def isp_pipeline_gpu(
        bayer16, pattern="GBRG",
        black_levels=None,
        gain_map=None,
        lsc_strength=1.28, #1.0,
        sigma_ratio=0.25, #0.125, 
        center_radius_ratio=0.14, #0.12,
        gain_clip=4.0, #1.7,
        raw_denoise=False,
        bilateral_sigma_color=8.0,
        bilateral_sigma_space=3.0,
        bilateral_iterations=1,
        bilateral_use_gpu=True,
        white_balance_temp=5200, #4600,
        ccm=None,
        white_level=65535.0,
        gamma=2.4,
        out_dtype=np.uint8
):
    """GPU-accelerated ISP pipeline"""
    if not HAS_CUPY:
        # Fallback to CPU version
        return isp_pipeline(
            bayer16, pattern=pattern,
            black_levels=black_levels,
            gain_map=gain_map,
            lsc_strength=lsc_strength,
            sigma_ratio=sigma_ratio,
            center_radius_ratio=center_radius_ratio,
            gain_clip=gain_clip,
            raw_denoise=False,
            bilateral_sigma_color=0,
            bilateral_sigma_space=0,
            bilateral_iterations=1,
            white_balance_temp=white_balance_temp,
            ccm=ccm,
            white_level=white_level,
            gamma=gamma,
            out_dtype=out_dtype,
        )
    
    try:
        # Transfer to GPU
        bayer16_gpu = cp.asarray(bayer16)
    except Exception as e:
        # GPU failed, fallback to CPU
        print(f"[Warning] GPU initialization failed: {e}. Falling back to CPU.")
        return isp_pipeline(
            bayer16, pattern=pattern,
            black_levels=black_levels,
            gain_map=gain_map,
            lsc_strength=lsc_strength,
            sigma_ratio=sigma_ratio,
            center_radius_ratio=center_radius_ratio,
            gain_clip=gain_clip,
            raw_denoise=raw_denoise,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            bilateral_iterations=bilateral_iterations,
            white_balance_temp=white_balance_temp,
            ccm=ccm,
            white_level=white_level,
            gamma=gamma,
            out_dtype=out_dtype,
        )
    
    try:
        # Apply bilateral denoise on raw data if requested
        raw_in_gpu = bayer16_gpu
        if raw_denoise:
            raw_in_gpu = raw16_bilateral_denoise_gpu(raw_in_gpu, pattern, 
                                                     bilateral_sigma_color, 
                                                     bilateral_sigma_space, 
                                                     bilateral_iterations,
                                                     use_gpu=bilateral_use_gpu)
        
        # Estimate or use provided gain map
        if gain_map is None:
            gain_map_gpu, black_levels = estimate_bayer_gain_map_gpu(
                raw_in_gpu, pattern, black_levels, sigma_ratio, center_radius_ratio, gain_clip
            )
        else:
            gain_map_gpu = cp.asarray(gain_map)
            if black_levels is None:
                black_levels = estimate_black_levels(bayer16, pattern, percentile=0.1)
        
        # Apply LSC
        raw_lsc_gpu = apply_lsc_on_raw16_gpu(raw_in_gpu, gain_map_gpu, pattern, 
                                              black_levels, lsc_strength, add_black_back=True)
        
        # Transfer back to CPU for demosaic and remaining operations
        # (OpenCV's demosaicing doesn't have direct GPU support via Python)
        raw_lsc = cp.asnumpy(raw_lsc_gpu)
    except Exception as e:
        # GPU processing failed, fallback to CPU
        print(f"[Warning] GPU processing failed: {e}. Falling back to CPU.")
        return isp_pipeline(
            bayer16, pattern=pattern,
            black_levels=black_levels,
            gain_map=gain_map,
            lsc_strength=lsc_strength,
            sigma_ratio=sigma_ratio,
            center_radius_ratio=center_radius_ratio,
            gain_clip=gain_clip,
            raw_denoise=raw_denoise,
            bilateral_sigma_color=bilateral_sigma_color,
            bilateral_sigma_space=bilateral_sigma_space,
            bilateral_iterations=bilateral_iterations,
            white_balance_temp=white_balance_temp,
            ccm=ccm,
            white_level=white_level,
            gamma=gamma,
            out_dtype=out_dtype,
        )
    
    # Demosaic
    bgr16 = demosaic_raw16_to_bgr16(raw_lsc, pattern)
    
    # White balance
    bgr16 = apply_white_balance(bgr16, white_balance_temp)
    
    # Color correction
    bgr16 = apply_ccm(bgr16, ccm)
    
    # Gamma correction
    bgr_srgb01 = apply_gamma_correction(bgr16, white_level, gamma)
    
    if out_dtype == np.uint8:
        bgr_out = to_unit8(bgr_srgb01)
    else:
        bgr_out = to_unit16(bgr_srgb01)
    
    return bgr_out, {"black_levels": black_levels, "gain_map": cp.asnumpy(gain_map_gpu) if gain_map is None else gain_map}

def run_isp_on_path_gpu(input_path, pattern='GBRG', 
                        enhance_shadow=False,
                        shadow_gamma=1.5,
                        shadow_white_level=40000.0,
                        apply_clahe=False):
    """GPU-accelerated ISP processing
    
    Args:
        enhance_shadow: If True, use adjusted gamma and white_level to enhance shadow contrast
        shadow_gamma: Gamma value for shadow enhancement (lower = darker shadows, more contrast)
        shadow_white_level: White level for shadow enhancement (lower = more contrast, brighter midtones)
        apply_clahe: If True, apply CLAHE (Contrast Limited Adaptive Histogram Equalization) post-processing
    """
    if not os.path.exists(input_path):
        print(f"Input path does not exist: {input_path}")
        return None, None
    
    raw_img = np.load(input_path)
    
    # Adjust parameters based on shadow enhancement mode
    if enhance_shadow:
        gamma_val = shadow_gamma
        white_level_val = shadow_white_level
    else:
        gamma_val = 2.4
        white_level_val = 65535.0
    
    img_srgb8_ori, meta_ori = isp_pipeline_gpu(
        raw_img, pattern='GBRG',
        black_levels=None,
        gain_map=None,             
        center_radius_ratio=0.12, #0.12, 
        lsc_strength=1.28, #1.642,          
        sigma_ratio=0.25,
        gain_clip=4.0, #8.0,
        raw_denoise=True,
        bilateral_sigma_color=200, #200.0,
        bilateral_sigma_space=8.0, #8.0,
        bilateral_iterations=1,
        bilateral_use_gpu=True,  # Use GPU for bilateral filtering
        white_balance_temp=5200, #4600, 
        ccm=None,              
        white_level=white_level_val,
        gamma=gamma_val,
        out_dtype=np.uint16
    )
    
    # Optional: Apply CLAHE for additional local contrast enhancement
    if apply_clahe and img_srgb8_ori is not None:
        # For uint16, convert to uint8 first (LAB conversion doesn't support uint16)
        if img_srgb8_ori.dtype == np.uint16:
            # Convert uint16 to uint8 for processing
            img_uint8 = (img_srgb8_ori.astype(np.float32) / 65535.0 * 255.0).astype(np.uint8)
            
            # Convert to LAB color space
            img_lab = cv2.cvtColor(img_uint8, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_lab)
            
            # Apply CLAHE on L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            # Merge back and convert to BGR
            img_lab_enhanced = cv2.merge([l_enhanced, a, b])
            img_bgr_uint8 = cv2.cvtColor(img_lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # Convert back to uint16
            img_srgb8_ori = (img_bgr_uint8.astype(np.float32) / 255.0 * 65535.0).astype(np.uint16)
        else:
            # For uint8, process directly
            img_lab = cv2.cvtColor(img_srgb8_ori, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(img_lab)
            
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            
            img_srgb8_ori = cv2.cvtColor(cv2.merge([l_enhanced, a, b]), cv2.COLOR_LAB2BGR)
    
    return img_srgb8_ori, meta_ori

def batch_isp_gpu(root_dataset, start_id: int, end_id: int):
    """
    GPU-accelerated batch ISP processing
    
    Args:
        root_dataset: Root directory of the dataset
        start_id: Start episode ID
        end_id: End episode ID (exclusive)
    """
    episode_dir = os.path.join(root_dataset, 'episodes')
    if not os.path.exists(episode_dir):
        raise FileNotFoundError(f"Directory not found: {episode_dir}")
    
    if HAS_CUPY:
        print(f"[INFO] Using GPU acceleration with CuPy")
        try:
            device_id = cp.cuda.Device().id
            device_props = cp.cuda.runtime.getDeviceProperties(device_id)
            print(f"[INFO] GPU: {device_props['name'].decode()}")
        except:
            print(f"[INFO] GPU: Device {device_id}")
    else:
        print(f"[INFO] CuPy not available, using CPU processing")
    
    total_images = 0
    total_time = 0
    
    for id in tqdm.tqdm(range(start_id, end_id), desc="Processing episodes"):
        episode_name = f"episode{id}"
        episode_path = os.path.join(episode_dir, episode_name)
        if not os.path.isdir(episode_path):
            continue
        
        rgb0_dir = os.path.join(episode_path, 'rgb_0')
        if not os.path.exists(rgb0_dir):
            print(f"[Warning] {rgb0_dir} does not exist, skip")
            continue
        
        rgb0_isped_dir = os.path.join(episode_path, 'rgb_0_isped')
        os.makedirs(rgb0_isped_dir, exist_ok=True)

        # Get all raw images
        img_files = sorted([
            f for f in os.listdir(rgb0_dir)
            if f.lower().endswith((".npy"))
        ])
        
        for img_name in tqdm.tqdm(img_files, desc=f"{episode_name}/rgb_0", leave=False):
            src_path = os.path.join(rgb0_dir, img_name)
            dst_name = os.path.splitext(img_name)[0] + ".png"
            dst_path = os.path.join(rgb0_isped_dir, dst_name)
            
            if os.path.exists(dst_path):
                # print(f"{dst_path} already exists, skip image {img_name}")
                continue

            try:
                start_time = time.time()
                srgb_img, _ = run_isp_on_path_gpu(src_path, apply_clahe=False, enhance_shadow=False)
                if srgb_img is not None:
                    cv2.imwrite(dst_path, srgb_img)
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    total_images += 1
            except Exception as e:
                print(f"[Error] failed in processing {src_path}: {e}")
                traceback.print_exc()
    
    if total_images > 0:
        avg_time = total_time / total_images
        print(f"\n[INFO] Processed {total_images} images in {total_time:.2f}s")
        print(f"[INFO] Average time per image: {avg_time:.3f}s ({1/avg_time:.1f} fps)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GPU-accelerated batch ISP processing for raw images')
    parser.add_argument('--root_dataset', type=str, required=True,
                       help='Root directory of the dataset')
    parser.add_argument('--start_id', type=int, required=True,
                       help='Start episode ID')
    parser.add_argument('--end_id', type=int, required=True,
                       help='End episode ID (exclusive)')
    args = parser.parse_args()
    
    batch_isp_gpu(root_dataset=args.root_dataset, 
                  start_id=args.start_id, 
                  end_id=args.end_id)
