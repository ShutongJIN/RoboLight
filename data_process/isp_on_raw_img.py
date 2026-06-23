
import os
import json
import glob
import numpy as np
import cv2
from scipy import ndimage
import time

# ================ Patterns for Bayer images ================

_BAYER2CV = {
    "RGGB": cv2.COLOR_BayerRG2BGR,
    "BGGR": cv2.COLOR_BayerBG2BGR,
    "GRBG": cv2.COLOR_BayerGR2BGR,
    "GBRG": cv2.COLOR_BayerGB2BGR,
}

_BAYER_INDEX = {
    "RGGB": ((0,0),(0,1),(1,0),(1,1)),
    "BGGR": ((1,1),(0,1),(1,0),(0,0)),
    "GRBG": ((0,1),(0,0),(1,1),(1,0)),
    "GBRG": ((1,0),(0,0),(1,1),(0,1)),
}

# ================ Bayer Tools ================

def split_bayer_channels(bayer16: np.ndarray, bayer_pattern: str):
    """
    Split a 2D Bayer16 image into its four channels: R, G1, G2, B
    """
    if bayer_pattern not in _BAYER_INDEX:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    R_idx, G1_idx, G2_idx, B_idx = _BAYER_INDEX[bayer_pattern]

    R  = bayer16[R_idx[0]::2, R_idx[1]::2]
    G1 = bayer16[G1_idx[0]::2, G1_idx[1]::2]
    G2 = bayer16[G2_idx[0]::2, G2_idx[1]::2]
    B  = bayer16[B_idx[0]::2, B_idx[1]::2]

    return {"R":R, "G1":G1, "G2":G2, "B":B}

def merge_bayer_channels(shape_hw, channels: dict, bayer_pattern: str):
    """
    Merge four channels: R, G1, G2, B into a 2D Bayer16 image
    """
    h, w = shape_hw
    bayer16 = np.zeros((h, w), dtype=channels["R"].dtype)
    if bayer_pattern not in _BAYER_INDEX:
        raise ValueError(f"Unsupported Bayer pattern: {bayer_pattern}")
    R_idx, G1_idx, G2_idx, B_idx = _BAYER_INDEX[bayer_pattern]
    bayer16[R_idx[0]::2, R_idx[1]::2] = channels["R"]
    bayer16[G1_idx[0]::2, G1_idx[1]::2] = channels["G1"]
    bayer16[G2_idx[0]::2, G2_idx[1]::2] = channels["G2"]
    bayer16[B_idx[0]::2, B_idx[1]::2] = channels["B"]

    return bayer16

# ================ Black level, ring mask, statistics ================

def estimate_black_levels(bayer16, pattern, percentile=0.1):
    """
    Estimate the black level of a Bayer16 image by computing the low percentile value
    across all channels.
    """
    channels = split_bayer_channels(bayer16, pattern)
    return {k: float(np.percentile(p, percentile)) for k, p in channels.items()}

def ellipse_mask(h, w, radius_ratio_y=0.12, radius_ratio_x=0.12, center=None):
    cy, cx = center if center is not None else (h // 2, w // 2)
    Ry = int(h * radius_ratio_y)
    Rx = int(w * radius_ratio_x)
    yy, xx = np.ogrid[:h, :w]
    return ((yy - cy) / Ry) ** 2 + ((xx - cx) / Rx) ** 2 <= 1

def ellipse_ring_mask(h, w, inner_ratio_y=0.80, inner_ratio_x=0.80, outer_ratio_y=0.98, outer_ratio_x=0.98, center=None):
    cy, cx = center if center is not None else (h // 2, w // 2)
    yy, xx = np.ogrid[:h, :w]
    inner = ((yy - cy) / (h * inner_ratio_y / 2)) ** 2 + ((xx - cx) / (w * inner_ratio_x / 2)) ** 2 <= 1
    outer = ((yy - cy) / (h * outer_ratio_y / 2)) ** 2 + ((xx - cx) / (w * outer_ratio_x / 2)) ** 2 <= 1
    return outer & (~inner)

def disk_mask(h, w, radius_ratio=0.12, center=None):
    """
    Create a disk mask with given radius ratio and center.
    returns a boolean mask of shape (h, w): True inside the disk, False outside.
    """
    cy, cx = center if center is not None else (h//2, w//2)
    R = int(min(h, w) * radius_ratio)
    yy, xx = np.ogrid[:h, :w]
    return (yy - cy)**2 + (xx - cx)**2 <= R**2

def corner_ring_masks(h, w, inner_ratio=0.80, outer_ratio=0.98):
    """
    Create four corner ring masks for vignetting estimation.
    returns a list of four boolean masks of shape (h, w).
    """
    cy, cx = h//2, w//2
    y = np.arange(h)[:, None] # [[0], [1], [2], ..., [h-1]]
    x = np.arange(w)[None, :] # [[0, 1, 2, ..., w-1]]
    rr = np.sqrt((y - cy)**2 + (x - cx)**2) # shape (h, w), distance to center
    R_inner = inner_ratio * min(h, w) / 2
    R_outer = outer_ratio * min(h, w) / 2
    ring = (rr >= R_inner) & (rr <= R_outer)
    masks = {}
    masks["c1"] = ring & (y < cy) & (x < cx)  # top-left
    masks["c2"] = ring & (y < cy) & (x >= cx) # top-right
    masks["c3"] = ring & (y >= cy) & (x < cx) # bottom-left
    masks["c4"] = ring & (y >= cy) & (x >= cx)# bottom-right

    return masks

def robust_center_and_corner_stats(bayer16, center_mask, corner_masks):
    """
    Compute robust statistics for center and corner pixels in a Bayer16 image."""
    bayer_f32 = bayer16.astype(np.float32)
    c = np.median(bayer_f32[center_mask])
    corners = np.concatenate([bayer_f32[m] for m in corner_masks.values()]) # get all corner pixels
    low, high = np.percentile(corners, [10, 90])
    clipped_corners = corners[(corners >= low) & (corners <= high)]
    k = clipped_corners.mean() if clipped_corners.size else corners.mean()

    return c, k

# ================ Estimate vignette in raw16 and estimate gain map ================

def estimate_chanel_vignette(chanel_data, sigma_ratio=0.125, center_radius_ratio=0.12):
    """
    Estimate the vignette for a single channel of a Bayer16 image.
    Returns normalized illuminate map of the same shape as chanel_data.
    """
    h, w = chanel_data.shape
    sigma = max(1.0, min(h, w) * sigma_ratio)
    illuminate_map = ndimage.gaussian_filter(chanel_data.astype(np.float32), sigma=sigma)
    center_mask = disk_mask(h, w, radius_ratio=center_radius_ratio)
    center_mean = np.mean(illuminate_map[center_mask])
    # ring_mask = corner_ring_masks(h, w, inner_ratio=0.2, outer_ratio=0.8)
    # corner_values = np.concatenate([illuminate_map[m] for m in ring_mask.values()])
    # corner_mean = np.mean(corner_values) if corner_values.size else 1.0
    center_ellipse_mask = ellipse_mask(h, w, radius_ratio_y=center_radius_ratio, radius_ratio_x=center_radius_ratio)
    center_ellipse_mean = np.mean(illuminate_map[center_ellipse_mask])
    # corner_ellipse_mask = ellipse_ring_mask(h, w, inner_ratio_y=0.4, inner_ratio_x=0.4*float(h)/w, outer_ratio_y=0.6, outer_ratio_x=0.6*float(h)/w)
    # corner_eppilise_mean = np.mean(illuminate_map[corner_ellipse_mask])
    global_mean = np.mean(illuminate_map)
    
    normalized_illuminate_map = illuminate_map / (center_mean + 1e-6)

    return normalized_illuminate_map # shape (h, w), approx 1.0 at center, <1.0 at corners

def estimate_bayer_gain_map(bayer16, pattern="GRBG",
                            black_levels=None,
                            sigma_ratio=0.125,
                            center_radius_ratio=0.12,
                            gain_clip=1.7,
                            vignette_floor_percentile=1.0):
    channels = split_bayer_channels(bayer16, pattern)
    if black_levels is None:
        black_levels = estimate_black_levels(bayer16, pattern, percentile=0.1)
    
    gain_channels, strength = {}, {}
    for channel_name, channel_data in channels.items():
        linear_data = np.clip(channel_data.astype(np.float32) - float(black_levels[channel_name]), 0, None)
        vignette_pattern = estimate_chanel_vignette(linear_data, sigma_ratio, center_radius_ratio)
        vignette_floor = max(np.percentile(vignette_pattern, vignette_floor_percentile), 0.05)
        gain = 1.0 / np.clip(vignette_pattern, vignette_floor, None)
        # smooth the gain map
        sigma = max(1.0, min(linear_data.shape) * sigma_ratio * 0.6)
        gain = ndimage.gaussian_filter(gain, sigma=sigma)
        gain = np.minimum(gain, gain_clip) # constrain the gain value with in [1.0, gain_clip]
        gain_channels[channel_name] = gain

        # log the per channel vignette strength
        h, w = channel_data.shape
        center_mask = disk_mask(h, w, radius_ratio=center_radius_ratio)
        illuminate = ndimage.gaussian_filter(linear_data, sigma=max(1.0, min(h, w) * sigma_ratio))
        corner_mask = corner_ring_masks(h, w, inner_ratio=0.80, outer_ratio=0.98)
        c_val, k_val = robust_center_and_corner_stats(illuminate, center_mask, corner_mask)
        strength[channel_name] = float(1.0 - k_val / max(c_val, 1e-6))
    gain_map = merge_bayer_channels(bayer16.shape, gain_channels, pattern)
    strength["MEAN_G"] = 0.5*(strength["G1"] + strength["G2"])
    strength["MEAN_RGB"] = float(np.mean([strength[x] for x in ("R","G1","G2","B")]))
    return gain_map.astype(np.float32), strength, black_levels

def apply_lsc_on_raw16(bayer16, gain_map, pattern="GBRG", 
                       black_levels=None, strength=1.0,
                       add_black_back=True, out_dtype=np.uint16):
    if black_levels is None:
        black_levels = estimate_black_levels(bayer16, pattern)
    raw_channels = split_bayer_channels(bayer16, pattern)
    gain_channels = split_bayer_channels(gain_map, pattern)

    corrected_channels = {}
    for k in ("R","G1","G2","B"):
        raw = raw_channels[k].astype(np.float32)
        bl  = float(black_levels[k])
        g   = np.power(np.maximum(gain_channels[k].astype(np.float32), 1e-6), strength)
        lin = np.clip(raw - bl, 0, None) * g
        out = lin + (bl if add_black_back else 0.0)
        corrected_channels[k] = np.clip(out, 0, np.iinfo(out_dtype).max).astype(out_dtype)
    return merge_bayer_channels(bayer16.shape, corrected_channels, pattern)

# ================ Denoising in RAW16 ====================
def raw16_bilateral_denoise(bayer16, pattern="GBRG", sigma_color=8.0, sigma_space=3.0, iterations=1):
    channels = split_bayer_channels(bayer16, pattern)
    out = {}
    h, w = bayer16.shape
    for k, p in channels.items():
        f = p.astype(np.float32)
        for _ in range(iterations):
            f = cv2.bilateralFilter(f, d=0, sigmaColor=sigma_color, sigmaSpace=sigma_space)
        out[k] = f.astype(bayer16.dtype)
    return merge_bayer_channels(bayer16.shape, out, pattern)

# ================ Demosaic and ISP on RAW16 ====================

def apply_black_level_correction(bgr16, black_level=512):
    """
    Apply black level correction to a demosaiced BGR16 linear img
    """
    corrected = np.maximum(bgr16.astype(np.float32) - black_level, 0)
    return corrected

def demosaic_raw16_to_bgr16(bayer16: np.ndarray, bayer_pattern: str) -> np.ndarray:
    """ 
    Lossless demosaic, converting a 16-bit Bayer image to a linear BGR image using OpenCV.
    - bayer16: a 2D numpy array of dtype uint16 representing the Bayer image
    - bayer_pattern: one of "RGGB", "BGGR", "GRBG", "GBRG"
    """
    if bayer_pattern not in _BAYER2CV:
        raise ValueError(f"[Error] Unsupported Bayer pattern: {bayer_pattern}. Supported patterns are: {list(_BAYER2CV.keys())}")
    bgr16 = cv2.cvtColor(bayer16, _BAYER2CV[bayer_pattern])
    return bgr16

def white_balance_temperature_to_gains(white_balance_temp, reference_temp=5500):
    """
    Convert a color temperature in Kelvin to approximate RGB gains for white balancing.
    Low temp (warm/yellow light) -> increase blue gain, decrease red gain
    High temp (cool/blue light) -> increase red gain, decrease blue gain
    
    Args:
        white_balance_temp: Color temperature in Kelvin
        reference_temp: Reference temperature (default 5500K)
        green_suppression: Green channel suppression factor (0.85-0.95 recommended)
                          Use 1.0 for no suppression, <1.0 to reduce green cast
    """
    
    temp_ratio = white_balance_temp / reference_temp
    
    # Use smooth scaling instead of hard clipping
    if white_balance_temp < reference_temp:
        # warm light (yellowish), need more blue, less red
        r_gain = temp_ratio
        g_gain = 0.55
        b_gain = 1.0 / temp_ratio
    else:
        # cool light (bluish), need more red, less blue  
        r_gain = temp_ratio
        g_gain = 0.55
        b_gain = 1.0 / temp_ratio
    
    # Optional: normalize so that the minimum gain is 1.0 to avoid darkening
    min_gain = min(r_gain, g_gain, b_gain)
    r_gain /= min_gain
    g_gain /= min_gain
    b_gain /= min_gain
    
    # Clip to reasonable range to avoid extreme values
    r_gain = np.clip(r_gain, 0.5, 3.0)
    g_gain = np.clip(g_gain, 0.5, 3.0)
    b_gain = np.clip(b_gain, 0.5, 3.0)
    
    return [r_gain, g_gain, b_gain]

def apply_white_balance(bgr16, white_balance_temp):
    """
    Apply white balance to BGR16 image.
    
    Args:
        bgr16: Input BGR16 image
        white_balance_temp: Color temperature in Kelvin
        green_suppression: Green suppression factor (default 0.90)
                          Reduce if image appears too green (try 0.85-0.95)
                          Increase if image appears pink/magenta (try 0.95-1.0)
    """
    wb_gains = white_balance_temperature_to_gains(white_balance_temp)
    bgr_f32 = bgr16.astype(np.float32)
    bgr_f32[:, :, 0] *= wb_gains[2]  # B
    bgr_f32[:, :, 1] *= wb_gains[1]  # G
    bgr_f32[:, :, 2] *= wb_gains[0]  # R
    return np.clip(bgr_f32, 0, 65535).astype(np.uint16)

def apply_ccm(bgr16, ccm=None):
    if ccm is None:
        return bgr16
    bgr_float = bgr16.astype(np.float32)
    H, W, _ = bgr_float.shape
    img = bgr_float.reshape(-1, 3)
    img = img @ np.asarray(ccm, dtype=np.float32).T
    return img.reshape(H, W, 3).clip(0, 65535).astype(np.uint16)

def apply_gamma_correction(bgr16, white_level=65535.0, gamma=2.4):
    """
    Apply gamma correction to a linear BGR16 image.
    """
    x = np.clip(bgr16 / float(white_level), 0.0, 1.0)
    a = 0.0031308
    low  = 12.92 * x
    high = 1.055 * np.power(x, 1.0/gamma) - 0.055
    return np.where(x <= a, low, high)

def to_unit8(img_01):
    return np.clip(np.round(img_01*255.0), 0, 255).astype(np.uint8)

def to_unit16(img_01):
    return np.clip(np.round(img_01*65535.0), 0, 65535).astype(np.uint16)


def isp_pipeline(
        bayer16, pattern="GBRG",
        demosaic=True,
        black_levels=None,
        gain_map=None, # can obtained by calibration or set to None to estimate online
        lsc=True, lsc_strength=1.0,
        sigma_ratio=0.125, center_radius_ratio=0.01,
        gain_clip=1.7, vignette_floor_percentile=1.0,
        # raw denoise
        raw_denoise=True, bilateral_sigma_color=8.0, bilateral_sigma_space=3.0, bilateral_iterations=1,
        # white balance
        white_balance=True,
        white_balance_temp=4600,
        ccm=None, # color correction matrix, 3x3 numpy array
        white_level=65535.0,
        gamma_correction=True, gamma=2.4,
        out_dtype=np.uint8
):
    raw_in = bayer16
    start = time.time()
    if raw_denoise:
        raw_in = raw16_bilateral_denoise(raw_in, pattern, bilateral_sigma_color, bilateral_sigma_space, bilateral_iterations)
    time_denoise = time.time()
    # print(f"ISP Denoise time: {time_denoise - start:.4f} seconds")
    if gain_map is None:
        gain_map, strengths, black_levels = estimate_bayer_gain_map(
            raw_in, pattern, black_levels, sigma_ratio, center_radius_ratio, gain_clip, vignette_floor_percentile
        )
    else:
        strengths = None
        if black_levels is None:
            black_levels = estimate_black_levels(raw_in, pattern, percentile=0.1)
    # LSC
    if lsc:
        raw_lsc = apply_lsc_on_raw16(raw_in, gain_map, pattern, black_levels, lsc_strength, add_black_back=True, out_dtype=np.uint16)
    else:
        raw_lsc = raw_in
    time_lsc = time.time()
    # print(f"ISP LSC time: {time_lsc - start:.4f} seconds")
    # Demosaic
    if demosaic:
        bgr16 = demosaic_raw16_to_bgr16(raw_lsc, pattern)
    else:
        bgr16 = raw_lsc
    # white balance
    if white_balance:
        bgr16 = apply_white_balance(bgr16, white_balance_temp)
    # color correction
    bgr16 = apply_ccm(bgr16, ccm)
    # gamma correction
    if gamma_correction:
        bgr_srgb01 = apply_gamma_correction(bgr16, white_level, gamma)
    else:
        # bgr_srgb01 = np.clip(bgr16.astype(np.float32) / float(white_level), 0.0, 1.0)
        bgr_srgb01 = bgr16.astype(np.float32) / float(white_level)
    if out_dtype == np.uint8:
        bgr_out = to_unit8(bgr_srgb01)
    else:
        bgr_out = to_unit16(bgr_srgb01)
    time_end = time.time()
    # print(f"WB/CC/gamma time: {time_end - time_lsc:.4f} seconds")
    return bgr_out, {"black_levels": black_levels, "strengths": strengths, "gain_map": gain_map}

def safe_average_raw16(img_a: np.ndarray, img_b: np.ndarray, 
                       weight_a: float = 0.5, weight_b: float = 0.5, 
                       white_level: int = 65535, black_level: int = 0) -> np.ndarray:
    """
    Safely compute a weighted average of two RAW16 (uint16) Bayer images 
    without overflow.
    Parameters
    img_a, img_b : np.ndarray Input Bayer images, dtype=uint16, same shape.
    weight_a, weight_b : float Weights for each image, default 0.5 and 0.5. 
    white_level : int Sensor white level (default 65535). 
    black_level : int Sensor black level (default 0).  
    Returns
    -------
    img_out : np.ndarray
        Averaged Bayer image, dtype=uint16, clipped to [black_level, white_level].
    """
    assert img_a.shape == img_b.shape, "Input images must have same shape"
    assert img_a.dtype == np.uint16 and img_b.dtype == np.uint16, "Inputs must be uint16 Bayer images"

    # convert to uint32 to avoid overflow
    img_a_32 = img_a.astype(np.uint32)
    img_b_32 = img_b.astype(np.uint32)

    # interpolation in unint32
    img_out_f32 = img_a_32 * weight_a + img_b_32 * weight_b

    # clip within uint16 sensor range
    img_out_f32 = np.clip(img_out_f32, black_level, white_level)

    # convert back to uint16
    img_out = img_out_f32.round().astype(np.uint16)
    return img_out

def safe_average_bgr8(img_a: np.ndarray, img_b: np.ndarray,
                      weight_a: float = 0.5, weight_b: float = 0.5) -> np.ndarray:

    assert img_a.shape == img_b.shape, "Input images must have the same shape"
    assert img_a.dtype == np.uint8 and img_b.dtype == np.uint8, "Inputs must be uint8"

    img_a_f = img_a.astype(np.float32)
    img_b_f = img_b.astype(np.float32)

    img_out_f = img_a_f * weight_a + img_b_f * weight_b

    img_out = np.clip(img_out_f, 0, 255).round().astype(np.uint8)

    return img_out
