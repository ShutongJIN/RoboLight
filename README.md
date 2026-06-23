# RoboLight

This repository is a toolkit for preparing RoboLight datasets for policy training. It currently focuses on two batch data-processing pipelines:

1. Interpolating between aligned datasets to generate new synthesized datasets.
2. Converting dataset images from HDR RAW space into RGB images suitable for policy training.

## Project Structure

- `data_process/dataset_interpolation.py` - Batch dataset synthesis by interpolating aligned trajectories and RAW images in each dataset.
- `data_process/batch_isp.py` - Batch RAW-to-PNG conversion with an ISP pipeline. It uses GPU acceleration through CuPy when available and falls back to CPU processing otherwise.
- `data_process/isp_on_raw_img.py` - Core ISP utilities for RAW image processing.

## Installation

```bash
conda create -n robolight python=3.10
conda activate robolight
pip install -r requirements.txt
```

`cupy-cuda11x` is used for optional GPU acceleration. If your machine does not have a compatible CUDA setup, remove or replace the CuPy package according to your CUDA version.

## Expected Dataset Layout

Both tools assume datasets are organized by episodes:

```text
data_root/
└── episodes/
    ├── episode0/
    │   ├── rgb_0/
    │   │   ├── image_0.npy
    │   │   └── ...
    │   └── state_action.json
    ├── episode1/
    └── ...
```

The RAW images are expected to be saved as 16-bit Bayer `.npy` arrays under each episode's `rgb_0/` directory.

## 1. Interpolation between datasets

`dataset_interpolation.py` fuses two or more aligned datasets into a new synthesized dataset. For each episode, it:

- Averages the corresponding `state_action.json` trajectories.
- Fuses matching RAW `.npy` images in `rgb_0/`.
- Saves the synthesized dataset to a new output directory.

Example usage:

```bash
python data_process/dataset_interpolation.py \
    --data_dirs \
        /path/to/dataset_1 \
        /path/to/dataset_2 \
        /path/to/dataset_3 \
    --dir_save /path/to/real_255_255_255_synth \
    --start_id 0 \
    --end_id 200
```

By default, all input datasets are interpolated with equal weights. To use custom fusion weights, pass one weight per input dataset:

```bash
python data_process/dataset_interpolation.py \
    --data_dirs \
        /path/to/dataset_1 \
        /path/to/dataset_2 \
        /path/to/dataset_3 \
    --dir_save /path/to/real_255_255_255_synth \
    --start_id 0 \
    --end_id 200 \
    --weights 0.5 0.3 0.2
```

The input datasets must share the same episode IDs, image names, and trajectory lengths within the selected episode range. This alignment is guaranteed during the collection of the `RoboLight_Real` datasets.

## 2. Batch RAW to PNG conversion

`batch_isp.py` converts all RAW `.npy` images in `rgb_0/` into `.png` images that is suitable for policy training. The output is written to `rgb_0_isped/` inside each episode.

```bash
python data_process/batch_isp.py \
    --root_dataset /path/to/data_root \
    --start_id 0 \
    --end_id 200 \
    --batch_size 1
```

For each episode, the script reads:

```text
data_root/episodes/episode{id}/rgb_0/*.npy
```

and writes:

```text
data_root/episodes/episode{id}/rgb_0_isped/*.png
```

Existing `.png` outputs are skipped, so the script can be resumed from an interrupted conversion.

## Notes

- `start_id` is inclusive and `end_id` is exclusive.
- Keep dataset folders aligned before synthesis; missing episodes, mismatched image names, or different trajectory lengths will cause warnings or errors.
- The default ISP settings are defined in `data_process/batch_isp.py` and `data_process/isp_on_raw_img.py`.


