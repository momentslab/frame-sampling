# Video Frame Sampling Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2509.14769-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.14769)

Open-source implementation of the experiments described in the paper *Frame Sampling Strategies Matter: A Benchmark for small vision language models*. The project evaluates modern vision-language models under a range of frame selection strategies to understand the trade-offs between temporal coverage, inference cost, and descriptive quality.

## Highlights
- Unified video backend with configurable sampling strategies (`first`, `center`, `fps`, `maxinfo`, `csta`).
- Ready-to-run wrappers for multi-modal models such as SmolVLM, Qwen2, Qwen2.5, InternVL, and Ovis.
- Benchmarking utilities that compute BLEU, ROUGE, METEOR, CIDEr, and BERTScore, plus optional Video-MME evaluation helpers.
- Scripts for model-to-model comparisons, FPS sensitivity analysis, and single-frame baselines.

## Installation

### Using PDM (Recommended)
```bash
# Install the project and its dependencies
pdm install

# Activate the virtual environment created by PDM
pdm shell
```

### Alternative: Using pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The dependency list is defined in `pyproject.toml`. PDM is recommended as it provides better dependency management and includes convenient scripts for running the benchmarks. GPU execution is recommended for the heavier models.

## Usage

### Using PDM scripts (Recommended)
If you installed with PDM, you can use the convenient scripts defined in `pyproject.toml`:

```bash
# Benchmark multiple models
pdm run bench \
  --folder_path /path/to/videos \
  --output_path ./outputs/model_comparison \
  --mode fps:1:4:96 \
  --prompt "Give a short description of the video:"

# Quick inference helper
pdm run vlm \
  --model_name ovis \
  --video_path /path/to/video.mp4 \
  --mode maxinfo:1000:96

# Single-frame baselines
pdm run bench_single \
  --folder_path /path/to/videos \
  --output_path ./outputs/single_frame

# Video-MME evaluation
pdm run vlm_VideoMME
pdm run vlm_VideoMME_evaluation
```

### Direct script execution
You can also run the scripts directly:

```bash
# Benchmark multiple models
python scripts/benchmark_models.py \
  --folder_path /path/to/videos \
  --output_path ./outputs/model_comparison \
  --mode fps:1:4:96 \
  --prompt "Give a short description of the video:"

# Sweep FPS settings for a single model
python scripts/benchmark_fps_model.py \
  --model qwen2_5 \
  --folder_path /path/to/videos \
  --output_path ./outputs/qwen2_5_fps \
  --fps_configs 0.25,0.5,1,2,3

# Single-frame baselines
python scripts/benchmark_single_frame.py \
  --folder_path /path/to/videos \
  --output_path ./outputs/single_frame

# Quick inference helper
python scripts/vlm.py \
  --model_name ovis \
  --video_path /path/to/video.mp4 \
  --mode maxinfo:1000:96
```

### Video-MME evaluation
The folder `Video_MME/` contains helpers for reproducing the Video-MME leaderboard submission. Use `VideoMME_results.txt` together with `scripts/vlm_VideoMME.py` and `scripts/vlm_VideoMME_evaluation.py` to regenerate predictions and compute category-wise accuracies.

## Repository Layout
- `src/video_model_research/`: Core library code (frame sampling, model wrappers, metrics, and utilities).
- `scripts/`: Command-line entry points used in the ICASSP experiments.

## Citation
If you use this repository in academic work, please cite 