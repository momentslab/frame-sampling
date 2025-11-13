# Video Frame Sampling Benchmark

[![arXiv](https://img.shields.io/badge/arXiv-2509.14769-B31B1B?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2509.14769)

Open-source implementation of the experiments described in the paper *Frame Sampling Strategies Matter: A Benchmark for small vision language models*. The project evaluates modern vision-language models under a range of frame selection strategies to understand the trade-offs between temporal coverage, inference cost, and descriptive quality.

## Highlights
- Unified video backend with configurable sampling strategies (`first`, `center`, `fps`, `maxinfo`, `csta`).
- Ready-to-run wrappers for multi-modal models such as SmolVLM, Qwen2, Qwen2.5, InternVL, and Ovis.

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

#### Download Datasets
```bash
# Download Video-MME dataset (~101 GB)
pdm run download_data VideoMME

# Download MVBench dataset
pdm run download_data MVBench
```

#### Run Video-MME Evaluation
```bash
# Basic evaluation with OVIS model (default mode: fps:2:4:96)
pdm run vlm_VideoMME --model ovis --video_dir ./data/VideoMME/data

# Test mode (limited samples)
pdm run vlm_VideoMME --model ovis --video_dir ./data/VideoMME/data --test

# Custom model and frame sampling
pdm run vlm_VideoMME --model qwen2_5 --mode fps:1:4:96 --video_dir /path/to/videos
```

#### Run MVBench Evaluation
```bash
# Basic evaluation (default mode: fps:2:4:96)
pdm run vlm_MVBench --model ovis --data_dir ./data/MVBench

# Test mode with custom output
pdm run vlm_MVBench --model smolvlm --test --output_dir ./results/test_run

# Different frame sampling strategies
pdm run vlm_MVBench --model intern --mode maxinfo:1000:96 --data_dir /path/to/mvbench
```


#### Available Options
- **Models**: `ovis`, `smolvlm`, `qwen2`, `qwen2_5`, `intern`
- **Frame modes**: 
  - Single frame: `first`, `center`
  - Multi-frame: `fps:rate:min:max` (e.g., `fps:2:4:96`)
  - Max info: `maxinfo:input:max` (e.g., `maxinfo:1000:96`)
  - CSTA: `csta:input:max` (e.g., `csta:1000:96`)
- **Other options**: `--test`, `--video_dir`/`--data_dir`, `--output_dir`

## Citation

If you use this repository in academic work, please cite:

```bibtex
@article{brkic2025frame,
  title={Frame Sampling Strategies Matter: A Benchmark for small vision language models},
  author={Brkic, Marija and Razzouki, Anas Filali and Tevissen, Yannis and Guetari, Khalil and Yacoubi, Mounim A El},
  journal={arXiv preprint arXiv:2509.14769},
  year={2025}
}
```