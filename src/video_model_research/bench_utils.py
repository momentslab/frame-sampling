"""
Common utility functions for video model benchmarking.

This module contains truly common helper functions that can be used by any benchmark script
without experiment-specific logic.
"""

import logging
import os
import sys
import gc
import time
import pandas as pd
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any

from .smolvlm.smolvlm import SmolVLM
from .qwen2_5.qwen2_5 import Qwen2_5
from .qwen2.qwen2 import Qwen2
from .intern.intern import Intern
from .ovis.ovis import Ovis
from .global_video_info import video_info_cache
from .metrics import compute_scores_from_df

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_frame_mode(mode):
    """
    Parse frame mode and return video configuration.

    Args:
        mode: String - either single frame ("first", "center") or multi-frame ("fps", "maxinfo", "csta")

    Returns:
        dict: Video configuration parameters
    """
    config = {}

    if mode in ["first", "center"]:
        logger.info(f"🖼️  Single frame mode: {mode}")
        config["single_frame"] = mode
        return config

    elif mode.startswith("fps:"):
        try:
            parts = mode.split(":")
            if len(parts) == 4 and parts[0] == "fps":
                fps_val, min_val, max_val = float(parts[1]), int(parts[2]), int(parts[3])
                logger.info(f"🎬 Multi-frame mode: fps with custom params (fps={fps_val}, min={min_val}, max={max_val})")
                config.update({
                    "selection_method": "fps",
                    "fps": fps_val,
                    "min_frames": min_val,
                    "max_frames": max_val
                })
                return config
            else:
                raise ValueError("Invalid format")
        except (ValueError, IndexError):
            raise ValueError(f"Invalid fps format: {mode}. Use 'fps:N:min:max' (e.g., 'fps:1:4:96')")

    elif mode.startswith("maxinfo:"):
        try:
            parts = mode.split(":")
            if len(parts) == 3 and parts[0] == "maxinfo":
                max_input_val, max_val = int(parts[1]), int(parts[2])
                logger.info(f"🧠 MaxInfo mode: max_input={max_input_val}, max={max_val}")
                config.update({
                    "selection_method": "maxinfo",
                    "max_input_frames": max_input_val,
                    "max_frames": max_val
                })
                return config
            else:
                raise ValueError("Invalid format")
        except (ValueError, IndexError):
            raise ValueError(f"Invalid maxinfo format: {mode}. Use 'maxinfo:input:max' (e.g., 'maxinfo:1000:96')")

    elif mode.startswith("csta:"):
        try:
            parts = mode.split(":")
            if len(parts) == 3 and parts[0] == "csta":
                max_input_val, max_val = int(parts[1]), int(parts[2])
                logger.info(f"🎯 CSTA mode: max_input={max_input_val}, max={max_val}")
                config.update({
                    "selection_method": "csta",
                    "max_input_frames": max_input_val,
                    "max_frames": max_val
                })
                return config
            else:
                raise ValueError("Invalid format")
        except (ValueError, IndexError):
            raise ValueError(f"Invalid csta format: {mode}. Use 'csta:input:max' (e.g., 'csta:1000:96')")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'first', 'center', 'fps:N:min:max', 'maxinfo:input:max', or 'csta:input:max'")


def setup_logging(output_path, log_filename="benchmark.log"):
    """
    Configure logging to write to the output directory.

    Args:
        output_path: Directory where log file should be created
        log_filename: Name of the log file (default: "benchmark.log")

    Returns:
        logger: Configured logger instance
    """
    log_file_path = os.path.join(output_path, log_filename)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file_path),
            logging.StreamHandler(sys.stdout)
        ],
        force=True  # Override any existing logging configuration
    )

    from transformers.utils import logging as transformers_logging
    transformers_logging.set_verbosity_error()

    return logging.getLogger(__name__)


def cleanup_memory():
    """Clean up GPU and system memory before loading new model."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def get_model(model_name: str):
    """Load and return the specified model with proper error handling."""
    logger.info(f"Loading {model_name} model...")
    start_load = time.time()
    
    cleanup_memory()
    
    try:
        if model_name == "smolvlm":
            model = SmolVLM()
        elif model_name == "qwen2":
            model = Qwen2()
        elif model_name == "qwen2_5":
            model = Qwen2_5()
        elif model_name == "intern":
            model = Intern()
        elif model_name == "ovis":
            model = Ovis()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        load_time = time.time() - start_load
        logger.info(f"{model_name} loaded successfully in {load_time:.2f}s")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        raise


def cleanup_model(model):
    """Properly cleanup model and free memory."""
    if hasattr(model, 'cleanup'):
        model.cleanup()
    del model
    cleanup_memory()





def process_model(model_name: str, data_config: Dict[str, Any], args, video_config: Dict = None, output_suffix: str = "") -> Optional[Dict[str, float]]:
    """Process videos with a specific model and configuration.

    Args:
        model_name: Name of the model to use
        data_config: Configuration containing video paths, metrics, etc.
        args: Arguments containing prompt, max_tokens, etc.
        video_config: Optional video configuration override (for FPS benchmarks)
        output_suffix: Optional suffix for output filename
    """
    logger.info(f"Starting inference with {model_name}")
    start_inference = time.time()

    model = get_model(model_name)

    if video_config is None:
        video_config = data_config['base_video_config'].copy()
        if model_name == "smolvlm":
            video_config["return_extra"] = True
    else:
        video_config = video_config.copy()

    batch_results = process_video_batch(
        data_config['video_paths'],
        video_config,
        model,
        args,
        data_config['progress_interval']
    )

    df = update_dataframe_with_results(data_config['df'], batch_results)

    if output_suffix:
        filename = f"{model_name}_{output_suffix}"
    else:
        filename = f"output_{model_name}"

    scores = save_and_compute_scores(df, args.output_path, filename, data_config['metrics'])

    cleanup_model(model)
    end_inference = time.time()
    inference_time = end_inference - start_inference
    logger.info(f"Inferences of {model_name} completed in {inference_time:.2f} seconds.")

    scores['inference_time'] = inference_time

    return scores


def load_video_files(folder_path: str) -> list:
    """Load video files from a folder."""
    video_paths = sorted(Path(folder_path).glob("*.mp4"))[:4]

    if not video_paths:
        raise ValueError(f"No video files found in {folder_path}")

    logger.info(f"Found {len(video_paths)} video files")
    return video_paths


def load_metadata_csv(csv_path: str = "./Asharq_data/metadata.csv") -> pd.DataFrame:
    """Load metadata CSV file."""
    try:
        df = pd.read_csv(csv_path, index_col=0)
        logger.info(f"Loaded metadata from {csv_path}")
        return df
    except FileNotFoundError:
        logger.warning(f"Metadata file {csv_path} not found, creating empty DataFrame")
        return pd.DataFrame()


def create_output_directory(output_path: str):
    """Create output directory if it doesn't exist."""
    os.makedirs(output_path, exist_ok=True)


def log_arguments(args, title: str = "Arguments"):
    """Log all arguments."""
    logger.info(f"{title}:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")


def process_video_batch(video_paths: list, video_config: dict, model, args, progress_interval: int = 40) -> list:
    """Process a batch of videos with a model and return results."""
    batch_results = []

    for idx, video_path in enumerate(video_paths):
        video_config["video"] = str(video_path)
        try:
            result = model.predict(video_items=video_config, prompt=args.prompt, max_tokens=args.max_tokens)
            info = video_info_cache.get(str(video_path), {})

            batch_results.append({
                'video_stem': video_path.stem,
                'prediction': result,
                'info': info
            })

        except Exception as e:
            logger.error(f"Cannot predict video file {str(video_path)}: {e}")
            batch_results.append({
                'video_stem': video_path.stem,
                'prediction': "",
                'info': {}
            })

        if (idx + 1) % progress_interval == 0:
            logger.info(f"Finished processing {idx + 1}/{len(video_paths)} videos.")

    return batch_results


def update_dataframe_with_results(df: pd.DataFrame, batch_results: list) -> pd.DataFrame:
    """Update DataFrame with batch results."""
    df_copy = df.copy()
    for result_data in batch_results:
        video_stem = result_data['video_stem']
        prediction = result_data['prediction']
        info = result_data['info']

        if info:
            df_copy.loc[video_stem, list(info.keys())] = list(info.values())
        df_copy.loc[video_stem, "Prediction"] = prediction

    return df_copy


def save_and_compute_scores(df: pd.DataFrame, output_path: str, filename: str, metrics: list) -> dict:
    """Save DataFrame and compute scores."""
    df = df.dropna()
    df.to_csv(os.path.join(output_path, f"{filename}.csv"))

    scores = compute_scores_from_df(df, metrics)
    return scores


def finalize_results_dataframe(results_df: pd.DataFrame, metrics: list, exclude_from_mean: list = None) -> pd.DataFrame:
    """Finalize results DataFrame with mean scores."""
    if exclude_from_mean is None:
        exclude_from_mean = []

    results_df[metrics] = results_df[metrics].apply(pd.to_numeric, errors="coerce")

    mean_metrics = [m for m in metrics if m not in exclude_from_mean]
    if mean_metrics:
        results_df["mean_scores"] = results_df[mean_metrics].mean(axis=1)

    return results_df.round(3)


