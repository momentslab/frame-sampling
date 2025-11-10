"""
Common utility functions for video model benchmarking.

This module contains essential helper functions for loading models and parsing frame configurations.
"""

import logging
import gc
import time
import torch

from .smolvlm.smolvlm import SmolVLM
from .qwen2_5.qwen2_5 import Qwen2_5
from .qwen2.qwen2 import Qwen2
from .intern.intern import Intern
from .ovis.ovis import Ovis

# Configure logger for this module
logger = logging.getLogger(__name__)


def parse_frame_mode(mode):
    """
    Parse frame mode and return video configuration.

    Args:
        mode: String - either single frame ("first", "center") or multi-frame ("fps", "maxinfo", "csta")

    Returns:
        dict: Video configuration parameters
    """
    config = {}

    # Single frame modes
    if mode in ["first", "center"]:
        logger.info(f"🖼️  Single frame mode: {mode}")
        config["single_frame"] = mode
        return config

    # FPS mode with parameters: fps:fps_val:min_frames:max_frames
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

    # MaxInfo mode: maxinfo:max_input_frames:max_frames
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

    # CSTA mode: csta:max_input_frames:max_frames
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
    
    # Clean memory before loading
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



