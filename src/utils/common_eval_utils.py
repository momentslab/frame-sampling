"""
Common utility functions shared between VideoMME and MVBench evaluation scripts.

This module contains shared functions for video model evaluation including
prompt creation, answer extraction, and video configuration.
"""

from typing import Dict, Any, Optional
import json
from pathlib import Path
from models.utils import parse_frame_mode
from video_manager.global_video_info import video_info_cache


# Supported models
SUPPORTED_MODELS = ["ovis", "smolvlm", "qwen2", "qwen2_5", "intern"]

# Answer options
ANSWER_OPTIONS = ['A', 'B', 'C', 'D']


def create_video_config(model_name: str, video_path: str, frame_mode: str) -> Dict[str, Any]:
    """
    Create video configuration for model inference.
    
    Args:
        model_name: Name of the model
        video_path: Path to the video file
        frame_mode: Frame mode configuration string
        
    Returns:
        Dictionary with video configuration
    """
    frame_config = parse_frame_mode(frame_mode)
    
    video_config = {
        "video": video_path,
        "return_extra": True if model_name == "smolvlm" else False,
        **frame_config
    }
    
    return video_config


def extract_answer(response: Optional[str]) -> Optional[str]:
    """
    Extract answer letter (A, B, C, or D) from model response.
    
    Args:
        response: Model response text
        
    Returns:
        Answer letter (A, B, C, or D) or None if not found
    """
    if not response:
        return None
    
    response_upper = response.upper().strip()
    
    # First, try to find answer at the start of response
    for option in ANSWER_OPTIONS:
        if response_upper.startswith(option):
            return option
    
    # Then, try to find answer anywhere in response
    for option in ANSWER_OPTIONS:
        if option in response_upper:
            return option
    
    print(f"⚠️ Could not extract valid answer from: {response}")
    return None


def is_answer_correct(predicted: Optional[str], ground_truth: Optional[str]) -> bool:
    """
    Check if predicted answer matches ground truth.
    
    Args:
        predicted: Predicted answer letter
        ground_truth: Ground truth answer letter
        
    Returns:
        True if answers match, False otherwise
    """
    if not predicted or not ground_truth:
        return False
    
    predicted_upper = predicted.strip().upper()
    ground_truth_upper = ground_truth.strip().upper()
    
    return predicted_upper == ground_truth_upper


def format_multiple_choice_options(options: list, option_format: str = "letter") -> str:
    """
    Format multiple choice options for prompt.
    
    Args:
        options: List of option texts
        option_format: Format type - "letter" (A, B, C, D) or "number" (1, 2, 3, 4)
        
    Returns:
        Formatted options string
    """
    if option_format == "letter":
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        formatted = []
        for i, option in enumerate(options):
            formatted.append(f"({letters[i]}) {option}")
        return "\n".join(formatted)
    elif option_format == "number":
        formatted = []
        for i, option in enumerate(options, 1):
            formatted.append(f"({i}) {option}")
        return "\n".join(formatted)
    else:
        return "\n".join(options)


def get_option_letter_for_answer(answer_text: str, options: list) -> Optional[str]:
    """
    Get the option letter (A, B, C, D) for a given answer text.

    Args:
        answer_text: The answer text to find
        options: List of option texts

    Returns:
        Option letter (A, B, C, D) or None if not found
    """
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for i, option in enumerate(options):
        if option == answer_text:
            return letters[i]
    return None


def save_frame_indices_json(output_dir: str = "./frame_indices", benchmark_name: str = "MVBench", mode: str = None, model_name: str = None) -> None:
    """
    Save frame indices for all processed videos to a JSON file.

    Args:
        output_dir: Directory to save the JSON file (default: "./frame_indices")
        benchmark_name: Name of the benchmark (default: "MVBench")
        mode: Frame mode used (optional)
        model_name: Name of the model (optional) - creates model-specific subdirectory
    """
    frames_data = {}

    for video_path, info in video_info_cache.items():
        # Parse the indices string to a list
        indices_str = info.get('Indices', '[]')
        try:
            indices = eval(indices_str)
        except:
            indices = []

        frames_data[video_path] = {
            "frames": indices,
            "num_frames": info.get('Nframes', 0),
            "total_frames": info.get('Total_frames', 0),
            "fps": info.get('Video_fps', 0)
        }

    # Create output path with model subdirectory if model_name is provided
    if model_name:
        output_path = Path(output_dir) / benchmark_name / model_name
    else:
        output_path = Path(output_dir) / benchmark_name

    output_path.mkdir(parents=True, exist_ok=True)

    # Create filename with mode if provided
    if mode:
        mode_str = mode.replace(":", "_").replace("/", "_")
        filename = f"{mode_str}.json"
    else:
        filename = "default.json"

    output_file = output_path / filename

    with open(output_file, 'w') as f:
        json.dump(frames_data, f, indent=2)

    print(f"✅ Frame indices saved to: {output_file}")

