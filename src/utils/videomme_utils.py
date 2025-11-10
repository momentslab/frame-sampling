"""
VideoMME-specific utility functions for video model evaluation.

This module contains functions specific to VideoMME dataset evaluation.
"""

import re
from pathlib import Path
from typing import Dict, Optional


def build_video_id_map(video_dir: str) -> Dict[str, str]:
    """
    Build a mapping of video IDs to video file paths.
    
    Searches for video files in the given directory and subdirectories,
    extracting video IDs from filenames.
    
    Args:
        video_dir: Path to directory containing video files
        
    Returns:
        Dictionary mapping video IDs to file paths
    """
    video_map = {}
    video_dir = Path(video_dir)

    # Check if video_dir itself contains video files (not subdirectories)
    for ext in ['mp4', 'avi', 'mov', 'mkv']:
        for file in video_dir.glob(f"*.{ext}"):
            # Extract video ID from filename (remove extension)
            video_id = file.stem.lower()
            video_map[video_id] = str(file)

    # Also check subdirectories for videos with numbered prefix pattern
    for subdir in video_dir.iterdir():
        if subdir.is_dir():
            for ext in ['mp4', 'avi', 'mov', 'mkv']:
                for file in subdir.glob(f"*.{ext}"):
                    # Try numbered prefix pattern first
                    match = re.search(r'^(\d+)_(.+?)\.(mp4|avi|mov|mkv)$', file.name)
                    if match:
                        video_id = match.group(2).lower()
                        video_map[video_id] = str(file)
                    else:
                        # Fall back to just using the stem
                        video_id = file.stem.lower()
                        video_map[video_id] = str(file)

    return video_map


def find_video_file_by_id(video_id: str, video_id_map: Dict[str, str]) -> str:
    """
    Find video file path by video ID.
    
    Args:
        video_id: Video ID to search for
        video_id_map: Mapping of video IDs to file paths
        
    Returns:
        Path to video file
        
    Raises:
        FileNotFoundError: If video file not found
    """
    stripped_id = video_id.strip()
    if stripped_id in video_id_map:
        print("✅ Found match!")
        return video_id_map[stripped_id]
    else:
        raise FileNotFoundError(f"❌ Video file not found for ID: '{video_id}'")


def create_videomme_prompt(question_data: Dict) -> str:
    """
    Create a prompt for VideoMME multiple-choice question.
    
    Args:
        question_data: Dictionary containing 'question' and 'options' keys
        
    Returns:
        Formatted prompt string
    """
    question = question_data["question"]
    options = "\n".join(question_data["options"])
    return (
        "Select the best answer to the following multiple-choice question based on the video.\n"
        "Respond with only the letter (A, B, C, or D) of the correct option.\n\n"
        f"{question}\n{options}\n\nThe best answer is:"
    )

