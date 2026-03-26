import torch
import time
import math
import torchvision
from packaging import version
import warnings
from torchvision import io
from qwen_vl_utils.vision_process import smart_nframes, calculate_video_frame_range
import transformers.image_utils as image_utils
import os
import logging
from torchcodec.decoders import VideoDecoder
import numpy as np
from adaptive_techniques.maxvolpy.maxinfo_selection import maxinfo_frame_selection
from adaptive_techniques.CSTA.config import get_config
from adaptive_techniques.CSTA.generate_video import main
logger = logging.getLogger(__name__)

global_frame_count = 0
global_call_count = 0

from video_manager.global_video_info import video_info_cache


def my_custom_read_video_torchvision_qwen_wrapper(ele: dict, **kwargs):
    """
    Wrapper for Qwen models that expect 3 return values: (video, video_metadata, sample_fps)
    Matches the format expected by qwen-vl-utils.vision_process
    """
    # Get video tensor and fps from base function
    video_tensor, sample_fps = my_custom_read_video_torchvision(ele, **kwargs)

    # Extract metadata from video_info_cache (populated by base function)
    video_path = ele["video"]
    cached_info = video_info_cache.get(video_path, {})

    # Create video_metadata dict in the format qwen-vl-utils expects
    video_metadata = dict(
        fps=cached_info.get("Video_fps", sample_fps),
        frames_indices=cached_info.get("Indices", []),
        total_num_frames=cached_info.get("Total_frames", video_tensor.shape[0]),
        video_backend="torchcodec",
    )

    return video_tensor, video_metadata, sample_fps


def my_custom_read_video_torchvision(ele: dict, **kwargs):
    """
    Read video using torchcodec.decoders.VideoDecoder with support for different frame selection methods.

    Args:
        ele (dict): Video configuration dictionary with supported keys:
            - video: Path to video file (supports "file://", "http://", "https://" and local paths)
            - video_start: Start time of video segment (used for multi-frame mode)
            - video_end: End time of video segment (used for multi-frame mode)
            - single_frame: Optional mode for single frame extraction ("first", "center", or None)
            - fps: Frames per second for multi-frame extraction (ignored in single_frame mode)
            - min_frames/max_frames: Frame count limits for multi-frame extraction (ignored in single_frame mode)
            - return_extra: Whether to return extra metadata (default: False)
            - selection_method: Frame selection method ("fps", "maxinfo", "csta", default: "fps")

    Returns:
        tuple: (video_tensor, metadata_or_fps)
            - video_tensor: torch.Tensor with shape (T, C, H, W) where T=1 for single frame mode
            - metadata_or_fps: VideoMetadata object if return_extra=True, else sample FPS float
    """
    global global_frame_count, global_call_count
    # Extract configuration parameters
    video_path = ele["video"]
    return_extra = ele.get("return_extra", False)
    single_frame_mode = ele.get("single_frame", None)
    selection_method = ele.get("selection_method", "fps")  # Default to fps selection
    num_threads = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))

    # Initialize video decoder and get metadata
    decoder = VideoDecoder(video_path, num_ffmpeg_threads=num_threads)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

    # Determine frame indices to extract based on mode
    if single_frame_mode:
        # For single frame, bypass complex frame range calculations
        frame_indices, num_frames = _get_single_frame_indices_optimized(
            single_frame_mode, total_frames, ele
        )
    else:
        # For multi-frame, use the specified selection method
        if selection_method == "fps":
            start_frame, end_frame, total_frames = calculate_video_frame_range(
                ele, total_frames, video_fps
            )
            frame_indices, num_frames = _get_multi_frame_indices(
                ele, start_frame, end_frame, total_frames, video_fps
            )
        elif selection_method == "maxinfo":
            frame_indices, num_frames = _get_multi_frame_indices_maxinfo(
                decoder, total_frames,
                max_frames=ele.get("max_frames"),
                max_input_frames=ele.get("max_input_frames")
            )
            
            global_frame_count += num_frames
            global_call_count += 1

        elif selection_method == "csta":
           frame_indices, num_frames = _get_multi_frame_indices_csta(
                decoder, total_frames, video_fps,
                max_frames=ele.get("max_frames"),
                max_input_frames=ele.get("max_input_frames"),
           )

           global_frame_count += num_frames
           global_call_count += 1

        elif selection_method == "uniform":
            frame_indices, num_frames = _get_uniform_frame_indices(ele, total_frames)

        elif selection_method == "clips":
            frame_indices, num_frames = _get_clip_frame_indices(ele, total_frames, video_fps)

        else:
            raise ValueError(f"Invalid selection_method: {selection_method}. Use 'fps', 'maxinfo', 'csta', 'uniform', or 'clips'")

    # Extract video frames
    video_tensor = decoder.get_frames_at(indices=frame_indices).data

    # Cache video information for debugging/analysis
    _cache_video_info(video_path, video_fps, total_frames, num_frames, frame_indices)

    # Return appropriate format based on return_extra flag
    if return_extra:
        metadata = _create_video_metadata(total_frames, video_fps, frame_indices)
        return video_tensor, metadata
    else:
        sample_fps = _calculate_sample_fps(num_frames, total_frames, video_fps)
        return video_tensor, sample_fps


def call_frames(decoder, total_frames, max_frames=1000):
    """
    Safely extract frames from decoder using batch processing.

    Args:
        decoder: Video decoder object
        total_frames: Total number of frames in the video
        max_frames: Maximum number of frames to process (default: 1000)

    Returns:
        list[PIL.Image.Image]: List of PIL Images
        list[int]: List of indices of presampled frames
        list[int]: List of indices of presampled frames
    """
    # Limit frames for memory efficiency
    if total_frames > max_frames:
        # Uniformly sample max_frames indices from the total frames
        indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int).tolist()
    else:
        indices = list(range(total_frames))

    if not indices:
        raise ValueError("No indices provided")

    from torchvision.transforms import ToPILImage

    batch_size = 100  # Process frames in batches
    to_pil = ToPILImage()
    pil_frames = []

    # Process and convert in single loop - no intermediate storage
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_frames = decoder.get_frames_at(indices=batch_indices).data

        # Convert each frame in the batch directly to PIL
        pil_frames.extend([to_pil(frame) for frame in batch_frames])

    return pil_frames, indices


def _get_single_frame_indices_optimized(mode, total_frames, _):
    """
    Get indices for single frame extraction with simple logic.

    Args:
        mode: "first" or "center"
        total_frames: Total number of frames in video
        ele: Video configuration dict (unused, kept for compatibility)

    Returns:
        tuple: (frame_indices_list, num_frames)
    """
    if mode == "first":
        return [0], 1
    elif mode == "center":
        return [total_frames // 2], 1
    else:
        raise ValueError(f"Invalid single_frame mode: {mode}. Use 'first' or 'center'")


def _get_multi_frame_indices(ele, start_frame, end_frame, total_frames, video_fps):
    """Get indices for multi-frame extraction."""
    num_frames = smart_nframes(ele, total_frames=total_frames, video_fps=video_fps)
    frame_indices = torch.linspace(start_frame, end_frame, num_frames).round().long().tolist()
    return frame_indices, num_frames


def _get_uniform_frame_indices(ele, total_frames):
    """Uniformly sample exactly N frames spread across the full video duration.

    Args:
        ele: Video configuration dict containing 'num_frames' (default 8).
        total_frames: Total number of frames in the video.

    Returns:
        tuple: (frame_indices_list, num_frames)
    """
    n = min(int(ele.get("num_frames", 8)), total_frames)
    frame_indices = torch.linspace(0, total_frames - 1, n).round().long().tolist()
    return frame_indices, n


def _get_clip_frame_indices(ele, total_frames, video_fps):
    """Sample frames from a video by splitting it into clips at a target FPS.

    Ported from MomentsVLM's clip_sample_indices logic.

    Args:
        ele: Video configuration dict containing:
            - frames_per_clip (int, default 8): frames sampled per clip
            - max_clips_per_video (int, default 32): maximum number of clips
            - target_fps (float, default 1.0): target sampling FPS
            - clip_sampling_ratio (float, default 1.0): controls clip density
        total_frames: Total number of frames in the video.
        video_fps: Native FPS of the video.

    Returns:
        tuple: (frame_indices_list, num_frames)
    """
    frames_per_clip = int(ele.get("frames_per_clip", 8))
    max_clips_per_video = int(ele.get("max_clips_per_video", 32))
    target_fps = float(ele.get("target_fps", 1.0))
    clip_sampling_ratio = float(ele.get("clip_sampling_ratio", 1.0))

    original_fps = video_fps
    video_duration = total_frames / original_fps if original_fps else 0

    if target_fps is None or target_fps <= 0:
        target_fps = original_fps

    # Duration of one clip at the target FPS
    clip_duration = frames_per_clip / target_fps

    # How many clips fit in the video
    desired_clips = math.ceil((video_duration / clip_duration) * clip_sampling_ratio) if clip_duration > 0 else 1
    num_clips = min(max(desired_clips, 1), max_clips_per_video)

    # How many original frames correspond to one target frame
    frame_step = original_fps / target_fps

    all_indices = []

    if frame_step > 0.5:
        # Normal case: target FPS <= original FPS
        frame_step_int = max(1, int(frame_step))
        clip_len = frames_per_clip * frame_step_int
        partition_len = total_frames // num_clips

        for i in range(num_clips):
            if partition_len > clip_len:
                start_idx = i * partition_len + (partition_len - clip_len) // 2
                indices = np.arange(start_idx, start_idx + clip_len, frame_step_int)
            else:
                sample_len = min(clip_len, total_frames)
                clip_step = (total_frames - sample_len) // max(1, num_clips - 1) if total_frames > sample_len else 0
                start_idx = i * clip_step
                indices = np.arange(start_idx, start_idx + sample_len, frame_step_int)

            if len(indices) > frames_per_clip:
                indices = indices[:frames_per_clip]
            elif len(indices) < frames_per_clip:
                last_valid = min(start_idx + clip_len - 1, total_frames - 1)
                padding = np.full(frames_per_clip - len(indices), last_valid)
                indices = np.concatenate((indices, padding))

            indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
            all_indices.extend(indices.tolist())

    else:
        # Low FPS case: need to repeat frames
        repeat_factor = int(np.ceil(1 / frame_step))
        clip_len = max(1, int(frames_per_clip * frame_step))
        sample_len = min(clip_len, total_frames)

        for i in range(num_clips):
            clip_step = (total_frames - sample_len) // max(1, num_clips - 1) if total_frames > sample_len else 0
            base_indices = np.arange(i * clip_step, i * clip_step + sample_len)
            indices = np.repeat(base_indices, repeat_factor)

            if len(indices) > frames_per_clip:
                indices = indices[:frames_per_clip]
            elif len(indices) < frames_per_clip:
                last_valid = min(i * clip_step + sample_len - 1, total_frames - 1)
                padding = np.full(frames_per_clip - len(indices), last_valid)
                indices = np.concatenate((indices, padding))

            indices = np.clip(indices, 0, total_frames - 1).astype(np.int64)
            all_indices.extend(indices.tolist())

    return all_indices, len(all_indices)


def _get_multi_frame_indices_maxinfo(decoder, total_frames, max_frames=96, max_input_frames=1000):
    """Get indices for multi-frame extraction using MaxInfo algorithm."""
    # Use ALL frames for maxinfo algorithm (up to max_input_frames limit)
    # Extract all frames safely using batch processing
    all_frames, initial_indices = call_frames(decoder, total_frames, max_frames=max_input_frames)

    # Apply maxinfo to select best frames from all frames
    selected_indices = maxinfo_frame_selection(
        all_frames,
        R=8,
        tol=0.3,
        max_n=min(max_frames, total_frames),
        normalize_svd_output='l2'
    )

    final_indices = [initial_indices[i] for i in sorted(selected_indices)]

    return final_indices, len(final_indices)

def _get_multi_frame_indices_csta(decoder, total_frames, video_fps, max_frames=96, max_input_frames=1000):
    """Get indices for multi-frame extraction using CSTA model."""
    
    start = time.time()
    all_frames, initial_indices = call_frames(decoder, total_frames, max_frames=max_input_frames)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    config = get_config(parse=False,
        input_is_file=True,
        weight_path='./src/adaptive_techniques/CSTA/weights/SumMe/split4.pt',
        device=device
    )

    frame_height = decoder.metadata.height
    frame_width = decoder.metadata.width
    selected_indices = main(
        all_frames, video_fps, frame_width, frame_height, config
        )
    
    final_indices = [initial_indices[i] for i in sorted(selected_indices)]

    if len(final_indices)>max_frames:
        indices = np.linspace(0, len(final_indices) - 1, num=max_frames, dtype=int).tolist()
        final_indices = [final_indices[i] for i in indices]

    return final_indices, len(final_indices)


def _cache_video_info(video_path, video_fps, total_frames, num_frames, frame_indices):
    """Cache video processing information for debugging."""
    video_info_cache[video_path] = {
        "Video_fps": int(video_fps),
        "Total_frames": total_frames,
        "Nframes": num_frames,
        "Indices": list(frame_indices)
    }


def _create_video_metadata(total_frames, video_fps, frame_indices):
    """Create VideoMetadata object with frame information."""
    duration = total_frames / video_fps if video_fps else 0
    metadata = image_utils.VideoMetadata(
        total_num_frames=int(total_frames),
        fps=float(video_fps),
        duration=float(duration),
        video_backend="io"
    )
    metadata.frames_indices = frame_indices
    return metadata


def _calculate_sample_fps(num_frames, total_frames, video_fps):
    """Calculate effective sampling FPS."""
    return num_frames / max(total_frames, 1e-6) * video_fps


