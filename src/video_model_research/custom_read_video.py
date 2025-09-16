import torch
import time
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
from video_model_research.maxinfo_selection import maxinfo_frame_selection
from .CSTA.config import get_config
from .CSTA.generate_video import main
from .CSTA.config import get_config
from .CSTA.generate_video import main
logger = logging.getLogger(__name__)

global_frame_count = 0
global_call_count = 0

from video_model_research.global_video_info import video_info_cache


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
    video_path = ele["video"]
    return_extra = ele.get("return_extra", False)
    single_frame_mode = ele.get("single_frame", None)
    selection_method = ele.get("selection_method", "fps")
    num_threads = int(os.environ.get('TORCHCODEC_NUM_THREADS', 8))

    decoder = VideoDecoder(video_path, num_ffmpeg_threads=num_threads)
    video_fps = decoder.metadata.average_fps
    total_frames = decoder.metadata.num_frames

    if single_frame_mode:
        frame_indices, num_frames = _get_single_frame_indices_optimized(
            single_frame_mode, total_frames, ele
        )
    else:
        if selection_method == "fps":
            start_frame, end_frame, total_frames = calculate_video_frame_range(
                ele, total_frames, video_fps
            )
            frame_indices, num_frames = _get_multi_frame_indices(
                ele, start_frame, end_frame, total_frames, video_fps
            )
            print(f"fps: {num_frames=}")
            
        elif selection_method == "maxinfo":
            frame_indices, num_frames = _get_multi_frame_indices_maxinfo(
                decoder, total_frames,
                max_frames=ele.get("max_frames"),
                max_input_frames=ele.get("max_input_frames")
            )
            
            print(f"maxinfo: {num_frames=}")
            global_frame_count += num_frames
            global_call_count += 1
            average_frames = global_frame_count / global_call_count
            print(f"Average frames per call: {average_frames:.2f}")
            

    elif selection_method == "csta":
           frame_indices, num_frames = _get_multi_frame_indices_csta(
                decoder, total_frames, video_fps,
                max_frames=ele.get("max_frames"),
                max_input_frames=ele.get("max_input_frames"),
           )
           
           print(f"csta: {num_frames=}")
           global_frame_count += num_frames
           global_call_count += 1
           average_frames = global_frame_count / global_call_count
           print(f"Average frames per call: {average_frames:.2f}")
           
        else:
            raise ValueError(f"Invalid selection_method: {selection_method}. Use 'fps', 'maxinfo', or 'csta'")

    video_tensor = decoder.get_frames_at(indices=frame_indices).data

    _cache_video_info(video_path, video_fps, total_frames, num_frames, frame_indices)

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
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int).tolist()
    else:
        indices = list(range(total_frames))

    if not indices:
        raise ValueError("No indices provided")

    from torchvision.transforms import ToPILImage

    batch_size = 100
    to_pil = ToPILImage()
    pil_frames = []

    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i:i + batch_size]
        batch_frames = decoder.get_frames_at(indices=batch_indices).data

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

def _get_multi_frame_indices_maxinfo(decoder, total_frames, max_frames=96, max_input_frames=1000):
    """Get indices for multi-frame extraction using MaxInfo algorithm."""
    start = time.time()

    all_frames, initial_indices = call_frames(decoder, total_frames, max_frames=max_input_frames)

    selected_indices = maxinfo_frame_selection(
        all_frames,
        R=8,
        tol=0.3,
        max_n=min(max_frames, total_frames),
        normalize_svd_output='l2'
    )

    final_indices = [initial_indices[i] for i in sorted(selected_indices)]
    end = time.time()

    print(f"Execution time: {end - start:.4f} seconds")

    return final_indices, len(final_indices)

def _get_multi_frame_indices_csta(decoder, total_frames, video_fps, max_frames=96, max_input_frames=1000):
    """Get indices for multi-frame extraction using CSTA model."""
    
    start = time.time()
    all_frames, initial_indices = call_frames(decoder, total_frames, max_frames=max_input_frames)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    config = get_config(parse=False,
        input_is_file=True,
        weight_path='./src/video_model_research/CSTA/weights/SumMe/split4.pt',
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

    end = time.time()

    print(f"Execution time: {end - start:.4f} seconds")
    return final_indices, len(final_indices)


def _cache_video_info(video_path, video_fps, total_frames, num_frames, frame_indices):
    """Cache video processing information for debugging."""
    video_info_cache[video_path] = {
        "Video_fps": int(video_fps),
        "Total_frames": total_frames,
        "Nframes": num_frames,
        "Indices": str(frame_indices)
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

