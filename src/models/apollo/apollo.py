import logging
import math
import os
import sys
from pathlib import Path

import torch

from models.ai_models import VideoModel
from video_manager.custom_read_video import my_custom_read_video_torchvision

REPO_SRC = Path(__file__).resolve().parents[4] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.append(str(REPO_SRC))

from moments_vlm.datasets.custom_image_utils import (
    is_av_available,
    is_cv2_available,
    is_decord_available,
    is_torchcodec_available,
    is_torchvision_available,
)
from moments_vlm.inference.batch_inference import ApolloBatchInference

logger = logging.getLogger(__name__)
REPO_DEFAULT_VIDEO_BACKEND = "torchcodec"


class Apollo(ApolloBatchInference, VideoModel):
    """Apollo wrapper that reuses the top-level inference path and only overrides frame selection."""

    def __init__(self, model_path: str, mode: str | None = None):
        if not model_path:
            raise ValueError("Apollo requires a checkpoint path via model_path")

        self.mode = mode
        device = self._resolve_device()
        super().__init__(
            model_path=model_path,
            device=str(device),
            device_map=str(device),
            batch_size=1,
            temperature=0.0,
            do_sample=False,
        )
        self.model.config.encode_batch_size = 16
        self.config.encode_batch_size = 16
        self.generation_kwargs.pop("top_p", None)
        self.data_loader.video_backend = self._resolve_video_backend(self.data_loader.video_backend)

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _resolve_video_backend(self, preferred_backend: str) -> str:
        backend_checks = {
            "torchcodec": is_torchcodec_available,
            "decord": is_decord_available,
            "torchvision": is_torchvision_available,
            "opencv": is_cv2_available,
            "pyav": is_av_available,
        }
        if backend_checks[REPO_DEFAULT_VIDEO_BACKEND]():
            if preferred_backend != REPO_DEFAULT_VIDEO_BACKEND:
                logger.info(
                    "Apollo overriding checkpoint video backend '%s' with frame-sampling default '%s'",
                    preferred_backend,
                    REPO_DEFAULT_VIDEO_BACKEND,
                )
            return REPO_DEFAULT_VIDEO_BACKEND

        if backend_checks.get(preferred_backend, lambda: False)():
            return preferred_backend

        for backend in ("torchvision", "opencv", "pyav", "decord"):
            if backend_checks[backend]():
                logger.warning(
                    "Apollo requested video backend '%s' but it is unavailable; falling back to '%s'",
                    preferred_backend,
                    backend,
                )
                return backend

        raise ImportError("Apollo requires at least one available video backend (torchcodec, torchvision, opencv, pyav, or decord)")

    def _configure_request(self, video_items: dict) -> None:
        request = dict(video_items)
        self.data_loader.sample_indices_fn = self._build_sample_indices_fn(request)
        if request.get("selection_method") == "clips":
            self.data_loader.frames_per_clip = max(int(request.get("frames_per_clip", self.data_loader.frames_per_clip)), 1)
            self.data_loader.max_clips_per_video = max(int(request.get("max_clips_per_video", 1)), 1)
            self.data_loader.target_fps = float(request.get("target_fps", self.data_loader.target_fps))
            self.data_loader.clip_sampling_ratio = float(request.get("clip_sampling_ratio", self.data_loader.clip_sampling_ratio))
            self.data_loader.video_sampling_type = "clip"
        elif request.get("single_frame"):
            self.data_loader.frames_per_clip = 1
            self.data_loader.max_clips_per_video = 1

    def _build_sample_indices_fn(self, request: dict):
        request = {**request, "return_extra": True}

        def sample_indices_fn(metadata, **kwargs):
            del kwargs
            _, sampled_metadata = my_custom_read_video_torchvision(request)
            sampled_indices = [
                int(i)
                for i in getattr(sampled_metadata, "frames_indices", [])
                if 0 <= int(i) < int(metadata.total_num_frames)
            ]
            if not sampled_indices:
                raise ValueError(f"Apollo received no frame indices from custom_read_video for {request['video']}")

            if request.get("selection_method") == "clips":
                frames_per_clip = max(int(request.get("frames_per_clip", self.data_loader.frames_per_clip)), 1)
                self.data_loader.frames_per_clip = frames_per_clip
                self.data_loader.max_clips_per_video = max(1, math.floor(len(sampled_indices) / frames_per_clip))
            elif request.get("single_frame"):
                self.data_loader.frames_per_clip = 1
                self.data_loader.max_clips_per_video = 1
            else:
                self.data_loader.frames_per_clip = len(sampled_indices)
                self.data_loader.max_clips_per_video = 1
            return sampled_indices

        return sample_indices_fn

    def predict(self, video_items: dict, prompt: str, max_tokens: int) -> str:
        self._configure_request(video_items)
        self.generation_kwargs["max_new_tokens"] = max_tokens
        return self.generate_single(prompt, [video_items["video"]])
