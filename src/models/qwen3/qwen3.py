import gc
import logging
import os

import torch
import qwen_vl_utils.vision_process as vp
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from models.ai_models import VideoModel
from video_manager.custom_read_video import my_custom_read_video_torchvision_qwen_wrapper
from video_manager.video_backend_manager import video_backend_manager

logger = logging.getLogger(__name__)
CUSTOM_QWEN_VIDEO_BACKEND = "qwen3_custom_video_reader"


class Qwen3(VideoModel):
    """Qwen3-VL wrapper using the custom Qwen video loader path."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        attn_implementation: str = "flash_attention_2",
        video_min_pixels: int = 4 * 32 * 32,
        video_max_pixels: int = 64 * 32 * 32,
        device: str | None = None,
        qwen3_root: str | None = None,
        **kwargs,
    ):
        self.model_id = model_id
        self.attn_implementation = attn_implementation
        self.video_min_pixels = video_min_pixels
        self.video_max_pixels = video_max_pixels
        self.qwen3_root = qwen3_root
        self.unused_kwargs = dict(kwargs)

        if self.unused_kwargs:
            logger.warning(
                "Ignoring unsupported Qwen3 kwargs: %s",
                sorted(self.unused_kwargs),
            )

        self.device = self._resolve_device(device)
        self._configure_cuda_runtime()

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = self._load_model()
        self.model.eval()

    def _resolve_device(self, device: str | None) -> torch.device:
        if device:
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _configure_cuda_runtime(self) -> None:
        if self.device.type != "cuda":
            return

        try:
            torch.backends.cuda.enable_flash_sdp(True)
        except Exception as exc:
            logger.debug("Could not enable flash SDP backend: %s", exc)

        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception as exc:
            logger.debug("Could not enable memory-efficient SDP backend: %s", exc)


    def _load_model(self):
        logger.info("Loading Qwen3-VL model %s on %s", self.model_id, self.device)

        model_kwargs: dict[str, object] = {
            "torch_dtype": torch.bfloat16 if self.device.type == "cuda" else "auto"
        }

        if self.attn_implementation and self.device.type == "cuda":
            model_kwargs["attn_implementation"] = self.attn_implementation

        if self.device.type == "cuda":
            model_kwargs["device_map"] = str(self.device)
            return Qwen3VLForConditionalGeneration.from_pretrained(
                self.model_id,
                **model_kwargs,
            )

        model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        return model.to(self.device)

    def _build_messages(self, video_items: dict, prompt: str) -> list[dict[str, object]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        **video_items,
                        "min_pixels": self.video_min_pixels,
                        "max_pixels": self.video_max_pixels,
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            }
        ]

    def _move_to_device(self, inputs: dict[str, object]) -> dict[str, object]:
        return {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def _should_patch_qwen_video_backend(self, messages: list[dict[str, object]]) -> bool:
        for message in messages:
            for content in message.get("content", []):
                if isinstance(content, dict) and isinstance(content.get("video"), str):
                    return True
        return False

    def _process_vision_info(self, messages: list[dict[str, object]]):
        backend_name = None
        should_restore = False

        if self._should_patch_qwen_video_backend(messages):
            try:
                backend_name = vp.get_video_reader_backend()
                if backend_name in vp.VIDEO_READER_BACKENDS:
                    video_backend_manager.register_and_patch_backend(
                        vp,
                        backend_name=backend_name,
                        custom_backend_name=CUSTOM_QWEN_VIDEO_BACKEND,
                        backend_func=my_custom_read_video_torchvision_qwen_wrapper,
                        description="Custom Qwen video wrapper for process_vision_info",
                    )
                    should_restore = True
            except Exception as exc:
                logger.warning("Could not patch Qwen video backend: %s", exc)

        try:
            return vp.process_vision_info(
                messages,
                image_patch_size=self.processor.image_processor.patch_size,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
        finally:
            if should_restore and backend_name is not None:
                try:
                    video_backend_manager.restore_backend(vp, backend_name)
                except Exception as exc:
                    logger.debug("Could not restore Qwen video backend %s: %s", backend_name, exc)

    def _prepare_inputs(self, video_items: dict, prompt: str) -> dict[str, torch.Tensor]:
        messages = self._build_messages(video_items, prompt)

        rendered_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs, video_kwargs = self._process_vision_info(messages)

        if video_inputs is not None:
            videos, video_metadatas = zip(*video_inputs)
            videos = list(videos)
            video_metadatas = list(video_metadatas)
        else:
            videos, video_metadatas = None, None

        model_inputs = self.processor(
            text=[rendered_prompt],
            images=image_inputs,
            videos=videos,
            video_metadata=video_metadatas,
            padding=True,
            return_tensors="pt",
            do_resize=False,   # important: qwen_vl_utils already resized
            **video_kwargs,
        )

        return self._move_to_device(model_inputs)

    def predict(self, video_items: dict, prompt: str, max_tokens: int) -> str:
        inputs = self._prepare_inputs(video_items, prompt)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                use_cache=True,
            )

        generated_suffix = generated_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated_suffix,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def cleanup(self):
        if hasattr(self, "model"):
            del self.model
            self.model = None

        if hasattr(self, "processor"):
            del self.processor
            self.processor = None

        gc.collect()

        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass