from video_model_research.ai_models import VideoModel
from transformers import AutoModelForImageTextToText, AutoProcessor
import torch
from video_model_research.common.video_backend_manager import video_backend_manager
from video_model_research.custom_read_video import my_custom_read_video_torchvision
import transformers.processing_utils as transformers_utils
import logging
import os

logger = logging.getLogger(__name__)


class SmolVLM(VideoModel):
    """A video model implementation for SmolVLM with unified backend management and multi-GPU support."""

    def __init__(self):
        video_backend_manager.register_and_patch_module_function(
            target_module=transformers_utils,
            function_name="load_video",
            custom_backend_name="smolvlm_unified",
            backend_func=my_custom_read_video_torchvision,
            description="SmolVLM unified video backend"
        )

        self._setup_model()

    def _setup_model(self):
        """Initialize the SmolVLM model components with multi-GPU support."""
        model_path = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        
        self.processor = AutoProcessor.from_pretrained(model_path)

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.cuda.set_per_process_memory_fraction(0.9)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
                device_map="balanced" if torch.cuda.device_count() > 1 else "auto"
            )
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)

    def predict(self, video_items: dict, prompt: str, max_tokens: int) -> str:
        """Predict the action in the video."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "path": video_items},
                    {"type": "text", "text": prompt}
                ]
            },
        ]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=torch.bfloat16)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_tokens
            )

        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]

        assistant_prefix = "Assistant: "
        if assistant_prefix in generated_texts:
            return generated_texts.split(assistant_prefix, 1)[1].strip()
        else:
            return generated_texts.strip()

    def cleanup(self):
        """Clean up resources and restore original backends."""
        video_backend_manager.restore_module_function(transformers_utils, "load_video")
        logger.info("SmolVLM cleanup completed")

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
