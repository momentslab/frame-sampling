from video_model_research.ai_models import VideoModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
import qwen_vl_utils.vision_process as vp
import torch
from video_model_research.custom_read_video import my_custom_read_video_torchvision
from video_model_research.common.video_backend_manager import video_backend_manager
import logging
import os

logger = logging.getLogger(__name__)



class Qwen2_5(VideoModel):
    """A video model implementation for Qwen2-VL with unified backend management."""

    def __init__(self):
        video_backend_manager.register_and_patch_backend(
            target_module=vp,
            backend_name="torchcodec",
            custom_backend_name="qwen25_torchcodec",
            backend_func=my_custom_read_video_torchvision,
            description="Qwen2.5 optimized TorchCodec video reader"
        )
        self._setup_model()

    def _setup_model(self):
        """Initialize the Qwen2.5-VL model components."""
        model_path = "Qwen/Qwen2.5-VL-3B-Instruct"

        self.processor = AutoProcessor.from_pretrained(model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)  # 20% memory savings
            torch.cuda.set_per_process_memory_fraction(0.9)  # Prevent GPU 0 overload
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="balanced" if torch.cuda.device_count() > 1 else None,
                attn_implementation="flash_attention_2"  
            )
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)


    def predict(self, video_items: dict, prompt: str, max_tokens: int) -> str:
        """Predict the action in the video."""
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "max_pixels": 360 * 420,
                    **video_items
                },
                
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }]      
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = vp.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_tokens)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        result = output_text[0]
        return result

    def cleanup(self):
        """Clean up resources and restore original backends."""
        video_backend_manager.restore_backend(vp, "torchcodec")
        logger.info("Qwen2 cleanup completed")

    def __del__(self):
        """Ensure cleanup on deletion."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup
