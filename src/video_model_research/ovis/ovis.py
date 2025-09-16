from video_model_research.ai_models import VideoModel
from transformers import AutoModelForCausalLM
import torch
from video_model_research.ovis.utils import load_video
import inspect
import os
import logging

logger = logging.getLogger(__name__)

class Ovis(VideoModel):
    """An optimized video model implementation for Ovis with multi-GPU support."""

    def __init__(self):
        self._setup_model()

    def _setup_model(self):
        """Initialize the Ovis model with optimized GPU configuration."""
        model_path = "AIDC-AI/Ovis2-2B"
        
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.9)  # Prevent GPU 0 overload
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=32768,
                trust_remote_code=True,
                device_map="auto" if torch.cuda.device_count() > 1 else None
            )
        else:
            self.device = torch.device("mps")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                multimodal_max_length=32768,
                trust_remote_code=True
            ).to(self.device)

        logger.info(f"Model loaded on devices: {self.model.hf_device_map if hasattr(self.model, 'hf_device_map') else 'single device'}")
        logger.info("Model defined in file: %s", inspect.getfile(type(self.model)))

    def predict(self, video_items: dict, prompt: str, max_tokens: int) -> str:
        """Predict the action in the video."""
        text_tokenizer = self.model.get_text_tokenizer()
        visual_tokenizer = self.model.get_visual_tokenizer()

        max_partition = 1
        images = load_video(video_items)
        query = '\n'.join(['<image>'] * len(images)) + '\n' + prompt
        
        prompt, input_ids, pixel_values = self.model.preprocess_inputs(
            query, 
            images, 
            max_partition=max_partition
        )
        
        attention_mask = torch.ne(input_ids, text_tokenizer.pad_token_id)
        input_ids = input_ids.unsqueeze(0).to(self.model.device)
        attention_mask = attention_mask.unsqueeze(0).to(self.model.device)

        if pixel_values is not None:
            pixel_values = pixel_values.to(
                dtype=visual_tokenizer.dtype, 
                device=visual_tokenizer.device
            )
        pixel_values = [pixel_values]

        with torch.inference_mode():
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "top_p": None,
                "top_k": None,
                "temperature": None,
                "repetition_penalty": None,
                "eos_token_id": self.model.generation_config.eos_token_id,
                "pad_token_id": text_tokenizer.pad_token_id,
                "use_cache": True
            }
            
            output_ids = self.model.generate(
                input_ids, 
                pixel_values=pixel_values, 
                attention_mask=attention_mask, 
                **gen_kwargs
            )[0]
            
            output = text_tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return output
