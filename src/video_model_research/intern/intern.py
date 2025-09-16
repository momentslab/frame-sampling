from video_model_research.ai_models import VideoModel
from transformers import AutoModel, AutoTokenizer
import torch
import os
from video_model_research.intern.utils import load_video

class Intern(VideoModel):
    """A video model implementation for InternVL with multi-GPU and FlashAttention2 support."""

    def __init__(self):
        model_path = "OpenGVLab/InternVL3-2B"

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.cuda.set_per_process_memory_fraction(0.9)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                _attn_implementation="flash_attention_2",  # IMPORTANT
                device_map="balanced" if torch.cuda.device_count() > 1 else "auto"
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True
            ).to(self.device)

        self.model.eval()
    
    def predict(self, video_items: dict, prompt: str, max_tokens: int) -> str:
        """Predict the action in the video."""
        pixel_values, num_patches_list = load_video(video_items, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.device)

        generation_config = dict(max_new_tokens=max_tokens, do_sample=True)

        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + prompt

        response, history = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=True
        )

        return response
        
