from models.ai_models import VideoModel
from transformers import AutoModel, AutoTokenizer
import torch
import os
import warnings
from models.intern.utils import load_video

class Intern(VideoModel):
    """A video model implementation for InternVL with multi-GPU and FlashAttention2 support."""

    def __init__(self):
        model_path = "OpenGVLab/InternVL3-2B"

        # When running under torchrun each rank owns one GPU (LOCAL_RANK).
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        # Set device config — pinned to this rank's GPU
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "mps")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

        # Set environment variables and enable flash attention (on supported devices)
        if torch.cuda.is_available():
            torch.backends.cuda.enable_flash_sdp(True)
            torch.cuda.set_per_process_memory_fraction(0.9)
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            # Load to CPU first, then move to the rank's GPU.
            # InternVL3's custom model class leaves _tp_plan=None which causes
            # transformers' caching_allocator_warmup to crash when device_map is set.
            # The "not initialized on GPU" warning from flash_attention_2 is expected —
            # the model IS moved to GPU immediately after via .to(self.device).
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*not initialized on GPU.*")
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    _attn_implementation="flash_attention_2",
                ).to(self.device)
        else:
            # Fallback for MPS or CPU
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

        response = self.model.chat(
            self.tokenizer,
            pixel_values,
            question,
            generation_config,
            num_patches_list=num_patches_list,
            history=None,
            return_history=False,
        )

        return response
        
