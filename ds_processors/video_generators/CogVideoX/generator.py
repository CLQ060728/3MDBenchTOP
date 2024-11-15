# Video generation using CogVideoX
# Author: Qian Liu

import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video


def get_CogVideoX_pipeline(cache_dir, device):
    pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b",
                                             cache_dir=cache_dir,
                                             torch_dtype=torch.bfloat16,
                                             use_safetensors=True).to(device) 
    # or "THUDM/CogVideoX-2b" bf16
    # pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()
    pipe.set_progress_bar_config(disable=True)
    
    return pipe


def run_cogvideox(pipe, prompt, output_file_path):
    video = pipe(prompt=prompt, guidance_scale=6, num_frames=49, num_inference_steps=50).frames[0]
    export_to_video(video, output_file_path, fps=8)