# Using PyramidFlow to generate videos
# Author: Qian Liu

import torch
from PIL import Image
from .code.pyramid_dit import PyramidDiTForVideoGeneration
from diffusers.utils import load_image, export_to_video


def get_pyramid_flow_model(cache_dir, resolution, gpu_id):
    
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

    torch.cuda.set_device(gpu_id)
    model_dtype, torch_dtype = 'bf16', torch.bfloat16   # Use bf16 (not support fp16 yet)
    
    model = PyramidDiTForVideoGeneration(
        cache_dir,                                        
        model_dtype,
        use_safetensors=True,
        model_name="pyramid_flux",
        model_variant=f'diffusion_transformer_{resolution}'
    )
    
    model.vae.to(device)
    model.dit.to(device)
    model.text_encoder.to(device) # 
    model.vae.enable_tiling()
    # model.enable_sequential_cpu_offload()

    return model


def run_pyramidflow(model, prompt, resolution, output_path):
    if resolution == "384p":
        # used for 384p model variant
        width = 640
        height = 384
        temp_val = 16
    elif resolution == "768p":
        # used for 768p model variant
        width = 1280
        height = 768
        temp_val = 31
    
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
        frames = model.generate(
            prompt=prompt,
            num_inference_steps=[20, 20, 20],
            video_num_inference_steps=[10, 10, 10],
            height=height,     
            width=width,
            temp=temp_val,                    # temp=16: 5s, temp=31: 10s
            guidance_scale=7.0,         # The guidance for the first frame, set it to 7 for 384p variant
            video_guidance_scale=5.0,   # The guidance for the other video latent
            output_type="pil",
            cpu_offloading=True,
            save_memory=True
        )
# If you have enough GPU memory, set it to `False` to improve vae decoding speed save_memory=False
    
    export_to_video(frames, output_path, fps=24)