# Generate images using Stable Diffusion XL, T2I
# Author: Qian Liu

import torch, os
from diffusers import DiffusionPipeline
from diffusers.utils import load_image


def get_sdxl_pipeline(base_repo_id, refiner_repo_id, cache_dir, gpu_id):
    base = DiffusionPipeline.from_pretrained(
    base_repo_id, torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
    cache_dir=cache_dir
    )
    base.enable_model_cpu_offload(gpu_id=gpu_id)
    base.set_progress_bar_config(disable=True)
    
    refiner = DiffusionPipeline.from_pretrained(
        refiner_repo_id,
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
        cache_dir=cache_dir
    )
    refiner.enable_model_cpu_offload(gpu_id=gpu_id)
    refiner.set_progress_bar_config(disable=True)
    
    return base, refiner


def run_sdxl_t2i(base, refiner, num_inference_steps, prompts, manual_seed=False, seed=None):
    images = []
    assert prompts is not None and type(prompts) == list, "Prompts should not be None and should be list!"

    if manual_seed:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    
    for prompt in prompts:
        img_base = base(
        prompt=prompt,
        prompt_2=prompt,
        num_inference_steps=num_inference_steps,
        denoising_end=0.8,
        guidance_scale=8.0,
        output_type="latent",
        generator=generator
        ).images 
        image = refiner( 
        prompt=prompt,
        prompt_2=prompt,
        num_inference_steps=num_inference_steps,
        denoising_start=0.8,
        guidance_scale=8.0,
        image=img_base,
        generator=generator
        ).images[0]
        images.append(image)
    
    return images


def run_sdxl_i2i(refiner, num_inference_steps, img_width, img_height, prompts, img_path, strength=0.35,
                 manual_seed=False, seed=None):
    images = []
    size = (img_width, img_height)
    ref_img = load_image(img_path).resize(size)
    assert prompts is not None and type(prompts) == list, "Prompts should not be None and should be list!"
    
    if manual_seed:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    
    for prompt in prompts:
        image = refiner( 
        prompt=prompt,
        prompt_2=prompt,
        strength=0.35,
        num_inference_steps=num_inference_steps,
        denoising_start=0.8,
        guidance_scale=8.0,
        image=ref_img,
        generator=generator
        ).images[0]
        images.append(image)
    
    return images