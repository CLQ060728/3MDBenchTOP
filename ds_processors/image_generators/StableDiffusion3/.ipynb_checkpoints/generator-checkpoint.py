# Generate images using Stable Diffusion 3 Medium, T2I
# Author: Qian Liu

import torch
from diffusers import StableDiffusion3Pipeline
import os

os.environ["TOKENIZERS_PARALLELISM"] = "True"


def get_sd3_pipeline(repo_id, cache_dir, device, torch_compile=False):
    if torch_compile:
        torch.set_float32_matmul_precision("high")
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
         
        # repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        pipe = StableDiffusion3Pipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                use_safetensors=True).to(device)
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)
        
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
    else:
        pipe = StableDiffusion3Pipeline.from_pretrained(
                repo_id,
                torch_dtype=torch.float16,
                cache_dir=cache_dir,
                use_safetensors=True).to(device)
        # pipe.enable_model_cpu_offload()
        pipe.set_progress_bar_config(disable=True)

    return pipe


def run_sd3_t2i(pipeline, num_inference_steps, img_width, img_height, prompts, prompts_3, manual_seed=False, seed=None):
    images = []
    assert prompts is not None and type(prompts) == list, "Prompts should not be None and should be list!"
    assert prompts_3 is not None and type(prompts_3) == list, "Prompts_3 should not be None and should be list!"
    assert len(prompts) == len(prompts_3), "The lengths of prompts and prompts_3 should be equal!"
    
    for prompt_index in range(len(prompts)):
        if manual_seed:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = pipeline(
            prompt=prompts[prompt_index],
            prompt_3=prompts_3[prompt_index],
            num_inference_steps=num_inference_steps,
            height=img_height,
            width=img_width,
            guidance_scale=8.0,
            max_sequence_length=512,
            generator=generator
            ).images[0]
        else:
            image = pipeline(
            prompt=prompts[prompt_index],
            prompt_3=prompts_3[prompt_index],
            num_inference_steps=num_inference_steps,
            height=img_height,
            width=img_width,
            guidance_scale=8.0,
            max_sequence_length=512
            ).images[0]
        images.append(image)

    return images

