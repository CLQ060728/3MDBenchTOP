# generate images using PixArt-Σ, T2I
# author Qian Liu

import torch
from diffusers import PixArtSigmaPipeline
from transformers import T5Tokenizer
import os


def get_pixart_sigma_pipeline(repo_id, cache_dir, device, torch_compile=False):
    if torch_compile:
        torch.set_float32_matmul_precision("high")
        torch._inductor.config.conv_1x1_as_mm = True
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.epilogue_fusion = False
        torch._inductor.config.coordinate_descent_check_all_directions = True
        
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-11b", legacy=False)
        pipe = PixArtSigmaPipeline.from_pretrained(
               repo_id, tokenizer=tokenizer,
               torch_dtype=torch.float16, 
               cache_dir=cache_dir, 
               use_safetensors=True
               ).to(device)
        
        pipe.set_progress_bar_config(disable=True)
        
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.vae.to(memory_format=torch.channels_last)
        
        pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=True)
        pipe.vae.decode = torch.compile(pipe.vae.decode, mode="max-autotune", fullgraph=True)
    else:
        tokenizer = T5Tokenizer.from_pretrained("google-t5/t5-11b", legacy=False)
        pipe = PixArtSigmaPipeline.from_pretrained(
               repo_id, tokenizer=tokenizer,
               torch_dtype=torch.float16, 
               cache_dir=cache_dir, 
               use_safetensors=True
               ).to(device)
        
        pipe.set_progress_bar_config(disable=True)

    return pipe


def run_pixart_sigma_t2i(pipeline, num_inference_steps, img_width, img_height, prompts, manual_seed=False, seed=None):
    images = []
    assert prompts is not None and type(prompts) == list, "Prompts should not be None and should be list!"

    for prompt in prompts:
        if manual_seed:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = pipeline(prompt,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=8.0, width=img_width, height=img_height, generator=generator
                            ).images[0]  # clean_caption=False
        else:
            image = pipeline(prompt,
                             num_inference_steps=num_inference_steps,
                             guidance_scale=8.0, width=img_width, height=img_height
                            ).images[0]  # clean_caption=False
        images.append(image)

    return images