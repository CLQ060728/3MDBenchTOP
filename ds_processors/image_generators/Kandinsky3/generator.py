# Generate images using Kandinsky 3.0, T2I and I2I
# Author: Qian Liu

from diffusers import AutoPipelineForText2Image
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image


def get_kandinsky3_pipeline(repo_id, variant, cache_dir, t2i_or_i2i):
    if t2i_or_i2i:
        pipe = AutoPipelineForText2Image.from_pretrained(repo_id, variant=variant, torch_dtype=torch.float16, cache_dir=cache_dir,
                                                         use_safetensors=True)
        pipe.enable_model_cpu_offload()
    else:
        pipe = AutoPipelineForImage2Image.from_pretrained(repo_id, variant=variant, torch_dtype=torch.float16, cache_dir=cache_dir,
                                                          use_safetensors=True)
        pipe.enable_model_cpu_offload()
        
    # to("cuda") can not be used with cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    
    return pipe


def run_kandinsky3_t2i(pipeline, num_inference_steps, img_width, img_height, prompts, manual_seed=False, seed=None):
    images = []
    assert prompts is not None and type(prompts) == list, "Prompts should not be None and should be list!"
    for prompt in prompts:
        if manual_seed:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = pipeline(prompt, num_inference_steps=num_inference_steps, generator=generator,
                         width=img_width, height=img_height).images[0]
        else:
            image = pipeline(prompt, num_inference_steps=num_inference_steps, width=img_width, height=img_height,
                             guidance_scale=8.0).images[0]
            # , clean_caption=False
        images.append(image)

    return images


def run_kandinsky3_i2i(pipeline, num_inference_steps, img_width, img_height, prompts, img_path, strength=0.35,
                       manual_seed=False, seed=None):
    images = []
    size = (img_width, img_height)
    ref_img = load_image(img_path).resize(size) # size must be width = height
    assert prompts is not None and type(prompts) == list, "Prompts can not be None and should be a list!"
    
    for prompt in prompts:
        if manual_seed:
            generator = torch.Generator(device="cpu").manual_seed(seed)
            image = pipeline(prompt, image=ref_img, strength=strength, num_inference_steps=num_inference_steps,
                         generator=generator, width=img_width, height=img_height).images[0]
        else:
            image = pipeline(prompt, image=ref_img, strength=strength, num_inference_steps=num_inference_steps,
                             width=img_width, height=img_height,
                             guidance_scale=8.0).images[0] 
            # lower the strength, higher the image influences, clean_caption=False
        images.append(image)

    return images