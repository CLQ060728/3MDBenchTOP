# generate images using DeepFloydIF T2I
# Author: Qian Liu


from diffusers import DiffusionPipeline
from diffusers.utils import pt_to_pil, make_image_grid
import torch


def get_deepfloydif_pipeline(repo_ids, cache_dir, gpu_id):
    assert repo_ids is not None and len(repo_ids) == 3, "repo_ids should not be None, and its length should be 3!"

    # stage 1
    stage_1 = DiffusionPipeline.from_pretrained(repo_ids[0], variant="fp16", torch_dtype=torch.float16,
                                                cache_dir=cache_dir, use_safetensors=True)
    stage_1.enable_model_cpu_offload(gpu_id=gpu_id)
    stage_1.set_progress_bar_config(disable=True)
    
    # stage 2
    stage_2 = DiffusionPipeline.from_pretrained(repo_ids[1], text_encoder=None, variant="fp16",
                                                torch_dtype=torch.float16, cache_dir=cache_dir, use_safetensors=True)
    stage_2.enable_model_cpu_offload(gpu_id=gpu_id)
    stage_2.set_progress_bar_config(disable=True)
    
    # stage 3
    safety_modules = {
        "feature_extractor": stage_1.feature_extractor,
        "safety_checker": stage_1.safety_checker,
        "watermarker": stage_1.watermarker,
    }
    stage_3 = DiffusionPipeline.from_pretrained(
        repo_ids[2], **safety_modules, torch_dtype=torch.float16,
        cache_dir=cache_dir, use_safetensors=True
    )
    stage_3.enable_model_cpu_offload(gpu_id=gpu_id)
    stage_3.set_progress_bar_config(disable=True)

    return stage_1, stage_2, stage_3


def run_deepfloydif_t2i(stage_1, stage_2, stage_3, prompts, manual_seed=False, seed=None):
    images = []
    if manual_seed:
        generator = torch.Generator(device="cpu").manual_seed(seed)
    else:
        generator = None
    for prompt in prompts:
        # text embeds
        prompt_embeds, negative_embeds = stage_1.encode_prompt(prompt)
        
        # stage 1
        stage_1_output = stage_1(
            prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_embeds, output_type="pt", generator=generator
        ).images
        #pt_to_pil(stage_1_output)[0].save("./if_stage_I.png")
        
        # stage 2
        stage_2_output = stage_2(
            image=stage_1_output,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_embeds,
            output_type="pt", generator=generator
        ).images
        #pt_to_pil(stage_2_output)[0].save("./if_stage_II.png")
        
        # stage 3
        stage_3_output = stage_3(prompt=prompt, image=stage_2_output, noise_level=100, generator=generator).images
        image_pair = [pt_to_pil(stage_2_output)[0], stage_3_output[0]]
        images.append(image_pair)

    return images


def run_centre_crop(img, new_width, new_height):
    width, height = img.size   # Get dimensions

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    
    # # Crop the center of the image
    image = img.crop((left, top, right, bottom))

    return image