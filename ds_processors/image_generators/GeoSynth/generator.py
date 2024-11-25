from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from PIL import Image


def get_geosynth_model(cache_dir):
    controlnet = ControlNetModel.from_pretrained("MVRL/GeoSynth-OSM",
                                                 cache_dir=cache_dir,
                                                 use_safetensors=True)

    pipe = StableDiffusionControlNetPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base",
                                                             controlnet=controlnet, 
                                                             cache_dir=cache_dir,
                                                             use_safetensors=True)
    # pipe = pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.set_progress_bar_config(disable=True)
    
    return pipe


def run_geosynth(pipe, prompt, osm_image, output_path):
    image = pipe(
        prompt,
        image=osm_image,
        num_inference_steps=50,
        guidance_scale=8.0
    ).images[0]
    
    image.save(output_path)