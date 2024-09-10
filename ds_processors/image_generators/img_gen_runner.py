# import builtins
# code_dir_root = "/home/jovyan/3MDBench/ds_processors/"
# builtins.CODE_DIR_ROOT_ = code_dir_root
import torch
import os
import json
from . import generation_script as gs


def run(project_root, ds_name, model_name, max_bound, img_width, img_height,
        prompt_type, device, gpu_id=0, t2i_or_i2i=False, num_inference_steps=50, manual_seed=False, seed=None):
    variant= ""
    img_path_root = os.path.join(project_root, "data/IMAGEs/")
    output_path_root = os.path.join(img_path_root, "generated")
    ds_path = os.path.join(img_path_root, ds_name)
    code_dir_root = os.path.join(project_root, "ds_processors")
    if model_name == "Kandinsky3":
        variant = "fp16"
        repo_id = "kandinsky-community/kandinsky-3"
        cache_dir = os.path.join(code_dir_root, "image_generators", model_name, "cache")
        pipe = gs.load_img_generator_pipeline(repo_id, variant, cache_dir, t2i_or_i2i, device)

    elif model_name == "PixArt_Σ":
        repo_id = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        cache_dir = os.path.join(code_dir_root, "image_generators", model_name, "cache")
        torch_compile = False
        pipe = gs.load_img_generator_pipeline(repo_id, variant, cache_dir, t2i_or_i2i, device, torch_compile)

    elif model_name == "StableDiffusion3":
        repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        cache_dir = os.path.join(code_dir_root, "image_generators", model_name, "cache")
        torch_compile = False
        pipe = gs.load_img_generator_pipeline(repo_id, variant, cache_dir, t2i_or_i2i, device, torch_compile)

    elif model_name == "DeepFloydIF":
        repo_ids = ["DeepFloyd/IF-I-XL-v1.0", "DeepFloyd/IF-II-L-v1.0", "stabilityai/stable-diffusion-x4-upscaler"]
        cache_dir=os.path.join(code_dir_root, "image_generators", model_name, "cache")
        gpu_id = 0
        stage1, stage2, stage3 = gs.load_deepfloyd_pipeline(repo_ids, cache_dir, gpu_id)
        pipe = [stage1, stage2, stage3]

    elif model_name == "StableDiffusionXL":
        base_repo_id = "stabilityai/stable-diffusion-xl-base-1.0"
        refiner_repo_id = "stabilityai/stable-diffusion-xl-refiner-1.0"
        cache_dir = os.path.join(code_dir_root, "image_generators", model_name, "cache")
        gpu_id = 0
        base, refiner = gs.load_sdxl_pipeline(base_repo_id, refiner_repo_id, cache_dir, gpu_id)
        pipe = [base, refiner]
    
    ### load dataset dict files
    # sel_cap_file_path = os.path.join(ds_path, "sel_captions_5000.txt") open(sel_cap_file_path, "r") as sel_cap_file
    img_cap_file_path = os.path.join(ds_path, "image_caption_dict.txt")
    captions_list_str = ""
    with open(img_cap_file_path, "r") as img_cap_file:
        # sel_cap_dict = json.load(sel_cap_file)
        img_cap_dict = json.load(img_cap_file)
    
    prompt_path = os.path.join(ds_path, "generated")
    raw_img_cap_file_path = os.path.join(prompt_path, f"raw_img_caption_{max_bound}.txt")
    raw_img_cap_dict = gs.load_img_captions_dict(raw_img_cap_file_path)
    para_img_cap_file_path = os.path.join(prompt_path, f"para_img_caption_{max_bound}.txt")
    para_img_cap_dict = gs.load_img_captions_dict(para_img_cap_file_path)
    captioner_img_cap_file_path = os.path.join(prompt_path, f"captioner_img_caption_{max_bound}.txt")
    captioner_img_cap_dict = gs.load_cap_img_captions_dict(captioner_img_cap_file_path)
    size = (512, 512)
    real_img_path_root = os.path.join(ds_path, "selected")
    if prompt_type == "raw_prompt":
        img_prompt_dict = raw_img_cap_dict
    elif prompt_type == "para_prompt":
        img_prompt_dict = para_img_cap_dict
    elif prompt_type == "cap_prompt":
        img_prompt_dict = captioner_img_cap_dict
    if t2i_or_i2i:
        T2I_I2I = "T2I"
    else:
        T2I_I2I = "I2I"
    output_path = os.path.join(output_path_root, model_name, ds_name, prompt_type, T2I_I2I, str(img_width))
    os.makedirs(output_path, exist_ok = True)
    
    gs.generate_imgs_in_batch(model_name, prompt_type, pipe, num_inference_steps, img_width, img_height, size, 
                              t2i_or_i2i, real_img_path_root, img_prompt_dict, img_cap_dict, output_path, device,
                              manual_seed=False, seed=None)


if __name__ == "__main__":
    img_path_root = "/home/jovyan/3MDBench/data/IMAGEs/"
    ds_name = "MSCOCO"
    ds_path = os.path.join(img_path_root, ds_name)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = "Kandinsky3"
    variant= "" # "fp16"
    