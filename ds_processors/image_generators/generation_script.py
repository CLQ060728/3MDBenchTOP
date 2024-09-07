# Generate images from different types of text prompts, raw combination, paraphrased, captioner generated.
# Author: Qian Liu

CODE_DIR_ROOT = None

import sys
import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
import builtins

if hasattr(builtins, "CODE_DIR_ROOT_"):
    CODE_DIR_ROOT = builtins.CODE_DIR_ROOT_
else:
    assert CODE_DIR_ROOT is not None, "Please specify CODE_DIR_ROOT!"
    
sys.path.insert(0, CODE_DIR_ROOT)

from prompt_processors import image_similarity_metrics as ism
from image_generators.Kandinsky3 import generator as kangen
from image_generators.PixArt_Σ import generator as pixgen
from image_generators.StableDiffusion3 import generator as sd3gen
from image_generators.DeepFloydIF import generator as dfgen
from image_generators.StableDiffusionXL import generator as sdxlgen
# import logging
# logger = logging.getLogger("GENERATION_SCRIPT")

# logging.basicConfig(filename='/home/jovyan/3MDBench/ds_processors/image_generators/out2.log', level=logging.INFO)

def load_img_captions_dict(img_cap_file_path):
    json_string = ""
    img_cap_dict = dict()
    with open(img_cap_file_path, "r") as img_cap_file:
        for line in img_cap_file:
            if not line.startswith(" counter:"):
                json_string += line
            else:
                if len(img_cap_dict) == 0:
                    img_cap_dict = json.loads(json_string)
                else:
                    img_cap_dict |= json.loads(json_string)
                json_string = ""
    
    return img_cap_dict


def load_cap_img_captions_dict(img_cap_file_path):
    with open(img_cap_file_path, "r") as img_cap_file:
        img_cap_dict = json.load(img_cap_file)
    
    return img_cap_dict


def convert_captioner_img_caption_files(captioner_img_cap_path_root, captioner_img_cap_file_prefix, sel_cap_dict):
    def get_img_cap_list(captioner_img_cap_file_path):
        json_string = ""
        img_cap_list = []
        with open(captioner_img_cap_file_path, "r") as captioner_img_cap_file:
            for line in captioner_img_cap_file:
                if not line.startswith(" counter:"):
                    json_string += line
                else:
                    if len(img_cap_list) == 0:
                        img_cap_list = json.loads(json_string)
                    else:
                        img_cap_list = [*img_cap_list, *(json.loads(json_string))]
                    json_string = ""
        img_cap_arr = np.array(img_cap_list)
        img_cap_list = img_cap_arr.squeeze()
        
        return img_cap_list

    # prompt_path = os.path.join(ds_path, "generated")
    img_cap_lists = np.array([])
    for idx in range(1, 11, 1):
        captioner_img_cap_file_path = os.path.join(captioner_img_cap_path_root, f"{captioner_img_cap_file_prefix}_{idx}.txt")
        if img_cap_lists.shape[0] == 0:
            img_cap_lists = get_img_cap_list(captioner_img_cap_file_path)
        else:
            img_cap_list = get_img_cap_list(captioner_img_cap_file_path)
            img_cap_lists = np.vstack((img_cap_lists, img_cap_list))
    captioner_img_caption_dict = dict()
    img_ids = list(sel_cap_dict.keys())
    for img_index in range(img_cap_lists.shape[1]):
        img_cap_arr = img_cap_lists[:, img_index]
        img_cap_arr = np.array([img_cap[1:] for img_cap in img_cap_arr], dtype=img_cap_arr.dtype)
        captioner_img_caption_dict[img_ids[img_index]] = img_cap_arr.tolist()
    with open(os.path.join(captioner_img_cap_path_root, 
                           f"{captioner_img_cap_file_prefix}.txt"), "w") as captioner_img_caption_file:
        json.dump(captioner_img_caption_dict, captioner_img_caption_file, indent=4)


def load_img_generator_pipeline(repo_id, variant, cache_dir, t2i_or_i2i, device, torch_compile=False):
    if repo_id.startswith("kandinsky"):
        pipe = kangen.get_kandinsky3_pipeline(repo_id, variant, cache_dir, t2i_or_i2i)
    elif repo_id.startswith("PixArt"):
        pipe = pixgen.get_pixart_sigma_pipeline(repo_id, cache_dir, device, torch_compile)
    elif repo_id.startswith("stabilityai"):
        pipe = sd3gen.get_sd3_pipeline(repo_id, cache_dir, device, torch_compile)
    
    return pipe


def load_deepfloyd_pipeline(repo_ids, cache_dir, gpu_id):
    stage1, stage2, stage3 = dfgen.get_deepfloydif_pipeline(repo_ids, cache_dir, gpu_id)
    
    return stage1, stage2, stage3


def load_sdxl_pipeline(base_repo_id, refiner_repo_id, cache_dir, gpu_id):
    base, refiner = sdxlgen.get_sdxl_pipeline(base_repo_id, refiner_repo_id, cache_dir, gpu_id)
    
    return base, refiner


def convert_img_to_imgtensor(img_path, img, size):
    if img_path is not None:
        img = Image.open(img_path)
    img = img.resize(size, resample=Image.BILINEAR)
    img_tensor = pil_to_tensor(img).type(torch.uint8)
    img_tensor = torch.unsqueeze(img_tensor, 0)
    
    return img_tensor


def generate_imgs_in_batch(model_name, prompt_type, pipe, num_inference_steps, img_width, img_height, size, 
                           t2i_or_i2i, real_img_path_root, img_prompt_dict, img_cap_dict, output_path, device,
                           manual_seed=False, seed=None):
    images = None
    img_id = 0
    for img_prompt_key, img_prompt_caps in img_prompt_dict.items():
        img_id = img_prompt_key
        
        if prompt_type == "raw_prompt" and model_name != "DeepFloydIF" and model_name != "StableDiffusionXL"\
                                       and model_name != "StableDiffusion3":
            prompts = [*img_prompt_caps, *img_prompt_caps]
        elif prompt_type == "para_prompt":
            if model_name == "StableDiffusion3":
                prompt0s = [*img_prompt_caps[-4:], *img_prompt_caps[-4:-2], *img_prompt_caps[-4:]]
                prompt1s = img_prompt_caps[0:10]
                prompts = [prompt0s, prompt1s]
            elif model_name == "DeepFloydIF" or model_name == "StableDiffusionXL":
                prompts = [*img_prompt_caps[-2:], *img_prompt_caps[-2:], *img_prompt_caps[-2:], *img_prompt_caps[-2:],
                           *img_prompt_caps[-2:]]
            else:
                prompts = img_prompt_caps
        elif prompt_type == "cap_prompt" and model_name != "DeepFloydIF" and model_name != "StableDiffusionXL"\
                                         and model_name != "StableDiffusion3":
            prompts = img_prompt_caps
        else:
            prompts = None
            print(f"prompts can not be captioner prompts when model is IF, SDXL or SD3!")
            return
        
        img_file_name = img_cap_dict[img_id][0]
        real_img_path = os.path.join(real_img_path_root, img_file_name)
        if model_name == "Kandinsky3":
            if t2i_or_i2i:
                images = kangen.run_kandinsky3_t2i(pipe, num_inference_steps, img_width, img_height, prompts, 
                                                   manual_seed=manual_seed, seed=seed)
            else:
                images = kangen.run_kandinsky3_i2i(pipe, num_inference_steps, img_width, img_height, prompts, 
                                                   real_img_path, manual_seed=manual_seed, seed=seed)
        elif model_name == "PixArt_Σ":
            images = pixgen.run_pixart_sigma_t2i(pipe, num_inference_steps, img_width, img_height, prompts, 
                                                 manual_seed=manual_seed, seed=seed)
        elif model_name == "StableDiffusion3":
            images = sd3gen.run_sd3_t2i(pipe, num_inference_steps, img_width, img_height, prompts[0], prompts[1], 
                                        manual_seed=manual_seed, seed=seed)
        elif model_name == "DeepFloydIF":
            images = dfgen.run_deepfloydif_t2i(pipe[0], pipe[1], pipe[2], prompts, manual_seed=manual_seed, seed=seed)
        elif model_name == "StableDiffusionXL":
            if t2i_or_i2i:
                images = sdxlgen.run_sdxl_t2i(pipe[0], pipe[1], num_inference_steps, prompts, 
                                              manual_seed=manual_seed, seed=seed)
            else:
                images = sdxlgen.run_sdxl_i2i(pipe[1], num_inference_steps, img_width, img_height, prompts, real_img_path,
                                              manual_seed=manual_seed, seed=seed)
        
        real_img_tensor = convert_img_to_imgtensor(real_img_path, None, size)
        print(f"real_img_tensor size: {real_img_tensor.size()}")
        similarities = torch.tensor([], dtype=torch.float32, device=device)
        for fake_img in images:
            if model_name == "DeepFloydIF":
                fake_img_tensor = convert_img_to_imgtensor(None, fake_img[1], size)
            else:
                fake_img_tensor = convert_img_to_imgtensor(None, fake_img, size)
            print(f"fake_img_tensor size: {fake_img_tensor.size()}")
            if real_img_tensor.size(dim=1) > fake_img_tensor.size(dim=1):
                fake_img_tensor = torch.repeat_interleave(fake_img_tensor, real_img_tensor.size(dim=1), dim=1)
            elif real_img_tensor.size(dim=1) < fake_img_tensor.size(dim=1):
                real_img_tensor = torch.repeat_interleave(real_img_tensor, fake_img_tensor.size(dim=1), dim=1)
            
            similarity_metrics = torch.tensor([[ism.compute_LPIPS(real_img_tensor, fake_img_tensor, "squeeze", device),
                                                ism.compute_PSNR(real_img_tensor, fake_img_tensor, device),
                                                ism.compute_SSIM(real_img_tensor, fake_img_tensor, device)]],
                                                dtype=torch.float32, device=device)
            similarities = torch.cat((similarities, similarity_metrics), 0)
        
        best_indices = torch.tensor([], dtype=torch.int32, device=device)
        for col_idx in range(similarities.size(dim=1)):
            if col_idx == 0:
                best_indices = torch.cat((best_indices, similarities[:,col_idx].min(dim=0)[1].unsqueeze(dim=0)), 0)
            else:
                best_indices = torch.cat((best_indices, similarities[:,col_idx].max(dim=0)[1].unsqueeze(dim=0)), 0)
        
        unique_counts = best_indices.unique(return_counts=True)[1]
        if (unique_counts == 1).all():
            unique_idx = torch.randint(0, unique_counts.size(dim=0), (1,))
        else:
            unique_idx = unique_counts.max(dim=0)[1]
        if model_name == "DeepFloydIF":
            result_img = images[best_indices[unique_idx]][1]
        else:
            result_img = images[best_indices[unique_idx]]
        out_file_path = os.path.join(output_path, img_file_name)
        result_img.save(out_file_path)

### ism.compute_LPIPS(real_img_tensor, fake_img_tensor, 'squeeze', device),