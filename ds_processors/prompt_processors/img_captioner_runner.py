# run image captioner in batch
# Author: Qian Liu

import builtins
builtins.LLAVA_PATH_ = "/home/jovyan/3MDBench/ds_processors/prompt_processors/LLaVA_NeXT/"
import image_captioner as ic
import os, json
import torch
import numpy as np


def convert_captioner_img_caption_files(captioner_img_cap_path_root, captioner_img_cap_file_prefix, sel_dict_file_path):
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
    with open(sel_dict_file_path, "r") as sel_dict_file:
        sel_cap_dict = json.load(sel_dict_file)
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


def generate_img_cap_captions(max_bound, last_bound, path_root, device):
    dict_file_path = os.path.join(path_root, f"sel_captions_{max_bound}.txt")
    img_cap_dict_file_path = os.path.join(path_root, "image_caption_dict.txt")
    
    with open(dict_file_path, "r") as dict_file, open(img_cap_dict_file_path, "r") as img_cap_dict_file:
        selected_img_cap_dict = json.load(dict_file)
        img_cap_dict = json.load(img_cap_dict_file)
    sel_path = os.path.join(path_root, "selected")
    
    img_paths = []
    for img_id in list(selected_img_cap_dict.keys()):
        if img_id in img_cap_dict:
            img_path = os.path.join(sel_path, img_cap_dict[img_id][0])
            img_paths.append(img_path)

    # for file_count in range(1, 6, 1):
    for file_count in range(6, 11, 1):
        out_file_path = os.path.join(path_root, "generated", f"captioner_img_caption_{max_bound}_{file_count}.txt")
        tokenizer = model = input_ids = image_processor = None
        tokenizer, model, input_ids, image_processor = ic.get_llava_next_llama3_8b_model(device)
        with open(out_file_path, "a") as out_file:
            for counter in range(1, 6, 1):
                image_captions = ic.get_image_captions(tokenizer, model, input_ids, image_processor,
                                                    img_paths[((counter-1)*1000):(counter*1000)], device)
                json.dump(image_captions, out_file, indent=4)
                out_file.write(f"\n counter:[{(last_bound+(counter-1)*1000)}:{(last_bound+counter*1000)}]\n")
                print(f"counter:[{(last_bound+(counter-1)*1000)}:{(last_bound+counter*1000)}]")  


if __name__ == "__main__":
    path_root = "/home/jovyan/3MDBench/data/IMAGEs/MSCOCO/"
    captioner_img_cap_path_root = os.path.join(path_root, "generated")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_bound = 15000
    last_bound = 10000
    captioner_img_cap_file_prefix = f"captioner_img_caption_{max_bound}"
    sel_dict_file_path = os.path.join(path_root, f"sel_captions_{max_bound}.txt")
    
    generate_img_cap_captions(max_bound, last_bound, path_root, device)
    
    # convert_captioner_img_caption_files(captioner_img_cap_path_root, captioner_img_cap_file_prefix, sel_dict_file_path)
    
    