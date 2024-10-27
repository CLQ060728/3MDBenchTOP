# run video captioner in batch
# Author: Qian Liu

import os, json
import torch
import numpy as np
import logging
from . import video_captioner as vc

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def convert_captioner_vid_caption_files(captioner_vid_cap_path_root,
                                        captioner_vid_cap_file_prefix, sel_dict_file_path):
    def get_vid_cap_list(captioner_vid_cap_file_path):
        json_string = ""
        vid_cap_list = []
        with open(captioner_vid_cap_file_path, "r") as captioner_vid_cap_file:
            vid_cap_list = json.load(captioner_vid_cap_file)
        vid_cap_list = np.array(vid_cap_list)
        
        return vid_cap_list

    # prompt_path = os.path.join(ds_path, "generated")
    with open(sel_dict_file_path, "r") as sel_dict_file:
        sel_cap_dict = json.load(sel_dict_file)
    vid_cap_lists = np.array([])
    for idx in range(1, 11, 1):
        captioner_vid_cap_file_path = os.path.join(captioner_vid_cap_path_root,
                                                   f"{captioner_vid_cap_file_prefix}_{idx}.txt")
        if vid_cap_lists.shape[0] == 0:
            vid_cap_lists = get_vid_cap_list(captioner_vid_cap_file_path)
        else:
            vid_cap_list = get_vid_cap_list(captioner_vid_cap_file_path)
            vid_cap_lists = np.vstack((vid_cap_lists, vid_cap_list))
    captioner_vid_caption_dict = dict()
    vid_ids = list(sel_cap_dict.keys())
    for vid_index in range(vid_cap_lists.shape[1]):
        vid_cap_arr = vid_cap_lists[:, vid_index]
        vid_cap_arr = np.array([vid_cap[1:] for vid_cap in vid_cap_arr], dtype=vid_cap_arr.dtype)
        captioner_vid_caption_dict[vid_ids[vid_index]] = vid_cap_arr.tolist()
    with open(os.path.join(captioner_vid_cap_path_root, 
                           f"{captioner_vid_cap_file_prefix}.txt"), "w") as captioner_vid_caption_file:
        json.dump(captioner_vid_caption_dict, captioner_vid_caption_file, indent=4)


def generate_video_cap_captions(max_bound, previous_bound, path_root, pretrained_model_name_or_path,
                                gpu_id, resolution):
    dict_file_path = os.path.join(path_root, f"sel_captions_{max_bound}.txt")
    img_cap_dict_file_path = os.path.join(path_root, "image_caption_dict.txt")
    
    with open(dict_file_path, "r") as dict_file, open(img_cap_dict_file_path, "r") as img_cap_dict_file:
        selected_img_cap_dict = json.load(dict_file)
        img_cap_dict = json.load(img_cap_dict_file)
    sel_path = os.path.join(path_root, "selected")
    
    vid_paths = []
    for vid_id in list(selected_img_cap_dict.keys()):
        if vid_id in img_cap_dict:
            vid_path = os.path.join(sel_path, img_cap_dict[vid_id][0])
            vid_paths.append(vid_path)
    
    use_lora = True
    lora_alpha = 4
    num_frames = 16
    model, processor = vc.load_model(pretrained_model_name_or_path,
                                     use_lora, lora_alpha, gpu_id, num_frames, resolution)
    
    vid_captions = vc.get_video_captions(gpu_id, model, processor, vid_paths, num_frames, resolution)
    
    return vid_captions


def run(data_root, cache_dir, max_bound, previous_bound, gpu_id, resolution, aggregate=False):
    captioner_vid_cap_path_root = os.path.join(data_root, "generated")
    captioner_vid_cap_file_prefix = f"captioner_vid_caption_{max_bound}"
    sel_dict_file_path = os.path.join(data_root, f"sel_captions_{max_bound}.txt")
    # n_gpus = torch.cuda.device_count()
    
    if not aggregate:
        for round in range(1, 11, 1):
            logger.info(f"Round {round} starts...")
            vid_captions = generate_video_cap_captions(max_bound, previous_bound, data_root,
                                        cache_dir, gpu_id, resolution)
            
            out_file_path = os.path.join(data_root, "generated",
                                         f"{captioner_vid_cap_file_prefix}_{round}.txt")
            with open(out_file_path, "w") as out_file:
                json.dump(vid_captions, out_file, indent=4)
    else:
        convert_captioner_vid_caption_files(captioner_vid_cap_path_root,
                                            captioner_vid_cap_file_prefix, sel_dict_file_path)