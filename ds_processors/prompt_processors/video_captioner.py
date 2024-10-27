# video captioner
# Author: Qian Liu

import builtins
import os, sys

PLLAVA_PATH = None
OPENSORA_PATH = None

if hasattr(builtins, "PLLAVA_PATH_"):
    PLLAVA_PATH = builtins.PLLAVA_PATH_
else:
    assert PLLAVA_PATH is not None, "Please specify PLLAVA_PATH!"

if hasattr(builtins, "OPENSORA_PATH_"):
    OPENSORA_PATH = builtins.OPENSORA_PATH_
else:
    assert OPENSORA_PATH is not None, "Please specify OPENSORA_PATH!"

code_dir = os.path.join(PLLAVA_PATH, "code")
sys.path.insert(0, code_dir)
sys.path.insert(0, OPENSORA_PATH)

import copy
import itertools
import logging
import numpy as np
import torch
import torchvision
import transformers
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm
from transformers.feature_extraction_utils import BatchFeature

from opensora.datasets.read_video import read_video_av
from utils.easydict import EasyDict
from tasks.eval.model_utils import load_pllava
from tasks.eval.eval_utils import (
    ChatPllava,
    conv_plain_v1,
    Conversation,
    conv_templates
)
from tasks.eval.demo import pllava_theme
from python_scripts import hf


# SYSTEM = """You are a powerful Video Magic ChatBot, a large vision-language assistant. 
# You are able to understand the video content that the user provides and assist the user in a video-language related task.
# The user might provide you with the video and maybe some extra noisy information to help you out or ask you a question. Make use of the information in a proper way to be competent for the job.
# ### INSTRUCTIONS:
# 1. Follow the user's instruction.
# 2. Be critical yet believe in yourself.
# """
SYSTEM = "Describe this video. Pay attention to all objects in the video. The description should be useful for AI to re-generate the video. The description should be no more than six sentences. Here are some examples of good descriptions: 1. A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about. 2. Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field. 3. Drone view of waves crashing against the rugged cliffs along Big Sur's garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff's edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway."
conv_template = Conversation(
    system=SYSTEM,
    roles=("USER:", "ASSISTANT:"),
    messages=[],
    sep=(" ", "</s>"),
    mm_token="<image>",
)
# INIT_CONVERSATION: Conversation = conv_plain_v1.copy()

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def is_folder_empty(folder_path):
    # Check if the folder is empty
    return not any(os.scandir(folder_path))


# ========================================
#     Model, dataloader Initialization
# ========================================
def load_model(pretrained_model_name_or_path, use_lora, lora_alpha,
               gpu_id, num_frames, resolution):
    logger.info('Initializing PLLaVA')
    if is_folder_empty(pretrained_model_name_or_path):
        hf.run_download(pretrained_model_name_or_path)

    pretrained_path = os.path.join(pretrained_model_name_or_path, "pllava-13b")
    model, processor = load_pllava(
        pretrained_path, num_frames, 
        use_lora=use_lora, 
        weight_dir=pretrained_path, 
        lora_alpha=lora_alpha)
    logger.info("loading llava done!")
    model = model.to(torch.device(gpu_id))
    model = model.eval()
    
    return model, processor


def pllava_answer(
    conv: Conversation,
    model,
    processor,
    video_list,
    do_sample=True,
    max_new_tokens=200,
    num_beams=1,
    min_length=1,
    top_p=0.9,
    repetition_penalty=1.0,
    length_penalty=1,
    temperature=1.0,
    stop_criteria_keywords=None,
    print_res=False,
):
    # torch.cuda.empty_cache()
    prompt = conv.get_prompt()
    inputs_list = [processor(text=prompt, images=video, return_tensors="pt") for video in video_list]
    inputs_batched = dict()  # add batch dimension by cat
    for input_type in list(inputs_list[0].keys()):
        inputs_batched[input_type] = torch.cat([inputs[input_type] for inputs in inputs_list])
    inputs_batched = BatchFeature(inputs_batched, tensor_type="pt").to(model.device)

    with torch.no_grad():
        output_texts = model.generate(
            **inputs_batched,
            media_type="video",
            do_sample=do_sample,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_texts = processor.batch_decode(
            output_texts, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
       
    for i in range(len(output_texts)):
        if print_res:  # debug usage
            print("### PROMPTING LM WITH: ", prompt)
            print("### LM OUTPUT TEXT:  ", output_texts[i])
        if conv.roles[-1] == "<|im_start|>assistant\n":
            split_tag = "<|im_start|> assistant\n"
        else:
            split_tag = conv.roles[-1]
        output_texts[i] = output_texts[i].split(split_tag)[-1]
        ending = conv.sep if isinstance(conv.sep, str) else conv.sep[1]
        output_texts[i] = output_texts[i].removesuffix(ending).strip()
        output_texts[i] = output_texts[i].replace("\n", " ")
        conv.messages[-1][1] = output_texts[i]
    return output_texts, conv


def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)])
    return offsets


def load_video(video_path, num_frames, resolution):
    transforms = torchvision.transforms.Resize(size=resolution)
    # vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    vframes, aframes, info = read_video_av(
        video_path,
        pts_unit="sec", 
        output_format="THWC"
    )
    logger.info(vframes.shape)
    logger.info(f"video_path: {video_path}")
    total_num_frames = len(vframes)
    # print("Video path: ", video_path)
    # print("Total number of frames: ", total_num_frames)
    frame_indices = get_index(total_num_frames, num_frames)
    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vframes[frame_index].numpy())
        images_group.append(transforms(img))
    
    return images_group


def collate_fn(batch):
    return batch


def get_video_captions(gpu_id, model, processor, video_paths, num_frames, resolution):
    # conv_mode = "plain"
    # INIT_CONVERSATION = conv_templates[conv_mode]
    
    result_list = []
    for vid_path in video_paths:
        conv = conv_template.copy()
        conv.user_query("Describe the video in details.", is_mm=True)
        video = [load_video(vid_path, num_frames, resolution=resolution)]
        try:
            vid_caption, _ = pllava_answer(
                conv=conv,
                model=model,
                processor=processor,
                video_list=video,
                max_new_tokens=256,
                do_sample=True,
                print_res=False
            )
        except Exception as e:
            logger.error(f"error in {video}: {str(e)}")
            
        result_list.append(vid_caption)
    
    return result_list