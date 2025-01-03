"""Inception Score (IS) from the paper "Improved techniques for training
GANs". Matches the original implementation by Salimans et al. at
https://github.com/openai/improved-gan/blob/master/inception_score/model.py"""

# from urllib.parse import urlparse
# import dnnlib
from tqdm import tqdm
# from PIL import Image

# import torch.nn as nn
import numpy as np
import argparse
# import pickle
import torch
import os
from decord import VideoReader, cpu
from .pytorch_gan_metrics.core import calculate_frechet_inception_distance, calculate_inception_score, get_inception_feature
import logging
import time
import ipdb


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@torch.no_grad()
def inception_score(paths, splits=10, verbose=False):

    features, split_scores = list(), list()
    for path in tqdm(paths):
        video = torch.tensor(read_video_to_np(path))  # t h w c
        video = video.permute(0, 3, 1, 2).contiguous()/255.  # t c h w 
        acts, probs = get_inception_feature(video, dims=[2048, 1008], use_torch=True)
        features.append(probs)
    
    inception_score, std, scores = calculate_inception_score(torch.cat(features), splits, use_torch=True)

    return inception_score, std, scores


def read_video_to_np(video_path):
    vidreader = VideoReader(video_path, ctx=cpu(0))
    vid_len = len(vidreader)
    frames = vidreader.get_batch(list(range(vid_len))).asnumpy()
    return frames


def read_prompt_from_txt(fpath):
    prompt_list = []
    with open(fpath, 'r') as f:
        for l in f.readlines():
            l = l.strip()
            if len(l) != 0:
                prompt_list.append(l)
    assert (len(prompt_list) == 1), len(prompt_list)
    return prompt_list[0]


def compute_inception_score(input_path, num_splits, output_path_root, gpu_id):
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--dir_videos", type=str, default='', help="Specify the path of generated videos")
    # parser.add_argument("--metric", type=str, default='IS', help="Specify the metric to be used")
    # parser.add_argument("--splits", type=int, default=10)
    # args = parser.parse_args()
   
    video_paths = [os.path.join(input_path, x) for x in os.listdir(input_path)]
    
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    # os.makedirs(f"../results", exist_ok=True)
    # # Set up logging
    # log_file_path = f"../results/{metric}_record.txt"
    # # Delete the log file if it exists
    # if os.path.exists(log_file_path):
    #     os.remove(log_file_path)
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # # File handler for writing logs to a file
    # file_handler = logging.FileHandler(filename=f"../results/{metric}_record.txt")
    # file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    # logger.addHandler(file_handler)
    # # Stream handler for displaying logs in the terminal
    # stream_handler = logging.StreamHandler()
    # stream_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    # logger.addHandler(stream_handler)  
    logging.info("Computing IS starts!")
    device = torch.device(f'cuda:{gpu_id}')
    mean, std, scores = inception_score(video_paths, num_splits, verbose=False)
    # ipdb.set_trace()
    # print(f"Inception score: {mean}, std: {std}.")

    output_path = os.path.join(output_path_root, "IS")
    os.makedirs(output_path, exist_ok = True)
    output_file_path = os.path.join(output_path, "inception_score_results.txt")
    with open(output_file_path, "w") as output_file:
        output_file.write(f"The final inception scores:\n Mean: {mean}; Std.: {std}; Scores:\n{scores}")
    
    logging.info("Computing IS is finished!")

# if __name__ == "__main__":
#     main()
