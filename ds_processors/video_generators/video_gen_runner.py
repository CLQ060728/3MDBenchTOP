import argparse
import os
import torch
import json
from glob import glob
from mmengine.config import Config
import numpy as np


def is_folder_empty(folder_path):
    # Check if the folder is empty
    return not any(os.scandir(folder_path))


def get_opensora_args(parser):
    # model config
    # parser.add_argument("--sora_config_path", required=True, default=None,
    #                     type=str, help="model config file path")

    # ======================================================
    # General
    # ======================================================
    # parser.add_argument("--seed", default=None, type=int, help="seed for reproducibility")
    parser.add_argument(
        "--ckpt-path",
        default=None,
        type=str,
        help="path to model ckpt; will overwrite cfg.model.from_pretrained if specified",
    )
    parser.add_argument("--batch-size", default=None, type=int, help="batch size")
    parser.add_argument("--outputs", default=None, type=str, help="the dir to save model weights")
    parser.add_argument("--flash-attn", default=None, type=str2bool, help="enable flash attention")
    parser.add_argument("--layernorm-kernel", default=None, type=str2bool, help="enable layernorm kernel")
    parser.add_argument("--resolution", default=None, type=str, help="multi resolution")
    parser.add_argument("--data-path", default=None, type=str, help="path to data csv")
    parser.add_argument("--dtype", default="bf16", type=str, help="data type")

    # ======================================================
    # Inference
    # ======================================================
    # if not training:
    # output
    # parser.add_argument("--save-dir", default=None, type=str, help="path to save generated samples")
    # parser.add_argument("--sample-name", default=None, type=str, help="sample name, default is sample_idx")
    # parser.add_argument("--start-index", default=None, type=int, help="start index for sample name")
    # parser.add_argument("--end-index", default=None, type=int, help="end index for sample name")
    parser.add_argument("--num-sample", default=None, type=int, help="number of samples to generate for one prompt")
    parser.add_argument("--prompt-as-path", action="store_true", help="use prompt as path to save samples")
    # parser.add_argument("--verbose", default=None, type=int, help="verbose level")

    # prompt
    parser.add_argument("--prompt-path", default=None, type=str, help="path to prompt txt file")
    parser.add_argument("--prompt", default=None, type=str, nargs="+", help="prompt list")
    parser.add_argument("--llm-refine", default=None, type=str2bool, help="enable LLM refine")
    parser.add_argument("--prompt-generator", default=None, type=str, help="prompt generator")

    # image/video
    parser.add_argument("--num-frames", default=None, type=str, help="number of frames")
    parser.add_argument("--fps", default=None, type=int, help="fps")
    parser.add_argument("--save-fps", default=None, type=int, help="save fps")
    parser.add_argument("--image-size", default=None, type=int, nargs=2, help="image size")
    parser.add_argument("--frame-interval", default=None, type=int, help="frame interval")
    parser.add_argument("--aspect-ratio", default=None, type=str, help="aspect ratio (h:w)")
    parser.add_argument("--watermark", default=None, type=str2bool, help="watermark video")

    # hyperparameters
    parser.add_argument("--num-sampling-steps", default=50, type=int, help="sampling steps")
    parser.add_argument("--cfg-scale", default=None, type=float, help="balance between cond & uncond")

    # reference
    parser.add_argument("--loop", default=None, type=int, help="loop")
    parser.add_argument("--condition-frame-length", default=None, type=int, help="condition frame length")
    parser.add_argument("--reference-path", default=None, type=str, nargs="+", help="reference path")
    parser.add_argument("--mask-strategy", default=None, type=str, nargs="+", help="mask strategy")
    parser.add_argument("--aes", default=None, type=float, help="aesthetic score")
    parser.add_argument("--flow", default=None, type=float, help="flow score")
    parser.add_argument("--camera-motion", default=None, type=str, help="camera motion")
    # ======================================================
    # Training
    # ======================================================
    # else:
    #     parser.add_argument("--lr", default=None, type=float, help="learning rate")
    #     parser.add_argument("--wandb", default=None, type=bool, help="enable wandb")
    #     parser.add_argument("--load", default=None, type=str, help="path to continue training")
    #     parser.add_argument("--start-from-scratch", action="store_true", help="start training from scratch")
    #     parser.add_argument("--warmup-steps", default=None, type=int, help="warmup steps")
    #     parser.add_argument("--record-time", default=False, action="store_true", help="record time of each part")
    
    return parser


def merge_args(cfg, args):
    training = False
    if args.ckpt_path is not None:
        cfg.model["from_pretrained"] = args.ckpt_path
        if cfg.get("discriminator") is not None:
            cfg.discriminator["from_pretrained"] = args.ckpt_path
        args.ckpt_path = None
    if args.flash_attn is not None:
        cfg.model["enable_flash_attn"] = args.flash_attn
        args.enable_flash_attn = None
    if args.layernorm_kernel is not None:
        cfg.model["enable_layernorm_kernel"] = args.layernorm_kernel
        args.enable_layernorm_kernel = None
    if args.data_path is not None:
        cfg.dataset["data_path"] = args.data_path
        args.data_path = None
    # NOTE: for vae inference (reconstruction)
    if not training and "dataset" in cfg:
        if args.image_size is not None:
            cfg.dataset["image_size"] = args.image_size
        if args.num_frames is not None:
            cfg.dataset["num_frames"] = args.num_frames
    if not training:
        if args.cfg_scale is not None:
            cfg.scheduler["cfg_scale"] = args.cfg_scale
            args.cfg_scale = None
        if args.num_sampling_steps is not None:
            cfg.scheduler["num_sampling_steps"] = args.num_sampling_steps
            args.num_sampling_steps = None

    for k, v in vars(args).items():
        if v is not None:
            cfg[k] = v

    return cfg


def read_config(config_path):
    cfg = Config.fromfile(config_path)
    return cfg


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")
##################################OPENSORA ABOVE##########################################
def get_prompt_list(prompt_type, prompt_root, max_bound):
    if prompt_type == "raw_prompt":
        prompt_file_path = os.path.join(prompt_root, f"raw_vid_caption_{max_bound}.txt")
    elif prompt_type == "para_prompt":
        prompt_file_path = os.path.join(prompt_root, f"para_vid_caption_{max_bound}.txt")
    elif prompt_type == "cap_prompt":
        prompt_file_path = os.path.join(prompt_root, f"captioner_vid_caption_{max_bound}.txt")
    with open(prompt_file_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    prompt_keys = list(prompt_dict.keys())
    # index_key_mapping = dict()
    # for prompt_key_index in range(len(list(prompt_dict.keys()))):
    #     index_key_mapping[prompt_key_index] = prompt_dict_keys[prompt_key_index]
    
    # prompt_keys = [] prompt_key
    prompt_list = []
    for _, prompt_val_list in prompt_dict.items():
        index_list = [val for val in range(len(prompt_val_list))]
        sel_index = np.random.choice(index_list, size=1, replace=False)
        # prompt_keys.append(prompt_key)
        prompt_list.append(prompt_val_list[sel_index[0]])
    
    return prompt_keys, prompt_list
    
    
def run(args_main):
    prompt_root = os.path.join(args_main.project_root, "data/VIDEOs", args_main.dataset_name,
                               "generated")
    prompt_keys, prompts_list = get_prompt_list(args_main.prompt_type, prompt_root,
                                                args_main.max_bound)
    args_main.save_dir = os.path.join(args_main.project_root, "data/VIDEOs", "generated",
                                      args_main.gen_model, args_main.dataset_name,
                                      args_main.prompt_type, args_main.resolution)
    os.makedirs(args_main.save_dir, exist_ok = True)
    device = torch.device(f"cuda:{args_main.gpu_id}" if torch.cuda.is_available() else "cpu")
    
    if args_main.gen_model == "OpenSora1_2":
        import builtins
        opensora_root = os.path.join(args_main.project_root, "ds_processors", "video_generators",
                                     "OpenSora1_2", "OpenSora")
        builtins.OPENSORA_PATH_ = opensora_root
        sora_config_path = os.path.join(opensora_root, "configs/opensora-v1-2/inference/sample.py")
        cfg = read_config(sora_config_path)
        from .OpenSora1_2.OpenSora.scripts import inference as infer
        prompts_index = 0
        for prompt in prompts_list:
            # print(f"Current prompt index: {prompts_index}; Current prompt string: {prompt}")
            args_main.seed = np.random.randint(0, 100000, size=1).item()
            args_main.prompt = [prompt]
            args_main.sample_name = f"{prompt_keys[prompts_index]}"
            print(f"Current sample name: {args_main.sample_name};")
            cfg = merge_args(cfg, args_main)
            infer.run_opensora(cfg)
            prompts_index += 1
    elif args_main.gen_model == "CogVideoX":
        from .CogVideoX import generator as gen
        cache_dir = os.path.join(args_main.project_root, "ds_processors", "video_generators",
                                 "CogVideoX", "cache")
        os.makedirs(cache_dir, exist_ok = True)
        prompts_index = 0
        for prompt in prompts_list:
            sample_name = f"{prompt_keys[prompts_index]}.mp4"
            output_file_path = os.path.join(args_main.save_dir, sample_name)
            print(f"Current sample name: {sample_name};")
            pipe = gen.get_CogVideoX_pipeline(cache_dir, device)
            gen.run_cogvideox(pipe, prompt, output_file_path)
            prompts_index += 1
    elif args_main.gen_model == "VideoCrafter":
        import builtins
        videocrafter_root = os.path.join(args_main.project_root, "ds_processors", "video_generators",
                                         "VideoCrafter")
        builtins.VIDEOCRAFTER_PATH_ = videocrafter_root
        from .VideoCrafter.scripts.evaluation import inference as vc_infer
        # args_main.ckpt_path = os.path.join(videocrafter_root, "cache")
        # os.makedirs(args_main.ckpt_path, exist_ok = True)
        args_main.mode = "base"
        args_main.bs = 1
        args_main.ddim_eta = 1.0
        
        if args_main.resolution == "512":
            args_main.config = os.path.join(videocrafter_root, "configs/inference_t2v_512_v2.0.yaml")
            args_main.ckpt_path = os.path.join(videocrafter_root, "cache", "VideoCrafter2", "model.ckpt")
        elif args_main.resolution == "1024":
            args_main.config = os.path.join(videocrafter_root, "configs/inference_t2v_1024_v1.0.yaml")
            args_main.ckpt_path = os.path.join(videocrafter_root, "cache", "VideoCrafter1_1024", "model.ckpt")
        # vc_args.savedir = args_main.save_dir
        # vc_args.savefps = args_main.save_fps
        # vc_args.frames = args_main.num_frames
        # vc_args.fps = args_main.fps
        args_main.unconditional_guidance_scale = 12.0
        if args_main.resolution == "512":
            args_main.gen_width = 512
            args_main.gen_height = 320
        elif args_main.resolution == "1024":
            args_main.gen_width = 1024
            args_main.gen_height = 576
        args_main.num_sample = 1
        prompts_index = 0
        for prompt in prompts_list:
            args_main.seed = np.random.randint(0, 100000, size=1).item()
            args_main.prompt = prompt
            args_main.sample_name = f"{prompt_keys[prompts_index]}"
            vc_infer.run_inference(args_main, device)
            prompts_index += 1
    elif args_main.gen_model == "PyramidFlow":
        pyramidflow_root = os.path.join(args_main.project_root, "ds_processors", "video_generators",
                                        "PyramidFlow")
        cache_dir = os.path.join(pyramidflow_root, "cache")
        code_dir = os.path.join(pyramidflow_root, "code")
        os.makedirs(cache_dir, exist_ok = True)
        assert args_main.resolution == "384p" or args_main.resolution == "768p", \
          "PyramidFlow resolution can only be 384p or 768p"
        
        if is_folder_empty(cache_dir):
            from huggingface_hub import snapshot_download
            model_path = cache_dir
            snapshot_download("rain1011/pyramid-flow-miniflux", local_dir=model_path,
                              local_dir_use_symlinks=False, repo_type='model')
        
        import builtins
        builtins.PYRAMIDFLOW_PATH_ = code_dir
        from .PyramidFlow import generator as pfgen
        model = pfgen.get_pyramid_flow_model(cache_dir, args_main.resolution, args_main.gpu_id)
        print(f"prompts_list length: {len(prompts_list)};")
        print(f"prompt_keys length: {len(prompt_keys)}")
        prompts_index = 0
        for prompt in prompts_list:
            # args_main.seed = np.random.randint(0, 100000, size=1).item()
            sample_name = f"{prompt_keys[prompts_index]}.mp4"
            print(f"prompts_index: {prompts_index}; sample_name: {sample_name}")
            output_path = os.path.join(args_main.save_dir, sample_name)
            pfgen.run_pyramidflow(model, prompt, args_main.resolution, output_path)
            prompts_index += 1