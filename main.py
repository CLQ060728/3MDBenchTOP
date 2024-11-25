import argparse
import os
import torch
import numpy as np
import json
from ds_processors.video_generators import video_gen_runner as vgr


def get_args_parser():
    parser = argparse.ArgumentParser('3MDBench', add_help=False)
    # parser.add_argument bool value, default must be False, even you make it to True, it is still False, when set in the command line, it's always True;  'surf_keypoints', """ + """'surf_matching', 'surf_descriptor'
    parser.add_argument('--functionality', default="", type=str, required=True,
        help="""Choose the functionality to use, which includes 'IMG_CAPTIONING', 'VIDEO_CAPTIONING', """
                + """'PROMPT_GENERATION', 'IMG_GENERATION', 'VIDEO_GENERATION', """
                + """'DCT', 'DFT', 'POWER', 'GLCM', 'TEXTURE_DESCRIPTORS'. For videos: """
                + """'VQA', 'IS', 'flow_score', 'warping_error', 'peak_est', 'key_frames', """
                + """'sift_keypoints', 'sift_matching', 'sift_descriptor'.""")
    parser.add_argument('--project_root', default="./", type=str,
                        help="""Specify the root directory for 3MDBench project.""")
    parser.add_argument('--gpu_id', default=0, type=int, help="""Specify the gpu id.""")
    parser.add_argument('--dataset_name', default="", type=str, required=True, 
                        help="""Specify dataset name, i.e., 'MSCOCO', 'CC3M', 'VISUAL_GENOME', """
                       + """ 'GEOSYNTH', 'MSR-VTT', 'CelebV-Text'.""")
    
    parser.add_argument('--aggregate', default=False, type=bool, 
           help="""Whether to combine the generated image captioning files (for image captioning).""")
    parser.add_argument('--max_bound', default=5000, type=int, 
                        help="""Specify the max file bound for 'IMG_CAPTIONING', """
                        +"""'VIDEO_CAPTIONING' and 'PROMPT_GENERATION'.""")
    parser.add_argument('--previous_bound', default=0, type=int, 
                        help="""Specify the previous file bound for 'IMG_CAPTIONING', """
                        + """'VIDEO_CAPTIONING' and 'PROMPT_GENERATION'.""")
    parser.add_argument('--cap_model', default="", type=str, 
                        help="""Specify the image captioning model for prompts generation,"""
                        + """ i.e., 'LLAVA', 'QWEN'.""")
    
    parser.add_argument('--gen_model', default="", type=str, 
                        help="""Specify the generation model for generation and profiling,"""
                        + """ i.e., 'Kandinsky3', 'PixArt_Σ', 'StableDiffusion3', 'DeepFloydIF',"""
                        + """ 'GeoSynth', 'StableDiffusionXL', 'OpenSora1_2', 'CogVideoX',"""      
                        + """ 'VideoCrafter', 'PyramidFlow'.""")
    parser.add_argument('--prompt_type', default="raw_prompt", type=str, 
                        help="""Specify prompt type, 'raw_prompt', 'para_prompt', 'cap_prompt'.""")
    parser.add_argument('--gen_width', default=512, type=int,
                        help="""The width of the generated images/frames.""")
    parser.add_argument('--gen_height', default=512, type=int,
                        help="""The height of the generated images/frames.""")
    parser.add_argument('--text2image', default=False, type=bool,
                        help="""Whether to use text prompts to generate images.""")
    parser.add_argument('--manual_seed', default=False, type=bool,
                        help="""Whether to use manual generated random"""
                        + """seed.""")
    parser.add_argument('--seed', default=None, type=int,
                        help="""The manually specified random seed.""")
    
    parser.add_argument('--real_path', default="", type=str, 
                        help="""Specify the path for real images/videos for profilers.""")
    parser.add_argument('--fake_path', default="", type=str, 
                        help="""Specify the path for fake images/videos for profilers.""")
    
    parser.add_argument('--colour_gray', default=False, type=bool, 
        help="""Profiling based on colour images/frames or gray-level images/frames, False - gray.""")
    parser.add_argument('--full_band', default=False, type=bool, 
           help="""Profiling based on full images/frames bands.""")
    # parser.add_argument('--real_or_fake', default=False, type=bool, 
           # help="""Profiling based on real or fake images/frames.""")
    parser.add_argument('--total_amount', default=80000, type=int, 
                        help="""Specify the total amount of profiling image/video files.""")
    parser.add_argument('--task_amount', default=1000, type=int, 
                        help="""Specify the amount of profiling image/video files per task.""")
    parser.add_argument('--output_path', default="./output/", type=str,
                        help="""Specify the output path for profilings.""")
    parser.add_argument('--diff_threshold', default=0.3, type=float, 
                        help="""Threshold of the image difference for peak estimation.""")
    # parser.add_argument('--out_sizes', nargs='+', type=int, 
    # help="""Embedding layer output feature sizes""")
    
    return parser


def main(args_main):
    if args_main.functionality == "IMG_CAPTIONING":
        device = torch.device(f'cuda:{args_main.gpu_id}' if torch.cuda.is_available() else 'cpu')
        import builtins
        builtins.LLAVA_PATH_ = os.path.join(args_main.project_root,
                                            "ds_processors/prompt_processors/LLaVA_NeXT")
        import ds_processors.prompt_processors.img_captioner_runner as icr
        data_root = f"{args_main.project_root}data/IMAGEs/{args_main.dataset_name}/"
        cache_dir = f"{args_main.project_root}ds_processors/prompt_processors/Qwen2VL/"
        icr.run(data_root, args_main.cap_model, cache_dir, args_main.max_bound,
                args_main.previous_bound, device, args_main.aggregate)
    elif args_main.functionality == "VIDEO_CAPTIONING":
        # num_of_gpus = torch.cuda.device_count()
        pllava_root = os.path.join(args_main.project_root,
                                   "ds_processors/prompt_processors/PLLaVA")
        import builtins
        builtins.PLLAVA_PATH_ = pllava_root
        builtins.OPENSORA_PATH_ = os.path.join(args_main.project_root,
                                              "ds_processors/video_generators/OpenSora1_2/OpenSora/")
        import ds_processors.prompt_processors.video_captioner_runner as vcr
        data_root = f"{args_main.project_root}data/VIDEOs/{args_main.dataset_name}/"
        cache_dir = os.path.join(pllava_root, "models")
        resolution = 672
        vcr.run(data_root, cache_dir, args_main.max_bound, args_main.previous_bound,
                args_main.gpu_id, resolution, args_main.aggregate)
    elif args_main.functionality == "PROMPT_GENERATION":
        import ds_processors.prompt_processors.prompt_runner as pr
        if args_main.dataset_name in ["MSCOCO", "CC3M", "VISUAL_GENOME"]:
            content_type = "image"
            data_root = f"{args_main.project_root}data/IMAGEs/{args_main.dataset_name}/"
        elif args_main.dataset_name in ["MSR-VTT", "CelebV-Text"]:
            content_type = "video"
            data_root = f"{args_main.project_root}data/VIDEOs/{args_main.dataset_name}/"
        pr.run(args_main.project_root, data_root, args_main.max_bound, content_type)
    elif args_main.functionality == "IMG_GENERATION":
        import builtins
        code_dir_root = os.path.join(args_main.project_root, "ds_processors")
        builtins.CODE_DIR_ROOT_ = code_dir_root
        import ds_processors.image_generators.img_gen_runner as igr
        device = torch.device(f'cuda:{args_main.gpu_id}' if torch.cuda.is_available() else 'cpu')
        if args_main.gen_model == "GeoSynth":
            igr.run_geo_synth(args_main.project_root, args_main.dataset_name, args_main.gen_model,
                              args_main.gpu_id)
        else:
            igr.run(args_main.project_root, args_main.dataset_name, args_main.gen_model, args_main.max_bound,
                    args_main.gen_width, args_main.gen_height, args_main.prompt_type, device, args_main.gpu_id, 
                    t2i_or_i2i = args_main.text2image, manual_seed=args_main.manual_seed, seed=args_main.seed)
    elif args_main.functionality == "VIDEO_GENERATION":
        vgr.run(args_main)
    elif args_main.functionality == "DCT":
        import ds_profiling.avg_dct_spectrum as ads
        profiler_name = "DCT"
        size = (512, 512)
        ds_paths = [f"{args_main.real_path},REAL", f"{args_main.fake_path}, FAKE"]
        out_path = os.path.join(args_main.output_path, profiler_name)
        os.makedirs(out_path, exist_ok = True)
        ads.compute_dct(args_main.total_amount, size, ds_paths, args_main.colour_gray, 
                        args_main.dataset_name, args_main.gen_model, out_path)
    elif args_main.functionality == "DFT":
        import ds_profiling.avg_dft_spectrum as ads
        size = (512, 512)
        profiler_name = "DFT"
        images_real = ads.load_images(args_main.real_path, args_main.total_amount, args_main.task_amount, size)
        images_fake = ads.load_images(args_main.fake_path, args_main.total_amount, args_main.task_amount, size)
        out_path = os.path.join(args_main.output_path, profiler_name, args_main.dataset_name,
                                args_main.gen_model)
        os.makedirs(out_path, exist_ok = True)
        ads.visualize_average_frequency_spectra(images_real, images_fake, "Average DFT Spectrums", 
                                                rgb_or_g=args_main.colour_gray, 
                                                full_band=args_main.full_band, out_path=out_path)
    elif args_main.functionality == "POWER":
        import ds_profiling.avg_dft_power_spectrum as adps
        size = (512, 512)
        fig_size = (20, 4)
        profiler_name = "POWER"
        images_real = adps.load_images(args_main.real_path, args_main.total_amount, args_main.task_amount, size)
        images_fake = adps.load_images(args_main.fake_path, args_main.total_amount, args_main.task_amount, size)
        out_path = os.path.join(args_main.output_path, profiler_name, args_main.dataset_name,
                                args_main.gen_model)
        os.makedirs(out_path, exist_ok = True)
        adps.visualize_average_frequency_power_spectra([images_real, images_fake], 
                                                       [f"{args_main.dataset_name} REAL", 
                                                        f"{args_main.dataset_name} {args_main.gen_model}"], 
                                                       rgb_or_g=args_main.colour_gray, out_path=out_path)
        adps.visualise_average_images_psd([images_real, images_fake], 
                                          [f"{args_main.dataset_name} REAL", 
                                           f"{args_main.dataset_name} {args_main.gen_model}"], fig_size, 
                                          args_main.colour_gray, out_path=out_path)
    elif args_main.functionality == "GLCM":
        import ds_profiling.img_ds_statistics_cooccurrence as idsc
        profiler_name = "GLCM"
        input_path = args_main.real_path
        real_or_fake = "real"
        out_path = os.path.join(args_main.output_path, profiler_name, args_main.dataset_name, 
                                args_main.gen_model, real_or_fake)
        os.makedirs(out_path, exist_ok = True)
        idsc.compute_glcm_textures(input_path, args_main.colour_gray, args_main.total_amount, args_main.task_amount, 
                                   output_path=out_path)
        input_path = args_main.fake_path
        real_or_fake = "fake"
        out_path = os.path.join(args_main.output_path, profiler_name, args_main.dataset_name, 
                                args_main.gen_model, real_or_fake)
        os.makedirs(out_path, exist_ok = True)
        idsc.compute_glcm_textures(input_path, args_main.colour_gray, args_main.total_amount, args_main.task_amount, 
                                   output_path=out_path)
    elif args_main.functionality == "TEXTURE_DESCRIPTORS":
        import ds_profiling.img_ds_texture_descriptors as idtd
        profiler_name = "TEXTURE_DESCRIPTORS"
        real_descriptors_dict = idtd.get_ds_channels_descriptors(args_main.real_path, 
                                                                 args_main.total_amount, args_main.task_amount)
        fake_descriptors_dict = idtd.get_ds_channels_descriptors(args_main.fake_path, 
                                                                 args_main.total_amount, args_main.task_amount)
        descriptor_names = ["LBP", "CoALBPs", "LPQ"]
        channels = ["Y", "Cr", "Cb", "H", "S", "V"]
        real_generator_name = f"{args_main.dataset_name}_REAL"
        fake_generator_name = f"{args_main.dataset_name}_{args_main.gen_model}"
        out_path = os.path.join(args_main.output_path, profiler_name)
        os.makedirs(out_path, exist_ok = True)
        for descriptor_name in descriptor_names:
            for channel in channels:
                descr_dict_key = f"{channel}_{descriptor_name}"
                if descriptor_name == "CoALBPs":
                    for idx in range(3):
                        idtd.histogram_full_image_comparison(real_descriptors_dict[descr_dict_key][idx], 
                                                             fake_descriptors_dict[descr_dict_key][idx], 
                                                             f"{descriptor_name}_{idx}", real_generator_name, 
                                                             fake_generator_name, channel, out_path)
                else:
                    idtd.histogram_full_image_comparison(real_descriptors_dict[descr_dict_key], 
                                                         fake_descriptors_dict[descr_dict_key], 
                                                         descriptor_name, real_generator_name, 
                                                         fake_generator_name, channel, out_path)
    elif args_main.functionality == "VQA" or args_main.functionality == "IS" or \
         args_main.functionality == "flow_score" or args_main.functionality == "warping_error":
        import ds_profiling.vid_ds_eval as vde
        if args_main.real_path != "":
            vid_set_name = f"{args_main.dataset_name}_real"
            data_path = args_main.real_path
        elif args_main.fake_path != "":
            vid_set_name = f"{args_main.dataset_name}_fake"
            data_path = args_main.fake_path
        else:
            raise ValueError('Please either specify real_path or fake_path!')
        
        output_path_root = os.path.join(args_main.output_path, vid_set_name)
        vde.run_evalcrafter_metrics(args_main.project_root, data_path, args_main.functionality,
                                    args_main.gpu_id, vid_set_name, output_path_root)
    elif args_main.functionality == "peak_est" or args_main.functionality == "key_frames":
        if args_main.real_path != "":
            vid_set_name = f"{args_main.dataset_name}_real"
            vid_file_path = args_main.real_path
        elif args_main.fake_path != "":
            vid_set_name = f"{args_main.dataset_name}_fake"
            vid_file_path = args_main.fake_path
        else:
            raise ValueError('Please either specify real_path or fake_path!')
            
        output_path = os.path.join(args_main.output_path, vid_set_name, args_main.functionality)
        import ds_profiling.vid_key_frames_extractor as vkfe
        if args_main.functionality == "peak_est":
            vkfe.compute_peak_est_key_frames(vid_file_path, output_path, args_main.diff_threshold)
        elif args_main.functionality == "key_frames":
            vkfe.compute_katna_key_frames(vid_file_path, output_path)
    elif args_main.functionality.endswith("keypoints") or \
         args_main.functionality.endswith("matching") or \
         args_main.functionality.endswith("descriptor"):
        descriptor_type = "sift" # args_main.functionality.split("_")[0]
        print(f"descriptor type: {descriptor_type}")
        if args_main.real_path == "" and args_main.fake_path == "":
           raise ValueError(
               'Please specify real_path, fake_path, or both (for image keypoints matching).')
        elif args_main.real_path != "":
            img_file_path = args_main.real_path
        elif args_main.fake_path != "":
            img_file_path = args_main.fake_path

        file_full_name = img_file_path.split("/")[-1]
        file_name = file_full_name[:file_full_name.index(".")]
        print(f"file name: {file_name}")
        output_path_root = os.path.join(args_main.output_path, args_main.dataset_name,
                                        args_main.functionality)
        os.makedirs(output_path_root, exist_ok = True)
        import ds_profiling.img_descriptors as ides
        if args_main.functionality.endswith("keypoints"):
            output_file_path = os.path.join(output_path_root, f"{file_name}_keypoints.png")
            ides.save_keypoints_annotated_img(img_file_path, output_file_path, descriptor_type)
        elif args_main.functionality.endswith("descriptor"):
            output_file_path = os.path.join(output_path_root, f"{file_name}_descriptor.txt")
            if descriptor_type == "sift":
                _, img_descriptor = ides.get_img_sift_descriptor(img_file_path)
            else:
                _, img_descriptor = ides.get_img_surf_descriptor(img_file_path)
            np.savetxt(output_file_path, img_descriptor, encoding="utf-8")
        else:
            output_file_path = os.path.join(output_path_root, f"{file_name}_matching.png")
            img1_file_path = args_main.real_path
            img2_file_path = args_main.fake_path
            ides.save_keypoints_matching_imgs(img1_file_path, img2_file_path,
                                              output_file_path, descriptor_type)
    

if __name__ == '__main__':
    parser_main = argparse.ArgumentParser('3MDBench_main', parents=[get_args_parser()])
    parser_main = vgr.get_opensora_args(parser_main)
    args_main = parser_main.parse_args()
    main(args_main)