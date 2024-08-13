# run image captioner in batch
# Author: Qian Liu

import builtins
builtins.LLAVA_PATH_ = "/home/jovyan/3MDBench/ds_processors/prompt_processors/LLaVA_NeXT/"
import image_captioner as ic
import os, json
import torch


if __name__ == "__main__":
    path_root = "/home/jovyan/3MDBench/data/IMAGEs/MSCOCO/"
    dict_file_path = os.path.join(path_root, "sel_captions_5000.txt")
    img_cap_dict_file_path = os.path.join(path_root, "image_caption_dict.txt")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(dict_file_path, "r") as dict_file, open(img_cap_dict_file_path, "r") as img_cap_dict_file:
        selected_img_cap_dict = json.load(dict_file)
        img_cap_dict = json.load(img_cap_dict_file)
    sel_path = os.path.join(path_root, "selected")
    out_file_path = os.path.join(path_root, "generated", "captioner_img_caption_5000_10.txt")
    img_paths = []
    for img_id in list(selected_img_cap_dict.keys()):
        if img_id in img_cap_dict:
            img_path = os.path.join(sel_path, img_cap_dict[img_id][0])
            img_paths.append(img_path)

    tokenizer, model, input_ids, image_processor = ic.get_llava_next_llama3_8b_model(device)
    with open(out_file_path, "a") as out_file:
        for counter in range(1, 6, 1):
            image_captions = ic.get_image_captions(tokenizer, model, input_ids, image_processor,
                                                img_paths[((counter-1)*1000):(counter*1000)], device)
            json.dump(image_captions, out_file, indent=4)
            out_file.write(f"\n counter:[{((counter-1)*1000)}:{(counter*1000)}]\n")
            print(f"counter:[{((counter-1)*1000)}:{(counter*1000)}]")
    