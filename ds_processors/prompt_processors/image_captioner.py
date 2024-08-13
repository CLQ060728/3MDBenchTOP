# generate image captions using LLaVA-NeXT
# Author: Qian Liu

LLAVA_PATH = None

import builtins
import sys

if hasattr(builtins, "LLAVA_PATH_"):
    LLAVA_PATH = builtins.LLAVA_PATH_
else:
    assert LLAVA_PATH is not None, "Please specify LLAVA_PATH!"
    
sys.path.insert(0, LLAVA_PATH)

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN
from llava.constants import DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle

from PIL import Image
import requests
import copy
import torch


def get_llava_next_llama3_8b_model(device):
    pretrained = "lmms-lab/llama3-llava-next-8b"
    model_name = "llava_llama3"  # 
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, None, model_name, device_map=device_map) 
    # Add any other thing you want to pass in llava_model_args
    
    model.eval()
    model.tie_weights()
    
    conv_template = "llava_llama_3" #  
    question = DEFAULT_IMAGE_TOKEN + "\nWhat is shown in this image?"
    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    return tokenizer, model, input_ids, image_processor


def get_image_captions(tokenizer, model, input_ids, image_processor, img_paths, device):
    image_captions = []
    
    for img_path in img_paths:
        image = Image.open(img_path)
        image_tensor = process_images([image], image_processor, model.config)
        image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
        image_sizes = [image.size]
        
        cont = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True,
            temperature=0.6,
            max_new_tokens=256,
        )
        text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        image_captions.append(text_outputs)

    return image_captions