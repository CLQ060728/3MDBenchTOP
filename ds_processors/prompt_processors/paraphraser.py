# paraphrase image captions into one single description
# Author: Qian Liu
import transformers
import torch


def get_sys_prompts(content_type):
    sys_prompt_template_1 = f"You are a precise captioner who can rephrase multiple sentences into a paragraph of a detailed {content_type} description "\
                            f"with all the information from the multiple sentences. The paragraph of the detailed {content_type} description "\
                            "must end with a complete sentence."
    sys_prompt_template_2 = f"You are a precise captioner who can paraphrase multiple sentences into a paragraph of a detailed image description "\
                            f"with all the information extracted from the multiple sentences. The paragraph of the detailed {content_type} "\
                            "description must end with a complete sentence."
    sys_prompt_template_3 = f"You can precisely paraphrase multiple sentences into a paragraph of a detailed {content_type} description."\
                            "The paraphrased paragraph has all the information from the multiple sentences. The paragraph must "\
                            "end with a complete sentence."
    sys_prompt_template_4 = f"You can precisely rephrase multiple sentences into a paragraph of a detailed {content_type} description."\
                            "The rephrased paragraph includes all the information from the multiple sentences. The paragraph must "\
                            "end with a complete sentence."
    sys_prompt_template_5 = f"You can precisely rephrase multiple sentences into a detailed {content_type} description."\
                            f"The rephrased {content_type} description has all the information from the multiple sentences. The rephrased "\
                            f"{content_type} description must end with a complete sentence."
    sys_prompt_template_6 = f"You can precisely paraphrase multiple sentences into a detailed {{content_type}} description."\
                            f"The paraphrased {content_type} description includes all the information from the multiple sentences. The "\
                            f"paraphrased {content_type} description must end with a complete sentence."
    sys_prompt_template_7 = f"You can precisely rephrase multiple sentences into a paragraph. The paragraph is a detailed {content_type} description that "\
                            "covers all the information from the multiple sentences. The paragraph must end with a complete sentence."  
    sys_prompt_template_8 = f"You can precisely paraphrase multiple sentences into a paragraph. The paragraph is a detailed {content_type} description that "\
                            "covers all the information from the multiple sentences. The paragraph must end with a complete sentence."
    # 50 stableDiffusion3; 45 deepfloydIF;
    sys_prompt_template_9 = f"You can precisely summarize multiple sentences into a paragraph. The paragraph is a concise {content_type} description that "\
                            "covers all the information from the multiple sentences. The length of the paragraph must be equal to 50 words."
    sys_prompt_template_10 = f"You can precisely summarize multiple sentences into a concise {content_type} description. The concise {content_type} description "\
                             "has all the information from the multiple sentences. The length of the concise image description must be equal to "\
                             "50 words."
    sys_prompt_template_11 = f"You can precisely summarize multiple sentences into a paragraph. The paragraph is a concise {content_type} description that "\
                             "covers all the information from the multiple sentences. The length of the paragraph must be equal to 45 words."
    sys_prompt_template_12 = f"You can precisely summarize multiple sentences into a concise {content_type} description. The concise {content_type} description "\
                             "has all the information from the multiple sentences. The length of the concise image description must be equal to "\
                             "45 words."
    sys_prompts = [sys_prompt_template_1, sys_prompt_template_2, sys_prompt_template_3,
                   sys_prompt_template_4, sys_prompt_template_5, sys_prompt_template_6,
                   sys_prompt_template_7, sys_prompt_template_8, sys_prompt_template_9,
                   sys_prompt_template_10, sys_prompt_template_11, sys_prompt_template_12]
    
    return sys_prompts


def get_paraphrase_pipeline(model_id):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    
    return pipeline, terminators


def paraphrase_image_captions(pipeline, terminators, caption_list, separator, content_type):
    caption_str = get_caption_string(caption_list, separator)
    
    messages = []
    for sys_str in get_sys_prompts(content_type):
        message = [
                {"role": "system", "content": sys_str},
                {"role": "user", "content": caption_str},
        ]
        messages.append(message)
    
    outputs = pipeline(
    messages,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.5,
    top_p=0.95,
    )

    para_captions = []
    for out_idx in range(len(outputs)):
        result_caption = outputs[out_idx][0]["generated_text"][-1]["content"]
        if ":\n\n" in result_caption:
            result_caption = result_caption.split(":\n\n")[1]
        if "\n\n" in result_caption:
            result_caption = result_caption.replace("\n\n", " ")
        if not result_caption.endswith("."):
            print(f"result_caption: {result_caption};\n len(result_caption): {len(result_caption)};")
            if result_caption.startswith('"'):
                result_caption = result_caption[1:]
            if result_caption.endswith('"'):
                result_caption = result_caption[:(len(result_caption) - 1)]
            
            point_index = (result_caption.rfind(".") + 1) if result_caption.rfind(".") != -1 else len(result_caption)
            result_caption = result_caption[:point_index]
            
            print(f"result_caption: {result_caption};\n len(result_caption): {len(result_caption)}; point_index: {point_index};")
        
        para_captions.append(result_caption)
    
    return para_captions


def get_caption_string(caption_list, separator="."):
    caption_str = "" 
    for caption in caption_list:
        caption_str += f"{caption}{separator} "
    
    return caption_str