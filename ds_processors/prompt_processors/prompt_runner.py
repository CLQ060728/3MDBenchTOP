import paraphraser as para
import os, json


def run(project_root, data_root, file_suffix):
    dict_file_path = os.path.join(data_root, f"sel_captions_{file_suffix}.txt")
    with open(dict_file_path, "r") as dict_file:
        selected_img_cap_dict = json.load(dict_file)

    raw_img_caption_prefixes = ["a high fidelity, high resolution image of ", "a high fidelity, high resolution picture of ",
                                "a high fidelity, high resolution photo of ", "a high fidelity, high resolution photograph of ",
                                "a high fidelity, high resolution, realistic image of "]
    try:
        raw_img_caption_dict = dict()
        para_img_caption_dict = dict()
        raw_out_file_path = os.path.join(data_root, "generated", f"raw_img_caption_{file_suffix}.txt")
        para_out_file_path = os.path.join(data_root, "generated", f"para_img_caption_{file_suffix}.txt")
        with open(raw_out_file_path, "a") as raw_out_file, open(para_out_file_path, "a") as para_out_file:
            model_id = os.path.join(project_root, "llama3/Meta-Llama-3-8B-Instruct")
            counter = 1
            for sel_key, sel_cap_list in selected_img_cap_dict.items():
                cap_string = para.get_caption_string(sel_cap_list, "")
                raw_img_caption_dict[sel_key] = []
                for raw_img_prefix in raw_img_caption_prefixes:
                    raw_img_caption_dict[sel_key].append(f"{raw_img_prefix}{cap_string}")
                pipeline, terminators = para.get_paraphrase_pipeline(model_id)
                para_captions = para.paraphrase_image_captions(pipeline, terminators, sel_cap_list, "")
                para_img_caption_dict[sel_key] = para_captions

                if counter % 1000 == 0:
                    json.dump(raw_img_caption_dict, raw_out_file, indent=4)
                    json.dump(para_img_caption_dict, para_out_file, indent=4)
                    raw_out_file.write(f"\n counter: {counter}\n")
                    para_out_file.write(f"\n counter: {counter}\n")
                    raw_img_caption_dict.clear()
                    para_img_caption_dict.clear()
                    raw_out_file.flush()
                    para_out_file.flush()
                    # Force write the data to disk
                    os.fsync(raw_out_file.fileno())
                    os.fsync(para_out_file.fileno())
                    print(f"counter: {counter}")
                counter += 1
    except IOError as e:
        print (f"IO Error: {e.strerror}")


if __name__ == "__main__":
    project_root = "/home/jovyan/3MDBench/"
    data_root = os.path.join(project_root, "data/IMAGEs/MSCOCO/")
    file_suffix = 20000
    run(project_root, data_root, file_suffix)