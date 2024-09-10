# 3MDBench The Multi-Modalities Multi-Domain Deep Fake Detection Benchmark
## Quick Start
### 1. Installation
```
# note that to guarantee successfully running our 3MDBench, please make sure to run it with python 3.11.*.
# Because of python's compatibility issues, we have to reconfigure flash-attn with below script file.
pip install --upgrade -r requirements.txt
./reset.sh
pip install --upgrade flash-attn
```
### 2. Functionalities
- Generate 'raw prompts' and 'paraphrased prompts'
```
# cd project directory
python ./main.py --functionality "PROMPT_GENERATION" --project_root [PROJECT_PATH] --max_bound [ORIGINAL_PROMPT_FILE_SPLIT_UPPER_BOUND] --previous_bound [ORIGINAL_PROMPT_FILE_SPLIT_LOWER_BOUND] --gpu_id 0 > output.log 2>&1 &
disown
```
- Generate 'image captioning prompts'
```
# cd project directory
python ./main.py --functionality "IMG_CAPTIONING" --project_root [PROJECT_PATH] --max_bound [ORIGINAL_PROMPT_FILE_SPLIT_UPPER_BOUND] --previous_bound [ORIGINAL_PROMPT_FILE_SPLIT_LOWER_BOUND] --gpu_id 0 > output.log 2>&1 &
disown
```
- Generate images with three different types of text prompts
```
# cd project directory
python main.py --functionality "IMG_GENERATION" --project_root [PROJECT_PATH] --gen_model [GENERATION MODEL, i.e., 'Kandinsky3', 'PixArt_Σ', 'StableDiffusion3', 'DeepFloydIF', 'StableDiffusionXL'] --prompt_type [PROMPT TYPE, i.e., 'raw_prompt', 'para_prompt' and 'cap_prompt'] --max_bound [ORIGINAL_PROMPT_FILE_SPLIT_UPPER_BOUND] --gen_width [WIDTH OF GENERATED IMAGES] --gen_height [HEIGHT OF GENERATED IMAGES] --gpu_id 0 > output.log 2>&1 &
disown
```