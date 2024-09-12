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
python ./main.py --functionality "PROMPT_GENERATION" --project_root [PROJECT_PATH] --dataset_name [“MSCOCO”, "CC3M", "VISUAL_GENOME"] --max_bound [ORIGINAL_PROMPT_FILE_SPLIT_UPPER_BOUND] --previous_bound [ORIGINAL_PROMPT_FILE_SPLIT_LOWER_BOUND] --gpu_id 0 > output.log 2>&1 &
disown
```
- Generate 'image captioning prompts'
```
# cd project directory
python ./main.py --functionality "IMG_CAPTIONING" --project_root [PROJECT_PATH] --dataset_name [“MSCOCO”, "CC3M", "VISUAL_GENOME"] --max_bound [ORIGINAL_PROMPT_FILE_SPLIT_UPPER_BOUND] --previous_bound [ORIGINAL_PROMPT_FILE_SPLIT_LOWER_BOUND] --gpu_id 0 > output.log 2>&1 &
disown
```
- Generate images with three different types of text prompts
```
# cd project directory
python main.py --functionality "IMG_GENERATION" --project_root [PROJECT_PATH] --dataset_name [“MSCOCO”, "CC3M", "VISUAL_GENOME"] --gen_model [GENERATION MODEL, i.e., 'Kandinsky3', 'PixArt_Σ', 'StableDiffusion3', 'DeepFloydIF', 'StableDiffusionXL'] --prompt_type [PROMPT TYPE, i.e., 'raw_prompt', 'para_prompt' and 'cap_prompt'] --max_bound [ORIGINAL_PROMPT_FILE_SPLIT_UPPER_BOUND] --gen_width [WIDTH OF GENERATED IMAGES] --gen_height [HEIGHT OF GENERATED IMAGES] --gpu_id 0 > output.log 2>&1 &
disown
```
- Dataset profiling
  - Discrete Cosine Transform (DCT)
  ```
  python main.py --functionality "DCT" --project_root [PROJECT_PATH] --real_path [PATH FOR REAL IMAGES/FRAMES] --fake_path [PATH FOR FAKE IMAGES/FRAMES] --total_amount [TOTAL AMOUNT OF IMAGES/FRAMES TO PROFILE] --task_amount [AMOUNT OF IMAGES/FRAMES TO PROFILE FOR EACH PARALLEL TASK] --output_path [OUTPUT PATH FOR PROFILING] > output.log 2>&1 &
  ```
  - Discrete Fourier Transform (DFT)
  ```
  python main.py --functionality "DFT" --project_root [PROJECT_PATH] --real_path [PATH FOR REAL IMAGES/FRAMES] --fake_path [PATH FOR FAKE IMAGES/FRAMES] --total_amount [TOTAL AMOUNT OF IMAGES/FRAMES TO PROFILE] --task_amount [AMOUNT OF IMAGES/FRAMES TO PROFILE FOR EACH PARALLEL TASK] --output_path [OUTPUT PATH FOR PROFILING] > output.log 2>&1 &
  ```
  - Power Spectrum of Discrete Fourier Transform (Power Spectrum of DFT)
  ```
  python main.py --functionality "POWER" --project_root [PROJECT_PATH] --real_path [PATH FOR REAL IMAGES/FRAMES] --fake_path [PATH FOR FAKE IMAGES/FRAMES] --total_amount [TOTAL AMOUNT OF IMAGES/FRAMES TO PROFILE] --task_amount [AMOUNT OF IMAGES/FRAMES TO PROFILE FOR EACH PARALLEL TASK] --output_path [OUTPUT PATH FOR PROFILING] > output.log 2>&1 &
  ```
  - Gray Level Cooccurrence Matrix (GLCM)
  ```
  python main.py --functionality "GLCM" --project_root [PROJECT_PATH] --real_path [PATH FOR REAL IMAGES/FRAMES] --fake_path [PATH FOR FAKE IMAGES/FRAMES] --total_amount [TOTAL AMOUNT OF IMAGES/FRAMES TO PROFILE] --task_amount [AMOUNT OF IMAGES/FRAMES TO PROFILE FOR EACH PARALLEL TASK] --output_path [OUTPUT PATH FOR PROFILING] > output.log 2>&1 &
  ```
  - Image/Frame Texture Descriptors (Local binary patterns (LBP), Co-occurrence among Adjacent LBPs (CoALBPs), Local Phase Quantization (LPQ))
  ```
  python main.py --functionality "TEXTURE_DESCRIPTORS" --project_root [PROJECT_PATH] --real_path [PATH FOR REAL IMAGES/FRAMES] --fake_path [PATH FOR FAKE IMAGES/FRAMES] --total_amount [TOTAL AMOUNT OF IMAGES/FRAMES TO PROFILE] --task_amount [AMOUNT OF IMAGES/FRAMES TO PROFILE FOR EACH PARALLEL TASK] --output_path [OUTPUT PATH FOR PROFILING] > output.log 2>&1 &
  ```