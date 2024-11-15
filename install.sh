pip install --upgrade setuptools==75.2.0
pip install --upgrade wheel==0.44.0
pip install --upgrade torch==2.1.2
pip install --upgrade -r requirements.txt
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ds_processors/video_generators/OpenSora1_2/apex/
pip install --upgrade ./ds_processors/video_generators/OpenSora1_2/colossalai/
pip install --upgrade ./ds_processors/prompt_processors/PLLaVA/moviepy/
mkdir -p ./ds_profiling/EvalCrafter/checkpoints/DOVER/pretrained_weights/
wget "https://github.com/w86763777/pytorch-gan-metrics/releases/download/v0.1.0/pt_inception-2015-12-05-6726825d.pth" --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/
wget https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/pretrained_weights/
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_1k_224_ema.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_small_1k_224_ema.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_base_1k_224_ema.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_224_ema.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_small_22k_224.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
wget https://dl.fbaipublicfiles.com/convnext/convnext_xlarge_22k_224.pth --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/DOVER/
mkdir ./ds_profiling/EvalCrafter/checkpoints/RAFT/
wget https://dl.dropboxusercontent.com/s/4j4z58wuv8o0mfz/models.zip --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/RAFT/
unzip ./ds_profiling/EvalCrafter/checkpoints/RAFT/models.zip -d ./ds_profiling/EvalCrafter/checkpoints/RAFT/
wget https://huggingface.co/RaphaelLiu/EvalCrafter-Models/resolve/main/FlowNet2_checkpoint.pth.tar --directory-prefix=./ds_profiling/EvalCrafter/checkpoints/
pip install --upgrade ./ds_profiling/EvalCrafter/metrics/RAFT/networks/channelnorm_package/
pip install --upgrade ./ds_profiling/EvalCrafter/metrics/RAFT/networks/correlation_package/
pip install --upgrade ./ds_profiling/EvalCrafter/metrics/RAFT/networks/resample2d_package/
pip install --upgrade ./ds_profiling/EvalCrafter/metrics/RAFT/alt_cuda_corr/
mkdir -p ./ds_processors/video_generators/VideoCrafter/cache/VideoCrafter2.0/
mkdir -p ./ds_processors/video_generators/VideoCrafter/cache/VideoCrafter1_1024/
wget https://huggingface.co/VideoCrafter/VideoCrafter2/resolve/main/model.ckpt --directory-prefix=./ds_processors/video_generators/VideoCrafter/cache/VideoCrafter2.0/
wget https://huggingface.co/VideoCrafter/Text2Video-1024/resolve/main/model.ckpt --directory-prefix=./ds_processors/video_generators/VideoCrafter/cache/VideoCrafter1_1024/
pip cache purge
