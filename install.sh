git clone https://github.com/hpcaitech/ColossalAI.git ./ds_processors/video_generators/OpenSora1_2/colossalai/
rm -rf ./ds_processors/video_generators/OpenSora1_2/colossalai/.git
rm -rf ./ds_processors/video_generators/OpenSora1_2/colossalai/.github
rm -rf ./ds_processors/video_generators/OpenSora1_2/colossalai/requirements/
cp -r ./ds_processors/video_generators/OpenSora1_2/requirements/ ./ds_processors/video_generators/OpenSora1_2/colossalai/
pip install --upgrade -r requirements.txt
pip install --upgrade ./ds_processors/video_generators/OpenSora1_2/colossalai/
git clone https://github.com/NVIDIA/apex.git ./ds_processors/video_generators/OpenSora1_2/apex/
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ds_processors/video_generators/OpenSora1_2/apex/