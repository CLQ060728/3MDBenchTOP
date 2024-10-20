pip install --upgrade -r requirements.txt
git clone https://github.com/NVIDIA/apex.git ./ds_processors/video_generators/OpenSora1_2/apex/
pip install --upgrade torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ds_processors/video_generators/OpenSora1_2/apex/