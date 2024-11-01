pip install --upgrade setuptools==75.2.0
pip install --upgrade wheel==0.44.0
pip install --upgrade torch==2.1.2
pip install --upgrade -r requirements.txt
pip install --upgrade ./ds_processors/video_generators/OpenSora1_2/colossalai/
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./ds_processors/video_generators/OpenSora1_2/apex/
pip install --upgrade ./ds_processors/prompt_processors/PLLaVA/moviepy/
pip cache purge
