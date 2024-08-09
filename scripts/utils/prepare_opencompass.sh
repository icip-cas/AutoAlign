#!/bin/bash
git clone https://github.com/open-compass/opencompass.git
cd opencompass
git checkout tags/0.2.3 # IMPORTANT
pip install -e .
# Download all dataset to data/ folder
wget --no-check-certificate https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;

# Download IFEval to data/ folder
git clone https://huggingface.co/datasets/HuggingFaceH4/ifeval
