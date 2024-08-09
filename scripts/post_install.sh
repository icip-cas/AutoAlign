git clone https://github.com/open-compass/opencompass.git
cd opencompass
git checkout tags/0.2.3
pip install -e .

# Download core dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
cd ./data

# Download all dataset to data/ folder
# wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
# unzip OpenCompassData-complete-20240207.zip
# cd ./data
# find . -name "*.zip" -exec unzip "{}" \;

# Download IFEval to data/ folder
git clone https://huggingface.co/datasets/HuggingFaceH4/ifeval
