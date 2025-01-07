git clone https://github.com/open-compass/opencompass.git
cd opencompass
git checkout tags/0.3.9
pip install -e .

# A bug of adafactor in the newest mmengine release is found in https://github.com/open-mmlab/mmengine/issues/1609
# The bugfix is in the commit of https://github.com/open-mmlab/mmengine/commit/2e0ab7a92220d2f0c725798047773495d589c548.
# However, the bugfix is not contained in any release version, so it is necessary to build mmengine from source
cd ..
pip uninstall mmengine-lite
git clone https://github.com/open-mmlab/mmengine.git
cd mmengine
pip install -e .
cd ..
cd opencompass


# Download core dataset to data/ folder
# wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
# unzip OpenCompassData-core-20240207.zip
# cd ./data

# Download all dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;

# Download IFEval to data/ folder
git clone https://huggingface.co/datasets/HuggingFaceH4/ifeval && cd ifeval && git checkout 8eea6e01f31913788bdd20ea8ffcff4d1541a761

# Download punkt to support IFEval evaluation
python -c "import nltk; nltk.download('punkt')"


