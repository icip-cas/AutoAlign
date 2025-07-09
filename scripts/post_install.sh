git clone https://github.com/open-compass/opencompass.git
cd opencompass
git checkout tags/0.3.9
pip install -e .

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

pip install nltk
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

cd ../../src/autoalign/eval/livecodebench
pip install -e .

pip install huggingface-hub==0.24.7
huggingface-cli download --repo-type dataset --resume-download livecodebench/code_generation_lite --local-dir livecodebench/code_generation_lite --local-dir-use-symlinks False

cd ../math_eval/latex2sympy
pip install -e .