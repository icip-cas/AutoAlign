git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
# Download core dataset to data/ folder
wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip
unzip OpenCompassData-core-20240207.zip
# Download all dataset to data/ folder
# wget https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
# unzip OpenCompassData-complete-20240207.zip
# cd ./data
# find . -name "*.zip" -exec unzip "{}" \;