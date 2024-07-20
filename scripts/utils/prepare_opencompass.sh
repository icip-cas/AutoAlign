#!/bin/bash
git clone https://github.com/open-compass/opencompass.git
cd opencompass
pip install -e .
# Download all dataset to data/ folder
wget --no-check-certificate https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-complete-20240207.zip
unzip OpenCompassData-complete-20240207.zip
cd ./data
find . -name "*.zip" -exec unzip "{}" \;