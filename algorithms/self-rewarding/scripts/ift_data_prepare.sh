#!/bin/bash

if [ ! -d "data" ]; then
    mkdir data
fi
cd data

# Download the oasst1 dataset
git lfs install
git clone https://huggingface.co/datasets/OpenAssistant/oasst1

gzip -d oasst1/2023-04-12_oasst_all.trees.jsonl.gz

cd ..
python src/ift_data_sample.py
python src/ift_data_form.py --in-file data/seed.jsonl --out-file data/seed.json
