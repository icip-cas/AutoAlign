export PYTHONPATH=$(dirname $(dirname $(pwd))):$PYTHONPATH
cd ../../src/autoalign/train_megatron/toolkits/sft_conv_data_preprocessing
bash run_build_idxmap_sft_conv_dataset.sh \
/mnt/userdata/data6/xudong2022/data/InfInstruct-Gen/infinite_2m.json \
conversations \
Qwen2Tokenizer \
4096 \
/ciphome/zhangqingyu2023/data/sft/InfInstruct-Gen_infinite_2m \
/ciphome/zhangqingyu2023/hf_models/Qwen2.5-7B