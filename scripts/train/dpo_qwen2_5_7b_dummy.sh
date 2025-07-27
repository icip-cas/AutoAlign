#!/bin/sh

#SBATCH --account=luxinyu2021
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -p a800
#SBATCH --qos=normal
#SBATCH -J ata_rl
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

export MODEL_PATH="/mnt/data2/hf_models/Qwen2.5-7B-Instruct"
export DATA_PATH="data/dummy_dpo.json"
export GA=4
export OUTPUT_DIR="./saved_models/qwen2.5-7b_ins_dummy_dpo"
export TRAIN_BATCH_SIZE=2

bash ./scripts/train_dpo.sh