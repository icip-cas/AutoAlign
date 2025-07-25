#!/bin/sh

#SBATCH --account=luxinyu2021
#SBATCH -o job.%j.out
#SBATCH -e job.%j.err
#SBATCH -p a800
#SBATCH --qos=normal
#SBATCH -J eval_obj
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --mem=1000G
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

autoalign-cli eval --config-path "configs/eval_obj.yaml"
