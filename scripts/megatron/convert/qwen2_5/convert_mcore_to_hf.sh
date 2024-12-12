#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-7}
ROOT=${ROOT:-"../../../autoalign"}

MODEL_SIZE=${MODEL_SIZE:-"7B"}
TP=${TP:-"2"}
PP=${PP:-"2"}
MG_MODEL_PATH=${MG_MODEL_PATH:-"${ROOT}/mg_models/Qwen2.5-${MODEL_SIZE}-hf-to-mcore-te-tp${TP}-pp${PP}"}
HF_MODEL_PATH=${HF_MODEL_PATH:-"${ROOT}/hf_models/Qwen2.5-${MODEL_SIZE}-mcore-to-hf-tp${TP}-pp${PP}"}
PRECISION=${PRECISION:-"fp32"}
USE_TE=${USE_TE:-"true"}
MG2HF=${MG2HF:-"true"}
HF_MODEL_SOURCE=${HF_MODEL_SOURCE:-"${ROOT}/hf_models/Qwen2.5-${MODEL_SIZE}"}

cd ${ROOT}/src/autoalign/train_megatron/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
    ${MODEL_SIZE} \
    ${MG_MODEL_PATH} \
    ${HF_MODEL_PATH} \
    ${TP} \
    ${PP} \
    ${PRECISION} \
    ${USE_TE} \
    ${MG2HF} \
    ${HF_MODEL_SOURCE}