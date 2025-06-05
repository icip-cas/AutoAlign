#!/bin/bash
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
ROOT=${ROOT:-"../../auto-alignment"}
HF_MODELS=${HF_MODELS:-"${ROOT}/hf_models"}
MODEL_SIZE=${MODEL_SIZE:-"7B"}
TP=${TP:-"2"}
PP=${PP:-"2"}
PRECISION=${PRECISION:-"bf16"}
USE_TE=${USE_TE:-"true"}
MG2HF=${MG2HF:-"false"}

cd ${ROOT}/megatron_autoalign/toolkits/model_checkpoints_convertor/qwen

bash hf2mcore_qwen2.5_convertor.sh \
    ${MODEL_SIZE} \
    ${HF_MODELS}/Qwen2.5-${MODEL_SIZE} \
    ${ROOT}/mg_models/Qwen2.5-${MODEL_SIZE}-hf-to-mcore-te-tp${TP}-pp${PP} \
    ${TP} \
    ${PP} \
    ${PRECISION} \
    ${USE_TE} \
    ${MG2HF}