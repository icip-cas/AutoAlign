## CAI
Building weakly supervised annotation data in the style of the Constitution class.

References:

HF team's ConstitutionAI experiment blog: https://huggingface.co/blog/constitutional_ai

Original Constitution from ConstitutionAI: https://raw.githubusercontent.com/anthropics/ConstitutionalHarmlessnessPaper/main/prompts/CritiqueRevisionInstructions.json

HF's revised anthropic Constitution: https://github.com/huggingface/llm-swarm/blob/main/examples/constitutional-ai/constitution_anthropic.json

These perference pairs can be collected by running the following scripts:

```bash
# stage 0: assign constitution
export PROMPTS_FILE="../../data/train/ultrafeedback_ata.json"
export OUTPUT_DIR="./outputs"
export OUTPUT_FILE_NAME="ultrafeedback_ata_stage0.json"
export STAGE=0
source prepare_for_stage.sh

```

```bash
# stage 1: generate vanilla response
export TEMPLATE_NAME="chatml"
export MODEL_NAME="qwen2-7b"
export SAVED_MODEL_PATH="./pretrained_models/Qwen2-7B"

export C_PROMPTS_FILE="${OUTPUT_DIR}/${OUTPUT_FILE_NAME}"
export OUTPUT_FILE_NAME="ultrafeedback_ata_stage1.json"
export START_STAGE=1
source prepare_for_stage.sh
```

```bash
# stage 2: generate cretique
export LAST_STAGE_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_cai_stage${START_STAGE}_${OUTPUT_FILE_NAME}"
export OUTPUT_FILE_NAME="ultrafeedback_ata_stage2.json"
export START_STAGE=2
source prepare_for_stage.sh
```

```bash
# stage 3: generate revision response
export LAST_STAGE_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}/${MODEL_NAME}_cai_stage${START_STAGE}_${OUTPUT_FILE_NAME}"
export OUTPUT_FILE_NAME="ultrafeedback_ata_stage3.json"
export START_STAGE=3
source prepare_for_stage.sh
```

```bash
# format chosen and rejected data
source prepare_for_dpo.sh
```

Then start training!

```bash
bash train_dpo.sh
```
