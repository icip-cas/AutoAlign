# CAI Process

## Paper Reference
[CAI Paper](https://arxiv.org/pdf/2212.08073)

## Data Sources
1. **Red Team Data** from HH-RLHF: [Dataset Link](https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/red-team-attempts)  
2. **Helpful Data** used in SFT, UltraChat_90k: [Dataset Link](https://hf-mirror.com/datasets/HuggingFaceH4/ultrachat_200k)  

## Workflow

### 1. Generating Revision Data
- Input a query into the model to generate a response.
- Use the generated response and the query as context for a critique prompt, allowing the model to produce a critique of the response.
- Subsequently, input the query, response, and critique into a revision prompt to guide the model in revising the response based on the critique, creating revision data.
- **Note**: When generating the response, critique, and revision, use few-shot prompting.

### 2. Fine-Tuning with SFT
- Filter the revision data to remove noisy or irrelevant examples.
- Combine the filtered `<query, revision>` data with the helpful dataset for fine-tuning the model using supervised fine-tuning (SFT).
- Ensure the number of helpful data samples is **2.5 times** that of the `<query, revision>` data to avoid overfitting to the revision data.

### 3. Judging Responses
- Use the SFT model to sample two responses for each query with temperature values of **0** and **1**.  
- Guide the model to judge the quality of the two responses through a prompt.  
  - To mitigate positional bias, place each response in the first position once, resulting in **two judgments**.  
  - For each judgment, increment the score of the preferred response by one.  
  - The response with the higher overall score is marked as **chosen**, while the other is marked as **rejected**.  
  - If the scores are tied, discard the data for that query.  
- **Note**: Few-shot prompting should be used during the judgment phase.

### 4. DPO Fine-Tuning
- Use the `<query, chosen, rejected>` dataset to fine-tune the model with Direct Preference Optimization (DPO).

## Key Notes

### Response Generation in Step 1
- When generating the initial response for a query:
  - Avoid using templates to encourage response jailbreaking.
  - Set the temperature to **0.7**.
- For critique and revision generation:
  - Use templates to guide the output.
  - Set the temperature to **0**.

### Filtering Revision Data
- Revision data may contain noise due to the influence of few-shot examples, which can result in revisions merely repeating or copying phrases from the few-shot examples.  
- **Filter out all such data associated with the few-shot examples**.

### Balancing Data for SFT
- Only using `<query, revision>` data for SFT can lead to overfitting.  
- Combine it with helpful data at a **1:2.5 ratio**.

### Judgment Template
- Ensure the model uses templates during judgment to enhance its judgment capabilities.  
- Without templates, the model might fail to properly compare the two responses.

``` bash
export PROMPTS_FILE="poison_en.json"
export POSITVE_CHAT_FILE="ultra_90k.json"
export OUTPUT_DIR="./outputs/"
export MODEL_NAME="mistral_7b_v0.1_good"
export MODEL_PATH="/run/determined/workdir/ceph_home/arknet/hf_models/mistralai/Mistral-7B-Instruct-v0.1"
export OUTPUT_CHOSEN_FILE_NAME="${MODEL_NAME}_poison_en_chosen.json"
export OUTPUT_REJECTED_FILE_NAME="${MODEL_NAME}_poison_en_rejected.json"
export OUTPUT_CAI_FILE_NAME="${MODEL_NAME}_cai_poison_en.json"
export OUTPUT_SFT_FILE_NAME="${MODEL_NAME}_sft_poison_en.json"
export OUTPUT_DPO_FILE_NAME="${MODEL_NAME}_dpo_poison_en.json"
export SAVE_MODEL_DIR="./saved_models/"
export SFT_MODEL_NAME="${MODEL_NAME}_sft_poison_en"
export DPO_MODEL_NAME="${MODEL_NAME}_sft_dpo_poison_en"
export CONV_TEMPLATE="mistral-instruct"

bash cai.sh
```