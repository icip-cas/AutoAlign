## CAI

cai论文:
https://arxiv.org/pdf/2212.08073

数据来源:
hh-rlhf中red team数据:https://huggingface.co/datasets/Anthropic/hh-rlhf/tree/main/red-team-attempts

sft中使用的helpful数据ultrachat_90k:https://hf-mirror.com/datasets/HuggingFaceH4/ultrachat_200k

cai流程：

1.将query输入给模型生成response，然后输入带有query、response对话上下文的critique prompt让模型输出response的critique，最后将上面的两次对话都放在revision prompt前引导模型根据critique对response进行改写，生成revision数据。注意模型生成response、critique、revision时使用few shot。

2.对revision数据进行过滤后，用<query, revision>数据混合helpful数据对模型进行sft微调。

3.使用sft微调后的模型，对query进行temperature参数分别为0、1的两次采样，得到两个response。通过prompt引导模型对两个response进行judge。注意judge时为了避免位置偏差的影响，将两个response依次放在prompt中的首个选项，共judge两次。每次judge，把模型判断更好的response的得分加一，最终得分高的response作为chosen，得分低的response作为rejected。如果两个response得分一样，就将此条数据舍弃。注意模型进行judge时使用few shot。

4.最后利用<query, chosen, rejected>对模型进行dpo微调。

注意事项：

1.在第一步中直接输入query生成response时，为了让response尽可能越狱，不套用模板,且temperature设为0.7。后面引导模型输出critique和revision时，这两步需要套用模板，temperature设为0。

2.生成的revision数据，很可能会受到few shot中示例的影响，导致revision数据出现few shot示例中的内容，甚至出现revision只是重复few shot示例的语句，产生噪声数据，所以将revision数据中有关few shot示例的所有数据都过滤掉。

3.进行sft微调时，不能只使用<query, revision>数据，不然会导致模型过拟合。所以要混合helpful数据，我们这里设置helpful数据数量是<query, revision>数据的2.5倍。

4.模型在judge时，要套用模板，提高模型judge能力，否则模型可能根本不进行judge。

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