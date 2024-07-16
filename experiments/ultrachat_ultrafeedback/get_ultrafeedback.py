from datasets import load_dataset
import json

def change_format(convs):
    for conv in convs:
        if conv['role']=='user':
            conv['from'] = 'human'
        elif conv['role'] == 'assistant':
            conv['from'] = 'gpt'
        conv.pop('role')
        conv['value'] = conv.pop('content')

# 加载数据集
ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split='train_sft')

# 遍历数据集并将 messages 键改为 conversations
modified_data = []
for example in ds:
    example['conversations'] = example.pop('messages')
    change_format(example['chosen'])
    change_format(example['rejected'])
    modified_data.append(example)

# 将修改后的数据集保存为 JSON 文件
with open('data/ultrafeedback.json', 'w') as f:
    json.dump(modified_data, f, indent=4)