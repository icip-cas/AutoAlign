import json
import random
from tqdm import tqdm

def clean_dict_in_place(data):
    """
    原位删除字典中除了id、conversations和source之外的所有键
    
    参数:
        data: 需要处理的字典
    """
    # 要保留的键列表
    keep_keys = ["id", "conversations", "source"]
    
    # 如果输入是字典类型
    if isinstance(data, dict):
        # 获取所有需要删除的键
        keys_to_remove = [k for k in list(data.keys()) if k not in keep_keys]
        
        # 删除这些键
        for key in keys_to_remove:
            del data[key]
    else:
        raise TypeError("输入必须是字典类型")

with open('data/train/sft_v3_filterd_harmful_identity.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)

with open('data/train/sft_v3_filterd_harmful_identity_cleaned.json', 'w', encoding='utf-8') as f:
    for item in tqdm(json_data):
        clean_dict_in_place(item)

    random.shuffle(json_data)
    
    json.dump(json_data, f, indent=4, ensure_ascii=False)