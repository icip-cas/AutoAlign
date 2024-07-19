import random
import json
import argparse
import os
from tqdm import tqdm
import re

random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument("--input-files-list", nargs='+', required=True)
parser.add_argument("--output-file-suffix", type=str, required=True)
parser.add_argument("--output-dir", type=str, default="./data/train/")
parser.add_argument("--add-identity", action="store_true")
parser.add_argument("--identity-file", type=str, default=None)
parser.add_argument("--n-split", type=int, default=3)
parser.add_argument("--use-prop", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# set seed
random.seed(args.seed)

use_prop = args.use_prop.split("|")
use_prop = [float(x) for x in use_prop]
assert len(use_prop) == len(args.input_files_list)

def should_skip(conv):
    wrong_indices_pattern = re.compile("\n1\. [^2]*\n1\. ")
    # Filter wrong list indices like https://sharegpt.com/c/1pREAGO
    if len(conv["conversations"]) == 0:
        return True

    for sentence in conv["conversations"]:
        val = sentence["value"]
        if 'openai' in val.lower() or 'open ai' in val.lower() or 'chatgpt' in val.lower():
            # print(val)
            # print(conv)
            return True
        sub = re.search(wrong_indices_pattern, val)
        if sub is not None:
            return True

    return False

def preprocess(content):
    new_content = []
    for conv in tqdm(content):
        if should_skip(conv):
            # print(f"{conv['id']} contains a wrong format.")
            pass
        else:
            if conv['conversations'][0]['from']=='system':
                to_test = conv['conversations'][1:]
            else:
                to_test = conv['conversations']
            flag = False
            if len(to_test) == 0:
                print(conv)
                flag = False
            for j, ele in enumerate(to_test):
                # print(ele)
                if j % 2 == 0 and ele['from'] != 'human':
                    # print(j, conv)
                    flag = True
                elif j % 2 == 1 and ele['from'] != 'gpt':
                    # print(j, conv)
                    flag = True
                if flag:
                    break
            if flag:
                continue

            if 'source' in ele and  (ele['source']=='rebert-test-identity' or ele['source']=='opensparrow-identity'):
                ele['conversations'][1] = ele['conversations'][1].replace('zhuque','Zhuque')

            new_content.append(conv)
    
    random.shuffle(new_content)
    print(f"#in: {len(content)}, #out: {len(new_content)}")

    return new_content

if args.add_identity:

    assert args.identity_file is not None, "Please provide identity file"

    with open(args.identity_file) as f:
        identity_data = json.load(f)

    print("We have {} identity_data".format(len(identity_data)))

count_num = 0
idx = 0
zh = 0
en = 0
all_data = []
for f in args.input_files_list:
    with open(f, 'r', encoding="utf-8") as f_r:
        print(f"Reading {f}")
        is_zh = "zh" in f.split("/")
        data = json.loads(f_r.read())
        data = preprocess(data)
        # randomly sample data according to the proportion
        if use_prop[idx] < 1:
            data = random.sample(data, int(len(data) * use_prop[idx]))
        print(f"Use Num of data: {int(len(data) * use_prop[idx])}")
        idx += 1
        all_data.extend(data)
        count_num += len(data)
        if is_zh:
            zh += len(data)
        else:
            en += len(data)

# randomly split the data for n_files, equal size for all splits
n_files = args.n_split
split_data = [[] for _ in range(n_files)]
random.shuffle(all_data)
split_size = len(all_data) // n_files
for i in range(n_files):
    split_data[i] = all_data[i*split_size:(i+1)*split_size]
    if args.add_identity:
        # add identity data after some conv
        added_data = random.sample(split_data[i], 1000)
        for d in added_data:
            d['conversations'].extend(random.choice(identity_data)['conversations'])
        # add identity for each split
        split_data[i].extend(identity_data)

print(f"Total data: {count_num}")
print(f"Total zh data: {zh}")
print(f"Total en data: {en}")

# save the split data for n files
for i in range(n_files):
    output_file = os.path.join(args.output_dir, "split_" + str(i) + "_" + args.output_file_suffix)
    with open(output_file, 'w', encoding="utf-8") as f:
        json.dump(
            split_data[i], 
            f, 
            indent=4,
            ensure_ascii=False
        )