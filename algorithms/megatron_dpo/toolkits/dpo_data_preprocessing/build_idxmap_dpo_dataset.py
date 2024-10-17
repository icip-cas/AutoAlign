# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



"""Processing data for SFT and Conversations."""

import argparse
import json
import multiprocessing
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import torch

from megatron_patch.tokenizer import build_tokenizer
from megatron_patch.data.indexed_dataset_dpo import make_dpo_builder



def custom_print(*args):
    formatted_message = ' '.join(str(arg) for arg in args)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
    BLUE = '\033[94m'
    END = '\033[0m'  
    print(f"{BLUE}[{current_time}] [INFO]{END} {formatted_message}")
    



class Encoder(object):
    def __init__(self, args):
        self.args = args
        Encoder.tokenizer = build_tokenizer(self.args) # acconding to args.tokenizer_type, return different tokenizer
        Encoder.mask_id = Encoder.tokenizer.vocab_size + 1   

    def initializer(self): 
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, json_line):

        if self.args.conversations: # multi lable mask
            return conv_encoder_provider(json_line)
        elif self.args.sft: # single lable mask
            return sft_encode_provide(json_line)
        else :
            raise ValueError("This script only supports SFT format data. Please ensure that either --conversations or --args.sft is passed.")
        

def conv_encoder_provider(json_line): # llama2 conversations templates from fastchat.conversation.py
    args = get_args()
    ids = {}
    processed_text = ""
    num_begin_mask_id = 0
    assert type(json_line) == dict
    doc_ids = []
    doc_lable_ids = []

    json_line = json_line["conversations"]
    

    if json_line[0]["from"]=='system':
        sys_mess = json_line[0]["value"]
        system_message = f"[INST] <<SYS>>\n{sys_mess}\n<</SYS>>\n\n"
        json_line = json_line[1:]
    else:
        system_message = "[INST] "
    
    conv_len = len(json_line)
    tmp_tokens = Encoder.tokenizer.tokenize(system_message)
    tmp_tokens = tmp_tokens [1:]
    doc_ids = doc_ids + tmp_tokens
    doc_lable_ids = doc_lable_ids + [Encoder.mask_id] * len(tmp_tokens)
    num_begin_mask_id += len(tmp_tokens)
    
    for idx,item in enumerate(json_line):
        processed_text += item["value"]
        assert item["from"] in ["human","gpt"]
        assert item["from"] == json_line[(idx + 2)% conv_len]["from"]
        if item["from"] == "human":
            if idx == 0 :
                tmp = item["value"] + " "
                tmp_tokens = Encoder.tokenizer.tokenize(tmp)
                tmp_tokens = tmp_tokens [1:]
                doc_ids = doc_ids + tmp_tokens
                if args.mask:
                    doc_lable_ids = doc_lable_ids + [Encoder.mask_id] * len(tmp_tokens)
                    num_begin_mask_id += len(tmp_tokens)
                else:
                    doc_lable_ids = doc_lable_ids + tmp_tokens
            else :
                tmp = "[INST]"+item["value"] + " "
                tmp_tokens = Encoder.tokenizer.tokenize(tmp)
                tmp_tokens = tmp_tokens [1:]
                doc_ids = doc_ids + tmp_tokens
                if args.mask:
                    doc_lable_ids = doc_lable_ids + [Encoder.mask_id] * len(tmp_tokens)
                else:
                    doc_lable_ids = doc_lable_ids + tmp_tokens
        else:
            tmp = "[/INST]"+ item["value"] +" </s><s>"
            tmp_tokens = Encoder.tokenizer.tokenize(tmp)
            tmp_tokens = tmp_tokens [1:]
            doc_ids = doc_ids + tmp_tokens 
            doc_lable_ids = doc_lable_ids + tmp_tokens
    
    assert len(doc_ids) == len(doc_lable_ids)
    if len(doc_ids) > args.model_max_length :
        doc_ids = doc_ids[:args.model_max_length]
        doc_lable_ids = doc_lable_ids[:args.model_max_length]
    if num_begin_mask_id  >= args.model_max_length :
        doc_lable_ids = []
        doc_ids =[]
    ids[args.json_keys[0]] = {}
    ids[args.json_keys[0]]["data"] = doc_ids
    ids[args.json_keys[0]]["label"] = doc_lable_ids 
    return ids,len(processed_text.encode('utf-8')),len(doc_ids)

def sft_encode_provide(json_line): # sft
    args = get_args()
    ids = {}
    processed_text = ""
    assert type(json_line) == dict
    doc_ids = []
    doc_lable_ids = []

    total_prompt,prompt_no_output = json_line["prompt"],json_line["prompt_no_output"]
    processed_text += total_prompt
    sentence_ids_total = Encoder.tokenizer.tokenize(total_prompt)
    sentence_ids_no_output = Encoder.tokenizer.tokenize(prompt_no_output)
    len_mask = len(sentence_ids_no_output)

    sentence_ids_total = np.array(sentence_ids_total)
    sentence_ids_label = sentence_ids_total.copy()

    assert len_mask > 0,"{} need mask".format(json_line)
    if args.mask:
        sentence_ids_label[:len_mask] = Encoder.mask_id

    sentence_ids_label = sentence_ids_label.tolist()
    sentence_ids_total = sentence_ids_total.tolist()

    if len(sentence_ids_total) > 0:
        doc_ids.append(sentence_ids_total)
        doc_lable_ids.append(sentence_ids_label)
    if len(doc_ids) > 0 and Encoder.args.append_eod:
        doc_ids[-1].append(Encoder.tokenizer.eod)
        doc_lable_ids[-1].append(Encoder.tokenizer.eod)  

    if len_mask >= args.model_max_length:
        doc_lable_ids = []
        doc_ids =[]
    ids["prompt"] = {}
    ids["prompt"]["data"] = doc_ids
    ids["prompt"]["label"] = doc_lable_ids
    return ids,len(processed_text.encode('utf-8')),len(doc_ids)

            
def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data') 
    group.add_argument("--conversations", action="store_true", help="Utilize conversations instructions to finetune the model.")
    group.add_argument("--sft", action="store_true", help="Utilize instructions to finetune the model.")
    group.add_argument("--dpo", action="store_true", help="Utilize dpo algorithm to alignment the model.")
    group.add_argument('--input', type=str,
                       help='Path to input JSON')
    group.add_argument('--datasets', nargs='+', default=None,
                       help='Paths to one or more input datasets to merge')
    group.add_argument('--json-keys', nargs='+', default=['text'],choices=['text', 'conversations','sft'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument("--mask", action="store_true", help="Utilize mask to finetune the model.")
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'PretrainedFromHF', "LLaMATokenizer","GPTSentensePieceTokenizer"],
                       help='What type of tokenizer to use.')
    group.add_argument("--model-max-length", type=int,required=True, default=1024, help="max sequence length")
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--append-bos', action='store_true',
                       help='Append a bos token to the end of a document.')
    group.add_argument('--prepend-space', action='store_true',
                    help='Prepends a space to the beginning of a document')
    group.add_argument("--tokenizer-name-or-path", type=str, default=None,
                       help="Name or path of the huggingface tokenizer.")
    group.add_argument('--make-vocab-size-divisible-by', type=int, default=128,
                       help='Pad the vocab size to be divisible by this value.'
                            'This is added for computational efficieny reasons.')
    group.add_argument('--pad-vocab-size-to', type=int, default=None,
                       help='Pad the vocab size to be divisible by this value.'
                            'Value of the size of the vocabulary of the tokenizer to reach. This value must be greater than'
                            ' the initial size of the tokenizer. If this argument is used the value of '
                            '`make-vocab-size-divisible-by` will be ignored.')
    group.add_argument('--patch-tokenizer-type',
        type=str,
        required=True,
        choices=['Qwen2Tokenizer', 'LLamaTokenizer', 'DeepSeekV2Tokenizer', 'LLama3Tokenizer'],
        help='What type of tokenizer to use.',
    )

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, default=None,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk-size', type=int, default=1,
                       help='size of chunk to per process')
    group.add_argument("--test-tokenize", action="store_true", help="test tokenize single json line")
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert'):
        if not args.split_sentences:
            custom_print("Bert tokenizer detected, are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    custom_print("Opening", args.input)
    fin = json.load(open(args.input, 'r', encoding='utf-8'))

    custom_print("number of json lines is: ", len(fin))
    if args.output_prefix is None :
        args.output_prefix = args.input[:-5]
    filtered_fin = []
    for i in tqdm(range(len(fin)),desc="Filtering Invalid Data"):
        # check if the data is valid
        if args.json_keys[0] == "conversations" and ((fin[i]["conversations"][0]["from"] != "system" and fin[i]["conversations"][0]["from"] != "human")  or fin[i]["conversations"][-1]["from"] != "gpt" ): 
            custom_print("invalid data", i, f"\n{json.dumps(fin[i], ensure_ascii=False)}")
        else:
            filtered_fin.append(fin[i])

    fin = filtered_fin
    custom_print("After filtering,number of json lines is", len(fin))
    breakpoint()
    encoder = Encoder(args)  
    custom_print(f"Vocab size: {Encoder.tokenizer.vocab_size}")
    custom_print(f"mask id: {Encoder.mask_id}")
    custom_print(f"Output prefix: {args.output_prefix}")

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    key = args.json_keys[0]
    num_large_doc_filterd = 0

    if args.test_tokenize:
        test_doc,_,_= encoder.encode(filtered_fin[0])
        custom_print("decode tokenized data: ",Encoder.tokenizer.detokenize(test_doc[key]['data']))
        custom_print("tokenized label: ",test_doc[key]['label'])
    else:
        output_bin_files[key] = "{}_{}_maxlen_{}.bin".format(args.output_prefix,
                                                    key, args.model_max_length)
        output_idx_files[key] = "{}_{}_maxlen_{}.idx".format(args.output_prefix,
                                                    key, args.model_max_length)
        builders[key] = make_dpo_builder(output_bin_files[key],impl=args.dataset_impl,vocab_size=Encoder.tokenizer.vocab_size)
        startup_end = time.time()
        custom_print("Time to startup:", startup_end - startup_start)

        proc_start = time.time()
        with multiprocessing.Pool(args.workers, initializer=encoder.initializer) as pool:

            encoded_docs = pool.imap(encoder.encode, filtered_fin, args.chunk_size)

            total_documents = 0
            total_bytes_processed = 0
            total_tokens_processed = 0

            progress_bar = tqdm(encoded_docs, total=len(filtered_fin), desc="Data tokenized and stored in binary format")
            for doc, bytes_processed, tokens_processed in progress_bar:
                if len(doc[key]["data"]) == 0:
                    num_large_doc_filterd += 1
                    continue
                total_documents += 1
                total_bytes_processed += bytes_processed
                total_tokens_processed += tokens_processed

                doc_data = doc[key]["data"]
                doc_label = doc[key]["label"]
               
                builders[key].add_item(
                    torch.IntTensor(doc_data),
                    torch.IntTensor(doc_label)
                )  # write in data.bin and lable.bin
                builders[key].end_document()
            

        custom_print(f"Time to tokenize and store: {time.time() - proc_start} s")
        custom_print(f"data size: {total_bytes_processed/1024/1024} MB")
        custom_print(f"documents number: {total_documents}")
        custom_print(f"numbers of too large doc filterd: {num_large_doc_filterd}")
        custom_print(f"processed tokens: {total_tokens_processed}")

        # To write indices to an IDX file in binary format, 
        # the binary data should include a checksum prefix, followed by an array of document sizes, an array of document byte sizes (i.e., offsets), 
        # and an array of document indices (i.e., base addresses)."
        builders[key].finalize(output_idx_files[key])
        custom_print("Time to finish preprocessing data:", time.time() - startup_start)
        
if __name__ == '__main__':
    main()