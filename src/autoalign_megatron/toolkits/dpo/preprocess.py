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

# Additional modifications and contributions by AutoAlign Team:
# - Added support for multi-turn dialogue training with SFT (Supervised Fine-Tuning).
# - Integrated DPO (Direct Preference Optimization) for alignment training.

# Copyright (c) 2024 AutoAlign Team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0



"""Processing data for DPO."""

import argparse
import json
import multiprocessing
import os
import sys
from tqdm import tqdm
from typing import Dict, Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from tqdm import tqdm
import numpy as np
import time
from datetime import datetime
import torch

from megatron_patch.tokenizer import build_tokenizer
from collections import defaultdict

from autoalign_megatron.patch.data.indexed_dataset_dpo import make_dpo_builder

from src.autoalign.conversation import TEMPLATES, Role, Conversation, IGNORED_TOKEN_ID

class ConversationDPO(Conversation):
    def fill_in_messages(
        self, conv: Dict[str, Any], replace_conv_system_message: bool = True
    ):
        """Fill in conversation messages from an external source."""
        self.messages.clear()  # Clear existing messages

        # Handle system message
        first_message = conv[0]
        if (
            first_message["from"] == Role.SYSTEM.value and replace_conv_system_message
        ):  # Use the system message from the conversation
            self._set_system_message(first_message["value"])
        else:
            self._set_system_message(
                self._system_message
            )  # Use the current system message

        # Fill in other messages
        for message_dict in conv:
            role_str, message_str = message_dict["from"], message_dict["value"]
            try:
                role = Role(role_str)
                if role != Role.SYSTEM:  # We've already handled the system message
                    self.messages.append((role, message_str))
            except ValueError:
                raise ValueError(
                    f"Invalid role: {role_str}. Must be one of {', '.join([r.value for r in Role])}"
                )
    def _generate_labels(self, tokenized_conversation, tokenizer, model_max_length, mask_id=IGNORED_TOKEN_ID):
        if self.template.strategy:
            return self.template.strategy.generate_labels(
                self.messages,
                tokenized_conversation,
                tokenizer,
                self.template.get_attributes(),
            )
        labels = [mask_id] * len(tokenized_conversation.input_ids)
        cur_inst = ""
        for role, message in self.messages:
            if role in [Role.SYSTEM, Role.HUMAN]:
                cur_inst += (
                    self.template.role_starts[role]
                    + message
                    + self.template.role_ends[role]
                )
            else:
                cur_inst += self.template.role_starts[role]
                start_idx = len(tokenizer(cur_inst, padding='do_not_pad', truncation=True, max_length=model_max_length).input_ids) - self.template.offset
                end_idx = len(
                    tokenizer(
                        cur_inst + message + self.template.role_ends[role],
                        padding='do_not_pad',
                        truncation=True, 
                        max_length=model_max_length
                    ).input_ids
                )
                labels[start_idx:end_idx] = tokenized_conversation.input_ids[
                    start_idx:end_idx
                ]
                cur_inst += message + self.template.role_ends[role]

        return labels


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
        pass
    
    def encode(self, json_line):

        # return doc, bytes_processed, tokens_processed, doc is a dic contain data and label
        if self.args.dpo: # include prompt chosen rejected data
            return dpo_encode_provide(json_line)
        else :
            raise ValueError("Only supports DPO format data. Please ensure that either --conversations/--dpo is passed.")
        
def dpo_encode_provide(json_line): # chatlm conversations templates
    args = get_args()
    template = args.template
    chosen_ids = {}
    rejected_ids = {}
    chosen_data = json_line['chosen']
    rejected_data = json_line['rejected']
    Chat_Template = TEMPLATES[template]
    conversation = ConversationDPO(Chat_Template)

    conversation.fill_in_messages(chosen_data)
    chosen_conversation_str = conversation.get_conversation_str()
    chosen_data_len = len(chosen_conversation_str.encode('utf-8'))
    chosen_doc_ids = Encoder.tokenizer(chosen_conversation_str,padding='do_not_pad',truncation=True, max_length=args.model_max_length)
    chosen_ids_len = len(chosen_doc_ids.input_ids)
    chosen_doc_lable_ids = conversation._generate_labels(
        chosen_doc_ids, Encoder.tokenizer, args.model_max_length, mask_id=Encoder.mask_id
    )
    
    assert len(chosen_doc_ids.input_ids) == len(chosen_doc_lable_ids )
    chosen_ids[args.json_keys[0]] = {}
    chosen_ids[args.json_keys[0]]["data"] = chosen_doc_ids.input_ids
    chosen_ids[args.json_keys[0]]["label"] = chosen_doc_lable_ids 
    
    conversation.fill_in_messages(rejected_data)
    rejected_conversation_str = conversation.get_conversation_str()
    rejected_data_len = len(rejected_conversation_str.encode('utf-8'))
    rejected_doc_ids = Encoder.tokenizer(rejected_conversation_str,padding='do_not_pad',truncation=True, max_length=args.model_max_length)
    rejected_ids_len = len(rejected_doc_ids.input_ids)
    rejected_doc_lable_ids = conversation._generate_labels(
        rejected_doc_ids, Encoder.tokenizer, args.model_max_length, mask_id=Encoder.mask_id
    )
    
    assert len(rejected_doc_ids.input_ids) == len(rejected_doc_lable_ids )
    rejected_ids[args.json_keys[0]] = {}
    rejected_ids[args.json_keys[0]]["data"] = rejected_doc_ids.input_ids
    rejected_ids[args.json_keys[0]]["label"] = rejected_doc_lable_ids
    

    ids = {
        'chosen': chosen_ids,
        'rejected': rejected_ids
    }
    processed_text_lens = {
        'chosen': chosen_data_len,
        'rejected': rejected_data_len
    }
    
    processed_ids_lens = {
        'chosen': chosen_ids_len,
        'rejected': rejected_ids_len
    }
    
    return ids, processed_text_lens, processed_ids_lens

            
def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data') 
    group.add_argument("--conversations", action="store_true", help="Utilize conversations instructions to finetune the model.")
    group.add_argument("--template", type=str, default="chatml-idsys", help="Template to use for the conversation data.")
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
    group.add_argument('--tokenizer-type', type=str, required=False,
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
    group.add_argument("--load", type=str, default=None,
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
    group.add_argument('--extra-vocab-size',
                    type=int,
                    default=0,
                    help='extra_vocab_size')

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

    if args.patch_tokenizer_type.lower().startswith('bert'):
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
    encoder = Encoder(args)  
    custom_print(f"Vocab size: {Encoder.tokenizer.vocab_size}")
    custom_print(f"mask id: {Encoder.mask_id}")
    custom_print(f"Output prefix: {args.output_prefix}")
    

    output_bin_files = defaultdict(dict)
    output_idx_files = defaultdict(dict)
    builders = defaultdict(dict)
    key = args.json_keys[0]
    num_large_doc_filterd = 0

    if args.test_tokenize:
        test_doc, _, _= encoder.encode(filtered_fin[1])
        custom_print("decode tokenized chosen data: ",Encoder.tokenizer.detokenize(test_doc['chosen'][key]['data']))
        custom_print("tokenized chosen label: ",test_doc['chosen'][key]['label'])
        custom_print("decode tokenized rejected data: ",Encoder.tokenizer.detokenize(test_doc['rejected'][key]['data']))
        custom_print("tokenized rejected label: ",test_doc['rejected'][key]['label'])
    else:
        for data_class in ['chosen','rejected'] :
            output_bin_files[data_class][key] = "{}_{}_maxlen_{}_{}".format(args.output_prefix,
                                                        key, 
                                                        args.model_max_length,
                                                        data_class
                                                        )
            output_idx_files[data_class][key] = "{}_{}_maxlen_{}_{}".format(args.output_prefix,
                                                        key, 
                                                        args.model_max_length,
                                                        data_class
                                                        )
            # return dpo_builder
            builders[data_class][key] = make_dpo_builder(output_bin_files[data_class][key],
                                                impl=args.dataset_impl,
                                                vocab_size=Encoder.tokenizer.vocab_size)
        startup_end = time.time()
        custom_print("Time to startup:", startup_end - startup_start)

        proc_start = time.time()
        with multiprocessing.Pool(args.workers, initializer=encoder.initializer) as pool:

            encoded_docs = pool.imap(encoder.encode, filtered_fin, args.chunk_size)

            total_documents = 0
            total_bytes_processed = 0
            total_tokens_processed = 0

            progress_bar = tqdm(encoded_docs, total=len(filtered_fin), desc="DPO data tokenized and stored in binary format")
            for doc, bytes_processed, tokens_processed in progress_bar:
                for data_class in ['chosen','rejected'] :
                    if len(doc[data_class][key]["data"]) == 0:
                        num_large_doc_filterd += 1
                        break
                    total_documents += 1
                    total_bytes_processed += bytes_processed[data_class]
                    total_tokens_processed += tokens_processed[data_class]

                    doc_data = doc[data_class][key]["data"]
                    doc_label = doc[data_class][key]["label"]
                
                    builders[data_class][key].add_item(
                        torch.IntTensor(doc_data),
                        torch.IntTensor(doc_label)
                    )  # write in data.bin and lable.bin
                    builders[data_class][key].end_document()
            

        custom_print(f"Time to tokenize and store: {time.time() - proc_start} s")
        custom_print(f"data size: {total_bytes_processed/1024/1024} MB")
        custom_print(f"documents number: {total_documents}")
        custom_print(f"numbers of too large doc filterd: {num_large_doc_filterd}")
        custom_print(f"processed tokens: {total_tokens_processed}")

        # To write indices to an IDX file in binary format, 
        # the binary data should include a checksum prefix, followed by an array of document sizes, an array of document byte sizes (i.e., offsets), 
        # and an array of document indices (i.e., base addresses)."
        
        for data_class in ['chosen','rejected'] :
            builders[data_class][key].finalize(output_idx_files[data_class][key])
        custom_print("Time to finish preprocessing data:", time.time() - startup_start)
        
if __name__ == '__main__':
    main()