# Copyright (c) 2023 Alibaba PAI and Nvidia Megatron-LM Team.
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

import torch
from megatron.core import mpu
try:
    from megatron import get_args
except:
    from megatron.training import get_args, print_rank_0
try:
    from megatron.utils import get_ltor_masks_and_position_ids
except:
    from megatron.training.utils import get_ltor_masks_and_position_ids

from megatron_patch.tokenizer import get_tokenizer


def get_batch_on_this_tp_rank_original(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)

        tokens_ = data['input_ids'].long()
        labels_ = data['labels'].long()
        tokens = tokens_[:, :-1].contiguous()
        labels = labels_[:, 1:].contiguous()
        # core/tensor_parallel/cross_entropy.py, target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        # labels[labels == tokenizer.eos_token_id] = -100
        labels[labels == tokenizer.pad_token_id] = -100

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            labels,
            -100,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True),
            'position_ids': position_ids.cuda(non_blocking=True)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])
            _broadcast(batch['position_ids'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        attention_mask = torch.empty((args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                     device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)
            _broadcast(position_ids)

        elif mpu.is_pipeline_last_stage():
            tokens = None
            position_ids = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }

    return batch

def get_batch_on_this_tp_rank_idxmap_sft(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:

        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)


        tokens = data['tokens'].long()
        labels = torch.roll(data['tokens'].long(), shifts=-1, dims=1)
        labels[:, -1] = tokenizer.pad_token_id
        # NOTE: assert lengths of tokens/labels are sequence-length
        assert args.seq_length == labels.shape[-1]

        # NOTE: mask labels on special tokens and input, only output_ids have labels
        for i in range(labels.shape[0]):
            sep_index = (tokens[i] == tokenizer.sep_token_id).nonzero(as_tuple=True)[0]
            
            if len(sep_index) % 2 != 0 or sep_index.numel() == 0:
                try:
                    input_str = tokenizer.decode(tokens[i])
                except:
                    input_str = tokenizer.tokenizer.decode(tokens[i])
                raise ValueError(f'Got a input with invalid format, input "{input_str}"')
            
            for start_idx, end_idx in sep_index.tensor_split(len(sep_index) // 2):
                labels[i, start_idx - 1:end_idx - 1] = -100        
        labels[labels == tokenizer.pad_token_id] = -100
        labels[labels == tokenizer.sep_token_id] = -100
        loss_mask = torch.ones_like(labels, dtype=torch.float)
        if args.eod_mask_loss:
            loss_mask[labels == -100] = 0.0

        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            False,
            args.create_attention_mask_in_dataloader
        )

        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True) if attention_mask is not None else None,
            'position_ids': position_ids.cuda(non_blocking=True)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
        
        _broadcast(batch['position_ids'])

    else:

        tokens = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        
        attention_mask = None
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                        device=torch.cuda.current_device())
        position_ids = torch.empty((args.micro_batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            _broadcast(tokens)
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None

            _broadcast(tokens)
            _broadcast(attention_mask)

        elif mpu.is_pipeline_last_stage():
            tokens = None

            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        _broadcast(position_ids)
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids
        }
    return batch

def get_batch_on_this_tp_rank_idxmap_dpo(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    mask_id = tokenizer.vocab_size + 1
    def _broadcast(item):
        if item is None:
            return
        torch.distributed.broadcast(item, mpu.get_tensor_model_parallel_src_rank(),
                                    group=mpu.get_tensor_model_parallel_group())

    if mpu.get_tensor_model_parallel_rank() == 0:
        if isinstance(data_iterator, dict):
            data = data_iterator
        else:
            data = next(data_iterator)


        # Process chosen and rejected data separately
        chosen_tokens = data['chosen_text'].long()
        rejected_tokens = data['rejected_text'].long()
        
        
        chosen_label = data['chosen_label'].long()
        rejected_label =data['rejected_label'].long()
        
        # Create labels by shifting tokens
        chosen_labels = torch.roll(chosen_label, shifts=-1, dims=1)
        rejected_labels = torch.roll(rejected_label, shifts=-1, dims=1)
        
        
        # Set the last token of each sequence to pad_token_id
        chosen_labels[:, -1] = tokenizer.pad_token_id
        rejected_labels[:, -1] = tokenizer.pad_token_id
        
        # Stack chosen and rejected data
        tokens = torch.cat([chosen_tokens, rejected_tokens], dim=0)
        labels = torch.cat([chosen_labels, rejected_labels], dim=0)

        

        # NOTE: assert lengths of tokens/labels are sequence-length
        assert args.seq_length == labels.shape[-1]
        
        # Set up loss mask
        loss_mask = torch.ones_like(labels, dtype=torch.float)
        loss_mask[labels == mask_id] = 0.0


        attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            tokenizer.eod,
            args.reset_position_ids,
            args.reset_attention_mask,
            False,
            args.create_attention_mask_in_dataloader
        )

        batch = {
            'tokens': tokens.cuda(non_blocking=True),
            'labels': labels.cuda(non_blocking=True),
            'loss_mask': loss_mask.cuda(non_blocking=True),
            'attention_mask': attention_mask.cuda(non_blocking=True) if attention_mask is not None else None,
            'position_ids': position_ids.cuda(non_blocking=True)
        }

        if args.pipeline_model_parallel_size == 1:
            _broadcast(batch['tokens'])
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])

        elif mpu.is_pipeline_first_stage():
            _broadcast(batch['tokens'])
            _broadcast(batch['attention_mask'])

        elif mpu.is_pipeline_last_stage():
            _broadcast(batch['labels'])
            _broadcast(batch['loss_mask'])
            _broadcast(batch['attention_mask'])
        
        _broadcast(batch['position_ids'])


    else:

        batch_size = args.micro_batch_size * 2  # Double the batch size for chosen and rejected
        tokens = torch.empty((batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        labels = torch.empty((batch_size, args.seq_length), dtype=torch.int64,
                             device=torch.cuda.current_device())
        loss_mask = torch.empty((batch_size, args.seq_length), dtype=torch.float32,
                                device=torch.cuda.current_device())
        
        attention_mask = None
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty((batch_size, 1, args.seq_length, args.seq_length), dtype=torch.bool,
                                        device=torch.cuda.current_device())
        position_ids = torch.empty((batch_size, args.seq_length), dtype=torch.int64,
                                   device=torch.cuda.current_device())

        if args.pipeline_model_parallel_size == 1:
            for tensor in [tokens, labels, loss_mask, attention_mask]:
                _broadcast(tensor)
        elif mpu.is_pipeline_first_stage():
            labels = None
            loss_mask = None
            _broadcast(tokens)
            _broadcast(attention_mask)
        elif mpu.is_pipeline_last_stage():
            tokens = None
            _broadcast(labels)
            _broadcast(loss_mask)
            _broadcast(attention_mask)

        _broadcast(position_ids)
        batch = {
            'tokens': tokens,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }


    return batch