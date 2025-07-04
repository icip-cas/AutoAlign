# Copyright (c) 2023 Alibaba PAI Team.
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

from megatron_pai import megatron_patch as megatron_patch
from megatron_patch.tokenizer import get_tokenizer


def get_batch_on_this_tp_rank_idxmap_dpo(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    mask_id = tokenizer.vocab_size + 1
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
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
        chosen_labels[:, -1] = mask_id
        rejected_labels[:, -1] = mask_id
        
        # Stack chosen and rejected data
        tokens = torch.cat([chosen_tokens, rejected_tokens], dim=0)
        labels = torch.cat([chosen_labels, rejected_labels], dim=0)
        
        
        pad_mask = tokens == pad_id  # (batch_size, seq_length)
        seq_lengths = (~pad_mask).sum(dim=1)
        cur_max_seq_length = seq_lengths.max()
        
        tokens = tokens[:, :cur_max_seq_length]
        labels = labels[:, :cur_max_seq_length]

        cur_max_seq_length = torch.tensor(
            cur_max_seq_length,
            dtype=torch.long,
            device=torch.cuda.current_device()
        )
        _broadcast(cur_max_seq_length)

        
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
        cur_max_seq_length_tensor = torch.empty(
            1,
            dtype=torch.long,
            device=torch.cuda.current_device()
        )
        _broadcast(cur_max_seq_length_tensor)
        cur_max_seq_length = cur_max_seq_length_tensor.item()
        
        batch_size = args.micro_batch_size * 2  # Double the batch size for chosen and rejected
        tokens = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        labels = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        loss_mask = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.float32,
            device=torch.cuda.current_device()
        )
        
        
        attention_mask = None
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (1, 1, cur_max_seq_length, cur_max_seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device()
            )
        position_ids = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )

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

def get_batch_on_this_tp_rank_idxmap_sft_conv(data_iterator):
    args = get_args()
    tokenizer = get_tokenizer()
    mask_id = tokenizer.vocab_size + 1
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
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
        
        # Process conv data 
        conv_tokens = data['conv_text'].long()
        conv_label = data['conv_label'].long()
        
        pad_mask = conv_tokens == pad_id  # (batch_size, seq_length)
        seq_lengths = (~pad_mask).sum(dim=1)
        cur_max_seq_length = seq_lengths.max()
        # cur_max_seq_length = (torch.ceil(cur_max_seq_length.float() / args.tensor_model_parallel_size) * args.tensor_model_parallel_size).long()
        
        conv_tokens = conv_tokens[:, :cur_max_seq_length]
        conv_label = conv_label[:, :cur_max_seq_length]

        cur_max_seq_length = torch.tensor(
            cur_max_seq_length,
            dtype=torch.long,
            device=torch.cuda.current_device()
        )
        _broadcast(cur_max_seq_length)
        
        # Create labels by shifting tokens
        conv_labels = torch.roll(conv_label, shifts=-1, dims=1)
  
        # Set the last token of each sequence to pad_token_id
        conv_labels[:, -1] = mask_id
        
        tokens = conv_tokens
        labels = conv_labels

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
        cur_max_seq_length_tensor = torch.empty(
            1,
            dtype=torch.long,
            device=torch.cuda.current_device()
        )
        _broadcast(cur_max_seq_length_tensor)
        cur_max_seq_length = cur_max_seq_length_tensor.item()
  
        batch_size = args.micro_batch_size 
        tokens = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        labels = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )
        loss_mask = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.float32,
            device=torch.cuda.current_device()
        )
        
        
        attention_mask = None
        if args.create_attention_mask_in_dataloader:
            attention_mask = torch.empty(
                (1, 1, cur_max_seq_length, cur_max_seq_length),
                dtype=torch.bool,
                device=torch.cuda.current_device()
            )
        position_ids = torch.empty(
            (batch_size, cur_max_seq_length),
            dtype=torch.int64,
            device=torch.cuda.current_device()
        )

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