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

import logging
import math
from typing import Dict, Literal, Optional, Tuple, Union
from torch import Tensor
import torch
import torch.nn.functional as F
from megatron.training import print_rank_0, get_args
from megatron.core import mpu
from megatron.core import InferenceParams, parallel_state, tensor_parallel
from megatron.core.dist_checkpointing.mapping import ShardedStateDict
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType, ModelType
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron_patch.tokenizer import get_tokenizer

from megatron_patch.model.qwen2.transformer_block import TransformerBlock
from megatron_patch.model.qwen2.model import GPTModel


class GPTModelDPO(GPTModel):
    """GPT Transformer language model.
    """
       
    def __init__(
        self,
        config: TransformerConfig,
        transformer_layer_spec: ModuleSpec,
        vocab_size: int,
        max_sequence_length: int,
        pre_process: bool = True,
        post_process: bool = True,
        fp16_lm_cross_entropy: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        position_embedding_type: Literal[
            "learned_absolute", "rope"
        ] = "learned_absolute",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        seq_len_interpolation_factor: Optional[float] = None,
        beta: float = 0.1,
        label_smoothing: float = 0,
        loss_type: str ='sigmoid',
    ) -> None:
        args = get_args()
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.tokenizer = get_tokenizer()
        
        super().__init__(
            config = config,
            transformer_layer_spec = transformer_layer_spec,
            vocab_size = vocab_size,
            max_sequence_length = max_sequence_length,
            pre_process = pre_process,
            post_process = post_process,
            fp16_lm_cross_entropy = fp16_lm_cross_entropy,
            parallel_output = parallel_output,
            share_embeddings_and_output_weights = share_embeddings_and_output_weights,
            position_embedding_type = position_embedding_type,
            rotary_percent = rotary_percent,
            rotary_base = rotary_base,
            seq_len_interpolation_factor = seq_len_interpolation_factor
        )
        
        
        self.ref_model = GPTModel(
                                config = config,
                                transformer_layer_spec = transformer_layer_spec,
                                vocab_size = vocab_size,
                                max_sequence_length = max_sequence_length,
                                pre_process = pre_process,
                                post_process = post_process,
                                fp16_lm_cross_entropy = fp16_lm_cross_entropy,
                                parallel_output = parallel_output,
                                share_embeddings_and_output_weights = share_embeddings_and_output_weights,
                                position_embedding_type = position_embedding_type,
                                rotary_percent = rotary_percent,
                                rotary_base = rotary_base,
                                seq_len_interpolation_factor = seq_len_interpolation_factor
                            )
        
        
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    

    def set_input_tensor(self, input_tensor: Tensor) -> None:
        """Sets input tensor to the model.

        See megatron.model.transformer.set_input_tensor()

        Args:
            input_tensor (Tensor): Sets the input tensor for the model.
        """
        # This is usually handled in schedules.py but some inference code still
        # gives us non-lists or None
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]

        assert len(input_tensor) == 1, 'input_tensor should only be length 1 for gpt/bert'
        
        if input_tensor[0] is None:
            super().set_input_tensor(None)
            self.ref_model.set_input_tensor(None)
        else:
            policy_, ref_ = input_tensor[0].chunk(2, dim=1)
            super().set_input_tensor(policy_)
            self.ref_model.set_input_tensor(ref_)

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        decoder_input: Tensor = None,
        labels: Tensor = None,
        inference_params: InferenceParams = None,
        packed_seq_params: PackedSeqParams = None,
        extra_block_kwargs: dict = None,
    ) -> Tensor:
        hidden_states = super().forward(
            input_ids,
            position_ids,
            attention_mask,
            decoder_input,
            labels=None,
            inference_params=inference_params,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            )
    
        with torch.no_grad():
            ref_hidden_states = self.ref_model(
                input_ids,
                position_ids,
                attention_mask,
                decoder_input,
                labels=None,
                inference_params=inference_params,
                packed_seq_params=packed_seq_params,
                extra_block_kwargs=extra_block_kwargs,
             )
     
        if not self.post_process:
            ret = torch.cat((hidden_states, ref_hidden_states), dim=1)
            
            return ret
        else:
            logits = hidden_states
            ref_logits = ref_hidden_states

            if labels is None:
                return torch.stack([logits.transpose(0,1).contiguous(), ref_logits.transpose(0,1).contiguous()]), {}

            ret, metrics = self.dpo(labels, logits, ref_logits)
            return ret, metrics

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Customized save for policy_model and ref_model, to avoid checkpoint mismatch in resume training"""
        state_dict_ = {}
        state_dict_['policy_model'] = super().state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
        
        state_dict_['ref_model'] = self.ref_model.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
    
    
        return state_dict_
    
    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if 'ref_model' in state_dict:
            print_rank_0("Resuming DPO training from checkpoint. Loading distinct states for policy and reference models...")
            ref_state_dict = state_dict.pop('ref_model')
            policy_state_dict = state_dict
            super().load_state_dict(policy_state_dict, strict=strict)
            self.ref_model.load_state_dict(ref_state_dict, strict=strict)

        else:
            print_rank_0("Starting new DPO training from a base model. Initializing policy and reference models to be identical...")
            policy_state_dict = state_dict
            super().load_state_dict(policy_state_dict, strict=strict)
            self.ref_model.load_state_dict(self.state_dict(), strict=strict)
            
        return [], []
    
    def get_batch_logps(self, logits, labels, average_log_prob=False):
        # Save the original batch size and sequence length

        batch_size, seq_length, vocab_size = logits.size()
        
        mask_id = self.tokenizer.vocab_size + 1 
        loss_mask = (labels != mask_id).float() 
        
        loss_mask = loss_mask.view(batch_size * seq_length) 
        
        # Flatten logits and labels
        logits = logits.reshape(batch_size * seq_length, vocab_size)
        labels = labels.reshape(batch_size * seq_length)
        
        # Compute per-token log probabilities
        per_token_logps = -1 * tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
        
        per_token_logps = per_token_logps * loss_mask
        
        # Reshape back to [batch_size, seq_length]
        per_token_logps = per_token_logps.view(batch_size, seq_length)
        
        # Compute sequence log probabilities
        seq_logps = per_token_logps.sum(dim=-1)  # Shape: [batch_size]
        
        
        return seq_logps  # Shape: [batch_size]

    def dpo_loss(self, 
                 policy_chosen_logps, 
                 policy_rejected_logps,
                 reference_chosen_logps, 
                 reference_rejected_logps, 
                 ):
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        logits = chosen_rewards - rejected_rewards

        losses = (-F.logsigmoid(logits) * (1 - self.label_smoothing) 
                    -F.logsigmoid(-logits) * self.label_smoothing)
    
        return losses.mean() , chosen_rewards.detach(), rejected_rewards.detach()

    def dpo(self, labels, logits, ref_logits):
        batch_size = logits.shape[0]
        if batch_size % 2 != 0 or batch_size < 2:
            raise ValueError(f"Batch size must be even and at least 2, but got batch_size = {batch_size}")
        
        # Split logits into chosen and rejected parts
        policy_chosen_logits, policy_rejected_logits = logits.chunk(2, dim=0)
        policy_chosen_logits_mean = policy_chosen_logits.detach().mean().float()
        policy_rejected_logits_mean = policy_rejected_logits.detach().mean().float()

        # Reduce for mean logits
        torch.distributed.all_reduce(
            policy_chosen_logits_mean,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group(),
        )
        torch.distributed.all_reduce(
            policy_rejected_logits_mean,
            op=torch.distributed.ReduceOp.SUM,
            group=mpu.get_tensor_model_parallel_group(),
        )
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        policy_chosen_logits_mean /= world_size
        policy_rejected_logits_mean /= world_size

        # Compute log probabilities
        logps = self.get_batch_logps(logits, labels)
        ref_logps = self.get_batch_logps(ref_logits, labels)
        

        # Split log probabilities into chosen and rejected parts
        policy_chosen_logps, policy_rejected_logps = logps.chunk(2, dim=0)
        reference_chosen_logps, reference_rejected_logps = ref_logps.chunk(2, dim=0)


        # Compute DPO loss
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        

        # Prepare metrics
        metrics = {
            "dpo-metrics/rewards-chosen": chosen_rewards.mean().float(),
            "dpo-metrics/rewards-rejected": rejected_rewards.mean().float(),
            "dpo-metrics/rewards-accuracies": reward_accuracies.mean().float(),
            "dpo-metrics/rewards-margins": (chosen_rewards - rejected_rewards).mean().float(),
            "dpo-metrics/logps-chosen": policy_chosen_logps.detach().mean().float(),
            "dpo-metrics/logps-rejected": policy_rejected_logps.detach().mean().float(),
            "dpo-metrics/logits-chosen": policy_chosen_logits_mean,
            "dpo-metrics/logits-rejected": policy_rejected_logits_mean,
        }

        return losses, metrics