# Copyright (c) 2024 Alibaba PAI and Nvidia Megatron-LM Team.
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

import logging
import math
from typing import Dict, Literal, Optional, Tuple, Union
from torch import Tensor
import torch
import torch.nn.functional as F
from megatron.training import get_args
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
from megatron_patch.tokenizer import build_tokenizer

from .transformer_block import TransformerBlock
from .model import GPTModel

class GPTModel_DPO(LanguageModule):
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
        ftx_gamma: float = 0.,
        model_using: str = 'both',
        forward_without_loss = False,
        dpo_loss_of_orion = False,
        orpo_loss = False,
    ) -> None:
        super().__init__(config=config)
        args = get_args()
        
        self.parallel_output = parallel_output
        self.pre_process = pre_process
        self.post_process = post_process
        self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
        self.untie_embeddings_and_output_weights = args.untie_embeddings_and_output_weights
        
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.ftx_gamma = ftx_gamma
        self.model_using = model_using
        self.forward_without_loss = forward_without_loss
        self.dpo_loss_of_orion = dpo_loss_of_orion
        self.orpo_loss = orpo_loss
        
        
        self.policy_model = GPTModel(
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
            self.policy_model.set_input_tensor(None)
            self.ref_model.set_input_tensor(None)
        else:
            # print("input_Tesor", input_tensor[0].shape) # seq_len, 2*micro_batch_size, hidden_size
            policy_, ref_ = input_tensor[0].split(2, dim=1)
            self.policy_model.set_input_tensor(policy_)
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
        
        hidden_states = self.policy_model(
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
            
            return ret, {}
        else:
            logits = hidden_states
            ref_logits = ref_hidden_states.transpose(0,1).contiguous()

            if labels is None:
                return torch.stack((logits.transpose(0,1), ref_logits), 1)

            if self.forward_without_loss:
                return self.margin_or_lopp(labels, logits, ref_logits)

            ret = self.dpo(labels, logits, ref_logits)
            return ret

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        """Customized save for policy_model and ref_model, to avoid checkpoint mismatch in resume training"""
        state_dict_ = {}
        state_dict_['policy_model'] = self.policy_model.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
        
        state_dict_['ref_model'] = self.ref_model.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)
        
        # # Save word_embeddings.
        # if self.post_process and not self.untie_embeddings_and_output_weights:
        #     state_dict_[self._word_embeddings_for_head_key] = \
        #         self.policy_model.word_embeddings.state_dict(prefix=prefix,
        #                                             keep_vars=keep_vars)
    
        return state_dict_
    
    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        if 'ref_model' not in state_dict: # for iter 0
            policy_state_dict = ref_state_dict = state_dict
        else: # for resume condition when policy param has been trained and both policy and ref param has been saved
            ref_model = state_dict.pop('ref_model')
            # print('state_dict keys:', state_dict['language_model'].keys())
            # print('ref_model keys:', ref_model['language_model'].keys())
            policy_state_dict, ref_state_dict = state_dict, ref_model
        

        self.policy_model.load_state_dict(policy_state_dict, strict=strict)
        self.ref_model.load_state_dict(ref_state_dict, strict=strict)
        
        if self.post_process and not self.untie_embeddings_and_output_weights: # all reduce embedding grad will call self.word_embeddings
            self.word_embeddings = self.policy_model.word_embeddings
        
        # for name, param in self.policy_model.named_parameters():
        #     ref_param = self.ref_model.state_dict()[name]
            # if not torch.equal(param.data, ref_param.data):
            #     print(f"Parameter {name} is different between policy and ref models.")
            # else:
            #     print(f"Checked {name}")
        
        return [],[]
    
    def get_batch_logps(self, logits, labels, average_log_prob=False):
        # Save the original batch size and sequence length
        batch_size, seq_length, vocab_size = logits.size()
        
        # Flatten logits and labels
        logits = logits.reshape(batch_size * seq_length, vocab_size)
        labels = labels.reshape(batch_size * seq_length)
        
        # Compute per-token log probabilities
        per_token_logps = -1 * tensor_parallel.vocab_parallel_cross_entropy(logits, labels)
        
        # Reshape back to [batch_size, seq_length]
        per_token_logps = per_token_logps.view(batch_size, seq_length)
        
        # Compute sequence log probabilities
        seq_logps = per_token_logps.sum(dim=-1)  # Shape: [batch_size]
        
        if average_log_prob:
            # Compute valid token counts per sequence
            args = get_args()
            tokenizer = build_tokenizer(args)
            pad_token_id = tokenizer.pad_token_id
            labels_reshaped = labels.view(batch_size, seq_length)
            seq_lengths = (labels_reshaped != pad_token_id).sum(dim=-1).clamp(min=1)
            # Compute average log probability per token
            seq_logps = seq_logps / seq_lengths
        
        return seq_logps  # Shape: [batch_size]

    def dpo_loss(self, policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps, chosen_kl, rejected_kl):
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps)
        logits = chosen_rewards - rejected_rewards

        if self.loss_type == 'sigmoid':
            losses = (-F.logsigmoid(logits) * (1 - self.label_smoothing) -
                    F.logsigmoid(-logits) * self.label_smoothing)
        elif self.loss_type == 'kto':
            # TRL implementation of KTO
            # eqn (7) of the HALOs paper
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(f'unknown loss type {self.loss_type}')
    
        return losses, chosen_rewards.detach(), rejected_rewards.detach()

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

        # Initialize Kullback-Leibler divergences if needed
        chosen_kl, rejected_kl = None, None
        if self.loss_type == 'kto':
            kl = self.beta * self.get_batch_logps(logits - ref_logits, labels, average_log_prob=True).clamp(min=0)
            chosen_kl, rejected_kl = kl.chunk(2, dim=0)

        # Compute DPO loss
        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            chosen_kl,
            rejected_kl,
        )

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        
        # Additional loss calculations if needed
        if self.orpo_loss:
            chosen_labels, _ = labels.chunk(2, dim=0)
            loss_mask = (chosen_labels != -100)
            policy_chosen_logps = policy_chosen_logps / loss_mask.sum(-1)
            policy_rejected_logps = policy_rejected_logps / loss_mask.sum(-1)
            log_odds = (policy_chosen_logps - policy_rejected_logps) \
                    - (torch.log1p(-torch.exp(policy_chosen_logps)) - torch.log1p(-torch.exp(policy_rejected_logps)))
            odds_ratio_loss = -F.logsigmoid(log_odds)
            losses = -policy_chosen_logps + self.ftx_gamma * odds_ratio_loss
            reward_accuracies = (policy_chosen_logps > policy_rejected_logps).float()
            
        elif self.dpo_loss_of_orion:
            sft_loss = 0.0
            if self.ftx_gamma > 1e-6:
                chosen_labels, _ = labels.chunk(2, dim=0)
                loss_mask = (chosen_labels != -100)
                sft_loss = -policy_chosen_logps / loss_mask.sum(-1)
                losses = losses * (1 - self.ftx_gamma) + self.ftx_gamma * sft_loss
            else:
                if self.ftx_gamma > 1e-6:
                    chosen_labels, _ = labels.chunk(2, dim=0)
                    loss_mask = (chosen_labels != -100)
                    losses += self.ftx_gamma * policy_chosen_logps / loss_mask.sum(-1)

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

        if self.dpo_loss_of_orion:
            metrics["dpo-metrics/sft_loss"] = sft_loss
        if self.orpo_loss:
            metrics["dpo-metrics/odds_ratio_loss"] = odds_ratio_loss.mean().float()
            metrics["dpo-metrics/log_odds"] = log_odds.mean().float()

        return losses, metrics
        
    
    def margin_or_logps(self, labels, logits, ref_logits):
        # DPO logits abn logps
        logps = None
        if self.policy_model:
            logits = self.convert_logits(logits, labels)
            logps = self.get_batch_logps(logits, labels)

        ref_logps = None
        if self.ref_model:
            ref_logits = self.convert_logits(ref_logits, labels)
            ref_logps = self.get_batch_logps(ref_logits, labels)

        if logps is None:
            return ref_logps
        if ref_logps is None:
            return logps
        return logps - ref_logps