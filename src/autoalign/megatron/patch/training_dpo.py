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

"""DPO training — thin wrapper over the shared training base (training_sft.py).

DPO uses the same training loop as SFT with two differences:
1. Dataset index files use ``chosen_idx_file_path`` (paired data format).
2. Micro-batch size is multiplied by 4 (chosen/rejected × policy/reference).
"""

from autoalign.megatron.patch.data.indexed_dataset_dpo import chosen_idx_file_path
from autoalign.megatron.patch.data.gpt_dataset_dpo import _get_train_valid_test_split_
from autoalign.megatron.patch.training_sft import sft as _pretrain


def dpo(
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    extra_args_provider=None,
    args_defaults={},
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
):
    """DPO training entry point.

    Delegates to the shared ``sft()`` implementation with DPO-specific
    parameters for dataset indexing and micro-batch sizing.
    """
    return _pretrain(
        train_valid_test_dataset_provider,
        model_provider,
        model_type,
        forward_step_func,
        process_non_loss_data_func=process_non_loss_data_func,
        extra_args_provider=extra_args_provider,
        args_defaults=args_defaults,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        idx_file_fn=chosen_idx_file_path,
        split_fn=_get_train_valid_test_split_,
        micro_batch_multiplier=4,
    )
