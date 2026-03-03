"""Online SFT and DPO datasets that tokenize on-the-fly.

These datasets read raw JSON conversation data and tokenize at training time
using AutoAlign's ``Conversation`` / ``TEMPLATES`` system, eliminating the
need for an offline ``preprocess`` step.

Usage in shell scripts::

    torchrun ... -m autoalign.megatron.entries.sft \\
        --dataset json \\
        --data-path ./data/sft_conversations.json \\
        --template chatml-idsys \\
        --model-path Qwen/Qwen2.5-3B-Instruct \\
        ...
"""

import json
import logging
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from autoalign.conversation import (
    IGNORED_TOKEN_ID,
    TEMPLATES,
    Conversation,
    Role,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _generate_labels_with_mask_id(conversation, tokenized, tokenizer, seq_length, mask_id):
    """Generate labels using *mask_id* instead of IGNORED_TOKEN_ID (-100).

    Re-implements the label generation logic from ``ConversationSFT`` in
    ``toolkits/sft/preprocess.py`` so that we don't need to import from
    the preprocessing module.
    """
    template = conversation.template
    if template.strategy:
        labels = template.strategy.generate_labels(
            conversation.messages,
            tokenized,
            tokenizer,
            template.get_attributes(),
        )
        # Replace IGNORED_TOKEN_ID → mask_id
        return [mask_id if l == IGNORED_TOKEN_ID else l for l in labels]

    labels = [mask_id] * len(tokenized.input_ids)
    cur_inst = ""
    for role, message in conversation.messages:
        if role in (Role.SYSTEM, Role.HUMAN):
            cur_inst += template.role_starts[role] + message + template.role_ends[role]
        else:
            cur_inst += template.role_starts[role]
            start_idx = len(
                tokenizer(cur_inst, padding="do_not_pad", truncation=True, max_length=seq_length).input_ids
            ) - template.offset
            end_idx = len(
                tokenizer(
                    cur_inst + message + template.role_ends[role],
                    padding="do_not_pad",
                    truncation=True,
                    max_length=seq_length,
                ).input_ids
            )
            labels[start_idx:end_idx] = tokenized.input_ids[start_idx:end_idx]
            cur_inst += message + template.role_ends[role]
    return labels


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _build_epoch_indices(n_docs, epochs, seed, shuffle_all_epochs=False):
    """Create shuffled epoch indices, matching ``GPTDatasetSFTConv`` behaviour."""
    rng = np.random.RandomState(seed)
    documents = np.arange(n_docs, dtype=np.int32)
    rng.shuffle(documents)
    indices = np.tile(documents, epochs)
    if shuffle_all_epochs:
        perm = rng.permutation(len(indices))
        indices = indices[perm]
    else:
        epoch_size = n_docs
        for i in range(epochs):
            start = i * epoch_size
            end = (i + 1) * epoch_size
            perm = rng.permutation(epoch_size)
            indices[start:end] = indices[start:end][perm]
    return indices


# ---------------------------------------------------------------------------
# Online SFT Dataset
# ---------------------------------------------------------------------------

class OnlineSFTDataset(torch.utils.data.Dataset):
    """SFT dataset that tokenizes conversations on-the-fly from raw JSON.

    Produces the same ``{"conv_text": ..., "conv_label": ...}`` dicts as
    ``GPTDatasetSFTConv`` so the downstream batch collation code in
    ``utils.py`` works unchanged.
    """

    def __init__(self, data, tokenizer, template_name, seq_length, seed, epochs=1, shuffle_all_epochs=False):
        self.data = data
        self.tokenizer = tokenizer
        self.template = TEMPLATES[template_name]
        self.seq_length = seq_length
        self.mask_id = tokenizer.vocab_size + 1
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self.indices = _build_epoch_indices(len(data), epochs, seed, shuffle_all_epochs)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        json_line = self.data[data_idx]

        conversation = Conversation(self.template)
        conversation.fill_in_messages(json_line)
        conversation_str = conversation.get_conversation_str()

        tokenized = self.tokenizer(
            conversation_str, padding="do_not_pad", truncation=True, max_length=self.seq_length
        )
        labels = _generate_labels_with_mask_id(
            conversation, tokenized, self.tokenizer, self.seq_length, self.mask_id
        )
        input_ids = tokenized.input_ids

        # Pad to seq_length (same as GPTDatasetSFTConv._pad_and_mask)
        pad_len = self.seq_length - len(input_ids)
        text = np.array(input_ids + [self.pad_id] * pad_len, dtype=np.int64)
        label = np.array(labels + [self.mask_id] * pad_len, dtype=np.int64)

        return {"conv_text": text, "conv_label": label}


# ---------------------------------------------------------------------------
# Online DPO Dataset
# ---------------------------------------------------------------------------

class OnlineDPODataset(torch.utils.data.Dataset):
    """DPO dataset that tokenizes chosen/rejected conversations on-the-fly.

    Produces ``{"chosen_text": ..., "chosen_label": ..., "rejected_text": ..., "rejected_label": ...}``
    dicts matching the MMap DPO dataset format.
    """

    def __init__(self, data, tokenizer, template_name, seq_length, seed, epochs=1, shuffle_all_epochs=False):
        self.data = data
        self.tokenizer = tokenizer
        self.template = TEMPLATES[template_name]
        self.seq_length = seq_length
        self.mask_id = tokenizer.vocab_size + 1
        self.pad_id = getattr(tokenizer, "pad_token_id", 0) or 0
        self.indices = _build_epoch_indices(len(data), epochs, seed, shuffle_all_epochs)

    def __len__(self):
        return len(self.indices)

    def _tokenize_branch(self, json_messages):
        """Tokenize one branch (chosen or rejected)."""
        conversation = Conversation(self.template)
        # DPO data uses {"conversations": [...]} or direct list format
        if isinstance(json_messages, dict):
            conversation.fill_in_messages(json_messages)
        else:
            conversation.fill_in_messages({"conversations": json_messages})
        conversation_str = conversation.get_conversation_str()

        tokenized = self.tokenizer(
            conversation_str, padding="do_not_pad", truncation=True, max_length=self.seq_length
        )
        labels = _generate_labels_with_mask_id(
            conversation, tokenized, self.tokenizer, self.seq_length, self.mask_id
        )
        input_ids = tokenized.input_ids

        pad_len = self.seq_length - len(input_ids)
        text = np.array(input_ids + [self.pad_id] * pad_len, dtype=np.int64)
        label = np.array(labels + [self.mask_id] * pad_len, dtype=np.int64)
        return text, label

    def __getitem__(self, idx):
        data_idx = self.indices[idx]
        json_line = self.data[data_idx]

        chosen_text, chosen_label = self._tokenize_branch(json_line["chosen"])
        rejected_text, rejected_label = self._tokenize_branch(json_line["rejected"])

        return {
            "chosen_text": chosen_text,
            "chosen_label": chosen_label,
            "rejected_text": rejected_text,
            "rejected_label": rejected_label,
        }


# ---------------------------------------------------------------------------
# Builder functions (drop-in replacements for the MMap builders)
# ---------------------------------------------------------------------------

def build_train_valid_test_datasets_online_sft(
    data_path,
    model_path,
    template_name,
    seq_length,
    seed,
    splits_string,
    epochs=1,
    shuffle_all_epochs=False,
):
    """Build train/valid/test SFT datasets from a JSON file (online tokenization).

    Returns ``(train_ds, valid_ds, test_ds)`` — same contract as
    ``build_train_valid_test_datasets_sft_conv``.
    """
    logger.info("Loading JSON data from %s", data_path)
    all_data = _load_json(data_path)
    total = len(all_data)
    logger.info("Loaded %d conversations", total)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Compute splits
    splits = _parse_splits(splits_string, total)

    def _build(start, end, name):
        if end <= start:
            return None
        data_slice = all_data[start:end]
        return OnlineSFTDataset(data_slice, tokenizer, template_name, seq_length, seed, epochs, shuffle_all_epochs)

    train_ds = _build(splits[0], splits[1], "train")
    valid_ds = _build(splits[1], splits[2], "valid")
    test_ds = _build(splits[2], splits[3], "test")
    return train_ds, valid_ds, test_ds


def build_train_valid_test_datasets_online_dpo(
    data_path,
    model_path,
    template_name,
    seq_length,
    seed,
    splits_string,
    epochs=1,
    shuffle_all_epochs=False,
):
    """Build train/valid/test DPO datasets from a JSON file (online tokenization)."""
    logger.info("Loading JSON data from %s", data_path)
    all_data = _load_json(data_path)
    total = len(all_data)
    logger.info("Loaded %d DPO pairs", total)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    splits = _parse_splits(splits_string, total)

    def _build(start, end, name):
        if end <= start:
            return None
        data_slice = all_data[start:end]
        return OnlineDPODataset(data_slice, tokenizer, template_name, seq_length, seed, epochs, shuffle_all_epochs)

    train_ds = _build(splits[0], splits[1], "train")
    valid_ds = _build(splits[1], splits[2], "valid")
    test_ds = _build(splits[2], splits[3], "test")
    return train_ds, valid_ds, test_ds


def _parse_splits(splits_string, size):
    """Parse split ratios and return boundary indices [0, train_end, valid_end, test_end]."""
    if "," in splits_string:
        splits = [float(s) for s in splits_string.split(",")]
    elif "/" in splits_string:
        splits = [float(s) for s in splits_string.split("/")]
    else:
        splits = [float(splits_string)]
    while len(splits) < 3:
        splits.append(0.0)
    splits = splits[:3]
    total = sum(splits)
    splits = [s / total for s in splits]
    boundaries = [0]
    for s in splits:
        boundaries.append(boundaries[-1] + int(round(s * size)))
    # Adjust last boundary
    diff = boundaries[-1] - size
    for i in range(1, len(boundaries)):
        boundaries[i] -= diff
    return boundaries
