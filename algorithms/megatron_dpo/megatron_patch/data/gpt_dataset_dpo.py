import time
from functools import lru_cache

import numpy as np
import torch
from megatron import get_tokenizer,print_rank_0,get_args
from megatron.data.blendable_dataset import BlendableDataset
from megatron.data.dataset_utils import (
    _build_train_valid_test_datasets,
    get_datasets_weights_and_num_samples,
    get_train_valid_test_split_,
)
from megatron.data.gpt_dataset import build_dataset_group
from megatron.data.indexed_dataset import (
    IndexedCachedDataset,
    IndexedDataset,
    infer_dataset_impl,
)


from .indexed_dataset_sft import MMapIndexedDataset_sft



class GPTDataset_sft(torch.utils.data.Dataset):

    def __init__(self, name, documents, indexed_dataset, # document means ranges of data
              seq_length, seed):
        self.args = get_args()
        self.name = name
        self.indexed_dataset = indexed_dataset
        
        np.random.seed(seed)

        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < indexed_dataset.sizes.shape[0]

        self.sft_idx = documents.copy()
        # for multi epoch
        self.sft_idx = np.concatenate([self.sft_idx]*self.args.epoch)

        if self.args.shuffle_all_epoch :
            perm = np.random.permutation(len(self.sft_idx))
            self.sft_idx = self.sft_idx[perm]
            
        self.sft_length = seq_length
        cur_tokenizer = get_tokenizer()
        self.mask_id = cur_tokenizer.vocab_size + 1
        if hasattr(cur_tokenizer, 'pad_token_id'):
           self.pad = cur_tokenizer.pad_token_id
        else:
            self.pad = 0

    def __len__(self):
        return len(self.sft_idx)

    def __getitem__(self, idx):
        # Get the shuffled index.
        assert idx < len(self.sft_idx) and idx >= 0
        idx = self.sft_idx[idx]
        
        sample,label_sample = self.indexed_dataset.get(idx)
        assert sample.shape == label_sample.shape
        pad_len = self.sft_length + 1 -len(sample)

        pad_value = np.ones(shape=(pad_len,),dtype=sample.dtype)
        label_pad_value = np.ones(shape=(pad_len,),dtype=sample.dtype)

        pad_value[:] =self.pad
        label_pad_value[:] = self.mask_id
        padded_sample = np.concatenate([sample,pad_value])
        padded_label_sample = np.concatenate([label_sample,label_pad_value])

        assert self.mask_id == padded_label_sample[0]
        assert self.mask_id != padded_sample[0]

        return {
            "text": np.array(padded_sample, dtype=np.int64),
            "label": np.array(padded_label_sample, dtype=np.int64),
        }

def make_dataset_sft(path,impl,skip_warmup=False):
    if not IndexedDataset.exists(path):
        raise FileNotFoundError("Dataset not found: {}".format(path))
    if impl == "infer":
        impl = infer_dataset_impl(path)
    if impl == "lazy" :
        return IndexedDataset(path)
    elif impl == "cached":
        return IndexedCachedDataset(path)
    elif impl == "mmap" and MMapIndexedDataset_sft.exists(path):
        return MMapIndexedDataset_sft(path, skip_warmup=skip_warmup)
    print(f"Unknown dataset implementation: {impl}")
    return None

def get_indexed_dataset_sft(data_prefix, data_impl, skip_warmup):
    """Build indexed dataset."""
    print_rank_0(' > building dataset index ...')
    start_time = time.time()
    indexed_dataset = make_dataset_sft(data_prefix, data_impl, skip_warmup)
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0(' > number of indexed dataset: {}'.format(indexed_dataset.sizes.shape[0]))
    return indexed_dataset

def _build_train_valid_test_datasets_sft(
    data_prefix,
    data_impl,
    splits_string,
    seq_length,
    seed,
    skip_warmup,
):
    """Build train, valid, and test datasets."""
    args = get_args()
    indexed_dataset = get_indexed_dataset_sft(data_prefix, data_impl, skip_warmup) # get MMapIndexedDataset_sft

    total_num_samples = indexed_dataset.sizes.shape[0]
    splits = get_train_valid_test_split_(splits_string, total_num_samples)

    print_rank_0(' > dataset split ...')
   
    def print_split_stats(split_name, index):
        print_rank_0("   {}".format(split_name))
        print_rank_0("    document indices in [{}, {}] total of {} "
                     "documents".format(splits[index], splits[index+1], splits[index+1] - splits[index]))
        
    print_rank_0("num train data :{args.train_data_sample}")
    print_split_stats("train", 0)
    print_split_stats("valid", 1)
    print_split_stats("test", 2)

    def build_dataset_sft(index,name):
        dataset = None
        if splits[index+1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index+1],step=1, dtype=np.int32)
            dataset = GPTDataset_sft(name, documents, indexed_dataset,
                                 seq_length, seed)
        return dataset
    
    train_dataset = build_dataset_sft(0,"train")
    valid_dataset = build_dataset_sft(1,"valid")
    test_dataset = build_dataset_sft(2,"test")

    return (train_dataset, valid_dataset, test_dataset)

def build_train_valid_test_datasets_dpo(
    data_prefix,
    data_impl,
    splits_string,
    seq_length,
    seed,
    skip_warmup,
):
    """Build train, valid, and test datasets."""
    if data_prefix :
        print_rank_0(' > Single data path provided for train / valid / test')

    
    return _build_train_valid_test_datasets_sft(
            data_prefix[0],
            data_impl,
            splits_string,
            seq_length,
            seed,
            skip_warmup,
        )
    




