import time
from functools import lru_cache
import numpy as np
import sys
import torch
from megatron_patch.tokenizer import build_tokenizer, get_tokenizer
from megatron_patch.data.indexed_dataset_dpo import MMapIndexedDataset_DPO
from megatron.training import get_args

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)
class GPTDataset_DPO(torch.utils.data.Dataset):

    def __init__(self, 
                 name, 
                 documents, 
                 indexed_dataset, # document means ranges of data
                 seq_length, seed):
        self.args = get_args()
        self.name = name
        self.indexed_dataset = indexed_dataset
        self.seq_length = seq_length
        self.seed = seed
        self.numpy_random_state = np.random.RandomState(seed)
        self.dpo_idx = self._initialize_and_shuffle_indices(documents)
            
        
        self.cur_tokenizer = build_tokenizer(self.args)
        self.mask_id = self.cur_tokenizer.vocab_size + 1
    
        if hasattr(self.cur_tokenizer, 'pad_token_id'):
            self.pad = self.cur_tokenizer.pad_token_id
        else:
            self.pad = 0

    def __len__(self):
        return len(self.dpo_idx)

    def __getitem__(self, idx):
        # Get the shuffled index.
        assert idx < len(self.dpo_idx) and idx >= 0
        idx = self.dpo_idx[idx]
        
        # Get the data from indexed_dataset
        data = self.indexed_dataset.get(idx)
        chosen_input, chosen_label = data['chosen']
        rejected_input, rejected_label = data['rejected']

        # Process chosen data
        chosen_padded = self._pad_and_mask(chosen_input, chosen_label)

        # Process rejected data
        rejected_padded = self._pad_and_mask(rejected_input, rejected_label)
        

        return {
            "chosen_text": np.array(chosen_padded['text'], dtype=np.int64),
            "chosen_label": np.array(chosen_padded['label'], dtype=np.int64),
            "rejected_text": np.array(rejected_padded['text'], dtype=np.int64),
            "rejected_label": np.array(rejected_padded['label'], dtype=np.int64),
        }

        
    def _pad_and_mask(self, sample, label_sample):
        assert sample.shape == label_sample.shape
        pad_len = self.seq_length - len(sample)
 
        pad_value = np.ones(shape=(pad_len,), dtype=sample.dtype)
        label_pad_value = np.ones(shape=(pad_len,), dtype=sample.dtype)

        pad_value[:] = self.pad
        label_pad_value[:] = self.mask_id
        padded_sample = np.concatenate([sample, pad_value])
        padded_label_sample = np.concatenate([label_sample, label_pad_value])

        
        assert self.mask_id == padded_label_sample[0]
        assert self.mask_id != padded_sample[0]

        return {
            "text": padded_sample,
            "label": padded_label_sample,
        }

    
    def _initialize_and_shuffle_indices(self, documents):
        # Checks
        assert np.min(documents) >= 0
        assert np.max(documents) < self.indexed_dataset.sizes.shape[0]
        
        shuffled_documents = documents.copy()
        self.numpy_random_state.shuffle(shuffled_documents)
    
        self.data_idx = np.tile(shuffled_documents, self.args.epochs)

        if self.args.shuffle_all_epochs:
            # Shuffle across all epochs
            perm = self.numpy_random_state.permutation(len(self.data_idx))
            self.data_idx = self.data_idx[perm]
        else:
            # Shuffle each epoch independently
            epoch_size = len(documents)
            for i in range(self.args.epochs):
                start = i * epoch_size
                end = (i + 1) * epoch_size
                perm = self.numpy_random_state.permutation(epoch_size)
                self.data_idx[start:end] = self.data_idx[start:end][perm]
        
        return self.data_idx

def _get_train_valid_test_split_(splits_string, size):
    """ Get dataset splits from comma or '/' separated string list."""
    splits = []
    if splits_string.find(',') != -1:
        splits = [float(s) for s in splits_string.split(',')]
    elif splits_string.find('/') != -1:
        splits = [float(s) for s in splits_string.split('/')]
    else:
        splits = [float(splits_string)]

    while len(splits) < 3:
        splits.append(0.)
    splits = splits[:3]
    splits_sum = sum(splits)
    assert splits_sum > 0.0

    splits = [split / splits_sum for split in splits]
    splits_index = [0]
    for index, split in enumerate(splits):
        splits_index.append(splits_index[index] +
                            int(round(split * float(size))))
    diff = splits_index[-1] - size
    for index in range(1, len(splits_index)):
        splits_index[index] -= diff
    assert len(splits_index) == 4
    assert splits_index[-1] == size
    return splits_index


def make_indexed_dataset_dpo(path, impl, skip_warmup=False):
    if not MMapIndexedDataset_DPO.exists(path):
        raise FileNotFoundError("Dataset not found: {}".format(path))
    elif impl == "mmap" and MMapIndexedDataset_DPO.exists(path):
        return MMapIndexedDataset_DPO(path, skip_warmup=skip_warmup)
    print_rank_0(f"Unknown dataset implementation: {impl}")
    return None



def _build_train_valid_test_datasets_dpo(
    data_prefix,
    data_impl,
    splits_string,
    seq_length,
    seed,
):
    """Build train, valid, and test datasets."""
    
    print_rank_0(' > building dataset indexed dataset ...')
    start_time = time.time()
    indexed_dataset =  make_indexed_dataset_dpo(data_prefix, data_impl) # get MMapIndexedDataset_DPO
    print_rank_0(' > finished creating indexed dataset in {:4f} '
                 'seconds'.format(time.time() - start_time))
    print_rank_0(' > number of indexed dataset: {}'.format(indexed_dataset.sizes.shape[0]))

    total_num_samples = indexed_dataset.sizes.shape[0]
    splits = _get_train_valid_test_split_(splits_string, total_num_samples) # return a list

    print_rank_0(' > dataset split ...')
   
    def print_split_stats(split_name, index):
        print_rank_0("   {}".format(split_name))
        print_rank_0("    document indices in [{}, {}] total of {} "
                     "documents".format(splits[index], splits[index+1], splits[index+1] - splits[index]))
        
    
    print_split_stats("train", 0)
    print_split_stats("valid", 1)
    print_split_stats("test", 2)

    def build_dataset_dpo(index, name):
        dataset = None
        if splits[index+1] > splits[index]:
            documents = np.arange(start=splits[index], stop=splits[index+1], step=1, dtype=np.int32)
            dataset = GPTDataset_DPO(name, documents, indexed_dataset,
                                 seq_length, seed)
        return dataset
    
    train_dataset = build_dataset_dpo(0,"train")
    valid_dataset = build_dataset_dpo(1,"valid")
    test_dataset = build_dataset_dpo(2,"test")

    return (train_dataset, valid_dataset, test_dataset)

def build_train_valid_test_datasets_dpo(
    data_prefix,
    data_impl,
    splits_string,
    seq_length,
    seed,
):
    """Build train, valid, and test datasets."""
    if data_prefix :
        print_rank_0(' > Single data path provided for train / valid / test')

    
    return _build_train_valid_test_datasets_dpo(
            data_prefix[0],
            data_impl,
            splits_string,
            seq_length,
            seed,
        )

if __name__ == "__main__" :
    train_dataset, valid_dataset, test_dataset = build_train_valid_test_datasets_dpo(
        data_prefix=("auto-alignment/algorithms/megatron_dpo/data/dummy_dpo_conversations_maxlen_2048",),
        data_impl="mmap",
        splits_string="99,1,1",
        seq_length=2048,
        seed=42
    )
    
    for pair_data in train_dataset :
        print(pair_data["chosen_text"])
        print(pair_data["chosen_label"])
        print(pair_data["rejected_text"])
        print(pair_data["rejected_label"])
    