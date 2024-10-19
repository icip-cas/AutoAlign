import os
import shutil
import stat
import struct
from functools import lru_cache
from itertools import accumulate
from megatron.training import get_args, print_rank_0
import numpy as np
import torch


from .indexed_dataset import (
    best_fitting_dtype,
    _warmup_mmap_file,
    code,
    data_file_path,
    dtypes,
    index_file_path,
    IndexedDatasetBuilder,
)



def data_label_file_path(prefix_path):
    return prefix_path + "_label.bin"

def chosen_data_file_path(prefix_path):
    return prefix_path + "_chosen_input.bin"

def chosen_label_file_path(prefix_path):
    return prefix_path + "_chosen_label.bin"

def chosen_idx_file_path(prefix_path):
    return prefix_path + "_chosen.idx"

def rejected_data_file_path(prefix_path):
    return prefix_path + "_rejected_input.bin"

def rejected_label_file_path(prefix_path):
    return prefix_path + "_rejected_label.bin"

def rejected_idx_file_path(prefix_path):
    return prefix_path + "_rejected.idx"

class MMapIndexedDataset_DPO(torch.utils.data.Dataset):
    class Index(object):# after the class inited, it can read three information about (sizes pointer doc_idx) from idx files that set up the index of dataset
        _HDR_MAGIC = b'MMIDIDX\x00\x00'

        @classmethod
        def writer(cls, path, dtype):
            class _Writer(object):
                def __enter__(self):
                    self._file = open(path, 'wb')

                    self._file.write(cls._HDR_MAGIC)
                    self._file.write(struct.pack('<Q', 1))
                    self._file.write(struct.pack('<B', code(dtype)))

                    return self

                @staticmethod
                def _get_pointers(sizes):
                    dtype_size = dtype().itemsize
                    address = 0
                    pointers = []

                    for size in sizes:
                        pointers.append(address)
                        address += size * dtype_size
                    return pointers

                def write(self, sizes, doc_idx):
                    pointers = self._get_pointers(sizes)
                    self._file.write(struct.pack('<Q', len(sizes)))
                    self._file.write(struct.pack('<Q', len(doc_idx)))

                    sizes = np.array(sizes, dtype=np.int32)
                    self._file.write(sizes.tobytes(order='C'))
                    del sizes

                    pointers = np.array(pointers, dtype=np.int64)
                    self._file.write(pointers.tobytes(order='C'))
                    del pointers

                    doc_idx = np.array(doc_idx, dtype=np.int64)
                    self._file.write(doc_idx.tobytes(order='C'))
                    del doc_idx


                def __exit__(self, exc_type, exc_val, exc_tb):
                    self._file.close()

            return _Writer()

        def __init__(self, path, skip_warmup=False):
            with open(path, 'rb') as stream:
                magic_test = stream.read(9)
                assert self._HDR_MAGIC == magic_test, (
                    'Index file doesn\'t match expected format. '
                    'Make sure that --dataset-impl is configured properly.'
                )
                version = struct.unpack('<Q', stream.read(8))
                assert (1,) == version

                dtype_code, = struct.unpack('<B', stream.read(1))
                self._dtype = dtypes[dtype_code]
                self._dtype_size = self._dtype().itemsize

                self._len = struct.unpack('<Q', stream.read(8))[0] # the size of tensor set. tensor set e.g. [3,4] - > [[1,2,3],[1,2,3,4]]
                self._doc_count = struct.unpack('<Q', stream.read(8))[0]
                offset = stream.tell()

            if not skip_warmup:
                print_rank_0("    warming up index mmap file...")
                _warmup_mmap_file(path)

            self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
            self._bin_buffer = memoryview(self._bin_buffer_mmap)
            print_rank_0("    reading sizes...")
            self._sizes = np.frombuffer(
                self._bin_buffer,
                dtype=np.int32,
                count=self._len,
                offset=offset)
            print_rank_0("    reading pointers...")
            self._pointers = np.frombuffer(self._bin_buffer, 
                                           dtype=np.int64,
                                           count=self._len,
                                           offset=offset + self._sizes.nbytes)
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer, 
                                          dtype=np.int64,
                                          count=self._doc_count,
                                          offset=offset + self._sizes.nbytes + self._pointers.nbytes)

        def __del__(self):
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

        @property
        def dtype(self):
            return self._dtype

        @property
        def sizes(self):
            return self._sizes

        @property
        def doc_idx(self):
            return self._doc_idx

        @lru_cache(maxsize=8)
        def __getitem__(self, i):
            return self._pointers[i], self._sizes[i]

        def __len__(self):
            return self._len

    def __init__(self, prefix_path, skip_warmup=False):
        super().__init__()

        self._prefix_path = None
        
        self._index = None
        self._chosen_index = None
        self._rejected_index = None
        
        self._bin_buffer = None
        self._chosen_bin_buffer = None
        self._rejected_bin_buffer = None

        self._do_init(prefix_path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, prefix_path, skip_warmup): # create index, data buffer,lable buffer
        self._prefix_path = prefix_path
        self._chosen_index = self.Index(chosen_idx_file_path(self._prefix_path), skip_warmup) # create chosen index 
        self._rejected_index = self.Index(rejected_idx_file_path(self._prefix_path), skip_warmup) # create reject index 

        if not skip_warmup:
            print_rank_0("    warming up data mmap files...")
            _warmup_mmap_file(chosen_data_file_path(self._prefix_path))
            _warmup_mmap_file(rejected_data_file_path(self._prefix_path))
            
            print_rank_0("    warming up label mmap files...")
            _warmup_mmap_file(chosen_label_file_path(self._prefix_path))
            _warmup_mmap_file(rejected_label_file_path(self._prefix_path))
        
        # input data
        print_rank_0("    creating numpy buffers of mmap for input data...")
        self._chosen_bin_buffer_mmap = np.memmap(chosen_data_file_path(self._prefix_path), mode='r', order='C')
        self._rejected_bin_buffer_mmap = np.memmap(rejected_data_file_path(self._prefix_path), mode='r', order='C')
        
        print_rank_0("    creating memory views of numpy buffers for input data...")
        self._chosen_bin_buffer = memoryview(self._chosen_bin_buffer_mmap)
        self._rejected_bin_buffer = memoryview(self._rejected_bin_buffer_mmap)

        # label data
        print_rank_0("    creating numpy buffers of mmap for label data...")
        self._chosen_label_bin_buffer_mmap = np.memmap(chosen_label_file_path(self._prefix_path), mode='r', order='C')
        self._rejected_label_bin_buffer_mmap = np.memmap(rejected_label_file_path(self._prefix_path), mode='r', order='C')
        
        print_rank_0("    creating memory views of numpy buffers for label data...")
        self._chosen_label_bin_buffer = memoryview(self._chosen_label_bin_buffer_mmap)
        self._rejected_label_bin_buffer = memoryview(self._rejected_label_bin_buffer_mmap)
        

    def __del__(self):
        # Close and delete chosen input data mmap
        if hasattr(self, '_chosen_bin_buffer_mmap'):
            self._chosen_bin_buffer_mmap._mmap.close()
            del self._chosen_bin_buffer_mmap
        
        # Close and delete rejected input data mmap
        if hasattr(self, '_rejected_bin_buffer_mmap'):
            self._rejected_bin_buffer_mmap._mmap.close()
            del self._rejected_bin_buffer_mmap
        
        # Close and delete chosen label data mmap
        if hasattr(self, '_chosen_label_bin_buffer_mmap'):
            self._chosen_label_bin_buffer_mmap._mmap.close()
            del self._chosen_label_bin_buffer_mmap
        
        # Close and delete rejected label data mmap
        if hasattr(self, '_rejected_label_bin_buffer_mmap'):
            self._rejected_label_bin_buffer_mmap._mmap.close()
            del self._rejected_label_bin_buffer_mmap
        
        # Delete indices
        if hasattr(self, '_chosen_index'):
            del self._chosen_index
        if hasattr(self, '_rejected_index'):
            del self._rejected_index

    def __len__(self):
        return len(self._chosen_index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx): # return data and label
        if isinstance(idx, int):
            # self._index.dtype was determined when the data and label were writen in .bin, which means the  method about add_item() used the best_fitting_dtype(vocab_size)
            chosen_ptr, chosen_size = self._chosen_index[idx]
            rejected_ptr, rejected_size = self._rejected_index[idx]
            
            chosen_input = np.frombuffer(self._chosen_bin_buffer, 
                                     dtype=self._chosen_index.dtype,
                                     count=chosen_size, 
                                     offset=chosen_ptr)
            chosen_label = np.frombuffer(self._chosen_label_bin_buffer, 
                                        dtype=self._chosen_index.dtype,
                                        count=chosen_size, 
                                        offset=chosen_ptr)

            rejected_input = np.frombuffer(self._rejected_bin_buffer, 
                                        dtype=self._rejected_index.dtype,
                                        count=rejected_size, 
                                        offset=rejected_ptr)
            rejected_label = np.frombuffer(self._rejected_label_bin_buffer, 
                                        dtype=self._rejected_index.dtype,
                                        count=rejected_size, 
                                        offset=rejected_ptr)
            
            return {
                'chosen': (chosen_input, chosen_label),
                'rejected': (rejected_input, rejected_label)
            }
            
        
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")

            chosen_ptr = self._chosen_index._pointers[start]
            chosen_sizes = self._chosen_index._sizes[idx]
            chosen_offsets = list(accumulate(chosen_sizes))
            chosen_total_size = sum(chosen_sizes)

            rejected_ptr = self._rejected_index._pointers[start]
            rejected_sizes = self._rejected_index._sizes[idx]
            rejected_offsets = list(accumulate(rejected_sizes))
            rejected_total_size = sum(rejected_sizes)

            chosen_input = np.frombuffer(self._chosen_bin_buffer, dtype=self._chosen_index.dtype,
                                        count=chosen_total_size, offset=chosen_ptr)
            chosen_input_sents = np.split(chosen_input, chosen_offsets[:-1])

            chosen_label = np.frombuffer(self._chosen_label_bin_buffer, dtype=self._chosen_index.dtype,
                                        count=chosen_total_size, offset=chosen_ptr)
            chosen_label_sents = np.split(chosen_label, chosen_offsets[:-1])

            rejected_input = np.frombuffer(self._rejected_bin_buffer, dtype=self._rejected_index.dtype,
                                        count=rejected_total_size, offset=rejected_ptr)
            rejected_input_sents = np.split(rejected_input, rejected_offsets[:-1])

            rejected_label = np.frombuffer(self._rejected_label_bin_buffer, dtype=self._rejected_index.dtype,
                                        count=rejected_total_size, offset=rejected_ptr)
            rejected_label_sents = np.split(rejected_label, rejected_offsets[:-1])
            
            
            return {
                'chosen': (chosen_input_sents, chosen_label_sents),
                'rejected': (rejected_input_sents, rejected_label_sents)
            }


    def get(self, idx, offset=0, length=None):
        """ 
        Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        chosen_ptr, chosen_size = self._chosen_index[idx]
        rejected_ptr, rejected_size = self._rejected_index[idx]

        if length is None:
            chosen_length = chosen_size - offset
            rejected_length = rejected_size - offset
        else:
            chosen_length = length
            rejected_length = length

        chosen_ptr += offset * np.dtype(self._chosen_index.dtype).itemsize
        rejected_ptr += offset * np.dtype(self._rejected_index.dtype).itemsize

        chosen_input = np.frombuffer(self._chosen_bin_buffer, dtype=self._chosen_index.dtype,
                                    count=chosen_length, offset=chosen_ptr)
        chosen_label = np.frombuffer(self._chosen_label_bin_buffer, dtype=self._chosen_index.dtype,
                                    count=chosen_length, offset=chosen_ptr)

        rejected_input = np.frombuffer(self._rejected_bin_buffer, dtype=self._rejected_index.dtype,
                                    count=rejected_length, offset=rejected_ptr)
        rejected_label = np.frombuffer(self._rejected_label_bin_buffer, dtype=self._rejected_index.dtype,
                                    count=rejected_length, offset=rejected_ptr)

        return {
            'chosen': (chosen_input, chosen_label),
            'rejected': (rejected_input, rejected_label)
        }

    @property
    def sizes(self):
        return self._chosen_index.sizes

    def size(self, index):
        return self._chosen_index.sizes[index]


    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(chosen_idx_file_path(path)) and 
            os.path.exists(chosen_data_file_path(path)) and
            os.path.exists(chosen_label_file_path(path))
        )


class MMapIndexedDatasetBuilder_DPO(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file + "_input.bin", 'wb')  
        self._lable_file = open(out_file +"_label.bin",'wb')
        self._dtype = dtype  # best_fitting_dtype(vocab_size)
        self._sizes = [] # data delta address
        self._doc_idx = [0] # data base address

    def add_item(self, tensor, tensor_label):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        label_np_array = np.array(tensor_label.numpy(), dtype=self._dtype)
        self._lable_file.write(label_np_array.tobytes(order='C'))
        self._sizes.append(np_array.size) 

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        # Concatenate index
        index = MMapIndexedDataset_DPO.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])

        # Concatenate data
        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()
        self._lable_file.close()

        with MMapIndexedDataset_DPO.Index.writer(index_file_path(index_file), self._dtype) as index:
                index.write(self._sizes, self._doc_idx)

def make_dpo_builder(out_file,impl,vocab_size=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder_DPO(
            out_file,
            dtype=best_fitting_dtype(vocab_size)
            )
    else:
        return IndexedDatasetBuilder(out_file)