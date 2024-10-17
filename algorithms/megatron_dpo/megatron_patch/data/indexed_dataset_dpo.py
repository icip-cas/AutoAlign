import os
import shutil
import stat
import struct
from functools import lru_cache
from itertools import accumulate

import numpy as np
import torch


from indexed_dataset import (
    best_fitting_dtype,
    _warmup_mmap_file,
    code,
    data_file_path,
    dtypes,
    index_file_path,
    MMapIndexedDataset,
    IndexedDatasetBuilder,
)

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def data_label_file_path(prefix_path):
    return prefix_path + "_label.bin"

class MMapIndexedDataset_DPO(torch.utils.data.Dataset):
    class Index(object):# 初始化类实例即可读取idx文件中的sizes pointer doc_idx三个数组的信息，以此建立数据集的索引
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

                self._len = struct.unpack('<Q', stream.read(8))[0]
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
            self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                           offset=offset + self._sizes.nbytes)
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
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

    def __init__(self, path, skip_warmup=False):
        super().__init__()

        self._path = None
        self._index = None
        self._bin_buffer = None

        self._do_init(path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, path, skip_warmup): # create index,data buffer,lable buffer
        self._path = path
        self._index = self.Index(index_file_path(self._path), skip_warmup)  # create index !!!

        if not skip_warmup:
            print_rank_0("    warming up data mmap file...")
            _warmup_mmap_file(data_file_path(self._path))
        
        print_rank_0("    creating numpy buffer of mmap...")
        self._bin_buffer_mmap = np.memmap(data_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy buffer...")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

        #newly added
        print_rank_0("    creating numpy label buffer of mmap...")
        self._label_bin_buffer_mmap = np.memmap(data_label_file_path(self._path), mode='r', order='C')
        print_rank_0("    creating memory view of numpy label buffer...")
        self._label_bin_buffer = memoryview(self._label_bin_buffer_mmap)

    def __del__(self):
        self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap
        self._label_bin_buffer_mmap._mmap.close()
        del self._index

    def __len__(self):
        return len(self._index)

    # @lru_cache(maxsize=8)
    def __getitem__(self, idx): # return data and label
        if isinstance(idx, int):
            ptr, size = self._index[idx]
            input_np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr) # self._index.dtype是整个数据集初始化时确定的dtype,data和lable在add item时使用的即是这个dtype,即best_fitting_dtype(vocab_size)
            
            #newly added
            label_np_array = np.frombuffer(self._label_bin_buffer, dtype=self._index.dtype,
                                     count=size, offset=ptr)
            return input_np_array, label_np_array
        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")
            ptr = self._index._pointers[start]
            sizes = self._index._sizes[idx]
            offsets = list(accumulate(sizes))
            total_size = sum(sizes)
            np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            sents = np.split(np_array, offsets[:-1])

            #get label_sents
            label_np_array = np.frombuffer(self._label_bin_buffer, dtype=self._index.dtype,
                                     count=total_size, offset=ptr)
            label_sents = np.split(label_np_array, offsets[:-1])
            return sents,label_sents

    def get(self, idx, offset=0, length=None):
        """ Retrieves a single item from the dataset with the option to only
        return a portion of the item.

        get(idx) is the same as [idx] but get() does not support slicing.
        """
        ptr, size = self._index[idx]
        if length is None:
            length = size - offset
        ptr += offset * np.dtype(self._index.dtype).itemsize
        input_np_array = np.frombuffer(self._bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        
        #get label_np_array
        label_np_array = np.frombuffer(self._label_bin_buffer, dtype=self._index.dtype,
                                 count=length, offset=ptr)
        return input_np_array,label_np_array

    @property
    def sizes(self):
        return self._index.sizes

    def size(self, index):
        return self._index.sizes[index]

    @property
    def doc_idx(self):
        return self._index.doc_idx

    def get_doc_idx(self):
        return self._index._doc_idx

    def set_doc_idx(self, doc_idx_):
        self._index._doc_idx = doc_idx_

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
                os.path.exists(index_file_path(path)) and os.path.exists(data_file_path(path))
        )


class MMapIndexedDatasetBuilder_DPO(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file, 'wb')  
        self._lable_file = open(out_file[:-4]+"_label.bin",'wb')
        self._dtype = dtype  # best_fitting_dtype(vocab_size)
        self._sizes = [] # data delta address
        self._doc_idx = [0] # data base address

    def add_item(self, tensor,tensor_label):
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

        with MMapIndexedDataset_DPO.Index.writer(index_file, self._dtype) as index:
                index.write(self._sizes, self._doc_idx)

def make_dpo_builder(out_file,impl,vocab_size=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilder_DPO(
            out_file,dtype=best_fitting_dtype(vocab_size)
            )
    else:
        return IndexedDatasetBuilder(out_file)