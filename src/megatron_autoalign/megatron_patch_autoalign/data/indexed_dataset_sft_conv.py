import os
import shutil
import stat
import struct
from functools import lru_cache
from itertools import accumulate
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

def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def data_label_file_path(prefix_path):
    return prefix_path + "_label.bin"

def conv_data_file_path(prefix_path):
    return prefix_path + "_conv_input.bin"

def conv_label_file_path(prefix_path):
    return prefix_path + "_conv_label.bin"

def conv_idx_file_path(prefix_path):
    return prefix_path + "_conv.idx"

class MMapIndexedDatasetSFTConv(torch.utils.data.Dataset):
    class Index(object):
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
            self._pointers = np.frombuffer(
                self._bin_buffer, 
                dtype=np.int64,
                count=self._len,
                offset=offset + self._sizes.nbytes)
            print_rank_0("    reading document index...")
            self._doc_idx = np.frombuffer(
                self._bin_buffer, 
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
        self._conv_index = None
        self._conv_bin_buffer = None
        self._do_init(prefix_path, skip_warmup)

    def __getstate__(self):
        return self._path

    def __setstate__(self, state):
        self._do_init(state)

    def _do_init(self, prefix_path, skip_warmup):
        self._prefix_path = prefix_path
        self._conv_index = self.Index(conv_idx_file_path(self._prefix_path), skip_warmup)

        if not skip_warmup:
            print_rank_0("    warming up data mmap files...")
            _warmup_mmap_file(conv_data_file_path(self._prefix_path))
            print_rank_0("    warming up label mmap files...")
            _warmup_mmap_file(conv_label_file_path(self._prefix_path))

        print_rank_0("    creating numpy buffers of mmap for input data...")
        self._conv_bin_buffer_mmap = np.memmap(conv_data_file_path(self._prefix_path), mode='r', order='C')
        print_rank_0("    creating memory views of numpy buffers for input data...")
        self._conv_bin_buffer = memoryview(self._conv_bin_buffer_mmap)

        print_rank_0("    creating numpy buffers of mmap for label data...")
        self._conv_label_bin_buffer_mmap = np.memmap(conv_label_file_path(self._prefix_path), mode='r', order='C')
        print_rank_0("    creating memory views of numpy buffers for label data...")
        self._conv_label_bin_buffer = memoryview(self._conv_label_bin_buffer_mmap)

    def __del__(self):
        if hasattr(self, '_conv_bin_buffer_mmap'):
            self._conv_bin_buffer_mmap._mmap.close()
            del self._conv_bin_buffer_mmap
        if hasattr(self, '_conv_label_bin_buffer_mmap'):
            self._conv_label_bin_buffer_mmap._mmap.close()
            del self._conv_label_bin_buffer_mmap
        if hasattr(self, '_conv_index'):
            del self._conv_index

    def __len__(self):
        return len(self._conv_index)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            conv_ptr, conv_size = self._conv_index[idx]
            conv_input = np.frombuffer(
                self._conv_bin_buffer, 
                dtype=self._conv_index.dtype,
                count=conv_size, 
                offset=conv_ptr)
            conv_label = np.frombuffer(
                self._conv_label_bin_buffer, 
                dtype=self._conv_index.dtype,
                count=conv_size, 
                offset=conv_ptr)
            return {
                'conv': (conv_input, conv_label),
            }

        elif isinstance(idx, slice):
            start, stop, step = idx.indices(len(self))
            if step != 1:
                raise ValueError("Slices into indexed_dataset must be contiguous")

            conv_ptr = self._conv_index._pointers[start]
            conv_sizes = self._conv_index._sizes[idx]
            conv_offsets = list(accumulate(conv_sizes))
            conv_total_size = sum(conv_sizes)

            conv_input = np.frombuffer(
                self._conv_bin_buffer, dtype=self._conv_index.dtype,
                count=conv_total_size, offset=conv_ptr)
            conv_input_sents = np.split(conv_input, conv_offsets[:-1])

            conv_label = np.frombuffer(
                self._conv_label_bin_buffer, dtype=self._conv_index.dtype,
                count=conv_total_size, offset=conv_ptr)
            conv_label_sents = np.split(conv_label, conv_offsets[:-1])

            return {
                'conv': (conv_input_sents, conv_label_sents),
            }

    def get(self, idx, offset=0, length=None):
        conv_ptr, conv_size = self._conv_index[idx]
        if length is None:
            conv_length = conv_size - offset
        else:
            conv_length = length

        conv_ptr += offset * np.dtype(self._conv_index.dtype).itemsize

        conv_input = np.frombuffer(
            self._conv_bin_buffer, dtype=self._conv_index.dtype,
            count=conv_length, offset=conv_ptr)
        conv_label = np.frombuffer(
            self._conv_label_bin_buffer, dtype=self._conv_index.dtype,
            count=conv_length, offset=conv_ptr)

        return {
                'conv': (conv_input, conv_label),
            }

    @property
    def sizes(self):
        return self._conv_index.sizes

    def size(self, index):
        return self._conv_index.sizes[index]

    @property
    def supports_prefetch(self):
        return False

    @staticmethod
    def exists(path):
        return (
            os.path.exists(conv_idx_file_path(path)) and 
            os.path.exists(conv_data_file_path(path)) and
            os.path.exists(conv_label_file_path(path))
        )

class MMapIndexedDatasetBuilderSFTConv(object):
    def __init__(self, out_file, dtype=np.int64):
        self._data_file = open(out_file + "_input.bin", 'wb')  
        self._label_file = open(out_file + "_label.bin", 'wb')
        self._dtype = dtype
        self._sizes = []
        self._doc_idx = [0]

    def add_item(self, tensor, tensor_label):
        np_array = np.array(tensor.numpy(), dtype=self._dtype)
        self._data_file.write(np_array.tobytes(order='C'))
        label_np_array = np.array(tensor_label.numpy(), dtype=self._dtype)
        self._label_file.write(label_np_array.tobytes(order='C'))
        self._sizes.append(np_array.size)

    def end_document(self):
        self._doc_idx.append(len(self._sizes))

    def merge_file_(self, another_file):
        index = MMapIndexedDatasetSFTConv.Index(index_file_path(another_file))
        assert index.dtype == self._dtype

        offset = len(self._sizes)
        self._sizes.extend(index.sizes)
        self._doc_idx.extend((offset + index.doc_idx)[1:])

        with open(data_file_path(another_file), 'rb') as f:
            shutil.copyfileobj(f, self._data_file)

    def finalize(self, index_file):
        self._data_file.close()
        self._label_file.close()

        with MMapIndexedDatasetSFTConv.Index.writer(index_file_path(index_file), self._dtype) as index:
            index.write(self._sizes, self._doc_idx)

def make_sft_conv_builder(out_file, impl, vocab_size=None):
    if impl == 'mmap':
        return MMapIndexedDatasetBuilderSFTConv(
            out_file,
            dtype=best_fitting_dtype(vocab_size)
        )
    else:
        return IndexedDatasetBuilder(out_file)