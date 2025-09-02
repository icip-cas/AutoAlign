"""utils"""
import json
import torch
from transformers.utils import (
    is_torch_cuda_available,
    is_torch_npu_available,
)
import logging
import sys
import os
import subprocess


def read_json(data_path):
    """read json data"""
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def save_json(data, save_path):
    """save json data"""
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def get_device_count() -> int:
    if is_torch_npu_available():
        return torch.npu.device_count()
    elif is_torch_cuda_available():
        return torch.cuda.device_count()
    else:
        return 0


def get_logger(name: str) -> logging.Logger:
    formatter = logging.Formatter(
        fmt="[%(asctime)s,%(msecs)03d] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    return logger


def remove_file_if_user_confirms(file_path):
    """
    Ask the user if they want to remove an existing file.
    Remove the file if the user confirms.

    :param file_path: Path to the file to potentially remove
    :return: True if file was removed, False otherwise
    """
    if os.path.exists(file_path):
        while True:
            user_choice = input(
                f"The file '{file_path}' already exists. Do you want to remove it? (y/N): "
            ).lower()
            if user_choice == "y":
                os.remove(file_path)
                return True
            elif user_choice == "n":
                return False
            else:
                pass
    else:
        print(f"File '{file_path}' does not exist.")
        return False


def get_available_gpu_list():
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"]
        )
        output = result.decode("utf-8")
        gpu_list = output.strip().split("\n")
        return gpu_list
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while getting GPU information: {e}")
        return []
