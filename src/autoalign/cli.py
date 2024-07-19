import os
import random
import subprocess
import sys
from enum import Enum, unique
import argparse

from .utils import get_logger, get_device_count

logger = get_logger(__name__)

@unique
class Command(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    DATA = "data"
    EVAL = "eval"
    INFER = "infer"

def run_distributed_task(file, args):
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
    logger.info(f"Initializing distributed tasks at: {master_addr}:{master_port}")
    
    command = (
        f"torchrun --nnodes {os.environ.get('NNODES', '1')} "
        f"--node_rank {os.environ.get('RANK', '0')} "
        f"--nproc_per_node {os.environ.get('NPROC_PER_NODE', str(get_device_count()))} "
        f"--master_addr {master_addr} --master_port {master_port} "
        f"{file} {' '.join(args)}"
    )
    process = subprocess.run(command, shell=True)
    sys.exit(process.returncode)

def run_inference(file, args, remaining_args):
    assert args.backend is not None, "Please specify the backend for inference"
    backend = args.backend
    if backend == 'hf':
        command = f"accelerate launch {file} --backend 'hf' {' '.join(remaining_args)}"
    elif backend == 'vllm':
        command = f"python {file} --backend 'vllm' {' '.join(remaining_args)}"
    
    process = subprocess.run(command, shell=True)
    sys.exit(process.returncode)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('command', type=str, choices=[c.value for c in Command])
    parser.add_argument('--backend', type=str, choices=["hf", "vllm"])
    args, remaining_args = parser.parse_known_args()

    if args.command == Command.SFT:
        from .train import sft
        run_distributed_task(sft.__file__, remaining_args)
    elif args.command == Command.DPO:
        from .train import dpo
        run_distributed_task(dpo.__file__, remaining_args)
    elif args.command == Command.EVAL:
        raise NotImplementedError()
    elif args.command == Command.INFER:
        from .inference import inference
        run_inference(inference.__file__, args, remaining_args)
    else:
        raise ValueError(f"Unknown command: {args.command}")
