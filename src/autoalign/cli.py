import os
import random
import subprocess
import sys
from enum import Enum, unique

from .train import sft, dpo
from .inference import inference
from .utils import get_logger, get_device_count

logger = get_logger(__name__)

def _help():

    print("="*10)
    print("autoalign-cli")
    print("="*10)
    exit()

@unique
class Command(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    DATA = "data"
    EVAL = "eval"
    INFER = "infer"

def main():

    command = sys.argv.pop(1) if len(sys.argv) != 1 else _help()
    if command == Command.SFT:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
        logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
        process = subprocess.run(
            (
                "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
            ).format(
                nnodes=os.environ.get("NNODES", "1"),
                node_rank=os.environ.get("RANK", "0"),
                nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
                master_addr=master_addr,
                master_port=master_port,
                file_name=sft.__file__,
                args=" ".join(sys.argv[1:]),
            ),
            shell=True,
        )
        sys.exit(process.returncode)
    elif command == Command.DPO:
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
        logger.info("Initializing distributed tasks at: {}:{}".format(master_addr, master_port))
        process = subprocess.run(
            (
                "torchrun --nnodes {nnodes} --node_rank {node_rank} --nproc_per_node {nproc_per_node} "
                "--master_addr {master_addr} --master_port {master_port} {file_name} {args}"
            ).format(
                nnodes=os.environ.get("NNODES", "1"),
                node_rank=os.environ.get("RANK", "0"),
                nproc_per_node=os.environ.get("NPROC_PER_NODE", str(get_device_count())),
                master_addr=master_addr,
                master_port=master_port,
                file_name=dpo.__file__,
                args=" ".join(sys.argv[1:]),
            ),
            shell=True,
        )
        sys.exit(process.returncode)
    elif command == Command.EVAL:
        NotImplementedError()

    elif command == Command.INFER:
        process = subprocess.run(
            (
                "accelerate launch {file_name} {args}"
            ).format(
                file_name=inference.__file__,
                args=" ".join(sys.argv[1:]),
            ),
            shell=True
        )
    else:
        raise NotImplementedError("Unknown command: {}".format(command))
