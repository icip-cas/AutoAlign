import os
import random
import subprocess
import sys
import time
import signal
from enum import Enum, unique
import argparse
from datetime import datetime

from .utils import get_logger, get_device_count

logger = get_logger(__name__)


@unique
class Command(str, Enum):
    SFT = "sft"
    DPO = "dpo"
    REWARD = "rm"
    DATA = "data"
    EVAL = "eval"
    INFER = "infer"
    SERVE = "serve"
    MERGE = "merge"
    RL = "rl"  # Added RL command


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
    if backend == "hf":
        command = f"accelerate launch {file} --backend 'hf' {' '.join(remaining_args)}"
    elif backend == "vllm":
        command = f"python {file} --backend 'vllm' {' '.join(remaining_args)}"

    process = subprocess.run(command, shell=True)
    sys.exit(process.returncode)


def run_serve(cli_file, webui_file, args, remaining_args):
    assert args.mode is not None, "Please specify the mode for serve"
    mode = args.mode
    if mode == "cli":
        command = f"python {cli_file} {' '.join(remaining_args)}"
    elif mode == "browser":
        command = f"python {webui_file} {' '.join(remaining_args)}"

    process = subprocess.run(command, shell=True)
    sys.exit(process.returncode)


def run_rl(file, args, remaining_args):
    """
    Run RL training with different algorithms.
    Currently supports:
    - grpo: GRPO training with VLLM server backend
    """
    assert args.algorithm is not None, "Please specify the RL algorithm (e.g., --algorithm grpo)"
    
    algorithm = args.algorithm.lower()
    
    if algorithm == "grpo":
        run_grpo_training(file, remaining_args)
    else:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")


def run_grpo_training(file, remaining_args):
    """
    Run GRPO training with VLLM server backend.
    This function:
    1. Starts a VLLM server in the background
    2. Runs the RL training with torchrun
    3. Handles cleanup on exit
    """
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vllm_log_file = f"{timestamp}_vllm_serve.log"
    
    # Parse remaining args to extract model path for VLLM server
    model_path = None
    for i, arg in enumerate(remaining_args):
        if arg == "--model_name_or_path" and i + 1 < len(remaining_args):
            model_path = remaining_args[i + 1]
            break

    if model_path is None:
        logger.error("Model path not specified. Please provide --model_name_or_path argument.")
        sys.exit(1)
    
    logger.info(f"Starting GRPO training with VLLM server")
    logger.info(f"Model: {model_path}")
    logger.info(f"VLLM server logs will be written to: {vllm_log_file}")
    
    # Start VLLM server in background
    vllm_command = f"CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model {model_path}"
    
    try:
        with open(vllm_log_file, 'w') as log_file:
            vllm_process = subprocess.Popen(
                vllm_command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group for easier cleanup
            )
        
        logger.info(f"VLLM server started with PID: {vllm_process.pid}")
        
        # Wait a bit for VLLM server to start up
        logger.info("Waiting for VLLM server to initialize...")
        time.sleep(30)
        
        # Check if VLLM server is still running
        if vllm_process.poll() is not None:
            logger.error("VLLM server failed to start. Check the log file.")
            sys.exit(1)
        
        # Get available GPU count for RL training (excluding GPU 0 used by VLLM)
        total_gpus = get_device_count()
        rl_gpus = max(1, total_gpus - 1)  # Use all GPUs except GPU 0
        gpu_list = ",".join(str(i) for i in range(1, total_gpus))
        
        logger.info(f"Starting GRPO training on GPUs: {gpu_list}")
        
        # Build torchrun command for RL training
        master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
        master_port = os.environ.get("MASTER_PORT", str(random.randint(20001, 29999)))
        
        if total_gpus > 1:
            rl_command = (
                f"CUDA_VISIBLE_DEVICES={gpu_list} torchrun "
                f"--nnodes 1 --nproc_per_node {rl_gpus} "
                f"--master_addr {master_addr} --master_port {master_port} "
                f"{file} --use-vllm True {' '.join(remaining_args)}"
            )
        else:
            rl_command = (
                f"torchrun "
                f"--nnodes 1 --nproc_per_node {rl_gpus} "
                f"--master_addr {master_addr} --master_port {master_port} "
                f"{file} --use-vllm True {' '.join(remaining_args)}"
            )
        
        logger.info(f"Running rl training command: {rl_command}")
        
        # Run RL training
        rl_process = subprocess.run(rl_command, shell=True)
        
        return_code = rl_process.returncode
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, cleaning up...")
        return_code = 130  # Standard exit code for Ctrl+C
    
    finally:
        # Cleanup: terminate VLLM server
        try:
            if 'vllm_process' in locals() and vllm_process.poll() is None:
                logger.info("Terminating VLLM server...")
                # Kill the entire process group
                os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
                
                # Wait a bit for graceful shutdown
                time.sleep(5)
                
                # Force kill if still running
                if vllm_process.poll() is None:
                    logger.warning("Force killing VLLM server...")
                    os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
                
                logger.info("VLLM server terminated")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    sys.exit(return_code)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("command", type=str, choices=[c.value for c in Command])
    parser.add_argument("--backend", type=str, choices=["hf", "vllm"])
    parser.add_argument("--mode", type=str, choices=["cli", "browser"])
    parser.add_argument("--algorithm", type=str, choices=["grpo"], 
                       help="RL algorithm to use (currently supports: grpo)")
    args, remaining_args = parser.parse_known_args()

    if args.command == Command.SFT:
        from .train import sft

        run_distributed_task(sft.__file__, remaining_args)
    elif args.command == Command.DPO:
        from .train import dpo

        run_distributed_task(dpo.__file__, remaining_args)
    elif args.command == Command.REWARD:
        from .train import rm

        run_distributed_task(rm.__file__, remaining_args)
    elif args.command == Command.RL:
        from .train import rl

        run_rl(rl.__file__, args, remaining_args)
    elif args.command == Command.EVAL:
        from .eval.run_eval import run_eval

        run_eval(remaining_args)
    elif args.command == Command.INFER:
        from .inference import inference

        run_inference(inference.__file__, args, remaining_args)
    elif args.command == Command.SERVE:
        from .serve import serve_cli, serve_webui

        run_serve(serve_cli.__file__, serve_webui.__file__, args, remaining_args)
    elif args.command == Command.MERGE:
        from .model_merging.merge_models import run_merge

        run_merge(remaining_args)
    else:
        raise ValueError(f"Unknown command: {args.command}")