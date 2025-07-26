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
        run_grpo_training(file, args, remaining_args)
    else:
        raise ValueError(f"Unsupported RL algorithm: {algorithm}")


def run_grpo_training(file, args, remaining_args):
    """
    Run GRPO training with VLLM server backend.
    This function:
    1. Starts a VLLM server in the background
    2. Optionally starts an xVerify VLLM server if --use_xverify is set
    3. Runs the RL training with torchrun
    4. Handles cleanup on exit
    """
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    vllm_log_file = f"{timestamp}_vllm_serve.log"
    xverify_log_file = f"{timestamp}_vllm_xv_serve.log"
    
    # Parse remaining args to extract model paths
    model_path = None
    use_xverify = False
    xverify_model_path = None
    
    # Create a new list without the args we want to remove
    new_remaining_args = []
    i = 0
    while i < len(remaining_args):
        arg = remaining_args[i]
        if arg == "--model_name_or_path" and i + 1 < len(remaining_args):
            model_path = remaining_args[i + 1]
            new_remaining_args.append(arg)
            new_remaining_args.append(remaining_args[i + 1])
            i += 2
        elif arg == "--xverify_model_path" and i + 1 < len(remaining_args):
            xverify_model_path = remaining_args[i + 1]
            i += 2  # Skip both the arg and its value
        elif arg == "--use_xverify":
            use_xverify = True
            i += 2  # Skip this arg
        else:
            new_remaining_args.append(arg)
            i += 1
    
    remaining_args = new_remaining_args

    if model_path is None:
        logger.error("Model path not specified. Please provide --model_name_or_path argument.")
        sys.exit(1)
    
    # Default xVerify model path if not specified
    if use_xverify and xverify_model_path is None:
        xverify_model_path = os.environ.get("XVERIFY_MODEL_PATH", "IAAR-Shanghai/xVerify-0.5B-I")
        logger.info(f"Using default xVerify model path: {xverify_model_path}")
    
    logger.info(f"Starting GRPO training with VLLM server")
    logger.info(f"Model: {model_path}")
    logger.info(f"VLLM server logs will be written to: {vllm_log_file}")
    if use_xverify:
        logger.info(f"xVerify model: {xverify_model_path}")
        logger.info(f"xVerify server logs will be written to: {xverify_log_file}")
    
    # Kill any existing VLLM processes
    logger.info("Cleaning up any existing VLLM processes...")
    subprocess.run("pkill -f vllm.entrypoints.openai.api_server", shell=True)
    subprocess.run("pkill -f vllm-serve", shell=True)
    time.sleep(2)  # Wait a bit for processes to die

    if use_xverify:
        xverify_random_port = random.randint(8000, 8999)
        logger.info(f"Starting xVerify server on port {xverify_random_port}")
        xverify_api_base_url = f"http://localhost:{xverify_random_port}/v1"
        os.environ["API_BASE_URL"] = xverify_api_base_url
    
    vllm_process = None
    xverify_process = None
    
    try:
        # Start main VLLM server in background
        vllm_command = f"CUDA_VISIBLE_DEVICES=0 trl vllm-serve --model {model_path}"
        
        with open(vllm_log_file, 'w') as log_file:
            vllm_process = subprocess.Popen(
                vllm_command,
                shell=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                preexec_fn=os.setsid  # Create new process group for easier cleanup
            )
        
        logger.info(f"VLLM server started with PID: {vllm_process.pid}")
        
        # Start xVerify server if requested
        if use_xverify:
            xverify_command = (
                f"CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server "
                f"--model {xverify_model_path} "
                f"--gpu_memory_utilization 0.45 "
                f"--cpu-offload-gb 10 "
                f"--host 0.0.0.0 "
                f"--port {xverify_random_port} "
                f"--served-model-name 'default'"
            )
            
            with open(xverify_log_file, 'w') as log_file:
                xverify_process = subprocess.Popen(
                    xverify_command,
                    shell=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    preexec_fn=os.setsid
                )
            
            logger.info(f"xVerify server started with PID: {xverify_process.pid}")
        
        # Wait for servers to start up
        wait_time = 5 if use_xverify else 10
        logger.info(f"Waiting {wait_time} seconds for server(s) to initialize...")
        time.sleep(wait_time)
        
        # Check if VLLM server is still running
        if vllm_process.poll() is not None:
            logger.error("VLLM server failed to start. Check the log file.")
            sys.exit(1)
        
        # Check if xVerify server is still running (if started)
        if use_xverify and xverify_process.poll() is not None:
            logger.error("xVerify server failed to start. Check the log file.")
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
        # Cleanup: terminate servers
        try:
            # Terminate VLLM server
            if vllm_process is not None and vllm_process.poll() is None:
                logger.info("Terminating VLLM server...")
                os.killpg(os.getpgid(vllm_process.pid), signal.SIGTERM)
                time.sleep(5)
                if vllm_process.poll() is None:
                    logger.warning("Force killing VLLM server...")
                    os.killpg(os.getpgid(vllm_process.pid), signal.SIGKILL)
                logger.info("VLLM server terminated")
            
            # Terminate xVerify server
            if xverify_process is not None and xverify_process.poll() is None:
                logger.info("Terminating xVerify server...")
                os.killpg(os.getpgid(xverify_process.pid), signal.SIGTERM)
                time.sleep(5)
                if xverify_process.poll() is None:
                    logger.warning("Force killing xVerify server...")
                    os.killpg(os.getpgid(xverify_process.pid), signal.SIGKILL)
                logger.info("xVerify server terminated")
                
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