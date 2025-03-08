'''
python run.py --interval 2 --command "python main.py" --gpu-flag=--gpu --single-threshold 79 --combined-threshold 125
'''
'''
--interval x: Monitor every x seconds
--gpu-flag=--xxx: How do you set the gpu in your code group? Usually it is --cuda or --gpu
--single-threshold xx: Start the task when a single graphics card exceeds xx
--combined-threshold xxx: Start the task when combined graphics cards exceeds xxx
--command how to run your project
the top is my example to run llm4gds
'''
import subprocess
import time
import re
import os
import argparse
import logging
from typing import Dict, List, Tuple

# Set up logging
os.makedirs("log", exist_ok=True)  # Create log directory if it doesn't exist
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("log/gpu_monitor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Monitor GPU available memory and execute tasks on available GPUs.')
    parser.add_argument('--interval', type=int, default=5, help='Monitoring interval in seconds (default: 5)')
    parser.add_argument('--single-threshold', type=int, default=70,
                        help='Available memory threshold for a single GPU in GB (default: 70)')
    parser.add_argument('--combined-threshold', type=int, default=80,
                        help='Combined available memory threshold for two GPUs in GB (default: 80)')
    parser.add_argument('--command', type=str, default='python main.py',
                        help='Command to execute (default: "python main.py")')
    # Fix: Use nargs='?' to make --gpu-flag accept a value that might start with --
    parser.add_argument('--gpu-flag', type=str, default='--gpu', nargs='?',
                        help='GPU flag for the command (default: --gpu)')
    return parser.parse_args()


def get_gpu_memory_info() -> Dict[int, Dict[str, float]]:
    """
    Get both total and free memory for each GPU using nvidia-smi.

    Returns:
        Dict[int, Dict[str, float]]: Dictionary mapping GPU IDs to memory info (total, used, free) in GB
    """
    try:
        # Run nvidia-smi command to get memory information
        result = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=index,memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'],
            universal_newlines=True)

        # Parse the output to extract GPU ID and memory info
        gpu_memory = {}
        for line in result.strip().split('\n'):
            if line:
                parts = [part.strip() for part in line.split(',')]
                gpu_id = int(parts[0])
                total_memory = float(parts[1]) / 1024  # Convert MB to GB
                used_memory = float(parts[2]) / 1024  # Convert MB to GB
                free_memory = float(parts[3]) / 1024  # Convert MB to GB

                gpu_memory[gpu_id] = {
                    'total': total_memory,
                    'used': used_memory,
                    'free': free_memory
                }

        return gpu_memory

    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run nvidia-smi: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error getting GPU memory info: {e}")
        return {}


def find_available_gpu(memory_info: Dict[int, Dict[str, float]], single_threshold: int, combined_threshold: int) -> int:
    """
    Find an available GPU based on free memory thresholds.

    Args:
        memory_info (Dict[int, Dict[str, float]]): Dictionary of GPU IDs and their memory info
        single_threshold (int): Available memory threshold for a single GPU in GB
        combined_threshold (int): Combined available memory threshold for two GPUs in GB

    Returns:
        int: The ID of an available GPU, or -1 if none available
    """
    # Check if any GPU exceeds the single threshold for free memory
    for gpu_id, memory in memory_info.items():
        free_memory = memory['free']
        if free_memory > single_threshold:
            logger.info(
                f"GPU {gpu_id} exceeds single threshold for free memory: {free_memory:.2f}GB > {single_threshold}GB")
            return gpu_id

    # Check if any pair of GPUs exceeds the combined threshold for free memory
    gpu_ids = list(memory_info.keys())
    for i in range(len(gpu_ids)):
        for j in range(i + 1, len(gpu_ids)):
            gpu_id1, gpu_id2 = gpu_ids[i], gpu_ids[j]
            combined_free_memory = memory_info[gpu_id1]['free'] + memory_info[gpu_id2]['free']
            if combined_free_memory > combined_threshold:
                logger.info(
                    f"GPUs {gpu_id1} and {gpu_id2} exceed combined threshold for free memory: {combined_free_memory:.2f}GB > {combined_threshold}GB")
                # Find the GPU with the most free memory
                if memory_info[gpu_id1]['free'] > memory_info[gpu_id2]['free']:
                    return gpu_id1
                else:
                    return gpu_id2

    return -1  # No GPU available based on thresholds


def run_command(command: str, gpu_flag: str, gpu_id: int) -> None:
    """
    Run the specified command with the GPU flag.

    Args:
        command (str): The command to run
        gpu_flag (str): The GPU flag to use
        gpu_id (int): The GPU ID to use
    """
    # Remove quotes from gpu_flag if present
    gpu_flag = gpu_flag.strip('"\'')

    full_command = f"{command} {gpu_flag} {gpu_id}"
    logger.info(f"Running command: {full_command}")

    try:
        # Run the command
        subprocess.Popen(full_command, shell=True)
        logger.info("Command started successfully")
    except Exception as e:
        logger.error(f"Failed to run command: {e}")


def main():
    """Main function to monitor GPU memory and execute tasks."""
    args = parse_arguments()

    logger.info(f"Starting GPU available memory monitor with interval {args.interval} seconds")
    logger.info(f"Single GPU available memory threshold: {args.single_threshold}GB")
    logger.info(f"Combined GPU available memory threshold: {args.combined_threshold}GB")
    logger.info(f"Command to run: {args.command} {args.gpu_flag} <gpu_id>")

    command_running = False

    try:
        while not command_running:
            # Get current GPU memory information
            memory_info = get_gpu_memory_info()

            if not memory_info:
                logger.warning("No GPU information available. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            # Log current memory information
            for gpu_id, memory in memory_info.items():
                logger.info(
                    f"GPU {gpu_id}: {memory['total']:.2f}GB total, {memory['used']:.2f}GB used, {memory['free']:.2f}GB free")

            # Check if we need to run the command
            available_gpu = find_available_gpu(memory_info, args.single_threshold, args.combined_threshold)

            if available_gpu >= 0:
                logger.info(f"Condition met, using GPU {available_gpu}")
                run_command(args.command, args.gpu_flag, available_gpu)
                command_running = True
            else:
                logger.info(f"Thresholds not met, checking again in {args.interval} seconds")
                time.sleep(args.interval)

    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
