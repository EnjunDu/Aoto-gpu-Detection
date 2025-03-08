





# GPU Memory Monitor & Task Execution Tool

## Overview

This tool monitors available GPU memory and executes a specified command when GPU memory availability meets predefined thresholds. It ensures efficient GPU utilization by automatically detecting available resources before launching computational tasks.

## Features

- Periodic GPU memory monitoring using `nvidia-smi`
- Configurable single and combined GPU memory thresholds
- Automatic command execution when GPU availability meets requirements
- Logging to both file and console for easy debugging

## Usage

Run the script with the following command:

```
python run.py --interval 2 --command "python main.py" --gpu-flag=--gpu --single-threshold 79 --combined-threshold 125
```

### Arguments:

- `--interval x` : Monitor every `x` seconds
- `--gpu-flag=--xxx` : Specify how to set the GPU in your code (e.g., `--cuda` or `--gpu`)
- `--single-threshold xx` : Execute task when a single GPU exceeds `xx` GB of free memory
- `--combined-threshold xxx` : Execute task when combined GPUs exceed `xxx` GB of free memory
- `--command` : Command to execute when conditions are met

## Logging

Logs are stored in the `log/gpu_monitor.log` file and printed to the console.

## Notes

- This script is designed for systems with NVIDIA GPUs and requires `nvidia-smi`.
- Ensure your environment has necessary GPU drivers installed.

## License

This project is open-source under the MIT License.
