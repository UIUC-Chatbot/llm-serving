#!/bin/bash
#SBATCH --job-name=ray-worker           # Job name
#SBATCH --partition=a100                # Partition name
#SBATCH --gres=gpu:2                    # Request GPU resource
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=16               # Number of CPU cores per task
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --output=ray-worker-%j.log      # Standard output and error log

ray start --address='10.10.109.244:6379' # Command to run
while true
do
    sleep 300
done