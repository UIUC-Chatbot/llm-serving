#!/bin/bash
#SBATCH --job-name=RayNode              # Job name
#SBATCH --partition=a100                # Partition name
#SBATCH --gres=gpu:2                    # Request GPU resource
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=16              # Number of CPU cores per task
#SBATCH --time=01:00:00                 # Time limit hrs:min:sec
#SBATCH --output=ray-worker-%j.log      # Standard output and error log

echo "$SLURMD_NODENAME"
ray start --address='10.10.109.243:6379' # Command to run

if [ $? -eq 0 ]; then
    echo "Ray worker started successfully"
    while true
    do
        sleep 300
    done

else
    echo "Ray worker failed to start"
    exit 1
fi