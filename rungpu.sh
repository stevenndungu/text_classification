#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128GB
#SBATCH --time=10:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100
#SBATCH --output=gpu_v2.out
#SBATCH --job-name=gpu_v3

module --force purge
#module load Python/3.10.8-GCCcore-12.2.0

#python3 -m venv /home4/$USER/.envs/deep_hashing
source /home4/$USER/.envs/deep_hashing/bin/activate

python text_classification.py
#squeue -u $USER

#srun --gpus-per-node=1 --time=01:00:00 --pty rungpu.sh