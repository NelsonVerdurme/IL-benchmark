#!/bin/bash
#SBATCH --job-name=simple_policy_peract
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --hint=nomultithread
#SBATCH --time=48:00:00
#SBATCH --mem=40G
#SBATCH --output=slurm_logs/%j.out
#SBATCH --error=slurm_logs/%j.out
#SBATCH -p willow 
#SBATCH -A willow

set -x
# set -e

# module purge
# pwd; hostname; date

# cd $HOME/codes/robot-3dlotus

# . $HOME/miniconda3/etc/profile.d/conda.sh
# conda activate gembench

# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
# export WORLD_SIZE=1
# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr

# ulimit -n 2048


python genrobo3d/train/train_simple_policy.py 