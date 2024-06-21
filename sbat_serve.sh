#!/bin/bash

#SBATCH --nodelist=slurm0-a3-ghpc-2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=train_model
#SBATCH --output=serve_model_%j.log
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=8

#export HUGGINGFACE_HUB_CACHE=/storage5/huggingface_cache/hub
#export TRANSFORMERS_CACHE=/storage5/huggingface_cache/hub
export HF_HUB_CACHE="/storage5/shared/huggingface_cache/hub"

bash run_reward_api.sh
