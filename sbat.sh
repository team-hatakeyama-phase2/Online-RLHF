#!/bin/bash

#SBATCH --nodelist=slurm0-a3-ghpc-2
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --job-name=train_model
#SBATCH --output=train_model_%j.log
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G


#export HUGGINGFACE_HUB_CACHE=/storage5/huggingface_cache/hub
#export TRANSFORMERS_CACHE=/storage5/huggingface_cache/hub
#export HF_HUB_CACHE="/storage5/shared/huggingface_cache/hub"
export HF_HUB_CACHE="/home/ext_at_y_takagi_gmail_com/.cache/huggingface_cache/hub"

#bash run_loop2.sh
#bash run_loop3.sh
#bash run_loop_multi_gpu.sh
#bash run_loop_large.sh
#bash run_loop_with_api.sh
#bash run_loop_with_reward_api.sh
bash run_loop_lora_with_reward_api.sh
