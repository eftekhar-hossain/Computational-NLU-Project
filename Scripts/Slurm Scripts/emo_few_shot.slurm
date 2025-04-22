#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=ai
#SBATCH --gres=gpu:1
#SBATCH --constraint=h100
#SBATCH --output=emo-few-slurm-%J.out
#SBATCH --job-name=deep-few


# Give this process 1 task and 1 GPU, then assign four CPUs per task
# (so 4 cores overall).

# Output some preliminaries before we begin
date
echo "Slurm nodes: $SLURM_JOB_NODELIST"
NUM_GPUS=`echo $GPU_DEVICE_ORDINAL | tr ',' '\n' | wc -l`
echo "You were assigned $NUM_GPUS gpu(s)"

# Load the Python and CUDA modules
module load anaconda
module load cuda

# List the modules that are loaded
module list

# Have Nvidia tell us the GPU/CPU mapping so we know
nvidia-smi topo -m
nvidia-smi

echo

# Activate the GPU version of PyTorch
conda activate CAP6640


# Run PyTorch Training



echo "Start:"
time python zero_few_shot.py --llm_id meta-llama/Llama-3.2-3B-Instruct --llm_name llama32-3B --dataset_name emo --prompt_type few
echo

echo "Start:"
time python zero_few_shot.py --llm_id mistralai/Mistral-7B-Instruct-v0.3 --llm_name mistral-7B --dataset_name emo --prompt_type few
echo

echo "Start:"
time python zero_few_shot.py --llm_id Qwen/Qwen2.5-72B-Instruct-AWQ --llm_name qwen-72B --dataset_name emo --prompt_type few
echo

echo "Start:"
time python zero_few_shot.py --llm_id google/gemma-2-27b-it --llm_name gemma2-27B --dataset_name emo --prompt_type few
echo

echo "Start:"
time python zero_few_shot.py --llm_id microsoft/Phi-4 --llm_name phi4-14B --dataset_name emo --prompt_type few
echo

echo "Start:"
time python zero_few_shot.py --llm_id deepseek-ai/DeepSeek-R1-Distill-Llama-8B --llm_name deepseek-8B --dataset_name emo --prompt_type few
echo


# You're done!
echo "Ending script..."
date
