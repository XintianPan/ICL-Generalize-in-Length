#!/bin/bash

#SBATCH --job-name=in_context_learning
#SBATCH --output=gpu_eval_4.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100:1
#SBATCH --mem-per-cpu=30000
#SBATCH --partition=gpu
#SBATCH --time=8:00:00

module load CUDA
module load cuDNN
# using your anaconda environment
module load miniconda
source activate in-context-learning
python src/get_eval.py
