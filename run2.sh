#!/bin/bash

#SBATCH --job-name=in_context_learning
#SBATCH --output=gpu_eval_8.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=a100.MIG.20gb:1
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=gpu_devel
#SBATCH --time=12:00:00

module load CUDA
module load cuDNN
# using your anaconda environment
module load miniconda
source activate in-context-learning
wandb login cc6ec8f1126b0f574d19718f6fc0232c274ac33c
python src/get_eval.py
