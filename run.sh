#!/bin/bash

#SBATCH --job-name=in_context_learning
#SBATCH --output=gpu_job_2.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus=a100:1
#SBATCH --partition=gpu
#SBATCH --mem-per-cpu=50000
#SBATCH --time=12:00:00

module load CUDA
module load cuDNN
# using your anaconda environment
module load miniconda
source activate in-context-learning
python src/train.py --config src/conf/linear_regression.yaml
