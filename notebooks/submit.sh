#!/bin/bash
#SBATCH --output="train_seqk4200.out"
#SBATCH --partition=l40s
#SBATCH --gres=gpu:1
#SBATCH --cluster=gpu
#SBATCH --job-name="train_seqk4200" 
#SBATCH --cpus-per-task 1
#SBATCH --time 0-12:00:00
#SBATCH --mem 50G


conda init
source ~/.bashrc
conda activate scGPT
echo $CONDA_DEFAULT_ENV

# python train_seq.py 
python train_seqk4.py
