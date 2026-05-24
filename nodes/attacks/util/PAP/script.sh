#!/bin/bash
#SBATCH --job-name=PAP_sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=60:00:00
#SBATCH --mem=16G

source ../../../../venv/bin/activate
python PAP_incontext_sampling_example.py