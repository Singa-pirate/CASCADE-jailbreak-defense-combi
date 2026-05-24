#!/bin/bash
#SBATCH --job-name=CASCADE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=72:00:00
#SBATCH --mem=32G

source venv/bin/activate
python main.py config.yaml
