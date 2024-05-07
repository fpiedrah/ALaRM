#!/bin/bash

#SBATCH -n 16
#SBATCH --mem=64g
#SBATCH -t 6:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --partition=3090-gcondo

# CUDA
module load cuda

# INSTALL ALARM
pip install -r requirements.txt
pip install -e trl/.

# WIN RATES
python ./eval/eval_compare.py --generations_dir ./long-form-QA/model_generations/seed42 \
                              --task_type qa
