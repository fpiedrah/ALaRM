#!/bin/bash

#SBATCH -n 16
#SBATCH --mem=64g
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH --partition=3090-gcondo

# CUDA
module load cuda

# INSTALL ALARM
pip install -r requirements.txt
pip install -e trl/.

# INSTALL AWQ KERNELS
# git clone https://github.com/casper-hansen/AutoAWQ_kernels
# pip install -e AutoAWQ_kernels/.

# DOWNLOAD DATASET
python -m spacy download en_core_web_sm

# RUN ALARM EVALUATION
accelerate launch --multi_gpu ./long-form-QA/train_ppo.py \
  --save_dir ./long-form-QA/model_generations/seed42/hierarchical.json \
  --sigmoid_shaping --reward_type hierarchical \
  --w_rel 0 --w_fact 1 --w_comp 0 --seed 42 --run_name test_hierarchical \
  --test --policy_ckpt ./long-form-QA/model_ckpts/t5-large-1k-train
