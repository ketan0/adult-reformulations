#!/usr/bin/env bash
#SBATCH -t 3:00:00
#SBATCH -p gpu
#SBATCH -G 1
#SBATCH -C GPU_MEM:16GB
#SBATCH --mem=32G

source /scratch/users/agrawalk/miniconda3/etc/profile.d/conda.sh
conda activate ml
python childes_mlm_pretraining.py \
    --batch-size 64 \
    --val-batch-size 256 \
    --num-epochs-per-save 1 \
    --train-csv-path pretraining_df_nt_train.csv \
    --val-csv-path pretraining_df_nt_val.csv
