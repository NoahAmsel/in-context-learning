#!/bin/bash

#SBATCH --cpus-per-task       8
#SBATCH --gpus-per-node       1
#SBATCH --mem                 32G
#SBATCH --open-mode           append
#SBATCH --output              /scratch/nia4240/garg-scratch/slurm_logs/%A/%A_%a.out
#SBATCH --time                08:00:00
#SBATCH --job-name            gilad
#SBATCH --array=0-19

Hs=(1 2 4 8 16)
ds=(48)
reps=(1 2 3 4)
Hs_reindexed=()
ds_reindexed=()
for H in ${Hs[@]}; do
    for d in ${ds[@]}; do
        for rep in ${reps[@]}; do
            Hs_reindexed+=( $H )
            ds_reindexed+=( $d )
        done
    done
done

H=${Hs_reindexed[$SLURM_ARRAY_TASK_ID]}
d=${ds_reindexed[$SLURM_ARRAY_TASK_ID]}

singularity exec --nv --overlay /scratch/nia4240/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate in-context-learning
cd /home/nia4240/in-context-learning/src
python train.py --config conf/linear_regression.yaml --model.n_head $H --model.n_embd $d --out_dir /scratch/nia4240/garg-scratch/models/$SLURM_JOB_NAME --wandb.name ${SLURM_JOB_NAME}/$SLURM_ARRAY_JOB_ID/$SLURM_ARRAY_TASK_ID
"
