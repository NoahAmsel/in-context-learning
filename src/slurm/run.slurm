#!/bin/sh

#SBATCH --cpus-per-task       8
#SBATCH --gpus-per-node       1
#SBATCH --mem                 32G
#SBATCH --open-mode           append
#SBATCH --output              /scratch/nia4240/garg-scratch/slurm_logs/%A.out
#SBATCH --time                10:00:00
#SBATCH --job-name            normal

singularity exec --nv --overlay /scratch/nia4240/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c "
source /ext3/env.sh
conda activate in-context-learning
cd /home/nia4240/in-context-learning/src
python train.py --config conf/linear_regression.yaml --model.n_embd 24
"
