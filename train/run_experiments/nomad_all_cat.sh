#!/bin/bash
#SBATCH --partition=savio3_gpu
#SBATCH --account=co_rail
#SBATCH --time=5-00:00:00
#SBATCH --mem=200G
#SBATCH --job-name=catjob
#SBATCH --qos=rail_gpu3_normal
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --job-name="cglossop_omni"
#SBATCH --output=/global/scratch/users/catherineglossop/omnimimic/outputs/exp-%j.out

source ~/.bashrc
cd /global/home/users/catherineglossop/visualnav-transformer/train
conda activate /global/scratch/users/catherineglossop/miniconda3/omni
export NCCL_P2P_DISABLE=1

python train.py -c config/vint_nomad_all.yaml --use-rlds --datasets all

#done


