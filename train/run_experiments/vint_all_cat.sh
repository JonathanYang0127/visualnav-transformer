#!/bin/bash
#SBATCH --partition=savio4_gpu
#SBATCH --account=co_rail
#SBATCH --time=1-20:00:00
#SBATCH --job-name=catjob
#SBATCH --qos=rail_gpu4_normal
#SBATCH --cpus-per-task=10
#SBATCH --mem=175G
#SBATCH --gres=gpu:A5000:1
#SBATCH --job-name="cglossop_omni"
#SBATCH --output=/global/scratch/users/catherineglossop/omnimimic/outputs/exp-%j.out

source ~/.bashrc
cd /global/home/users/catherineglossop/visualnav-transformer/train
conda activate /global/scratch/users/catherineglossop/miniconda3/omni
export NCCL_P2P_DISABLE=1

python train.py -c config/vint_all.yaml --use-rlds --datasets all

#done


