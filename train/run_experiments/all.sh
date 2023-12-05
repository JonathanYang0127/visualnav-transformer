#!/bin/bash
#SBATCH --partition=iris-hi
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --mem=200G

# only use the following on partition with GPUs
#SBATCH --gres=gpu:2

#SBATCH --job-name="Jonathan_Exp"
#SBATCH --output=/iris/u/jyang27/cluster/outputs/exp-%j.out
source ~/.bashrc
conda deactivate 
conda activate navigation_env

cd /iris/u/jyang27/dev/visualnav-transformer/train
python train.py -c config/vint_all.yaml --use-rlds --datasets all

#done


