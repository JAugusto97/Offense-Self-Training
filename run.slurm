#!/bin/bash
#SBATCH --job-name=st
#SBATCH --ntasks=1
#SBATCH --mem=82G
#SBATCH --output olid_default.out
#SBATCH --mail-type BEGIN,END
#SBATCH --mail-user="jaleite1@sheffield.ac.uk"
#SBATCH --time=1-08:00:00

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2019.07
source activate selftrain
cd experiments

make olid SEED=$1 BATCH_SIZE=$2 AUGMENTATION_TYPE=$3 PRETRAINED_NAME=$4

