#!/bin/bash
#SBATCH --job-name=df_43
#SBATCH --ntasks=1
#SBATCH --mem=34G
#SBATCH --output olid_default.out
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user="jaleite1@sheffield.ac.uk"

#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=4

export SLURM_EXPORT_ENV=ALL
module load Anaconda3/2019.07
module load cuDNN/7.6.4.38-gcccuda-2019b
source activate selftrain
cd experiments

make olid-default SEED=43