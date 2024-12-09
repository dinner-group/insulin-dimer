#!/bin/bash

#SBATCH --job-name=step4_rate_div

#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1

#SBATCH --time=2-00:00:00
#SBATCH --partition=dinner
#SBATCH --ntasks=48
#SBATCH --mem=375G

source ~/.bashrc
export NUMEXPR_MAX_THREADS=$SLURM_NTASKS

python step4_rate_div.py