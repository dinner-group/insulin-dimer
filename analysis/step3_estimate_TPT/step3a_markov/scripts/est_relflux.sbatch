#!/bin/bash

#SBATCH --job-name=cg_ninter_full
#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1

#SBATCH --time=2-00:00:00

#SBATCH --partition=dinner
#SBATCH --partition=dinner-hm
#SBATCH --qos=dinner
#SBATCH --ntasks=48
##SBATCH --mem=375G
#SBATCH --mem=1500G

source ~/.bashrc
export NUMEXPR_MAX_THREADS=$SLURM_NTASKS

python dga_ninter.py
echo "Completed!"
