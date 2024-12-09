#!/bin/bash

#SBATCH --job-name=q_unq_hk

#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1

#SBATCH --time=2-00:00:00
#SBATCH --partition=dinner-hm
#SBATCH --qos=dinner
#SBATCH --ntasks=48
#SBATCH --mem=1500G

source ~/.bashrc
export NUMEXPR_MAX_THREADS=48

python step2_q_unq.py