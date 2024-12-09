#!/bin/bash

#SBATCH --job-name=step3_memory_hk

#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1

#SBATCH --time=2-00:00:00
##SBATCH --partition=beagle3-bigmem
##SBATCH --constraint=512g
##SBATCH --ntasks=32
##SBATCH --mem=500G
#SBATCH --partition=dinner
#SBATCH --ntasks=48
#SBATCH --mem=375G

source ~/.bashrc
export NUMEXPR_MAX_THREADS=$SLURM_NTASKS

workdir=/project/dinner/kjeong/insulin/pipeline/step7_5ns
output=${workdir}/step3_lag_mem/output
#state_arr=${workdir}/state_arr/state_arr_OG.npy
state_arr=${workdir}/step1_cvs_state/state_arr_hk.npy
#state_arr=${workdir}/state_arr/state_arr_middle_randomseed_k200.npy

python step3_memory.py --state_arr ${state_arr}\
                    --output ${output}\
                    --lags 100 200 400 600 800 900 --mems 0 1 4