#!/bin/bash

#SBATCH --job-name=CV_GridDensity
####Minimum Image convention
####ISolv::: water_distance = capped_distance(ag, water_o, **kwargs) # (n_pairs, 2) pairs of indices 
####CV_SolvResidue:::water_distance = capped_distance(ags, water_o, **kwargs) # (n_pairs, 2) pairs of indices
####CV_hbridge::: pbc=True

#SBATCH --output=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/whole_5ns/log_jobarray/dinner/%x_%a.out
#SBATCH --error=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/whole_5ns/log_jobarray/dinner/%x_%a.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1
#SBATCH --time=48:00:00

##SBATCH --partition=caslake
##SBATCH --ntasks=48
##SBATCH --mem=180G
##SBATCH --array=1-16

#SBATCH --partition=dinner
#SBATCH --ntasks=48
#SBATCH --mem=375G
#SBATCH --array=1-4

source ~/.bashrc
export NUMEXPR_MAX_THREADS=$SLURM_NTASKS

workdir=/beagle3/dinner/kjeong
ref_file=${workdir}/Insulin_md/input/parm/charmm-gui-3w7y-dimer-ins/step3_pbcsetup.pdb
parm_file=${workdir}/Insulin_md/input/parm/charmm-gui-3w7y-dimer-ins/step3_pbcsetup.psf

#My
feat=GridDensity

#Extended/Whole
coord_dir=${workdir}/Insulin_md/run_openmm/output

job_id=${SLURM_ARRAY_TASK_ID}

#Whole
#output=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/whole_5ns/${feat}
output=/beagle3/dinner/kjeong/output/whole_5ns/dinner/${feat}_${job_id}
python analysis_nocat.py ${coord_dir} ${ref_file} ${parm_file} -o ${output} -c ${feat} -w -i ${job_id}
