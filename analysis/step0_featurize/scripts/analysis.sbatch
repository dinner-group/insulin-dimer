#!/bin/bash

## Change `job-name`, `feat`

##SBATCH --job-name=CV_dist
##SBATCH --job-name=CV_BBagchi
##SBATCH --job-name=CV_DEShaw
##SBATCH --job-name=CV_angle
##SBATCH --job-name=CV_angle_open
##SBATCH --job-name=CV_IRMSD
##SBATCH --job-name=CV_HeavyContact
##SBATCH --job-name=CV_ISolv
##SBATCH --job-name=CV_NativeContact
##SBATCH --job-name=CV_hbridge
##SBATCH --job-name=CV_NonnatContact
##SBATCH --job-name=CV_Euler
##SBATCH --job-name=CV_Allcontact1
##SBATCH --job-name=CV_Allcontact_hm
##SBATCH --job-name=CV_SolvAtom
##SBATCH --job-name=CV_SolvResidue_wrap_water
##SBATCH --job-name=CV_ZIP
##SBATCH --job-name=CV_IntraCont_ch1
##SBATCH --job-name=CV_IntraCont_ch2
##SBATCH --job-name=CV_BBagchiQ
##SBATCH --job-name=CV_Detach
##SBATCH --job-name=CV_GridDensity
#SBATCH --job-name=CV_Luis

#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --qos=dinner
#SBATCH --partition=dinner-hm
#SBATCH --ntasks=48
#SBATCH --mem=1500G

source ~/.bashrc
export NUMEXPR_MAX_THREADS=$SLURM_NTASKS

workdir=${REPOPATH}/insulin-dimer/mdrun
ref_file=${workdir}/inputs/ins-dim.pdb
parm_file=${workdir}/inputs/ins-dim.psf

## feat (shoud've used $SLURM_JOB_NAME parsing)
#feat=distance
#feat=BBagchi
#feat=DEShaw
#feat=angle
#feat=angle_open
#feat=IRMSD
#feat=HeavyContact
#feat=ISolv
#feat=NativeContact
#feat=hbridge
#feat=NonnatContact
#feat=Euler
#feat=Allcontact1
#feat=Allcontact
#feat=SolvAtom-atom
#feat=SolvAtom-residue
#feat=ZIP
#feat=IntraContacts1
#feat=IntraContacts2
#feat=BBagchiQ
#feat=Detach
#feat=GridDensity
feat=LuisDisorder


coord_dir=${workdir}/outputs
output=${REPOPATH}/insulin-dimer/analysis/step0_featurize/outputs

## A. The first iteration of simulation (2.5 ns length trajectory)
#python analysis.py ${coord_dir}/concat ${ref_file} ${parm_file} -o ${output}/${feat} -c ${feat}

## B. Extended/Whole (after extending trajectories to 5 ns)
## B.1. Extended (for the CVs which had already been computed for the first half of trajectories, 2.5 ns )
#python analysis.py ${coord_dir} ${ref_file} ${parm_file} -o ${output}/extend_5ns/$feat -c ${feat} -e

## B.2. Whole (for the CVs which had not been computed for the first half of trajectories, 2.5 ns)
python analysis.py ${coord_dir} ${ref_file} ${parm_file} -o ${output}/whole_5ns/${feat} -c ${feat} -w

## When job_array is needed due to memory issue.
##SBATCH --array=1-4
##job_id=${SLURM_ARRAY_TASK_ID}
##python analysis.py ${coord_dir} ${ref_file} ${parm_file} -o ${output}/whole_5ns/${feat}_${job_id} -c ${feat} -w -i ${job_id}