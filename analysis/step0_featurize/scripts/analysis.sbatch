#!/bin/bash

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

#SBATCH --output=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/whole_5ns/%x.out
#SBATCH --error=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/whole_5ns/%x.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1

#SBATCH --time=48:00:00

#SBATCH --qos=dinner
#SBATCH --partition=dinner-hm
#SBATCH --ntasks=48
#SBATCH --mem=1500G

##SBATCH --partition=dinner
##SBATCH --ntasks=48
##SBATCH --mem=375G

##SBATCH --partition=beagle3-bigmem
##SBATCH --ntasks=32
##SBATCH --mem=500G


source ~/.bashrc
export NUMEXPR_MAX_THREADS=$SLURM_NTASKS

workdir=/beagle3/dinner/kjeong
ref_file=${workdir}/Insulin_md/input/parm/charmm-gui-3w7y-dimer-ins/step3_pbcsetup.pdb
parm_file=${workdir}/Insulin_md/input/parm/charmm-gui-3w7y-dimer-ins/step3_pbcsetup.psf

#My
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

#Adam
#coord_dir=${workdir}/Insulin_md/Adam_Exploration/step0_catdcd/output
#output=${workdir}/Insulin_md/Adam_Exploration/step1_feat/output/${feat}

##Original
#coord_dir=${workdir}/Insulin_md/run_openmm/output_cat
#output=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/${feat}
#
#python analysis.py ${coord_dir} ${ref_file} ${parm_file} -o ${output} -c ${feat}

#Extended/Whole
coord_dir=${workdir}/Insulin_md/run_openmm/output
#Extended
#output=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/extend_5ns/${feat}
#python analysis_nocat.py ${coord_dir} ${ref_file} ${parm_file} -o ${output} -c ${feat} -e
#Whole

#output=/project/dinner/kjeong/insulin/pipeline/step1_feat/output/whole_5ns/${feat}
output=/beagle3/dinner/kjeong/output/whole_5ns/${feat}
python analysis_nocat.py ${coord_dir} ${ref_file} ${parm_file} -o ${output} -c ${feat} -w
