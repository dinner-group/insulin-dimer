#!/bin/bash

#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --account=pi-dinner
#SBATCH --nodes=1

###beagle3-long
#SBATCH --time=4-00:00:00
#SBATCH --partition=beagle3
#SBATCH --qos=beagle3-long

###beagle3
##SBATCH --time=2-00:00:00
##SBATCH --partition=beagle3


#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=beagle3-0029
source ~/.bashrc
export NUMEXPR_MAX_THREADS=8
#export NUMEXPR_MAX_THREADS=$SLURM_NTASKS
module unload cuda
module load cuda/11.5

###S_START(=S_END), itfr_START, itfr_END are passed from run_module.sh

I_START=00
I_END=27

workdir=/beagle3/dinner/kjeong/Insulin_md

parm_dir=${workdir}/input/parm/charmm-gui-3w7y-dimer-ins/toppar
parm_file=${workdir}/input/parm/charmm-gui-3w7y-dimer-ins/step3_pbcsetup.psf

itfr_file=${workdir}/input/itfr.txt
ITFRS=$(ex +${itfr_START},${itfr_END}p -scq ${itfr_file})

for itfr_id in ${ITFRS}
do
	for s in $(eval echo {${S_START}..${S_END}})
	do
    		for i in $(eval echo {${I_START}..${I_END}})
		    do
		        coord_file=${workdir}/input/coord/ac${s}bc${i}/itfr${itfr_id}.gro
		        output=${workdir}/run_openmm/output/ac${s}bc${i}/itfr${itfr_id}
			#Round0: Initiation 1 250 000 * 2fs = 2.5 ns (2500 frames)
		        #python run.py ${coord_file} ${parm_dir} ${parm_file} -p cuda -t 1250000 -s 2500 -o ${output}
			#Round1: Restart from last checkpoint file#Only change -t 5ns
		        python run.py ${coord_file} ${parm_dir} ${parm_file} -p cuda -t 2500000 -s 2500 -o ${output} -r ${output}.chk
		    done
	done
done
