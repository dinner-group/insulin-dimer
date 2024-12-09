#!/bin/bash

#SBATCH --output=%x.out
#SBATCH --error=%x.err

#SBATCH --nodes=1
##SBATCH --time=2-00:00:00
#SBATCH --partition=beagle3
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source ~/.bashrc
export NUMEXPR_MAX_THREADS=8
module unload cuda
module load cuda/11.5

###S_START(=S_END), itfr_START, itfr_END are passed from run_module.sh

I_START=00
I_END=27

workdir=${REPOPATH}/insulin-dimer/mdrun
parm_dir=${workdir}/inputs/toppar
parm_file=${workdir}/inputs/ins-dim.psf
itfr_file=${workdir}/inputs/itfr.txt
ITFRS=$(ex +${itfr_START},${itfr_END}p -scq ${itfr_file})

for itfr_id in ${ITFRS}
do
	for s in $(eval echo {${S_START}..${S_END}})
	do
    		for i in $(eval echo {${I_START}..${I_END}})
		    do
		        coord_file=${workdir}/inputs/coord/ac${s}bc${i}/itfr${itfr_id}.gro
		        output=${workdir}/outputs/ac${s}bc${i}/itfr${itfr_id}
			#Round0: Initiation 1 250 000 * 2fs = 2.5 ns (2500 frames)
		        #python run.py ${coord_file} ${parm_dir} ${parm_file} -p cuda -t 1250000 -s 2500 -o ${output}
			#Round1: Restart from last checkpoint file#Only change -t 5ns
		        python run.py ${coord_file} ${parm_dir} ${parm_file} -p cuda -t 2500000 -s 2500 -o ${output} -r ${output}.chk
		    done
	done
done
