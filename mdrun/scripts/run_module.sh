#!/bin/bash

###The only thing that needs to be changed is 
#1) qos=beagle3 itfr_len=5 && for itr_val in 1 6
#2) qos=beagle3-long itfr_len=10 && for itfr_val in 11
#3) qos=beagle3 itfr_len=5 && for itfr_val in 21

#Jun 21 2024 extend trajectory from 2.5 ns length to 5.0 ns
#1) qos=beagle3 itfr_len=6 && S_START=00, S_END=27 && for itfra_val in 2 8
#2) qos=beagle3-long itfr_len=12 && S_START=00, S_END=27 && for itfr_val in 14
itfr_len=12
S_START=00
S_END=27
for itfr_val in 14
do
	for S_VAL in $(eval echo {${S_START}..${S_END}})
	do
		#echo --job-name=s${S_VAL}itfr${itfr_val} --export=S_START=${S_VAL},S_END=${S_VAL},itfr_START=${itfr_val},itfr_END=$(($itfr_val+$itfr_len-1)) run.sbatch
		sbatch --job-name=s${S_VAL}itfr${itfr_val} --export=S_START=${S_VAL},S_END=${S_VAL},itfr_START=${itfr_val},itfr_END=$(($itfr_val+$itfr_len-1)) run.sbatch
	done
done
