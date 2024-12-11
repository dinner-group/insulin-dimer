## mdrun  
This directory contains files for running molecular dynamics (MD) simulations.  

1. **`inputs`**  
   - Contains the `parm` folder, which stores topology and force-field files.  

2. **`scripts`**  
   - Contains Python and Bash scripts for running OpenMM, with parameters such as timestep, temperature, methods for handling long-range interactions, and etc.  
     - `run.py`: Python script for running OpenMM simulations.  
     - `run.sbatch`: Slurm job submission script.  
     - `run_module.sh`: Bash script for running `run.sbatch` in parallel (similar to a Slurm job array).  
