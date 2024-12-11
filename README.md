# insulin-dimer  
This repository contains custom code for running and analyzing all-atom molecular dynamics (MD) simulations of insulin dimer dissociation, as described in [ref. 1][1].  

> **Note:**  
> The code is shared to document the workflow implementation in [ref. 1][1] and is not intended for direct execution.  
> 
> **Missing Components:**  
> - Initial coordinates for running MD simulations  
> - Outputs such as MD trajectories, featurized trajectories, estimated committors, and TPT quantities, which are needed to link workflow steps.

## Repository Structure  
- `mdrun`: Files for running MD simulations.  
  - `scripts`: Scripts for MD runs  
  - `inputs`: Input topologies and force fields (initial coordinates are stored in [this][2])  
- `analysis`: Files for DGA/TPT analysis of MD trajectories.  

## References
1. Jeong, et al. [Analysis of the Dynamics of a Complex, Multipathway Reaction: Insulin Dimer Dissociation][1]

[1]: https://doi.org/10.1021/acs.jpcb.4c06933
