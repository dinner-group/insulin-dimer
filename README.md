# insulin-dimer
This repository contains custom code for analyzing all-atom Molecular dynamics simulations of insulin dimer dissocation for [ref. 1][1].

* This repository is not stable, and there are missing information, which I hope myself to include for full reproduction of data reported in [ref. 1][1]. 
Following is the non-exhaustive list of potentially missing components. 
    - Initial coordinates for running MD simulation
    - All the outputs, MD trajectories, featurized trajectories, estimated committors and TPT quantities

## Repository Structure
- mdrun    
    - scripts: scripts for running Molecular Dynamics
    - inputs: input topology and forcefield (Initial coordinates are stored in [this][2])
- analysis : 
    - notebooks
        - analysis_projection.ipynb     :
        - analysis_coarse-grained.ipynb :
    - step1_build_MSM:
    - step2_validate_choice
        - step2a_nstate_lag
        - step2b_nmem_lag
        - step2c_rate
    - step3_estimate_TPT

## TO DO
- Dependencies on environment set up
- Add initial coordinates for MD simulation

## References
1. Jeong, et al. [Analysis of the Dynamics of a Complex, Multipathway Reaction: Insulin Dimer Dissociation][1]

[1]: https://doi.org/10.1021/acs.jpcb.4c06933