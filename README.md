# 
This repository contains custom code for analyzing all-atom Molecular dynamics simulations of insulin dimer dissocation for [ref. 1][1].

## Repository Structure
.
├── analysis
│   ├── cg.ipynb
│   ├── proj.ipynb
│   ├── step1_build_MSM
│   │   ├── buildMSM.ipynb
│   │   └── outputs
│   ├── step2_validate_choice
│   │   ├── step2a_nstate_lag
│   │   │   ├── outputs
│   │   │   ├── q_nstate_lag.ipynb
│   │   │   └── scripts
│   │   │       ├── est_q.py
│   │   │       └── est_q.sbatch
│   │   ├── step2b_nmem_lag
│   │   │   ├── outputs
│   │   │   ├── q_nmem_lag.ipynb
│   │   │   └── scripts
│   │   │       ├── est_q.py
│   │   │       └── est_q.sbatch
│   │   └── step2c_rate
│   │       ├── outputs
│   │       ├── rate.ipynb
│   │       └── scripts
│   │           ├── rate_div.py
│   │           ├── rate_div.sbatch
│   │           ├── rate_mem.py
│   │           └── rate_mem.sbatch
│   └── step3_estimate_TPT
│       ├── outputs
│       └── scripts
│           ├── est_relflux.py
│           └── est_relflux.sbatch
├── copy_src.sh
├── mdrun
│   └── scripts
└── README.md


## Requirements
- 
- Additional dependencies: see `requirements.txt`

## References
1. Jeong, et al. [Analysis of the Dynamics of a Complex, Multipathway Reaction: Insulin Dimer Dissociation][1]

[1]: https://doi.org/10.1021/acs.jpcb.4c06933