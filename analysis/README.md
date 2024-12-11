## analysis  
This directory contains files for DGA/TPT analysis of MD trajectories.  

The analysis is divided into sequential steps, each in its own directory. Each step may include scripts for analysis and Jupyter notebooks for visualizing outcomes.

> **Note:**  
Due to storage limitations, all analysis outcomes have been removed. As this repository was transferred from a cluster to GitHub, file paths for outputs (step $i$) or inputs (step $i+1$) have not been updated. As a result, the scripts may not run as-is. These scripts are intended to document the workflow described in [this work][1] rather than function out of the box.


0. `step0_featurize`: compute a variety of collective variables from MD trajectories using MDAnalysis.
1. `step1_build_MSM`: define Markov state to build MSM
2. `step2_validate_choice`: valide choices for MSM construction (the number of Markov states, $k$, lag time, $\tau$, and the number of memroy terms, $\tau/\sigma -1$, where $\sigma$ is the time interval used to compute the memory terms) by comparing
    a. `step2a_nstate_lag`: investigate $(k, \tau)$-space based on $q_B$ estimated by reducing the amount of data set.
    b. `step2b_nmem_lag`: investigate $(\sigma, \tau)$-space based on stability of estimated $q_B$ 
    c. `step2c_rate` : investigate $(k, \sigma, \tau)$-space based on the rate estimates.
3. `step3_estimate_TPT`: Estimate reactive flux with state $i$ excluded, where the state $i$ is
    a. each Markov state
    b. coarse-grained state defined in `cg.ipynb`

Two Jupyter notebooks, `proj.ipynb` and `cg.ipynb`, shows how we present the result from the aforementioned sequence of analysis in [the paper][1].

[1]: https://doi.org/10.1021/acs.jpcb.4c06933