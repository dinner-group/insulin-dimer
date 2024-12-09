"""
This script estimates dga with memoryruns 

TPT quantities computed includes
    1. change of measure
    2. forward committor
    ...

Inputs:
------
Command line input: list[int], list[int]
Outputs: ./estimates/w_mem.pkl
         ./estimates/qf_mem.pkl
         
-------


"""

from time import time
import argparse

import numpy as np
import dill

from dga_prep import dga_construct
import extq2

#hk: line (30, 31), (80, 81)
#k_ls = [100, 200, 400, 600]
k_ls = [1000, 1500, 2000, 3000]

rn_ls = [0, 1, 2]

def main(state_arr_f, output, ntraj, length, lags, mems):
    """
    Parameters:
    ----------
    state_trajs: ndarray (ntraj, length)[int]
        state index from 0 to ncluster - 1
    lags: list[int]
    mems: list[int]
    
    Returns:
    -------
    
    """
    #ntraj, length =  state_trajs.shape
    state_arr = np.load(state_arr_f)

    qf_mem_unq = [np.zeros((len(rn_ls), len(lags), len(mems), k)) for k in k_ls] #proj, solv

    for i_rn, rn in enumerate(rn_ls): #0, 1, 2
        for i_k, k in enumerate(k_ls): 
            #Basis construct
            st_uq, st_id = np.unique(state_arr[i_k,i_rn], return_index=True)

            state_trajs = state_arr[i_k, i_rn].reshape((ntraj, length))
            basis_dic, D_dic, guess_dic = dga_construct(state_trajs)
            basisF = basis_dic['committor']
            inD = D_dic['q']
            gf = guess_dic['qf']
            for i_lag, lag in enumerate(lags):
                noweight = np.ones((ntraj, length))
                noweight[:, -lag:]=0
                for i_mem, mem in enumerate(mems):
                    assert mem >= 0 
                    assert lag % (mem + 1) == 0
                    t1 = time()
                    qf_mem = extq2.dgamem.forward_committor(basisF, noweight, inD, gf, 
                                                            lag = lag,
                                                            mem = mem,
                                                            return_projection = True,
                                                            return_solution = False)
                    qf_mem_unq[i_k][i_rn, i_lag, i_mem] = np.copy(np.ravel(np.vstack(qf_mem))[st_id])
                    del qf_mem
                    t2 = time()
                    print(f"compute qf w/ lag {lag} mem {mem}: {t2-t1} secs")

                    with open(f"{output}/qf_unq_mem_hk.pkl", "wb") as f:
                        dill.dump(qf_mem_unq, f)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "")
    parser.add_argument(
        "--state_arr",
        type = str,
        help = "path to the state_trajectory"
    )
    parser.add_argument(
        "--output",
        type = str,
        default = "/project/dinner/kjeong/insulin/pipeline/extra_test/validation/memory/output",
        help = "path to save the tpt"
    )
    parser.add_argument(
        "--ntraj",
        type = int,
        default = 18816,
        help = "number of trajectories"
    )
    parser.add_argument(
        "--length",
        type = int,
        default = 1000,
        help = "length of trajectory"
    )
    parser.add_argument(
        "--lags",
        type = int,
        nargs = '+',
        default = [100],
        help = "list of lag time to compute tpt estimates in unit of step (2500 * 0.2 fs = 5 ps)"
    )
    parser.add_argument(
        "--mems",
        type = int,
        nargs = '+',
        default = [0],
        help = "list of the number of mem kernels to compute tpt estimates"
    )
    args = parser.parse_args()
    
    main(args.state_arr, args.output, args.ntraj, args.length, args.lags, args.mems)
    