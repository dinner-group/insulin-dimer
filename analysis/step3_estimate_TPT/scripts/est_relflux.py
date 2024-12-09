from time import time
import numpy as np
import pandas as pd
import scipy
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from time import time
import dill
import extq2
ncpu = cpu_count()

# 1. tpt_df; k, lag, ntraj, length, nframe, states_alongq, qf
t0 = time()
workdir = "/project/dinner/kjeong/insulin/pipeline/step7_5ns"
k, lag, ntraj, length = 600, 400, 28*28*24, 1000
nframe = int(ntraj*length)

state_arr = np.load(f"{workdir}/step1_cvs_state/state_arr.npy")[3,0]
state_trajs = state_arr.reshape((ntraj, length))

# Load US weight
P_beagle=f"/beagle3/dinner/kjeong"
us_path=f"{P_beagle}/insulin_backedup/NVT_US_ins/cvs"
psis=np.load(f'{us_path}/input_PMFdata/t_50/Psis50.npy')
z=np.load(f'{us_path}/input_PMFdata/t_50/Zs50.npy')

nwindow, nframe_us = psis.shape[:-1]
nwindow_ax = int(np.sqrt(nwindow))
ntraj_w=24
ntraj = nwindow*ntraj_w
weight_us_all = np.zeros((nwindow, nframe_us))# (window, nframe per each traj 1050 )
for i0 in range(nwindow):
    psi_sum = np.sum(psis[i0], axis=1)
    weight_us_all[i0] = z[i0]/(psi_sum*nframe_us)
#/project/dinner/kjeong/insulin/pipeline/step0_catdcd
weight_us = weight_us_all[:, 42:1050:42]

w_us = np.repeat(np.ravel(weight_us), length).reshape(ntraj, length)
w_us[:,-lag:]=0
w_us = w_us/np.sum(w_us)                
weights = {
    'us': w_us
}
weights['no'] = np.ones((ntraj, length))
weights['no'][:, -lag:] = 0

dt=2e-15
log=2500

def sps_basis(state_arr):
    """Converts a discretized trajectory (e.g. from k-means clustering)
    into a sparse basis of indicator functions.

    Parameters
    ----------
    state_arr : ndarray (nframes,)
        discretized trajectories (having the values from 0 to k-1),
        where k is the number of basis.

    Return
    ------
    basis : scipy.sparse.csr_matrix (nframes, nclusters)
    """
    nclusters = k
    rows, cols = [], []
    for i in range(nclusters):
        pts = np.argwhere(state_arr == i)
        # indices of which frames are in the cluster i
        rows.append(pts.squeeze())
        # all assigned as 1 in the basis
        cols.append(np.repeat(i, len(pts)))
    rows = np.hstack(rows)
    cols = np.hstack(cols)
    data = np.ones(len(rows), dtype=float)
    basis = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(len(state_arr), nclusters))

    return basis
t1 = time()
print(f"Elapsed time for loading: {t1-t0:.2f}s")
basis_raw = sps_basis(state_arr)
t2 = time()
print(f"Elapsed time for loading: {t2-t1:.2f}s")

gf_reg = np.zeros((ntraj, length), dtype=float)
gf_reg[np.isin(state_trajs, k-1)] = 1.0

gb_reg = np.zeros((ntraj, length), dtype=float)
gb_reg[np.isin(state_trajs, 0)] = 1.0


def make_excluded_basis(state_id):
    # Define in_D_tmp
    id_inD = [x for x in range(k) if not x in np.hstack(([0, k-1], state_id))]
    in_D = np.isin(state_trajs, id_inD)
    st_inD = np.isin(np.arange(k), id_inD) #to remove the chosen state column from basisD
    
    # Define basis
    basisD = [basis_raw[i0*length:(i0+1)*length, st_inD] for i0 in range(ntraj)]
    return basisD, in_D

basisD, in_D = make_excluded_basis([0])
qf_0 = np.vstack(
    extq2.dgamem.forward_committor(
        basisD, weights['no'],in_D, gf_reg, 
        lag, mem=1,
        return_projection=True, return_solution=False))

def case_flux(state_id, mem):
    basisD, in_D = make_excluded_basis(state_id)

    #q_+(B)
    qf_tmp = np.vstack(extq2.dgamem.forward_committor(basisD, weights['no'],\
        in_D, gf_reg, lag, mem=mem,
        return_projection=True, return_solution=False))
    
    #q_-(A) = q_+(A)
    qb_tmp = np.vstack(extq2.dgamem.forward_committor(basisD, weights['no'],\
        in_D, gb_reg, lag, mem=mem,
        return_projection=True, return_solution=False))

    rate = 1/(extq2.tpt.rate(qf_tmp, qb_tmp, weights['us'], in_D, qf_0, lag)/(dt*log))

    return rate

def parallel_run(function, ncpu, state_id_set, mem):
    """
    Input:
    -----
    function: callable function which is expensive and needs to be parallelized.
    args_list: list of arguments which will be passed to function
    
    """
    return Parallel(n_jobs=ncpu, verbose=11)(delayed(function)(state_id, mem) for state_id in state_id_set)

#1. 
state_id_set = np.arange(1, k-1) #The first one will be just a full DGA
outfname = f"{workdir}/step6_atpt/output/fine_flux_mem.npy"

"""
#2
state_id_set = []
for i0 in np.arange(1, k-1):
    for i1 in np.arange(1,k-1):
        if i0 < i1:
            state_id_set.append([i0, i1])
#3. 
with open(f"{workdir}/step7_structure/stid_cg.pkl", "rb") as f:
    stid_cg = dill.load(f)
state_id_set_1 = []
for key, tmp in stid_cg.items():
    state_id_set_1 += tmp
#4
state_id_set = []
for i0, arr0 in enumerate(state_id_set_1):
    for i1, arr1 in enumerate(state_id_set_1):
        if i0 < i1:
            state_id_set.append(np.hstack((arr0, arr1)))
outfname = f"{workdir}/step6_atpt/output/cg_flux2_2ns_mem.npy"
#5. All combinations
n_meta = len(state_id_set_1)
extensions = np.zeros((2**n_meta, n_meta), dtype=bool)
for i0 in range(len(extensions)):
    binary_str = bin(i0)[2:]
    extensions[i0] = np.array([0]*(n_meta-len(binary_str))+[int(i) for i in binary_str], dtype=bool)
state_id_set_1_arr = np.array(state_id_set_1, dtype=object)
state_id_set = [np.hstack(state_id_set_1_arr[comb_inst]) for comb_inst in extensions[1:]]
outfname = f"{workdir}/step6_atpt/output/cg_flux_full_2ns_mem.npy"
"""

result = parallel_run(case_flux, ncpu, state_id_set, mem=1)
np.save(outfname, np.array(result))

print("Elapsed time for parallel run: ", time()-t2)