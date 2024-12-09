import numpy as np
from time import time
import dill
import extq2
#hk: line (7, 8), (9, 10), (18, 19) (87)
workdir = "/project/dinner/kjeong/insulin/pipeline/step7_5ns"
#state_arr = np.load(f"{workdir}/step1_cvs_state/state_arr.npy")
state_arr = np.load(f"{workdir}/step1_cvs_state/state_arr_hk.npy")
#with open(f"{workdir}/step2_k_div/qf_unq.npy", "rb") as f:
with open(f"{workdir}/step2_k_div/qf_unq_hk.npy", "rb") as f:
    qf_unq = dill.load(f)

ntraj = 28*28*24
length = 1000
dt=2e-15
log=2500

#k_ls = [100, 200, 400, 600]
k_ls = [1000, 1500, 2000, 3000]
rn_ls = [0, 1, 2]
lag_ls = [100, 200, 400, 600, 800, 900]
div_ls = [1, 4, 2, -4]

dim_arr = state_arr[0, 0]==0
mon_arr = state_arr[0, 0]==k_ls[0]-1

inA, inB, inD =dim_arr.reshape((ntraj, length)), mon_arr.reshape((ntraj, length)), np.ones((ntraj, length), dtype=bool)
inD[inB] = False
inD_mfpt = np.copy(inD)
inD[inA] = False
gf, gb, gmfpt = np.zeros((ntraj, length)), np.zeros((ntraj, length)), np.zeros((ntraj, length))
gf[inB] = 1
gb[inA] = 1

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

inv_rate_arr = np.zeros((len(k_ls), len(rn_ls), len(lag_ls), len(div_ls)))
for i_rn, rn in enumerate(rn_ls): #0, 1, 2
    for i_k, k in enumerate(k_ls): #100, 200, 400, 600
        #Basis construct
        #basisL_arr = basis_construct(state_arr[i_k, i_rn])
        t0 = time()
        for i_div, div in enumerate(div_ls): #1, 4, 2, -4
            #Mask
            mask = np.ones(ntraj, dtype=bool)
            if div > 1:
                mask[np.arange(ntraj)[::div]] = False
            elif div < 0:
                mask = np.zeros(ntraj, dtype=bool)
                mask[::-div] = True
            
            gf_div = gf[mask]
            inD_div = inD[mask]
            ntraj_div = np.count_nonzero(mask)
            st_uq, st_id, st_inv = np.unique(state_arr[i_k,i_rn].reshape(ntraj, length)[mask], return_index=True, return_inverse=True)
            if len(st_uq) != k:
                print(f"Error: {len(st_uq)} states are found in k={k}, rn={rn}, div={ntraj_div/ntraj}")
                continue
            
            for i_lag, lag in enumerate(lag_ls): #100, 200, 400, 600, 800, 900
                w_us = np.repeat(np.ravel(weight_us), length).reshape(ntraj, length)[mask]
                w_us[:,-lag:]=0
                w_us = w_us/np.sum(w_us)                

                qf = np.ravel(np.zeros(w_us.shape))
                for i_st, q_val in enumerate(qf_unq[i_k][i_rn, i_lag, i_div]):
                    qf[st_inv==i_st] = q_val
                qf = qf.reshape(ntraj_div, length)
                inv_rate_arr[i_k, i_rn, i_lag, i_div] = 1/(extq2.tpt.rate(qf, 1-qf, w_us, inD_div, qf, lag)/(dt*log))

                print(f"Done: k={k}, rn={rn}, div={ntraj_div/ntraj}, lag={lag}")
        np.save(f"{workdir}/step4_rate/div/inv_rate_arr_hk.npy", inv_rate_arr)    
        print(f"Done: k={k}, rn={rn}, ({time()-t0:.2f} sec)")