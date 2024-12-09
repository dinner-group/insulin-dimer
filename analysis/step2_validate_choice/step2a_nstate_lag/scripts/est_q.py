import numpy as np
from time import time
import dill
import extq2
import sys
sys.path.append(f"/project/dinner/kjeong/insulin/notebooks/pylocal")
from utils_rwdga import basis_construct
#hk: line (10, 11), (15, 16), (72, 73)
workdir = "/project/dinner/kjeong/insulin/pipeline/step7_5ns"
#state_arr = np.load(f"{workdir}/step1_cvs_state/state_arr.npy")
state_arr = np.load(f"{workdir}/step1_cvs_state/state_arr_hk.npy")
ntraj = 28*28*24
length = 1000

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

qf_unq = [np.zeros((len(rn_ls), len(lag_ls), len(div_ls), k)) for k in k_ls]

for i_k, k in enumerate(k_ls): #100, 200, 400, 600
    for i_rn, rn in enumerate(rn_ls): #0, 1, 2
        #Basis construct
        basisL_arr = basis_construct(state_arr[i_k, i_rn])
        t0 = time()
        for i_div, div in enumerate(div_ls): #1, 4, 2, -4
            #Mask
            mask = np.ones(ntraj, dtype=bool)
            if div > 1:
                mask[np.arange(ntraj)[::div]] = False
            elif div < 0:
                mask = np.zeros(ntraj, dtype=bool)
                mask[::-div] = True
            
            #Basis, inD, gf
            gf_div = gf[mask]
            inD_div = inD[mask]
            basisW, basisF= [], []
            ntraj_div = np.count_nonzero(mask)
            for i0 in np.nonzero(mask)[0]:
                basisW.append(basisL_arr[i0*length:(i0+1)*length, 1:])
                basisF.append(basisL_arr[i0*length:(i0+1)*length, 1:-1])

            st_uq, st_id = np.unique(state_arr[i_k,i_rn].reshape(ntraj, length)[mask], return_index=True)
            if len(st_uq) != k:
                print(f"Error: {len(st_uq)} states are found in k={k}, rn={rn}, div={ntraj_div/ntraj}")
                continue
            
            for i_lag, lag in enumerate(lag_ls): #100, 200, 400, 600, 800, 900
                noweight = np.ones((ntraj_div, length))
                noweight[:,-lag:]=0           
                qf_unq[i_k][i_rn, i_lag, i_div] = np.ravel(np.vstack(extq2.dga.forward_committor(basisF, noweight, inD_div, gf_div, lag)))[st_id]
                print(f"Done: k={k}, rn={rn}, div={ntraj_div/ntraj}, lag={lag}")
        print(f"Done: k={k}, rn={rn}, ({time()-t0:.2f} sec)")
        del basisL_arr, basisW, basisF

#with open(f"{workdir}/step2_k_div/qf_unq.npy", "wb") as f:
with open(f"{workdir}/step2_k_div/qf_unq_hk.npy", "wb") as f:
    dill.dump(qf_unq, f)