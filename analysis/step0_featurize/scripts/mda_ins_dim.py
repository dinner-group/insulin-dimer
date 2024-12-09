import numpy as np
import MDAnalysis as mda

from MDAnalysis.analysis.rms import RMSD, rmsd
from MDAnalysis.analysis.contacts import Contacts

from MDAnalysis.lib.distances import distance_array, self_distance_array, calc_angles, calc_dihedrals, calc_bonds, capped_distance
from MDAnalysis.analysis.base import AnalysisBase, AnalysisFromFunction
from MDAnalysis.analysis.hydrogenbonds import WaterBridgeAnalysis

# Defining Class
"""
1. Distance:
    def def_dist(uni):
    def make_dist(uni):
2. Disorder:
    def def_disorder(uni):
    def make_disorder(uni, ref):
3. Reltaive Rotation:
    class rel_rot(AnalysisBase):
    def def_relrot(uni):
    def make_relrot(uni):
"""
        
chainA = ["PROA", "PROC"]
chainB = ["PROB", "PROD"]

def def_dist(uni):
    inters = []; intras = []
    for chB in chainB:
        inters.append(uni.select_atoms(f"segid {chB} and name CA and resid 9 12 13 16 21 23 24 25 26 29"))
    return inters

def make_dist(uni):
    inters = def_dist(uni)
    return AnalysisFromFunction(distance_array, uni.trajectory, inters[0], inters[1], box=uni.dimensions, backend="OpenMP")

def RCOM_BBagchi(ag1, ag2, **kwargs):
    dist=calc_bonds(ag1.center_of_mass(), ag2.center_of_mass(),**kwargs)
    return dist

def Contact_BBagchi(ag1, ag2, **kwargs):
    dists = distance_array(ag1, ag2, **kwargs)
    r6 = (dists/7)**6
    r12 = r6**2
    return ((1-r6)/(1-r12)).sum()

def make_BBagchi(uni, Q=False): 
    #R_COM: Distnace between center of mass
    ch1=uni.select_atoms(f"segid PROA PROB")
    ch2=uni.select_atoms(f"segid PROC PROD")
    RCOM_inst = AnalysisFromFunction(RCOM_BBagchi, uni.trajectory, ch1, ch2, box=ch1.dimensions, backend='OpenMP')
    
    #N: The nubmer of cross-contact
    ch1_ca=uni.select_atoms(f"name CA and segid PROA PROB")
    ch2_ca=uni.select_atoms(f"name CA and segid PROC PROD")
    N_inst = AnalysisFromFunction(Contact_BBagchi, uni.trajectory, ch1_ca, ch2_ca, box=ch1_ca.dimensions, backend='OpenMP')
    if Q:
        return N_inst
    else:
        return RCOM_inst, N_inst


def make_IRMSD(uni, ref):
    #1. I-RMSD
    ch1_ca=ref.select_atoms(f"name CA and segid PROA PROB")
    ch2_ca=ref.select_atoms(f"name CA and segid PROC PROD")
    pairs_ca = capped_distance(ch1_ca, ch2_ca, max_cutoff=10.0, box=ref.dimensions, return_distances=False)
    
    interCA_ch1=np.unique(pairs_ca[:,0])+1 #1~51
    interCA_ch2=np.unique(pairs_ca[:,1])+1 #1~51

    interface_CAs_str = f"name CA and\
    ((segid PROA and resid {' '.join(str(x) for x in interCA_ch1[interCA_ch1<=21])})\
     or (segid PROB and resid {' '.join(str(x) for x in interCA_ch1[interCA_ch1>21]-21)})\
     or (segid PROC and resid {' '.join(str(x) for x in interCA_ch2[interCA_ch2<=21])})\
     or (segid PROD and resid {' '.join(str(x) for x in interCA_ch2[interCA_ch2>21]-21)})\
    )"
    
    IRMSD_inst = RMSD(uni, ref, select=interface_CAs_str)
    return IRMSD_inst
"""
def make_NativeContact(uni, ref):
    #2. Fractinal Native Contacts
    ch1_heavy_str = f'segid PROA PROB and not (same mass as (name HA))'
    ch2_heavy_str = f'segid PROC PROD and not (same mass as (name HA))'
    
    sel_ag1, sel_ag2 = uni.select_atoms(ch1_heavy_str), uni.select_atoms(ch2_heavy_str)
    ref_ag1, ref_ag2 = ref.select_atoms(ch1_heavy_str), ref.select_atoms(ch2_heavy_str)
    
    NC_inst = Contacts(uni,\
                       select=(sel_ag1, sel_ag2),\
                       refgroup=(ref_ag1, ref_ag2),\
                       radius=4.5, pbc=True,\
                       method='soft_cut', beta=5.0, lambda_constant=2)
    return NC_inst
"""
def hydration_DEShaw(ags, water_o, **kwargs):
    results=np.zeros((len(ags), 2), dtype=int)
    for i0, ag in enumerate(ags):
        cri_tmp = len(ag.select_atoms(f"segid PROA PROB"))
        water_distance = capped_distance(ag, water_o, **kwargs) # (n_pairs, 2) pairs of indices
        wd_uq, wd_cnt = np.unique(water_distance[:, 1], return_counts=True)
        nw_intersect_cnt = 0
        for w_id_tmp in wd_uq[wd_cnt>1]: # We count # of waters between two chains so wd_cnt should be at least 2
            hb_res_tmp = water_distance[water_distance[:, 1]==w_id_tmp, 0]
            if ((hb_res_tmp[0]<cri_tmp) and (hb_res_tmp[-1]>=cri_tmp)):
                nw_intersect_cnt+=1
        results[i0] = len(wd_uq), nw_intersect_cnt
    return results

def make_ISolv(uni, ref):
    #3. Fractional Hydration
    ch1_heavy_str = f'segid PROA PROB and not (same mass as (name HA))'
    ch2_heavy_str = f'segid PROC PROD and not (same mass as (name HA))'
    sel_ag1_h, sel_ag2_h = uni.select_atoms(ch1_heavy_str), uni.select_atoms(ch2_heavy_str)
    ref_ag1_h, ref_ag2_h = ref.select_atoms(ch1_heavy_str), ref.select_atoms(ch2_heavy_str)
    water_o = uni.select_atoms(f"segid SOLV and name OH2")
    
    pairs_heavy = capped_distance(ref_ag1_h, ref_ag2_h, max_cutoff=4, box=ref_ag1_h.dimensions, return_distances=False)
    
    ch1_inter_HB = sel_ag1_h[np.unique(pairs_heavy[:, 0])]
    ch2_inter_HB = sel_ag2_h[np.unique(pairs_heavy[:, 1])]

    solv_inter_HB = ch1_inter_HB+ch2_inter_HB
    solvated_ags = [solv_inter_HB,
                    solv_inter_HB.select_atoms("segid PROB PROD and resid 9 to 19"),
                    solv_inter_HB.select_atoms("segid PROB PROD and resid 23 to 27")]
    Hydr_inst = AnalysisFromFunction(hydration_DEShaw, uni.trajectory, solvated_ags, water_o, max_cutoff=4, box=ch1_inter_HB.dimensions, return_distances=False)
    return Hydr_inst

def N_Heavy(ag1, ag2, **kwargs):
    return len(capped_distance(ag1, ag2, **kwargs))
def make_HeavyContact(uni):
    #N: The nubmer of cross-contact
    ch1_heavy = uni.select_atoms(f'segid PROA PROB and not (same mass as (name HA))')
    ch2_heavy = uni.select_atoms(f'segid PROC PROD and not (same mass as (name HA))')
    HeavyC_inst = AnalysisFromFunction(N_Heavy, uni.trajectory, ch1_heavy, ch2_heavy, max_cutoff=4.5, box=ch1_heavy.dimensions, return_distances=False)
    return HeavyC_inst

def make_DEShaw(uni, ref):    
    IRMSD_inst = make_IRMSD(uni, ref)
    NC_inst = make_NativeContact(uni, ref)
    Hydr_inst = make_ISolv(uni, ref)
    return IRMSD_inst, NC_inst, Hydr_inst


def angle_calc(ags, **kwargs):
    npoint=len(ags)
    nangle=len(ags[0])
    coord = np.zeros((npoint, nangle, 3))
    for i0, ag_point in enumerate(ags):
        for i1, ag in enumerate(ag_point):
            coord[i0, i1]=ag.positions if len(ag)==1 else ag.center_of_geometry()
    return calc_angles(*coord, **kwargs) if npoint==3 else calc_dihedrals(*coord, **kwargs)


def def_angle(uni):
    #plumed.dat /beagle3/dinner/kjeong/insulin_backedup/NVT_US_insplumed.dat
    pnts_psi_a = [['17-19', '9-11','9-11', '17-19'], ['9-11', '12-16', '12-16', '9-11']] # geometric center of 'backbone'
    pnts_psi_b = [['24', '26', '24', '26'], ['26', '25', '25', '26']]
    pnts_phi_d = [['13', '20', '24'], ['20', '24', '25'], ['24', '25', '26'], ['25', '26', '29']]

    psi_ags=[[],[],[],[]]# Four (Adam alpha, Me alpha, Adam beta, Me beta) four points angle

    phi_ags=[[], [], []]# Eight (13-20-24 ch1, .. ch2, 20-24-25 ch1, .. ch2, ...)three points angle in 

    for pnts_psi in pnts_psi_a+pnts_psi_b: #Four dihedrals
        for i0, pnt in enumerate(pnts_psi): #Four points for each dihedral
            backbone = 'backbone' if len(pnt) > 2 else 'name CA'
            segID = 'PROB' if i0 <2 else 'PROD'
            str_tmp = f"segid {segID} and resid {pnt} and {backbone}"
            psi_ags[i0].append(uni.select_atoms(str_tmp))

    for pnts_phi in pnts_phi_d:
        for i0, pnt in enumerate(pnts_phi):
            for segID in ['PROB', 'PROD']:
                str_tmp = f"segid {segID} and resid {pnt} and name CA"
                phi_ags[i0].append(uni.select_atoms(str_tmp))
    return psi_ags, phi_ags

def def_angle_open(uni):
    pnts_psi_open = ['12-16', '25', '25', '12-16']
    psi_ags=[[],[],[],[]]
    phi_ags=[[], [], []]

    for i0, pnt in enumerate(pnts_psi_open): #Four points for each dihedral
        backbone = 'backbone' if len(pnt) > 2 else 'name CA'
        segID = 'PROB' if i0 <2 else 'PROD'
        str_tmp = f"segid {segID} and resid {pnt} and {backbone}"
        psi_ags[i0].append(uni.select_atoms(str_tmp))
    
    for i1, pnts_phi in enumerate([pnts_psi_open[:-1], pnts_psi_open[1:]]):
        segID_tmp = np.array(chainB)[[0, i1, 1]]
        for i0, pnt in enumerate(pnts_phi):
            backbone = 'backbone' if len(pnt) > 2 else 'name CA'
            segID = segID_tmp[i0]            
            str_tmp = f"segid {segID} and resid {pnt} and {backbone}"
            phi_ags[i0].append(uni.select_atoms(str_tmp))
    return psi_ags, phi_ags

def make_angle(uni, opening=False):
    ags_four, ags_three = def_angle_open(uni) if opening else def_angle(uni)
    Four_inst = AnalysisFromFunction(angle_calc, uni.trajectory, ags_four, box=uni.dimensions, backend='OpenMP')
    Three_inst = AnalysisFromFunction(angle_calc, uni.trajectory, ags_three, box=uni.dimensions, backend='OpenMP')
    return Four_inst, Three_inst

def make_hbridge(uni, order = 3, distance = 3.0, angle = 120.0):
    chA_str= f"segid PROB and resid 24 to 26 and name N O"
    chB_str = f"segid PROD and resid 24 to 26 and name N O"
    wt_str = f"segid SOLV"
    hba = WaterBridgeAnalysis(
            uni, selection1 = chA_str, selection2 = chB_str, water_selection = wt_str,
            pbc=True,
            order = order, distance = distance, angle = angle)
    return hba

def nonnat_contact(ag1_set, ag2_set, **kwargs):
    
    assert len(ag1_set) == len(ag2_set)
    length=len(ag1_set)
    result = np.zeros(length, dtype=int)
    for i0, (ag1, ag2) in enumerate(zip(ag1_set, ag2_set)):
        result[i0]=len(capped_distance(ag1, ag2, **kwargs))
    return result

residue_pairs=np.array([
    [ 9, 13],#Alpha helix
    [13,  9],
    [24, 26],
    [26, 24],
    [12, 24],
    [24, 12],
    [16, 26],
    [26, 16],
    [21, 29],
    [29, 21],
    [ 9,  9],
    [ 9, 21],
    [21,  9],
    [16, 29],
    [29, 16],
    [13, 29],
    [29, 13],
    [ 9, 29],
    [29,  9],
    [13, 26],
    [26, 13],
    [ 9, 26],
    [26,  9]
    ])
def make_NonnatContact(uni):
    #Interest of residue
    #N: The nubmer of cross-contact
    ch1_heavy_set, ch2_heavy_set = [], []
    for resid1, resid2 in residue_pairs:
        ch1_heavy_set.append(uni.select_atoms(f'segid PROB and not (same mass as (name HA)) and resid {resid1}'))
        ch2_heavy_set.append(uni.select_atoms(f'segid PROD and not (same mass as (name HA)) and resid {resid2}'))
    NonNatC_inst = AnalysisFromFunction(nonnat_contact, uni.trajectory, ch1_heavy_set, ch2_heavy_set, max_cutoff=7.0, box=uni.dimensions, return_distances=False)
    return NonNatC_inst
#####################

def euler_angle(ag_sets, **kwargs):
    coords = np.vstack([ag_set.center_of_mass() for ag_set in ag_sets])
    #00, 01, 02, 03, 04, 05
    #P1, P2, P3, L1, L2, L3
    #P1, L1
    r_com = calc_bonds(coords[0], coords[3], **kwargs)
    #theta: (L1-P1-P2); Theta: (P1-L1-L2)
    theta, Theta = calc_angles(coords[[3, 0]], coords[[0, 3]], coords[[1, 4]], **kwargs)
    #phi: (L1-P1-P2-P3); PHI: (P1-L1-L2-L3) PSI: (P2-P1-L1-L2)
    phi, Phi, Psi = calc_dihedrals(coords[[3, 0, 1]], coords[[0, 3, 0]], coords[[1, 4, 3]], coords[[2, 5, 4]], **kwargs)
    return r_com, theta, phi, Theta, Phi, Psi
def make_Euler(uni):
    strs = [" and segid PROA PROB", " and segid PROA and resid 13 to 19", " and segid PROB and resid 9 to 18",\
            " and segid PROC PROD", " and segid PROC and resid 13 to 19", " and segid PROD and resid 9 to 18"]
    point_strs=np.hstack([np.array([f"name C{str_tmp}" for str_tmp in strs])])
    ag_sets = [uni.select_atoms(pnt_str) for pnt_str in point_strs]
    Euler_inst = AnalysisFromFunction(euler_angle, uni.trajectory, ag_sets, box=uni.dimensions, backend='OpenMP')
    return Euler_inst

#####Heavy Contact (All = Native + Non-native)
#405 is the number of heavy atoms in each monomer
from scipy import sparse
def N_Allcontact(ag1, ag2, **kwargs):
    dist_tmp = capped_distance(ag1, ag2, **kwargs)
    #result_bool = np.zeros((405, 405), dtype=bool)
    #result_bool[(dist_tmp[:, 0], dist_tmp[:, 1])] = True
    d_csr, r_csr, c_csr = np.ones(len(dist_tmp), dtype=bool), *(dist_tmp.T)
    bool_sps_arr = sparse.csr_array((d_csr, (r_csr, c_csr)), shape=(405, 405), dtype=bool)
    return bool_sps_arr
def make_Allcontact(uni, contact_type='inter'):
    #N: The nubmer of cross-contact
    ch1_heavy = uni.select_atoms(f'segid PROA PROB and not (same mass as (name HA))')
    ch2_heavy = uni.select_atoms(f'segid PROC PROD and not (same mass as (name HA))')
    if contact_type == 'inter':
        pairs = (ch1_heavy, ch2_heavy)
    elif contact_type == 'intra1':
        pairs = (ch1_heavy, ch1_heavy)
    elif contact_type == 'intra2':
        pairs = (ch2_heavy, ch2_heavy)
    else: 
        raise ValueError("contact_type should be either 'inter', 'intra1', or 'intra2'")
    AllC_inst = AnalysisFromFunction(N_Allcontact, uni.trajectory, *pairs, max_cutoff=4.5, box=ch1_heavy.dimensions, return_distances=False)
    return AllC_inst

### Atom-wise Redidue-wise Solvation
def hydration_atom(ags, water_o, resol, **kwargs):
    water_distance = capped_distance(ags, water_o, **kwargs) # (n_pairs, 2) pairs of indices
    if resol == 'atom':
        results = np.zeros(len(ags), dtype=np.uint8)
    elif resol == 'residue':            
        results = np.zeros(len(ags.residues), dtype=np.uint8)
        atom2res_id = np.copy(ags.resids) - 1
        PROBid = ags.segids == "PROB"
        PROCid = ags.segids == "PROC"
        PRODid = ags.segids == "PROD"
        atom2res_id[PROBid] += 21
        atom2res_id[PROCid] += 21+30
        atom2res_id[PRODid] += 21+30+21
        water_distance[:, 0] = atom2res_id[water_distance[:, 0]]
        water_distance = np.unique(water_distance, axis=0)
    else:
        raise ValueError("resol should be either 'atom' or 'residue'")
    wd_uq, wd_cnt = np.unique(water_distance[:, 0], return_counts=True)
    results[wd_uq] = wd_cnt
    return results

def make_SolvAtom(univ, resol='atom'):
    """AnalysisBase
    """
    #select heavy atoms
    heavy = univ.select_atoms("protein and not (same mass as (name HA))")
    water_o = univ.select_atoms(f"segid SOLV and name OH2")
    nwater_inst = AnalysisFromFunction(hydration_atom, univ.trajectory, heavy, water_o, resol, max_cutoff=4, box=heavy.dimensions, return_distances=False)
    return nwater_inst

def zip_calc(ags, **kwargs):
    npoint=3
    nangle=len(ags[0]) # 4 angles (PROB, 21), (PROB, 8 29), (PROD, 21), (PROD, 8 29)
    coord = np.zeros((npoint, nangle, 3))
    for i1 in range(nangle): #angle to define
        coord[0, i1]=(ags[0][i1].center_of_geometry() + ags[1][i1].positions) / 2
        coord[1, i1]=(ags[2][i1].center_of_geometry() + ags[3][i1].positions) / 2
        coord[2, i1]=ags[4][i1].positions if len(ags[4][i1])==1 else ags[4][i1].center_of_geometry()
    return calc_angles(*coord, **kwargs)

def make_ZIP(uni):
    # Def 
    zip_ags = [[], [], [], [], []]
    for i0 in range(2):#PROB PROD
        segID_tmp = np.array(chainB)[[i0-1, i0-1, i0, i0, i0]]
        for res_tmp in ["21", "8 29"]:
            for i1, (seg_ID, res_ID, backbone) in enumerate(zip(segID_tmp, 
                                                                ["12-16", "25", "12-16", "25", res_tmp],
                                                                ["backbone", "name CA", "backbone", "name CA", "name CA"])):
                str_tmp = f"segid {seg_ID} and resid {res_ID} and {backbone}"
                zip_ags[i1].append(uni.select_atoms(str_tmp))
    zip_inst = AnalysisFromFunction(zip_calc, uni.trajectory, zip_ags, box=uni.dimensions, backend='OpenMP')
    return zip_inst

def detach_calc(ags, **kwargs):
    npoint=3
    nangle=len(ags[0]) # 2 angles ch1, ch2
    coord = np.zeros((npoint, nangle, 3))
    for i0, ag_point in enumerate(ags):
        for i1, ag in enumerate(ag_point):
            coord[i0, i1]=ag.positions
    return calc_angles(*coord, **kwargs)

def make_detach(uni):
    detach_ags = [[], [], []]
    for seg_tmp in ["PROB", "PROD"]:
        for i1, res_tmp in enumerate(["22", "24", "26"]):
            str_tmp = f"segid {seg_tmp} and resid {res_tmp} and name CA"
            detach_ags[i1].append(uni.select_atoms(str_tmp))
    detach_inst = AnalysisFromFunction(detach_calc, uni.trajectory, detach_ags, box=uni.dimensions, backend='OpenMP')
    return detach_inst

def grid_density(ags, ch1, ch2, **kwargs):
    arr = np.zeros((3, 10), dtype=int)
    box = kwargs['box'][:3]
    segids = ags.segids
    ch1_mask = np.isin(segids, ['PROA', 'PROB'])
    ch2_mask = np.isin(segids, ['PROC', 'PROD'])
    water_mask = np.isin(segids, ['SOLV'])
    type_mask = np.vstack((ch1_mask, ch2_mask, water_mask)).T

    vecs = ags.positions
    com_c1 = ch1.center_of_mass()
    com_c2 = ch2.center_of_mass()
    z_vec = com_c2 - com_c1

    #Move chain2 to close to chain1 (minimum image convention)
    mask_out = np.abs(z_vec) > box/2
    com_c2[mask_out] -= np.sign(z_vec[mask_out])*box[mask_out]
    
    #Center of mass and displacement vector between center of mass of each chain 
    center = (com_c1 + com_c2) / 2
    z_vec = com_c2 - com_c1

    #Translate the whole system by the vector to 'the center of box' from 'the center of mass'
    vecs += box/2-center
    #Wrapping
    vecs = vecs % box
    #Then move the whole system back such that the origin of coordinate becomes the center of mass again.
    vecs -= box/2


    #z_vec = ch2.center_of_mass() - ch1.center_of_mass()
    z_uni = z_vec / np.linalg.norm(z_vec)

    #center = (ch1.center_of_mass() + ch2.center_of_mass()) / 2
    #vecs = ags.positions - center
    z_mag = np.dot(vecs, z_uni)

    mask_z = (z_mag < 20) & (z_mag > -20)# -2nm ~ +2nm
    z_mag_trunc = z_mag[mask_z]

    vecs_z = np.matmul(z_mag_trunc[:, np.newaxis], z_uni[np.newaxis, :])
    r2_mag = np.sum((vecs[mask_z]-vecs_z)**2, axis=1)
    mask_r2 = r2_mag < 25
    for i0 in range(3):
        arr[i0] = np.histogram(z_mag_trunc[mask_r2][type_mask[mask_z][mask_r2][:, i0]], bins=np.linspace(-20, 20, 11))[0]
    return arr

def make_GridDensity(uni):
    ags = uni.select_atoms('not (same mass as (name HA))')
    ch1 = uni.select_atoms('segid PROA PROB')
    ch2 = uni.select_atoms('segid PROC PROD')
    grid_inst = AnalysisFromFunction(grid_density, uni.trajectory, ags, ch1, ch2, box=uni.dimensions)
    return grid_inst


#Retrieved from GitHub Plumed ([https://github.com/plumed/plumed2/blob/master/src/secondarystructure/AlphaRMSD.cpp])
ref_helix = np.array([
( 0.733,  0.519,  5.298 ),#; // N    i
( 1.763,  0.810,  4.301 ),#; // CA
( 3.166,  0.543,  4.881 ),#; // CB
( 1.527, -0.045,  3.053 ),#; // C
( 1.646,  0.436,  1.928 ),#; // O
( 1.180, -1.312,  3.254 ),#; // N    i+1
( 0.924, -2.203,  2.126 ),#; // CA
( 0.650, -3.626,  2.626 ),#; // CB
(-0.239, -1.711,  1.261 ),#; // C
(-0.190, -1.815,  0.032 ),#; // O
(-1.280, -1.172,  1.891 ),#; // N    i+2
(-2.416, -0.661,  1.127 ),#; // CA
(-3.548, -0.217,  2.056 ),#; // CB
(-1.964,  0.529,  0.276 ),#; // C
(-2.364,  0.659, -0.880 ),#; // O
(-1.130,  1.391,  0.856 ),#; // N    i+3
(-0.620,  2.565,  0.148 ),#; // CA
( 0.228,  3.439,  1.077 ),#; // CB
( 0.231,  2.129, -1.032 ),#; // C
( 0.179,  2.733, -2.099 ),#; // O
( 1.028,  1.084, -0.833 ),#; // N    i+4
( 1.872,  0.593, -1.919 ),#; // CA
( 2.850, -0.462, -1.397 ),#; // CB
( 1.020,  0.020, -3.049 ),#; // C
( 1.317,  0.227, -4.224 ),#; // O
(-0.051, -0.684, -2.696 ),#; // N    i+5
(-0.927, -1.261, -3.713 ),#; // CA
(-1.933, -2.219, -3.074 ),#; // CB
(-1.663, -0.171, -4.475 ),#; // C
(-1.916, -0.296, -5.673 )#; // O
])

def alpha_rmsd(ags_arr, id_noCB_ls):
    results = np.zeros(len(ags_arr), dtype=float)
    for i0, (ags, id_noCB) in enumerate(zip(ags_arr, id_noCB_ls)):
        mask_ref = np.ones(len(ref_helix), dtype=bool)
        mask_ref[id_noCB*5+2] = False
        r_rmsd = rmsd(ref_helix[mask_ref], ags.positions, center=True, superposition=True)
        results[i0] = (1-(r_rmsd/0.8)**8) / (1-(r_rmsd/0.8)**12)
    return results

def bond_calc(ags, **kwargs):
    #Return: (2) chB: CA20-CA23, chD: CA20-CA23
    npoint=len(ags)
    ndist=len(ags[0])
    coord = np.zeros((npoint, ndist, 3))
    for i0, ag_point in enumerate(ags):
        for i1, ag in enumerate(ag_point):
            coord[i0, i1]=ag.positions
    return calc_bonds(*coord, **kwargs)

def calc_Luis(ags_p4_arr, ags_p3_arr, ags_p2_arr, ags_rmsd_arr, id_noCB_ls, **kwargs):
    results_4p = angle_calc(ags_p4_arr, **kwargs) # (10)
    results_3p = angle_calc(ags_p3_arr, **kwargs) #Return: (4) B1B7_detach, D1D7_detach, B25B30_detach, D25D30_detach
    results_2p = bond_calc(ags_p2_arr, **kwargs) #Return: (2) chB: CA20-CA23, chD: CA20-CA23
    results_rmsd = alpha_rmsd(ags_rmsd_arr, id_noCB_ls) # (14)
    return np.hstack((results_4p, results_3p, results_2p, results_rmsd))

def make_LuisDisorder(uni):
    """
    4p, 3p, alpha_rmsd
    """
    # 1 Define strings for selection of dihedral angles (angles_4_ext) and three points angles (angles_3_ext)
    a_rmsd = [
        ['B', '9'], #B9-B14 melting : 1
        ['B', '18'], #B Helix extension
        ['B', '22'],#B Helix extension
        ['A', '1'], #AN-helix melting: 4
        ['A', '2'],
        ['A', '3'],
        ['A', '4'], 
    ]
    
    a_rmsd_ext = np.repeat(a_rmsd, 2, axis=0)
    maskA = a_rmsd_ext[1::2, 0] == 'A'
    maskB = ~maskA
    a_rmsd_ext[1::2, 0][maskA] = 'C'
    a_rmsd_ext[1::2, 0][maskB] = 'D'

    angles_4 = np.array([
        [['B', '3'], ['A', '15'], ['B', '18'], ['B', '15']], #B1-B7 flippling
        [['A', '13'], ['A', '18'], ['B', '19'], ['B', '25']], #B20-B30 detachment
        [['A', '11'], ['A', '14'], ['B', '19'], ['B', '16']]#B-helix rotation
    ])

    angles_3 = [
        [['B', '2'], ['B', '9'], ['B', '20']], #B1-B7 detachment
        [['B', '16'], ['B', '20'], ['B', '27']], #B25-B30 detachment
    ]

    ##Extend to two monomers
    angles_4_ext = np.repeat(angles_4, 2, axis=0)
    angles_3_ext = np.repeat(angles_3, 2, axis=0)

    for angles_ext in [angles_4_ext, angles_3_ext]:
        maskA = angles_ext[1::2, :, 0] == 'A'
        maskB = ~maskA
        angles_ext[1::2, :, 0][maskA] = 'C'
        angles_ext[1::2, :, 0][maskB] = 'D'

    # 2 Define atomgroups for dihedral angles and three points angles
    ags_rmsd_arr = np.zeros(len(a_rmsd_ext), dtype=object)
    id_noCB_ls = []
    for i0, points in enumerate(a_rmsd_ext):
        ags_rmsd_arr[i0] = uni.select_atoms(f"(backbone or name CB) and segid PRO{points[0]} and resid {points[1]}:{int(points[1])+5}")
        CB_id = ags_rmsd_arr[i0].select_atoms("name CB").resids-int(points[1])
        mask_CB = np.ones(6)
        mask_CB[CB_id] = 0
        id_noCB_ls.append(np.nonzero(mask_CB)[0])
    ## 2.1 Dihedral angles
    ## 2.1.1 Dihedral angles
    nangle, npoint = angles_4_ext.shape[:2]
    ags_p4_arr = np.zeros((npoint, nangle+4), dtype=object)
    for i0, points in enumerate(angles_4_ext):
        for i1, point in enumerate(points):
            ags_p4_arr[i1, i0] = uni.select_atoms(f"name CA and segid PRO{point[0]} and resid {point[1]}")

    ## 2.1.2 Additional points for defining phi angles for 21 23
    cnt = nangle
    for i0, phi_res in enumerate([21, 23]):
        for i1, chid in enumerate(['B', 'D']):
            ags_p4_arr[0, cnt] = uni.select_atoms(f"name C and segid PRO{chid} and resid {phi_res-1}")
            for i2, atom_tmp in enumerate(['N', 'CA', 'C'], 1):
                ags_p4_arr[i2, cnt] = uni.select_atoms(f"name {atom_tmp} and segid PRO{chid} and resid {phi_res}")
            cnt += 1
    ## 2.2 Three points angles
    nangle, npoint = angles_3_ext.shape[:2]
    ags_p3_arr = np.zeros((npoint, nangle), dtype=object)
    for i0, points in enumerate(angles_3_ext):
        for i1, point in enumerate(points):
            ags_p3_arr[i1, i0] = uni.select_atoms(f"name CA and segid PRO{point[0]} and resid {point[1]}")

    ## 2.3 Two points distances
    ndist, npoint = 2, 2
    ags_p2_arr = np.zeros((npoint, ndist), dtype=object)
    ags_p2_arr[0, 0] = uni.select_atoms("name CA and segid PROB and resid 20")
    ags_p2_arr[1, 0] = uni.select_atoms("name CA and segid PROB and resid 23")
    ags_p2_arr[0, 1] = uni.select_atoms("name CA and segid PROD and resid 20")
    ags_p2_arr[1, 1] = uni.select_atoms("name CA and segid PROD and resid 23")

    return AnalysisFromFunction(calc_Luis, uni.trajectory, ags_p4_arr, ags_p3_arr, ags_p2_arr, ags_rmsd_arr, id_noCB_ls, box=uni.dimensions, backend='OpenMP')