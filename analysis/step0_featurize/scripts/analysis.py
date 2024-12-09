import MDAnalysis as mda
import numpy as np

import argparse
import os, glob, sys
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from MDAnalysis import transformations
from mda_ins_dim import *

import dill

##Generate mask_5n to cover trajectory only from 2.5 to 5ns
#sel_id = np.arange(24*2+1)*500
#sel_range = np.vstack((sel_id[:-1], sel_id[1:])).T
#mask_5ns = np.zeros(24000, dtype=bool) #24 * 1000 = len(univ.trajectory) = # fo seed for each windows * 1000 frames (5ns)
#for i, f in sel_range[1::2]:
#    mask_5ns[i:f] = True

def run_analysis(an, **kwargs):
    an.run(verbose=True, **kwargs)# frames=mask_5ns, **kwargs)
    return an.results
def parallel_run(instance_list, **kwargs):
    return Parallel(n_jobs=cpu_count(), verbose=1)(delayed(run_analysis)(inst, **kwargs) for inst in instance_list)


def main(
        coord_dir,
        ref_file,
        parm_file,
        out_file,
        feat,
        extended
        ):
    ##Create Universe & Add Transformation
    if extended:
        coord_files = sorted(glob.glob(f"{coord_dir}/*/*.dcd"))
        #coord_files = np.array([file for file in coord_files if not '0000' in file]).reshape(28*28, 24)
        #univs = [mda.Universe(parm_file, *f) for f in coord_files[:2]]
        
        ##Generate mask_5n to cover trajectory only from 2.5 to 5ns
        #sel_id = np.arange(24*2+1)*500
        #sel_range = np.vstack((sel_id[:-1], sel_id[1:])).T
        #mask_5ns = np.zeros(24000, dtype=bool) #24 * 1000 = len(univ.trajectory) = # fo seed for each windows * 1000 frames (5ns)
        #for i, f in sel_range[1::2]:
        #    mask_5ns[i:f] = True

        coord_files = np.array([file for file in coord_files if not '0000' in file])#.reshape(28*28, 24)
        univs = [mda.Universe(parm_file, f) for f in coord_files]
        
        #Generate mask_5n to cover trajectory only from 2.5 to 5ns
        mask_5ns = np.zeros(1000, dtype=bool)
        mask_5ns[500:] = True

    else:
        coord_files = sorted(glob.glob(f"{coord_dir}/*.dcd"))
        univs = [mda.Universe(parm_file, f) for f in coord_files]
    
    ref = mda.Universe(parm_file, ref_file)

    for u in univs:
        workflow = [transformations.unwrap(u.atoms)]
        u.trajectory.add_transformations(*workflow)
    boxdim = u.dimensions
    workflow = [transformations.boxdimensions.set_dimensions(boxdim), transformations.unwrap(ref.atoms)]
    ref.trajectory.add_transformations(*workflow)

    if feat == "distance":
        inst = [make_dist(uni) for uni in univs]
    elif feat == "BBagchi":
        inst = [make_BBagchi(uni) for uni in univs]
    elif feat == "DEShaw":
        inst = [make_DEShaw(uni, ref) for uni in univs]
    elif feat == "angle":
        inst = [make_angle(uni) for uni in univs]
    elif feat == "angle_open":
        inst = [make_angle(uni, opening=True) for uni in univs]
    elif feat == "NativeContact":
        inst = [make_NativeContact(uni, ref, method='soft_cut', beta=5.0, lambda_constant=2) for uni in univs]
    elif feat == "IRMSD":
        inst = [make_IRMSD(uni, ref) for uni in univs]
    elif feat == "ISolv":
        inst = [make_ISolv(uni, ref) for uni in univs]
    elif feat == "HeavyContact":
        inst = [make_HeavyContact(uni) for uni in univs]
    elif feat == "hbridge":
        inst = [make_hbridge(uni, order = 2, distance = 4.0, angle = 150.0) for uni in univs]
    elif feat == "NonnatContact":
        inst = [make_NonnatContact(uni) for uni in univs]
    elif feat == "Euler":
        inst = [make_Euler(uni) for uni in univs]
    elif feat == "Allcontact0":#
        inst = [make_Allcontact(uni) for uni in univs[:392]]
    elif feat == "Allcontact1":#
        inst = [make_Allcontact(uni) for uni in univs[392:784]]
    #elif feat == "Allcontact2":#
    #    inst = [make_Allcontact(uni) for uni in univs[392:588]]
    #elif feat == "Allcontact3":#
    #    inst = [make_Allcontact(uni) for uni in univs[588:784]]        
    elif feat == "SolvAtom-atom":
        inst = [make_SolvAtom(u, resol='atom') for u in univs]
    elif feat == "SolvAtom-residue":
        inst = [make_SolvAtom(u, resol='residue') for u in univs]
    else: 
        raise ValueError("-c or --cvs can only take distance, disroder, rotation, or hbridge")
        
    if isinstance(inst[0], tuple):
        result=[]
        for i0 in range(len(inst[0])):
            inst_tmp = [dt[i0] for dt in inst]
            if extended:
                result.append(parallel_run(inst_tmp, frames=mask_5ns))
            else:
                result.append(parallel_run(inst_tmp))
    else:
        if extended:
            result = parallel_run(inst, frames=mask_5ns)
        else:
            result = parallel_run(inst)

    with open(f"{out_file}.pkl", 'wb') as f:
        dill.dump(result, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='mda_ins_dim')
    parser.add_argument('coord_dir', type=str, help="Directory containing *.dcd input coordinate files")
    parser.add_argument('ref_file', type=str, help="Input reference file .pdb")
    parser.add_argument('parm_file', type=str, help="Input topology file .psf")
    parser.add_argument(
            "-o",
            "--out_file",
            type=str,
            default=None,
            help="Output directory"
            )
    parser.add_argument(
            '-c', 
            '--cvs', 
            type=str,
            choices=['distance', 'BBagchi', 'DEShaw', 'angle', 'angle_open', "NativeContact", "IRMSD", "ISolv", "HeavyContact", "hbridge", "NonnatContact", "Euler", "Allcontact", "Allcontact0", "Allcontact1", "Allcontact2", "Allcontact3", "SolvAtom-atom", "SolvAtom-residue"],
            default=None,
            help="What features to compute"
            )
    parser.add_argument(
            '-e',
            '--extended',
            action='store_true',
            help="If you're computing features from extended trjaectories"
            )
    args = parser.parse_args()
    main(
            args.coord_dir,
            args.ref_file,
            args.parm_file,
            args.out_file,
            args.cvs,
            args.extended
            )
