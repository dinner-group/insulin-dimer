import MDAnalysis as mda
import numpy as np

import argparse
import os, glob, sys, time
from multiprocessing import cpu_count
from joblib import Parallel, delayed

from MDAnalysis import transformations
from mda_ins_dim import *

import dill

def run_analysis(coord_file, *args, **kwargs):
    start_time = time.time()
    coord_dir, ref_file, parm_file, out_file, feat, extended, whole, jobid = args
    ref = mda.Universe(parm_file, ref_file)
    if extended|whole:
        univ = mda.Universe(parm_file, *coord_file)
        #Generate mask_5n to cover trajectory only from 2.5 to 5ns
        sel_id = np.arange(24*2+1)*500
        sel_range = np.vstack((sel_id[:-1], sel_id[1:])).T
        mask_5ns = np.zeros(24000, dtype=bool) #24 * 1000 = len(univ.trajectory) = # fo seed for each windows * 1000 frames (5ns)
        for i, f in sel_range[1::2]:
            mask_5ns[i:f] = True
    else:
        univ = mda.Universe(parm_file, coord_file)

    workflow = [
        transformations.unwrap(univ.atoms),
        #transformations.center_in_box(univ.select_atoms('protein'), center='geometry'),
        #transformations.wrap(univ.select_atoms('segid SOLV'), compound='residues'),
    ]
    univ.trajectory.add_transformations(*workflow)
    boxdim = univ.dimensions
    workflow = [transformations.boxdimensions.set_dimensions(boxdim), transformations.unwrap(ref.atoms)]
    ref.trajectory.add_transformations(*workflow)

    if feat == "distance":
        inst = make_dist(univ)
    elif feat == "BBagchi":
        inst = make_BBagchi(univ)
    elif feat == "BBagchiQ":
        inst = make_BBagchi(univ, Q=True)
    elif feat == "DEShaw":
        inst = make_DEShaw(univ, ref)
    elif feat == "angle":
        inst = make_angle(univ)
    elif feat == "angle_open":
        inst = make_angle(univ, opening=True)
    elif feat == "NativeContact":
        inst = make_NativeContact(univ, ref, method='soft_cut', beta=5.0, lambda_constant=2)
    elif feat == "IRMSD":
        inst = make_IRMSD(univ, ref)
    elif feat == "ISolv":
        inst = make_ISolv(univ, ref)
    elif feat == "HeavyContact":
        inst = make_HeavyContact(univ)
    elif feat == "hbridge":
        inst = make_hbridge(univ, order = 2, distance = 4.0, angle = 150.0)
    elif feat == "NonnatContact":
        inst = make_NonnatContact(univ)
    elif feat == "Euler":
        inst = make_Euler(univ)
    elif feat == "Allcontact":
        inst = make_Allcontact(univ)
    elif feat == "SolvAtom-atom":
        inst = make_SolvAtom(univ, resol='atom')
    elif feat == "SolvAtom-residue":
        inst = make_SolvAtom(univ, resol='residue')
    elif feat == "ZIP":
        inst = make_ZIP(univ)
    elif feat == "IntraContacts1":
        inst = make_Allcontact(univ, contact_type='intra1')
    elif feat == "IntraContacts2":
        inst = make_Allcontact(univ, contact_type='intra2')
    elif feat == "Detach":
        inst = make_detach(univ)
    elif feat == "GridDensity":
        inst = make_GridDensity(univ)
    elif feat == "LuisDisorder":
        inst = make_LuisDisorder(univ)
    else: 
        raise ValueError("-c or --cvs can only take distance, disroder, rotation, or hbridge")

    if isinstance(inst, tuple):
        result=[]
        for inst_tmp in inst:
            if extended:
                result_tmp = inst_tmp.run(frames=mask_5ns, **kwargs).results
                result.append(result_tmp)
            else:
                result_tmp = inst_tmp.run(**kwargs).results
                result.append(result_tmp)
    else:
        if extended:
            result = inst.run(frames=mask_5ns, **kwargs).results
        else:
            result = inst.run(**kwargs).results
    end_time = time.time()
    if extended|whole:
        print(f"Time taken for {coord_file[0].split('/')[-2:]} is {end_time - start_time}")
    else:
        print(f"Time taken for {coord_file.split('/')[-2:]} is {end_time - start_time}")
    return result

def parallel_run(coord_files, *args, **kwargs):
    return Parallel(n_jobs=cpu_count(), verbose=1)(delayed(run_analysis)(coord_file, *args, **kwargs) for coord_file in coord_files)


def main(
        coord_dir,
        ref_file,
        parm_file,
        out_file,
        feat,
        extended,
        whole,
        jobid
        ):
    ##Create Universe & Add Transformation
    args = coord_dir, ref_file, parm_file, out_file, feat, extended, whole, jobid
    if extended|whole:
        coord_files = sorted(glob.glob(f"{coord_dir}/*/*.dcd"))
        coord_files = np.array([file for file in coord_files if not '0000' in file]).reshape(28*28, 24)
    else:
        coord_files = sorted(glob.glob(f"{coord_dir}/*.dcd"))

    if jobid ==0:
        result = parallel_run(coord_files, *args)
    else:
        idx = jobid-1
        njob = 4 # Number of nodes to be used
        assert len(coord_files)%njob == 0 #"Number of files should be divisible by number of nodes"
        ntask_job = int(len(coord_files)/njob) # Number of tasks per each node
        result = parallel_run(coord_files[ntask_job*idx:ntask_job*(idx+1)], *args)

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
            choices=['distance', 'BBagchi', 'DEShaw', 'angle', 'angle_open', "NativeContact", "IRMSD", "ISolv", "HeavyContact", 
                     "hbridge", "NonnatContact", "Euler", "Allcontact", "Allcontact0", "Allcontact1", "Allcontact2", "Allcontact3", 
                     "SolvAtom-atom", "SolvAtom-residue", "ZIP",
                     "IntraContacts1", "IntraContacts2",
                     "BBagchiQ", "Detach",
                     "GridDensity",
                     "LuisDisorder"
                     ],
            default=None,
            help="What features to compute"
            )
    parser.add_argument(
            '-e',
            '--extended',
            action='store_true',
            help="If you're computing features from extended trjaectories"
            )
    parser.add_argument(
            '-w',
            '--whole',
            action='store_true',
            help="If you're computing features from the whole trjaectories"
            )
    parser.add_argument(
            '-i',
            '--jobid',
            type=int,
            default=0,
            help="If you're using jobarrays start the index from 1, \
                default is just running over whole trajectoreis"
            )
    args = parser.parse_args()
    main(
            args.coord_dir,
            args.ref_file,
            args.parm_file,
            args.out_file,
            args.cvs,
            args.extended,
            args.whole,
            args.jobid
            )
