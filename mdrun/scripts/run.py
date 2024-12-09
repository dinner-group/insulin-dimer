import sys
import os
import glob
import time
import argparse

import numpy as np

from openmm import *
from openmm.app import *
from openmm.unit import *

TEMPERATURE = 303.15 * kelvin
TIMESTEP = 0.002 * picoseconds


def prepare(
    coord_file,
    parm_dir,
    parm_file,
    platform,
    total_steps,
    save_interval,
    output,
    restart_file=None,
):

    gro = GromacsGroFile(f"{coord_file}")
    top = CharmmPsfFile(f"{parm_file}", periodicBoxVectors=gro.getPeriodicBoxVectors()) 
    pfiles = [f"{parm_dir}/{dt}" for dt in ["par_all36m_prot.prm", "top_all36_prot.rtf", "toppar_water_ions.str"]]
    params=CharmmParameterSet(*pfiles)
    system = top.createSystem(
        params,
        nonbondedMethod=PME,
        nonbondedCutoff=1.2*nanometer,
        constraints=HBonds,
        switchDistance = 1.0*nanometer,
        removeCMMotion = True,
        rigidWater = True
    )

    # Langevin dynamics with 1 ps^-1 friction coefficient and 2 fs timestep
    integrator = LangevinMiddleIntegrator(TEMPERATURE, 1/12 / picosecond, TIMESTEP)
    if platform == "cuda":
        platform = Platform.getPlatformByName("CUDA")
    elif platform == "cpu":
        platform = Platform.getPlatformByName("CPU")
    simulation = Simulation(top.topology, system, integrator, platform)
    
    if restart_file is None:
        simulation.context.setPositions(gro.positions)
        simulation.context.setVelocitiesToTemperature(TEMPERATURE)
        ifappend=False
    else:
        with open(restart_file, "rb") as f:
            simulation.context.loadCheckpoint(f.read())
        ifappend=True
        if output != restart_file[:-4]:
            ifappend=False

    # save trajectory
    simulation.reporters.append(
        DCDReporter(f"{output}.dcd", save_interval, append=ifappend)
    )
    simulation.reporters.append(
        StateDataReporter(
            f"{output}.log",
            reportInterval=save_interval * 100,
            step=True,
            time=True,
            progress=True,
            remainingTime=True,
            speed=True,
            elapsedTime=True,
            totalSteps=total_steps,
            separator="\t",
            append=ifappend
        )
    )
    simulation.reporters.append(
        CheckpointReporter(f"{output}.chk", save_interval * 100)
    )

    return simulation


def main(
    coord_file,
    parm_dir,
    parm_file,
    platform,
    total_steps,
    save_interval,
    output,
    restart_file,
):
    if not os.path.exists(f"{output[:-9]}"):
        os.mkdir(f"{output[:-9]}")
    if restart_file is not None:
        if not os.path.exists(restart_file):
            raise FileNotFoundError

    print(f"Simulation information")
    print("--------------------------------")
    print(f"OpenMM Version: {Platform.getOpenMMVersion()}")
    print(f"Input coordinates: {coord_file}")
    print(f"Parameter/topology: {parm_file}")
    print(f"Using {platform} platform")
    print(f"Output directory: {output[:-9]}")
    if restart_file is not None:
        print(f"Restarting, using {restart_file}")
    print("\n")
    print(f"Preparing simulation...")
    simulation = prepare(
        coord_file, parm_dir, parm_file, platform, total_steps, save_interval, output, restart_file=restart_file
    )
    run_steps = total_steps - simulation.context.getStepCount()
    print(
        f"Running simulation for {run_steps} * {TIMESTEP} = {run_steps * TIMESTEP}"
    )
    print(f"Saving every {save_interval} * {TIMESTEP} = {save_interval * TIMESTEP}")
    start_time = time.time()
    print(f"Started at {time.localtime(start_time)}")
    simulation.step(run_steps)
    simulation.saveCheckpoint(f"{output}.chk")
    end_time = time.time()
    print(f"Finished simulation at {time.localtime(end_time)}.")
    print(f"Simulation took {(end_time - start_time)//3600}h {((end_time - start_time)%3600)//60}m.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run simulation of insulin dimer")
    parser.add_argument("coord_file", type=str, help="Input coordinate file, must be gro type")
    parser.add_argument("parm_dir", type=str, help="prf, psf, tpr topology directory")
    parser.add_argument("parm_file", type=str, help=".psf topology file")
    parser.add_argument("-t", "--time", type=int, default=50000, help="Number of time steps")
    parser.add_argument(
        "-s",
        "--save",
        type=int,
        default=2500,
        help="Frequency to save information and check end",
    )
    parser.add_argument(
        "-p",
        "--platform",
        type=str,
        choices=["cpu", "cuda"],
        help="What hardware to use; must be 'cpu' or 'cuda'",
    )
    parser.add_argument(
        "-o", "--output", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "-r", "--restart", type=str, default=None, help="Restart checkpoint file"
    )
    args = parser.parse_args()

    main(
        args.coord_file,
        args.parm_dir,
        args.parm_file,
        args.platform,
        args.time,
        args.save,
        args.output,
        args.restart,
    )
