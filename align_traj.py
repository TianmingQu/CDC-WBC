"""
align_full_pipeline.py

Combined script to perform two-step alignment on MD trajectories:
1) Align protein-only trajectory based on backbone atoms.
2) Align water-copied trajectory to the first processed frame.

Usage (command-line):
    python align_full_pipeline.py \
        <protein_pdb> <protein_dcd> <aligned_protein_pdb> <aligned_protein_dcd> \
        <water_reference_pdb> <water_trajectory_dcd> <water_output_directory> \
        [protein_max_frames] [water_max_frames]

Arguments:
    protein_pdb             : Path to the original protein PDB file
    protein_dcd             : Path to the original protein DCD trajectory
    aligned_protein_pdb     : Path to save the aligned protein-only PDB (first frame)
    aligned_protein_dcd     : Path to save the aligned protein-only DCD
    water_reference_pdb     : PDB file of the first processed frame (reference for water alignment)
    water_trajectory_dcd    : DCD trajectory file for water-copied alignment
    water_output_directory  : Directory where the aligned water trajectory will be saved
    protein_max_frames      : (Optional) Maximum frames for protein alignment (default: 2500)
    water_max_frames        : (Optional) Maximum frames for water alignment (default: all frames)
"""

import sys
import os
import time
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis import align as md_align


def align_protein(pdb_path: str,
                  dcd_path: str,
                  output_pdb: str,
                  output_dcd: str):
    """
    Align protein trajectory based on all heavy (non-hydrogen) atoms and save outputs.

    Parameters:
        pdb_path   : Path to the original protein PDB file
        dcd_path   : Path to the original DCD trajectory file
        output_pdb : Output path for aligned protein PDB (first frame)
        output_dcd : Output path for aligned DCD trajectory
    """
    start_time = time.perf_counter()
    print(f"[Protein Alignment] Started at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Load the full protein universe
    u = mda.Universe(pdb_path, dcd_path)

    # Select heavy atoms (non-hydrogen) for alignment
    heavy_selection = 'protein and not name H*'
    heavy_atoms = u.select_atoms(heavy_selection)

    # Align trajectory in memory using heavy atoms as reference and mobile
    aligner = md_align.AlignTraj(u, u, select=heavy_selection, in_memory=True)
    aligner.run()

    # Write the aligned trajectory with all atoms (including hydrogens)
    total_atoms = u.atoms.n_atoms
    with mda.Writer(output_dcd, total_atoms) as writer:
        for ts in u.trajectory:
            writer.write(u.atoms)

    # Save the first aligned frame as PDB (all atoms)
    u.atoms.write(output_pdb)

    elapsed = time.perf_counter() - start_time
    print(f"[Protein Alignment] Saved PDB: {output_pdb}")
    print(f"[Protein Alignment] Saved DCD: {output_dcd}")
    print(f"[Protein Alignment] Elapsed time: {elapsed:.2f} seconds\n")


def align_water_trajectory(reference_pdb: str,
                           trajectory_dcd: str,
                           output_directory: str,
                           max_frames: int = None):
    """
    Align water-copied trajectory to a reference and save aligned DCD.

    Parameters:
        reference_pdb     : Reference PDB for alignment
        trajectory_dcd    : Input DCD trajectory to align
        output_directory  : Directory for aligned DCD output
        max_frames        : Maximum frames to write (default: all)
    """
    start = time.perf_counter()
    print(f"[Water Alignment] Start at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(output_directory, exist_ok=True)
    u = mda.Universe(reference_pdb, trajectory_dcd)
    ref = mda.Universe(reference_pdb)

    # Backbone selection string
    sel = 'name CA or name C or name N or name O'
    mobile = u.select_atoms(sel)

    # Perform alignment
    aligner = md_align.AlignTraj(u, ref, select=sel, in_memory=True)
    aligner.run()

    out_path = os.path.join(output_directory, 'aligned_water_copied_trajectory.dcd')
    with mda.Writer(out_path, u.atoms.n_atoms) as w:
        for idx, ts in enumerate(u.trajectory):
            if max_frames is not None and idx >= max_frames:
                break
            w.write(u.atoms)

    elapsed = time.perf_counter() - start
    print(f"[Water Alignment] Saved aligned trajectory: {out_path}")
    print(f"[Water Alignment] Elapsed: {elapsed:.2f} s\n")


if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) not in (7, 8, 9):
        print("Usage:\n"
              "  python align_full_pipeline.py "
              "<protein_pdb> <protein_dcd> <aligned_protein_pdb> <aligned_protein_dcd> "
              "<water_reference_pdb> <water_trajectory_dcd> <water_output_dir> "
              "[protein_max_frames] [water_max_frames]")
        sys.exit(1)

    (pdb_in, dcd_in, pdb_out, dcd_out,
     water_ref, water_traj, water_out) = args[:7]

    prot_max = int(args[7]) if len(args) >= 8 else 2500
    water_max = int(args[8]) if len(args) == 9 else None

    # Step 1: Protein alignment
    align_protein(
        pdb_path=pdb_in,
        dcd_path=dcd_in,
        output_pdb=pdb_out,
        output_dcd=dcd_out,
        max_frames=prot_max
    )

    # Step 2: Water trajectory alignment
    align_water_trajectory(
        reference_pdb=water_ref,
        trajectory_dcd=water_traj,
        output_directory=water_out,
        max_frames=water_max
    )
