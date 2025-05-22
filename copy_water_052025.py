"""
copy_water.py

Module for copying water oxygen atoms in an MD trajectory and saving processed files.

Usage (command-line):
    python copy_water.py <protein_pdb> <unaligned_dcd> <output_dir> <temp_dir> [max_frames]

    - protein_pdb      : Path to the original protein PDB file
    - unaligned_dcd    : Path to the original unaligned DCD trajectory
    - output_dir       : Directory where final PDB and DCD files will be saved
    - temp_dir         : Directory for temporary intermediate files
    - max_frames (opt) : Maximum number of frames to process (default: 2500)
"""

import os
import warnings
import numpy as np
import MDAnalysis as mda
from tqdm.autonotebook import tqdm as tq

warnings.filterwarnings("ignore")

def copy_water(protein_pdb: str,
               protein_unaligned_dcd: str,
               output_directory: str,
               temp_directory: str,
               max_frames: int = 2500,
               frame_stride: int = 1):
    """
    Process a trajectory by selecting protein and water oxygen atoms,
    copying water molecules into six directions, and writing new files.

    Parameters:
        protein_pdb: Path to the original protein PDB file.
        protein_unaligned_dcd: Path to the original unaligned DCD trajectory.
        output_directory: Directory where final PDB and DCD files will be saved.
        temp_directory: Directory for temporary intermediate files.
        max_frames: Maximum number of frames to process (default: 2500).
        frame_stride: Stride for frame processing (default: 1, process every frame).
    """
    # Create output and temporary directories if they do not exist
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(temp_directory, exist_ok=True)

    print('Starting trajectory processing...')

    # Load the original universe with protein and full trajectory
    u = mda.Universe(protein_pdb, protein_unaligned_dcd)

    # Select protein atoms and water oxygen atoms
    protein_water = u.select_atoms('protein or name OH2')

    # Save an intermediate PDB containing protein and water oxygens
    temp_pdb = os.path.join(temp_directory, 'protein_water.pdb')
    protein_water.write(temp_pdb)

    # Save an intermediate unaligned DCD
    temp_dcd = os.path.join(temp_directory, 'protein_water_unaligned.dcd')
    with mda.Writer(temp_dcd, protein_water.n_atoms) as writer:
        for ts in u.trajectory:
            writer.write(protein_water)

    # Reload the intermediate universe
    u_new = mda.Universe(temp_pdb, temp_dcd)
    protein = u_new.select_atoms('protein')
    water_oxygens = u_new.select_atoms('name OH2')

    # Determine box dimensions (lx, ly, lz) and floor them to integers
    box_dims = np.floor(u_new.dimensions[:3]).astype(int)

    # Define displacement vectors for six periodic images
    directions = [
        np.array([ box_dims[0], 0, 0]), np.array([-box_dims[0], 0, 0]),
        np.array([0,  box_dims[1], 0]), np.array([0, -box_dims[1], 0]),
        np.array([0, 0,  box_dims[2]]), np.array([0, 0, -box_dims[2]])
    ]

    PROT = "PROT"
    WAT  = "WAT"
    all_frames = []

    # Loop over trajectory frames
    for frame_idx, ts in enumerate(tq(u_new.trajectory)):
        # Stop after processing max_frames
        if frame_idx >= max_frames:
            break
        # Skip frames according to stride
        if frame_idx % frame_stride != 0:
            continue

        # Extract coordinates
        prot_coords = protein.positions
        wat_coords  = water_oxygens.positions

        # Create copies of water oxygen positions in six directions
        water_copies = [wat_coords] + [wat_coords + d for d in directions]
        merged_water = np.vstack(water_copies)
        merged_all   = np.vstack([prot_coords, merged_water])

        # Counts for atoms and residues
        n_prot      = protein.n_atoms
        n_wat_orig  = water_oxygens.n_atoms
        n_wat_total = n_wat_orig * len(water_copies)
        n_residues  = len(protein.residues) + len(water_oxygens.residues) * len(water_copies)

        # Create an empty Universe for the merged frame
        new_universe = mda.Universe.empty(
            n_atoms        = n_prot + n_wat_total,
            n_residues     = n_residues,
            atom_resindex = np.concatenate([
                protein.atoms.resindices,
                np.repeat(water_oxygens.atoms.resindices, len(water_copies))
            ]),
            trajectory = True
        )

        # Assign merged positions
        new_universe.atoms.positions = merged_all

        # Define topology attributes
        new_universe.add_TopologyAttr('name',   protein.atoms.names.tolist() + water_oxygens.atoms.names.tolist() * len(water_copies))
        new_universe.add_TopologyAttr('type',   protein.atoms.types.tolist() + ['OH2'] * n_wat_total)
        new_universe.add_TopologyAttr('resname', [PROT] * len(protein.residues) + [WAT] * (len(water_oxygens.residues) * len(water_copies)))
        new_universe.add_TopologyAttr('resid',   list(range(1, len(protein.residues)+1)) + list(range(1, len(water_oxygens.residues)*len(water_copies)+1)))

        # Save the first processed frame as a PDB
        if frame_idx == 0:
            first_pdb = os.path.join(output_directory, 'protein_copied_Water_0.pdb')
            new_universe.atoms.write(first_pdb)
            print(f"Saved first frame to {first_pdb}")

        all_frames.append(new_universe)

    # Write all processed frames to a new DCD
    output_dcd = os.path.join(output_directory, 'protein_copied_water_unaligned.dcd')
    with mda.Writer(output_dcd, n_atoms = n_prot + n_wat_total) as writer:
        for uni in all_frames:
            writer.write(uni.atoms)

    print(f"Processed trajectory saved to {output_dcd}")


if __name__ == "__main__":
    import sys
    argc = len(sys.argv)
    if argc not in (5, 6):
        print("Usage: python copy_water.py <protein_pdb> <unaligned_dcd> <output_dir> <temp_dir> [max_frames]")
        sys.exit(1)

    # Parse required arguments
    pdb_file     = sys.argv[1]
    dcd_file     = sys.argv[2]
    out_dir      = sys.argv[3]
    tmp_dir      = sys.argv[4]

    # Optional max_frames argument
    if argc == 6:
        try:
            max_fr = int(sys.argv[5])
        except ValueError:
            print("Error: max_frames must be an integer")
            sys.exit(1)
    else:
        max_fr = 2500

    # Run the copy_water workflow
    copy_water(pdb_file, dcd_file, out_dir, tmp_dir, max_frames=max_fr)
