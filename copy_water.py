import MDAnalysis as mda
import numpy as np
import sys, os
from tqdm.autonotebook import tqdm as tq
from MDAnalysis.analysis import align
from MDAnalysis.coordinates import base
import warnings

warnings.filterwarnings("ignore")

# Usage: python script_name.py protein_pdb protein_unaligned_dcd output_directory temp_directory

# Gather command line inputs
try:
    protein_pdb = sys.argv[1]
    protein_unaligned_dcd = sys.argv[2]
    output_directory = sys.argv[3]  # Directory used for saving the final aligned DCD and the first copied PDB
    temp_directory = sys.argv[4]  # Directory used for saving the temporary unaligned protein+water trajectory and PDB
except IndexError:
    print("Not enough arguments provided. Expected usage:")
    print("python script_name.py protein_pdb protein_unaligned_dcd output_directory temp_directory")
    sys.exit(1)

# Create output and temp directories if they don't exist
os.makedirs(output_directory, exist_ok=True)
os.makedirs(temp_directory, exist_ok=True)

print('Start processing the trajectory...')

# Load the original trajectory
u = mda.Universe(protein_pdb, protein_unaligned_dcd)

# Select only protein and water oxygen atoms
protein_water = u.select_atoms('protein or name OH2')

# Save the new unaligned trajectory and PDB with only protein and water oxygens
protein_water.write(f'{temp_directory}/protein_water.pdb')
with mda.Writer(f'{temp_directory}/protein_water_unaligned.dcd', protein_water.n_atoms) as W:
    for ts in u.trajectory:
        W.write(protein_water)

# Load the new unaligned trajectory
u_new = mda.Universe(f'{temp_directory}/protein_water.pdb', f'{temp_directory}/protein_water_unaligned.dcd')

# Selection: protein and water oxygen atoms only
protein = u_new.select_atoms('protein')
water_oxygens = u_new.select_atoms('name OH2')

# Read the box dimensions from the trajectory and ground them down to integers
d = np.floor(u_new.dimensions[:3]).astype(int)  # box dimensions: [lx, ly, lz]
directions = [
    np.array([d[0], 0, 0]), np.array([-d[0], 0, 0]),
    np.array([0, d[1], 0]), np.array([0, -d[1], 0]),
    np.array([0, 0, d[2]]), np.array([0, 0, -d[2]])
]

# Define the desired residue names
protein_residue_name = "PROT"
water_residue_name = "WAT"

# List to store frames for the new DCD
all_frames = []

# Process each frame in the trajectory
for frame_idx, ts in enumerate(tq(u_new.trajectory)):
    # Perform the analysis only on every second frame
    if frame_idx >= 2500:
        break
    if frame_idx % 1 != 0:
        continue
    # Extract the coordinates of the selected atoms
    water_coords = water_oxygens.positions
    protein_coords = protein.positions

    # Copy water molecules to six directions
    all_coords = [water_coords]
    for direction in directions:
        all_coords.append(water_coords + direction)

    # Merge all the coordinates (original + six translated)
    merged_water_coords = np.concatenate(all_coords, axis=0)
    merged_coords = np.vstack([protein_coords, merged_water_coords])

    # Create a new Universe with the merged protein and water coordinates
    num_protein_atoms = protein.n_atoms
    num_water_atoms = water_oxygens.n_atoms
    total_water_atoms = num_water_atoms * 7  # Total number of water oxygen atoms after copying

    # Determine the correct number of residues
    num_protein_residues = len(protein.residues)
    num_water_residues = len(water_oxygens.residues) * 7  # Each original water residue gets copied 7 times
    n_residues = num_protein_residues + num_water_residues

    new_universe = mda.Universe.empty(n_atoms=num_protein_atoms + total_water_atoms, 
                                      n_residues=n_residues, 
                                      atom_resindex=np.concatenate([protein.atoms.resindices, 
                                                                    np.repeat(water_oxygens.atoms.resindices, 7)]), 
                                      trajectory=True)

    # Copy the coordinates into the new universe
    new_universe.atoms.positions = merged_coords

    # Add topology attributes
    new_universe.add_TopologyAttr('name', protein.atoms.names.tolist() + water_oxygens.atoms.names.tolist() * 7)
    new_universe.add_TopologyAttr('type', protein.atoms.types.tolist() + ['OH2'] * num_water_atoms * 7)  # Assuming type 'O' for water oxygens
    new_universe.add_TopologyAttr('resname', [protein_residue_name] * num_protein_residues + [water_residue_name] * num_water_residues)
    new_universe.add_TopologyAttr('resid', list(range(1, num_protein_residues + 1)) + list(range(1, num_water_residues + 1)))

    # Save the first processed frame as a PDB file
    if len(all_frames) == 0:
        output_pdb = os.path.join(output_directory, 'protein_copied_Water_0.pdb')
        new_universe.atoms.write(output_pdb)
        print(f'First processed frame saved to {output_pdb}')

    # Append the new frame's universe to the list
    all_frames.append(new_universe)

# Save the processed trajectory into a new DCD file
output_dcd = os.path.join(output_directory, 'protein_copied_water_unaligned.dcd')
with mda.Writer(output_dcd, n_atoms=num_protein_atoms + total_water_atoms) as W:
    for universe in all_frames:
        W.write(universe.atoms)

print(f'Processed trajectory saved to {output_dcd}')
