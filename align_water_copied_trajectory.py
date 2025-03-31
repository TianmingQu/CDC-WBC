import MDAnalysis as mda
from MDAnalysis.analysis import align
import sys, os

# Usage: python align_trajectory.py first_processed_frame.pdb processed_trajectory.dcd output_directory

# Gather command line inputs
try:
    reference_pdb = sys.argv[1]
    trajectory_dcd = sys.argv[2]
    output_directory = sys.argv[3]  # Directory to save the aligned trajectory
except IndexError:
    print("Not enough arguments provided. Expected usage:")
    print("python align_trajectory.py first_processed_frame.pdb processed_trajectory.dcd output_directory")
    sys.exit(1)

# Create output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Load the reference structure and trajectory
u = mda.Universe(reference_pdb, trajectory_dcd)

# Select the protein backbone for alignment
protein_backbone = u.select_atoms('name CA or name C or name N or name O')

# Reference structure for alignment
reference = mda.Universe(reference_pdb)
reference_backbone = reference.select_atoms('name CA or name C or name N or name O')

# Perform the alignment and calculate the RMSD
aligner = align.AlignTraj(u, reference, select="name CA or name C or name N or name O", in_memory=True)
aligner.run(verbose=True)

# Save the aligned trajectory
aligned_trajectory = os.path.join(output_directory, 'aligned_water_copied_trajectory.dcd')
with mda.Writer(aligned_trajectory, n_atoms=u.atoms.n_atoms) as W:
    for ts in u.trajectory:
        W.write(u.atoms)

print(f'Aligned trajectory saved to {aligned_trajectory}')
