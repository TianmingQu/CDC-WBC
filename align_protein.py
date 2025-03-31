import MDAnalysis as mda
from MDAnalysis.analysis import align
import sys

def align_protein(pdb_path, dcd_path, output_pdb, output_dcd):
    # Load the universe (PDB and DCD files)
    u = mda.Universe(pdb_path, dcd_path)

    # Select only the protein atoms
    protein = u.select_atoms("protein and not name H*")

    # Align the protein based on backbone atoms
    # Create an empty Universe for storing aligned trajectory
    aligned_universe = mda.Merge(protein)

    # Initialize writer for the aligned DCD output
    with mda.Writer(output_dcd, protein.n_atoms) as dcd_writer:
        # Align based on the backbone atoms (backbone selection: CA, N, C, O atoms)
        ref = u.select_atoms("protein and backbone")
        aligner = align.AlignTraj(u, u, select='protein and backbone', in_memory=True)
        aligner.run(verbose=True)

        # Write the aligned protein-only frames to the DCD file, only every 2nd frame
        for frame_idx, ts in enumerate(u.trajectory):
            if frame_idx >= 2500:
                break
            if frame_idx % 1 == 0:  # Only write every 2nd frame
                dcd_writer.write(protein)

    # Write out the protein-only PDB (first frame)
    protein.write(output_pdb)
    print(f"Protein-only PDB saved to {output_pdb}")
    print(f"Aligned protein-only DCD saved to {output_dcd}")

if __name__ == "__main__":
    pdb_path = sys.argv[1]
    dcd_path = sys.argv[2]
    output_pdb = sys.argv[3]
    output_dcd = sys.argv[4]

    align_protein(pdb_path, dcd_path, output_pdb, output_dcd)
