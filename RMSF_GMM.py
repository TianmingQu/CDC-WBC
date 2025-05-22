"""
rmsf_cluster.py

Module for RMSF calculation and clustering of high-fluctuation residues.

Provides:
    cluster_high_rmsf_residues(pdb_path, dcd_path, atom_selection, cutoff_fraction,
                               n_components, output_csv='cluster_results.csv',
                               output_plot='rmsf_plot.png')

Usage (command-line):
    python rmsf_cluster.py \
        <pdb_path> <dcd_path> <atom_selection> <cutoff_fraction> <n_components> \
        [output_csv] [output_plot]

Arguments:
    pdb_path        : Path to the input PDB file
    dcd_path        : Path to the input DCD trajectory file
    atom_selection  : MDAnalysis atom selection string for analysis, e.g.
                      "protein and backbone and resid 30 to 280"
    cutoff_fraction : Fraction of max RMSF to use as cutoff (e.g., 0.25)
    n_components    : Number of mixture components for clustering
    output_csv      : (Optional) Output CSV filename (default: 'cluster_results.csv')
    output_plot     : (Optional) Output PNG filename for RMSF plot (default: 'rmsf_plot.png')

Output:
    CSV file with columns: cluster_id, residue_ids (comma-separated)
    PNG file of RMSF plot saved as specified
"""

import sys
import numpy as np
import MDAnalysis as mda
from MDAnalysis.analysis.rms import RMSF
from sklearn.mixture import BayesianGaussianMixture
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt


def cluster_high_rmsf_residues(pdb_path: str,
                               dcd_path: str,
                               atom_selection: str,
                               cutoff_fraction: float,
                               n_components: int,
                               output_csv: str = 'cluster_results.csv',
                               output_plot: str = 'rmsf_plot.png'):
    """
    Calculate RMSF on C-alpha atoms within a specified selection, plot RMSF,
    select high-fluctuation residues, cluster their average heavy-atom coordinates,
    and write cluster results to CSV.

    Parameters:
        pdb_path        : Path to the input PDB file
        dcd_path        : Path to the input DCD trajectory file
        atom_selection  : Atom selection string for RMSF (MDAnalysis syntax)
        cutoff_fraction : Fraction of max RMSF to define cutoff
        n_components    : Number of mixture model components for clustering
        output_csv      : Output CSV filename for cluster results
        output_plot     : Output PNG filename for RMSF plot
    """
    # Load the trajectory
    u = mda.Universe(pdb_path, dcd_path)

    # Select C-alpha atoms within the provided selection for RMSF calculation
    ca_selection = f"({atom_selection}) and name CA"
    calphas = u.select_atoms(ca_selection)

    # Compute RMSF for each C-alpha atom
    rmsf_calc = RMSF(calphas, verbose=False).run()
    rmsf = rmsf_calc.results.rmsf
    resids = calphas.resids

    # Determine dynamic cutoff based on fraction of max RMSF
    max_rmsf = np.max(rmsf)
    cutoff = cutoff_fraction * max_rmsf
    print(f"Maximum RMSF = {max_rmsf:.3f}; cutoff = {cutoff:.3f} ({cutoff_fraction} x max)")

    # Plot RMSF and cutoff
    plt.figure()
    plt.plot(resids, rmsf, label='CA RMSF')
    plt.axhline(y=cutoff, color='r', linestyle='--', label=f'Cutoff = {cutoff:.2f}')
    plt.xlabel('Residue Number')
    plt.ylabel('RMSF (Å)')
    plt.title(f'CA RMSF with cutoff = {cutoff_fraction:.2f}×max')
    plt.legend()
    plt.savefig(output_plot)
    plt.close()
    print(f"RMSF plot saved to {output_plot}")

    # Identify high-RMSF residues
    high_idx = np.where(rmsf >= cutoff)[0]
    high_resids = np.unique(resids[high_idx])

    # Select heavy atoms from those residues within the original selection
    heavy_selection = f"({atom_selection}) and protein and not name H* and resid {' '.join(map(str, high_resids))}"
    heavy_atoms = u.select_atoms(heavy_selection)

    # Collect heavy-atom coordinates over all frames
    n_frames = len(u.trajectory)
    n_heavy = heavy_atoms.n_atoms
    coords = np.zeros((n_frames, n_heavy, 3), dtype=np.float32)
    for i, ts in enumerate(tqdm(u.trajectory, desc='Reading frames')):
        coords[i] = heavy_atoms.positions

    # Compute average coordinates per heavy atom
    avg_coords = coords.mean(axis=0)

    # Cluster average coordinates
    bgm = BayesianGaussianMixture(
        n_components=n_components,
        covariance_type='full',
        weight_concentration_prior=1e2,
        weight_concentration_prior_type='dirichlet_process',
        max_iter=100,
        random_state=0
    )
    bgm.fit(avg_coords)
    labels = bgm.predict(avg_coords)

    # Map clusters to residue IDs
    heavy_resids = heavy_atoms.resids
    cluster_map = {}
    for cid in range(n_components):
        idx = np.where(labels == cid)[0]
        if idx.size == 0:
            continue
        resid_list = np.unique(heavy_resids[idx]).tolist()
        cluster_map[cid] = resid_list

    # Write cluster results to CSV
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['cluster_id', 'residue_ids'])
        for cid, res_list in cluster_map.items():
            writer.writerow([cid, ','.join(map(str, res_list))])

    print(f"Cluster results written to {output_csv}")


if __name__ == '__main__':
    argc = len(sys.argv)
    if argc < 5 or argc > 7:
        print('Usage: python rmsf_cluster.py <pdb_path> <dcd_path> <atom_selection> <cutoff_fraction> <n_components> [output_csv] [output_plot]')
        sys.exit(1)

    pdb_file = sys.argv[1]
    dcd_file = sys.argv[2]
    atom_sel = sys.argv[3]
    cutoff_frac = float(sys.argv[4])
    n_comp = int(sys.argv[5])
    out_csv = sys.argv[6] if argc >= 7 else 'cluster_results.csv'
    out_plot = sys.argv[7] if argc == 8 else 'rmsf_plot.png'

    cluster_high_rmsf_residues(
        pdb_path=pdb_file,
        dcd_path=dcd_file,
        atom_selection=atom_sel,
        cutoff_fraction=cutoff_frac,
        n_components=n_comp,
        output_csv=out_csv,
        output_plot=out_plot
    )
