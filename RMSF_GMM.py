import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
from MDAnalysis.analysis.rms import RMSF
from sklearn import mixture
from tqdm import tqdm as tq

def write_pdb(filename, coords, resid_list=None, atomnames=None):
    """
    Write coordinates to a simple PDB file.
    - coords: array of shape (N, 3)
    - resid_list: array of shape (N,), residue IDs for annotation
    - atomnames: array of shape (N,), atom names such as ["C", "O", "N", ...]
    """
    with open(filename, 'w') as f:
        for i, xyz in enumerate(coords):
            x, y, z = xyz
            at_name = "C" if atomnames is None else atomnames[i]
            f.write(
                "ATOM  {:5d} {:>2s}  PTH     1    {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}\n".format(
                    i+1, at_name, x, y, z, 1.0, 0.0
                )
            )
        if resid_list is not None:
            f.write("REMARK  Residue IDs:\n")
            f.write("REMARK  {}\n".format(resid_list))

def Read_Protein_Coordinates(pdb_file, dcd_file, atom_selection):
    """
    Read PDB/DCD files and return:
    1) Coordinates of selected atoms in each frame (list of arrays)
    2) Selected AtomGroup
    3) Universe object
    """
    u = mda.Universe(pdb_file, dcd_file)
    protein = u.select_atoms(atom_selection)
    coords_list = []
    for ts in tq(u.trajectory):
        coords_list.append(protein.positions.copy())
    return coords_list, protein, u
# ---------------------- Main Logic ----------------------
pdb_path = "/scratch2/users/tq19b/analysis_2024/1jwp_2024_water_analysis_data/1jwp_noh.pdb"
dcd_path = "/scratch2/users/tq19b/analysis_2024/1jwp_2024_water_analysis_data/1jwp_noh.dcd"

# 1) Calculate RMSF using backbone atoms (or only CA atoms)
protein_info = Read_Protein_Coordinates(
    pdb_file=pdb_path,
    dcd_file=dcd_path,
    atom_selection="protein and backbone and resid 30 to 280"
)
u = protein_info[2]           # Universe
protein = protein_info[1]     # AtomGroup
calphas = protein.select_atoms("name CA")

# 2) Compute RMSF
rmsfer = RMSF(calphas, verbose=True).run()
rmsf = rmsfer.results.rmsf      # RMSF for each CA atom
resids = calphas.resids         # Residue IDs for each CA
resnums = calphas.resnums       # Residue numbers for each CA
# 3) Identify high RMSF residues
max_rmsf = np.max(rmsf)
cutoff = max_rmsf * 0.15
print("Maximum RMSF = {:.3f}, cutoff = {:.3f}".format(max_rmsf, cutoff))

highRMSF_indices = np.where(rmsf >= cutoff)[0]  # Indices of CA atoms with high RMSF
highRMSF_resids = resids[highRMSF_indices]     # Residue IDs of these CA atoms 
highRMSF_resnums = resnums[highRMSF_indices]    # Residue numbers

unique_highRMSF_resnums = np.unique(highRMSF_resnums)
print("{} CA atoms >= cutoff, corresponding to {} unique residues".format(
    len(highRMSF_indices), len(unique_highRMSF_resnums))
)
print("High RMSF residue numbers:", unique_highRMSF_resnums)

# 4) Select all heavy atoms from these high-RMSF residues (exclude hydrogens)
resnum_str_list = [f"resid {r}" for r in unique_highRMSF_resnums]
selection_str = " or ".join(resnum_str_list) + " and protein and not name H*"
print("Selection for all heavy atoms in high RMSF residues:", selection_str)

heavy_atoms = u.select_atoms(selection_str)

# Collect coordinates of heavy atoms across all frames
n_frames = len(u.trajectory)
n_heavy = len(heavy_atoms)
all_heavy_coords = np.zeros((n_frames, n_heavy, 3), dtype=np.float32)

# Save residue IDs and atom names for PDB output
heavy_resids = heavy_atoms.resids
heavy_atomnames = heavy_atoms.names

for iframe, ts in enumerate(tq(u.trajectory)):
    all_heavy_coords[iframe, :, :] = heavy_atoms.positions

# Calculate average coordinates (shape: [n_heavy, 3])
avg_heavy_coords = np.mean(all_heavy_coords, axis=0)

# 5) Cluster average coordinates using Gaussian Mixture Model (GMM)
n_components = 10
clf = mixture.BayesianGaussianMixture(
    n_components=n_components,
    covariance_type="full",
    weight_concentration_prior=1e2,
    weight_concentration_prior_type="dirichlet_process",
    max_iter=100,
    random_state=42
)
clf.fit(avg_heavy_coords)
labels = clf.predict(avg_heavy_coords)

# Count atoms per cluster
cluster_dict = {}
for i in range(n_components):
    cluster_dict[i] = np.where(labels == i)[0] 

for cid, idx_array in cluster_dict.items():
    resids_in_cluster = heavy_resids[idx_array]
    unique_resids_in_cluster = np.unique(resids_in_cluster)
    print(f"Cluster {cid}: {len(idx_array)} atoms, Residue IDs => {unique_resids_in_cluster}")

# 6) Write clustered atoms from first frame to PDB files
u.trajectory[0]
coords_first_frame = heavy_atoms.positions  # shape = (n_heavy, 3)

for cid, idx_array in cluster_dict.items():
    coords_cluster = coords_first_frame[idx_array]
    resids_cluster = heavy_resids[idx_array]
    names_cluster = heavy_atomnames[idx_array]
    out_pdb = f"cluster_{cid}.pdb"
    write_pdb(
        out_pdb,
        coords_cluster,
        resid_list=resids_cluster,
        atomnames=names_cluster
    )
    print(f"Cluster {cid} PDB file written to => {out_pdb}")

# 7) (Optional) Plot RMSF results
plt.figure()
plt.plot(resnums, rmsf, label="CA RMSF")
plt.axhline(y=cutoff, color='r', linestyle='--', label='Cutoff')
plt.xlabel("Residue Number", fontsize=12)
plt.ylabel("RMSF (A)", fontsize=12)
plt.title("RMSF with Cutoff = {:.2f}".format(cutoff), fontsize=12)
plt.legend()
plt.show()
