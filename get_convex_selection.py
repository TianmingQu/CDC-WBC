import pickle
import sys

def get_selections(pdb_id, pkl_file_path):
    # Load the dictionary from the pkl file
    with open(pkl_file_path, 'rb') as f:
        convex_selection_dict = pickle.load(f)
    
    # Get the convex selection residues for the given PDB ID
    convex_residues = convex_selection_dict.get(pdb_id, None)
    
    if convex_residues is None:
        raise ValueError(f"No convex selection found for PDB ID: {pdb_id}")
    
    # Generate the convex_selection string
    #convex_selection = f"name CA and ({' or '.join([f'resid {res}' for res in convex_residues])})"
    convex_selection = f"({' or '.join([f'resid {res}' for res in convex_residues])})"
    # Generate the target_selection string for backbone atoms (CA, C, N, O)
    #target_selection = f"(not name H*) and ({' or '.join([f'resid {res}' for res in convex_residues])})"
    target_selection = f"protein and not name H*" 
    return convex_selection, target_selection

if __name__ == "__main__":
    pdb_id = sys.argv[1]
    pkl_file_path = sys.argv[2]

    convex_selection, target_selection = get_selections(pdb_id, pkl_file_path)

    print(f"convex_selection = \"{convex_selection}\"")
    print(f"target_selection = \"{target_selection}\"")
