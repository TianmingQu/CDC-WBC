import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.spatial.distance import cdist
from scipy.spatial import Delaunay
from gridData import Grid
from tqdm import tqdm as tq
from scipy.optimize import least_squares
from scipy.spatial import cKDTree  # For efficient neighbor searches

def atom_name_to_element(atom_name):
    """
    Infers the element symbol from the atom name using PDB conventions.
    """
    atom_name = atom_name.strip()
    # If atom name starts with a digit, the element is in columns 2-3
    if atom_name[0].isdigit():
        element = atom_name[1]
    else:
        # Adjusted list excludes 'CA' to correctly map alpha-carbons
        if len(atom_name) >= 2 and atom_name[:2] in ['FE', 'MG', 'ZN', 'CL', 'BR', 'NA']:
            element = atom_name[:2]
        else:
            element = atom_name[0]
    return element.capitalize()

def best_fit_sphere(points):
    """
    Compute the best-fit sphere to a set of 3D points using least squares optimization.
    """
    center_initial = np.mean(points, axis=0)
    radius_initial = np.mean(np.linalg.norm(points - center_initial, axis=1))
    x0 = np.append(center_initial, radius_initial)

    def residuals(params):
        center = params[:3]
        radius = params[3]
        distances = np.linalg.norm(points - center, axis=1)
        return distances - radius

    result = least_squares(residuals, x0, method='lm')

    if result.success:
        center_opt = result.x[:3]
        radius_opt = result.x[3]
        return center_opt, radius_opt
    else:
        raise RuntimeError("Least squares fitting did not converge")

def ritter_sphere(points):
    """
    Compute an approximate minimal enclosing sphere using Ritter's algorithm.
    """
    xmin = points[np.argmin(points[:, 0])]
    xmax = points[np.argmax(points[:, 0])]
    center = (xmin + xmax) / 2
    radius = np.linalg.norm(xmax - center)

    for point in points:
        dist = np.linalg.norm(point - center)
        if dist > radius:
            new_radius = (radius + dist) / 2
            direction = (point - center) / dist
            center = center + (dist - radius) / 2 * direction
            radius = new_radius

    return center, radius

def in_hull(p, hull):
    """Test if points in `p` are in `hull`."""
    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p) >= 0

def write_pdb(name, coordinates):
    with open(name, 'w') as pdb:
        for idx, coord in enumerate(coordinates):
            pdb.write('ATOM  {:5d}  C   PTH     1    {:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n'.format(
                idx + 1, coord[0], coord[1], coord[2]))

def analyze_binding_site(pdb_path, dcd_path, convex_selection, target_selection, density_map_dir, output_prefix):
    print("Loading trajectory and initializing...")
    # Load the trajectory
    u = mda.Universe(pdb_path, dcd_path)

    print("Loading density maps...")
    # Load the density maps
    density_map_files = sorted(os.listdir(density_map_dir), key=lambda x: int(x.split('.')[0]))
    Density_maps = [Grid(os.path.join(density_map_dir, i)) for i in tq(density_map_files, desc="Loading density maps")]
    Density_map_1d_list = [grid.grid.ravel() for grid in Density_maps]

    print("Preparing grid coordinates...")
    # Prepare grid coordinates from the first density map
    Density_map_example = Density_maps[0]
    Delta = Density_map_example.delta
    Origin = Density_map_example.origin
    grid_shape = Density_map_example.grid.shape

    # Generate grid points coordinates
    x = np.linspace(Origin[0] + Delta[0]/2, Origin[0] + Delta[0]*(grid_shape[0]-0.5), grid_shape[0])
    y = np.linspace(Origin[1] + Delta[1]/2, Origin[1] + Delta[1]*(grid_shape[1]-0.5), grid_shape[1])
    z = np.linspace(Origin[2] + Delta[2]/2, Origin[2] + Delta[2]*(grid_shape[2]-0.5), grid_shape[2])
    xv, yv, zv = np.meshgrid(x, y, z, indexing='ij')
    pnts = np.column_stack([xv.ravel(), yv.ravel(), zv.ravel()])

    # Constants
    GRID_VOLUME = Delta[0]*Delta[1]*Delta[2]
    if GRID_VOLUME == 0:
        GRID_VOLUME = 1.0  # Replace with actual grid volume if different

    # Initialize lists to store results
    Convex_Density = []
    Cryptic_Volume_weighted_sum = []
    Average_density = []
    Std_density = []  # New list to store standard deviations
    Frame_indices = []  # List to store frame indices

    # Create directory for volume PDBs if it doesn't exist
    volume_pdb_dir = os.path.join('volume_pdbs')
    if not os.path.exists(volume_pdb_dir):
        os.makedirs(volume_pdb_dir)
        
    weight_plots_dir = os.path.join('weight_plots_dir')
    if not os.path.exists(weight_plots_dir):
        os.makedirs(weight_plots_dir)  # <-- Added Line
        
    print("Processing frames...")
    # For each frame
    for frame_idx in tq(range(len(u.trajectory)), desc="Frames"):
        ts = u.trajectory[frame_idx]

        print(f"\nProcessing frame {frame_idx + 1}/{len(u.trajectory)}")
        # Select convex region atoms in this frame
        Convex_region = u.select_atoms(convex_selection)
        print(Convex_region)
        Convex_region_coords = Convex_region.positions

        # Compute sphere using best-fit or Ritter's algorithm
        print("Computing sphere...")
        try:
            center, radius = best_fit_sphere(Convex_region_coords)
            if radius >= 20:
                center, radius = ritter_sphere(Convex_region_coords)
        except RuntimeError:
            center, radius = ritter_sphere(Convex_region_coords)
        print('center:', center)
        print('radius:', radius)
        # Ensure center is a tuple of floats
        center = tuple(map(float, center))

        # Expand the sphere radius by 1 + 1.09 Å
        radius += 0#+= 1 + 1.09

        # Remove grid points outside the sphere
        print("Filtering grid points inside the sphere...")
        distances_to_center = np.linalg.norm(pnts - np.array(center), axis=1)
        inside_sphere_mask = distances_to_center <= radius
        pnts_inside_sphere = pnts[inside_sphere_mask]

        # Use all protein heavy atoms (excluding hydrogens) for the convex hull
        protein_heavy_atoms = u.select_atoms('protein and (not name H*)')
        #protein_heavy_atoms = u.select_atoms(convex_selection) 
        protein_heavy_coords = protein_heavy_atoms.positions

        # Generate convex hull
        print("Generating convex hull using all heavy atoms...")
        try:
            hull = Delaunay(protein_heavy_coords)
        except Exception as e:
            print(f"Convex hull generation failed: {e}. Skipping frame.")
            continue

        # Refine pocket grids (keep only grids inside convex hull)
        print("Refining pocket grids inside convex hull...")
        in_hull_mask = in_hull(pnts_inside_sphere, hull)
        if not np.any(in_hull_mask):
            print("No grids inside convex hull. Skipping frame.")
            continue
        refined_grids = pnts_inside_sphere[in_hull_mask]

        # Remove grids that are too close to protein atoms
        # Use protein heavy atom VDW + 1.09 Å as cutoff
        print("Removing grids too close to protein atoms...")
        protein_heavy_atom_names = protein_heavy_atoms.names
        protein_heavy_elements = [atom_name_to_element(name) for name in protein_heavy_atom_names]

        # Define VDW radii for common elements
        vdw_radii = {
            'H': 1.2,
            'C': 1.7,
            'N': 1.55,
            'O': 1.52,
            'F': 1.47,
            'P': 1.80,
            'S': 1.80,
            'Cl': 1.75,
            'Br': 1.85,
            'I': 1.98,
            'Mg': 1.73,
            'Fe': 1.63,
            'Zn': 1.39,
            'Na': 2.27,
            'K': 2.75,
            'Ca': 2.31,
            'Mn': 1.79,
            'Co': 1.72,
            'Ni': 1.63,
            'Cu': 1.40,
            'Se': 1.90
            # Add more elements as needed
        }

        # Assign VDW radii to heavy atoms
        vdw = np.array([vdw_radii.get(elem, 1.7) for elem in protein_heavy_elements])

        # Add 1.09 Å to VDW radii to get cutoff distances
        cutoff_distances = vdw + 1.09 
        #cutoff_distances = 0
        # Build cKDTree for protein heavy atoms
        protein_tree = cKDTree(protein_heavy_coords)

        # Find grid points within the maximum cutoff distance from any heavy atom
        max_cutoff = cutoff_distances.max()
        #max_cutoff = 0
        print("Computing distances between grids and protein atoms...")
        indices = protein_tree.query_ball_point(refined_grids, r=max_cutoff)

        # Filter out grids that are too close
        mask = np.ones(len(refined_grids), dtype=bool)
        for i, idxs in enumerate(indices):
            if idxs:
                atom_coords = protein_heavy_coords[idxs]
                atom_cutoffs = cutoff_distances[idxs]
                #atom_cutoffs = 0
                distances = np.linalg.norm(atom_coords - refined_grids[i], axis=1)
                # Check if any atom is within its VDW + 1.09 Å
                if np.any(distances < atom_cutoffs):
                    mask[i] = False
        final_grids = refined_grids[mask]

        if len(final_grids) == 0:
            print("No grids left after removing close grids. Skipping frame.")
            continue

        # Get density values for the grids
        density_map_1d = Density_map_1d_list[frame_idx]
        # Get the indices of final grids in the pnts array
        # Need to map back to original indices in pnts
        inside_sphere_indices = np.where(inside_sphere_mask)[0]
        in_hull_indices = inside_sphere_indices[np.where(in_hull_mask)[0]]
        final_grid_indices = in_hull_indices[mask]

        convex_density = density_map_1d[final_grid_indices]

        # Further refine: Keep grids with density higher than 0.003
        print("Filtering grids with density higher than 0.01...")
        density_threshold = np.average(convex_density) - np.std(convex_density)
        density_mask = convex_density > 0.01

        if not np.any(density_mask):
            print("No grids left after density filtering. Skipping frame.")
            continue

        # Update final_grids and convex_density based on density_mask
        final_grids = final_grids[density_mask]
        convex_density = convex_density[density_mask]
        final_grid_indices = final_grid_indices[density_mask]

        # Save each volume grids as a single PDB file
        print("Saving volume grids as PDB file...")
        pdb_filename = os.path.join(volume_pdb_dir, f'{output_prefix}_volume_grids_frame_{frame_idx}.pdb')
        write_pdb(pdb_filename, final_grids)
        
        Frame_indices.append(frame_idx)
        
        # Calculate average density
        average_frame_density = np.mean(convex_density)
        std_frame_density = np.std(convex_density)  # Calculate standard deviation
        Average_density.append(average_frame_density)
        Std_density.append(std_frame_density)  # Append std to list
        Convex_Density.append(convex_density)

        
        #def reciprocal_sqrt_weighting(density):
        #    weights = 0.5 + 0.5 / (1 + np.exp(-500 * (density - np.average(density))))
        #   return weights
        
        def reciprocal_sqrt_weighting(density):
            mean_density = np.mean(density)
            std_density = np.std(density)
            # Gaussian weighting with scaling and shifting
            weights = 0.5 * np.exp(-((density - mean_density) ** 2) / (2 * std_density ** 2)) + 0.5
            return weights
        
        # Apply reweighting
        print("Applying Reweighting function...")
        # Modify h to have a stronger dependence on average_frame_density
        #h = (average_frame_density / 0.01)  # Using quadratic dependence
        weights = reciprocal_sqrt_weighting(convex_density)
        #weights /= np.max(weights)
        # Compute weighted volume
        #weighted_density = convex_density * weights
        sum_weighted_density = np.sum(weights) * GRID_VOLUME  # Use actual grid volume
        Cryptic_Volume_weighted_sum.append(sum_weighted_density)
        # Plot sorted convex_density against weights
        sorted_indices = np.argsort(convex_density)
        sorted_convex_density = convex_density[sorted_indices]
        sorted_weights = weights[sorted_indices]
        plt.figure()
        plt.plot(sorted_convex_density, sorted_weights, label=f'Frame {frame_idx + 1}')
        plt.xlabel('Convex Density (sorted)')
        plt.ylabel('Weight')
        plt.title(f'Sorted Weight vs Convex Density (Frame {frame_idx + 1})')
        plt.grid(True)
        plt.legend()
#
        weight_plot_filename = os.path.join(weight_plots_dir, f'{output_prefix}_weights_sorted_frame_{frame_idx + 1}.png')
        plt.savefig(weight_plot_filename)
        plt.close()

    print("\nProcessing completed. Saving results...")
# Save the results as CSV files with ',' delimiter, no headers, include frame indices

    # Save average and standard deviation of frame density together with frame indices
    Average_and_Std_density = np.column_stack((Frame_indices, Average_density, Std_density))
    np.savetxt(f'{output_prefix}_average_frame_density.csv', Average_and_Std_density, delimiter=',', fmt='%d,%f,%f', header='', comments='')
    
    # Save cryptic volume weighted sum with frame indices
    cryptic_volume_data = np.column_stack((Frame_indices, Cryptic_Volume_weighted_sum))
    np.savetxt(f'{output_prefix}_volume_crypwater_new.csv', cryptic_volume_data, delimiter=',', fmt='%d,%f', header='', comments='')
    
    print("Plotting results...")
    # Plotting
    simulation_time = np.arange(len(Cryptic_Volume_weighted_sum)) * 0.004  # Adjust time step as needed

    
    # Plot Cryptic Volume (Weighted)
    plt.figure()
    plt.plot(simulation_time, Cryptic_Volume_weighted_sum)
    plt.xlabel('Time (ns)', fontsize=10, fontweight='bold')
    plt.ylabel('Cryptic Volume (Weighted)', fontsize=10, fontweight='bold')
    plt.title('Cryptic Volume (Weighted) Over Time', fontsize=11, fontweight='bold')
    plt.savefig(f'{output_prefix}_cryptic_volume_weighted_plot.png')
    plt.close()
    
    # Plot Average Frame Density with Error Bars (Standard Deviation)
    plt.figure()
    plt.errorbar(simulation_time, Average_density, yerr=Std_density, fmt='-o', ecolor='lightgray', elinewidth=2, capsize=0)
    plt.xlabel('Time (ns)', fontsize=10, fontweight='bold')
    plt.ylabel('Average Frame Density', fontsize=10, fontweight='bold')
    plt.title('Average Frame Density Over Time', fontsize=11, fontweight='bold')
    plt.savefig(f'{output_prefix}_average_frame_density_plot.png')
    plt.close()
    
    print("All tasks completed successfully.")


if __name__ == "__main__":

    pdb_path = sys.argv[1]
    dcd_path = sys.argv[2]
    convex_selection = sys.argv[3]
    target_selection = sys.argv[4]
    print(f"Convex selection: {convex_selection}")
    density_map_dir = sys.argv[5]
    output_prefix = sys.argv[6]

    analyze_binding_site(pdb_path, dcd_path, convex_selection, target_selection, density_map_dir, output_prefix)
