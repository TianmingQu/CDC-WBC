import MDAnalysis as mda
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import math
from tqdm.autonotebook import tqdm as tq
import torch
from gridData import OpenDX
from MDAnalysis.transformations import translate
from scipy.interpolate import griddata
import psutil
import sympy

def smallest_factor(n):
    factors = sympy.factorint(n)
    smallest_factor = min(factors.keys())
    return smallest_factor if smallest_factor != n else min(factors.keys(), key=lambda x: n // x)

def generate_water_density_maps(pdb_path, dcd_path, ref_pdb_path, ref_dcd_path, resolution1, resolution2, gpu_id):
    
    # Set the CPU affinity
    p = psutil.Process(os.getpid())
    p.cpu_affinity(list(range(16)))
    
    # Select the GPU ID used for calculation
    device1 = torch.device(f"cuda:{gpu_id}")
    
    # Read in aligned copied water trajectory
    u_aligned = mda.Universe(ref_pdb_path, ref_dcd_path)

    # Origin Trajectory is needed for dimensions
    u = mda.Universe(pdb_path, dcd_path)
    
    # Read the dimension
    dim_float = u.trajectory.ts.dimensions[0]

    # Round up to the next integer
    dim_rounded = math.ceil(dim_float)

    # If the rounded dimension is odd, add 1 to make it even
    if dim_rounded % 2 != 0:
        dim_rounded += 1

    dim = dim_rounded
    print('The current dim is:', dim)
    split_number = int(dim/smallest_factor(dim))

    # Define Grid Box1
    protein_box = np.array([dim, dim, dim]) 

    # Determine the number of grid points along each axis for both resolutions
    n_points = np.ceil(protein_box / resolution1).astype(int)
    n_points2 = np.ceil(protein_box / resolution2).astype(int)

    # Determine the box dimensions of the grid boxes
    grid_box = n_points * resolution1
    grid_box2 = n_points2 * resolution2

    # Define the coordinates of the grid points for both resolutions
    x_coords = np.linspace(-grid_box[0]/2, grid_box[0]/2, n_points[0])
    y_coords = np.linspace(-grid_box[1]/2, grid_box[1]/2, n_points[1])
    z_coords = np.linspace(-grid_box[2]/2, grid_box[2]/2, n_points[2])

    x_coords2 = np.linspace(-grid_box2[0]/2, grid_box2[0]/2, n_points2[0])
    y_coords2 = np.linspace(-grid_box2[1]/2, grid_box2[1]/2, n_points2[1])
    z_coords2 = np.linspace(-grid_box2[2]/2, grid_box2[2]/2, n_points2[2])

    # Create 3D grids of the coordinates
    xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    xx2, yy2, zz2 = np.meshgrid(x_coords2, y_coords2, z_coords2, indexing='ij')

    grid_coords1 = np.stack((xx,yy,zz), axis=-1).reshape(-1, 3)
    grid_coords2 = np.stack((xx2,yy2,zz2), axis=-1).reshape(-1, 3)

    num_grids = np.prod(n_points)
    num_grids2 = np.prod(n_points2)

    # Generate corners of the high-resolution grid box for inbox water selection
    X_coords, Y_coords, Z_coords = zip(*grid_coords1)
    min_X, max_X = min(X_coords), max(X_coords)
    min_Y, max_Y = min(Y_coords), max(Y_coords)
    min_Z, max_Z = min(Z_coords), max(Z_coords)

    ow_within_cube = u_aligned.select_atoms(
        f'name OH2 and prop x > {min_X} and prop x < {max_X} and prop y > {min_Y} and prop y < {max_Y} and prop z > {min_Z} and prop z < {max_Z}', 
        updating=True
    )

    ow_within_cube_coords = [ow_within_cube.positions for ts in tq(u_aligned.trajectory)]

    ### Gaussian Convolution function
    def Gaussian_convolution(e, d, r):
        g = ((2 * torch.pi * (e ** 2)) ** (-1 * d / 2)) * torch.exp(-1 * torch.square(r.to(device1)) / (2 * (e ** 2) ))
        return g

    ### Here, the split number is used for splitting the large grid box into multiple small grid boxes.
    Grid_Coords_Split = np.split(grid_coords1, split_number)

    Density_map = []
    Density_map_2A = []
    Delta = np.array([resolution2, resolution2, resolution2])
    Origin = np.array([min_X, min_Y, min_Z])

    for i in tq(range(len(u_aligned.trajectory))):
        ow_coords_i = ow_within_cube_coords[i]
        inside_waters_torch = torch.from_numpy(ow_coords_i).to(torch.float16).to(device1)
        Water_Density = np.array([0,0,0])
        for j in range(split_number):
            Grid_Coordinates_torch = torch.from_numpy(Grid_Coords_Split[j]).to(torch.float16).to(device1)
            dist = torch.cdist(Grid_Coordinates_torch.float(), inside_waters_torch.float())
            Gaussian = Gaussian_convolution(1.7682, 3, dist)
            Grid_density = torch.sum(Gaussian, 1)
            Grid_density_np = Grid_density.cpu().numpy()
            del dist
            del Grid_Coordinates_torch
            del Gaussian
            del Grid_density
            Water_Density = np.concatenate((Water_Density, Grid_density_np), axis = 0)
        Water_Density = Water_Density[3:]
        del inside_waters_torch
        density_map_2A = griddata(points=grid_coords1, values=Water_Density, xi=grid_coords2, method='nearest')
        Density_map_2A.append(density_map_2A)
        Density_map.append(Water_Density)
        Grid_Density_rs = density_map_2A.reshape(n_points2[0], n_points2[1], n_points2[2])
        dx = OpenDX.field('density')
        dx.add('positions', OpenDX.gridpositions(1, Grid_Density_rs.shape, Origin, Delta))
        dx.add('connections', OpenDX.gridconnections(2, Grid_Density_rs.shape))
        dx.add('data', OpenDX.array(3, Grid_Density_rs))
        dx.write(f'water_density_maps/{i:04d}.dx')

if __name__ == "__main__":
    # Get input from the command line arguments
    pdb_path = sys.argv[1]
    dcd_path = sys.argv[2]
    ref_pdb_path = sys.argv[3]
    ref_dcd_path = sys.argv[4]
    resolution1 = float(sys.argv[5])
    resolution2 = float(sys.argv[6])
    gpu_id = int(sys.argv[7])

    generate_water_density_maps(pdb_path, dcd_path, ref_pdb_path, ref_dcd_path, resolution1, resolution2, gpu_id)
