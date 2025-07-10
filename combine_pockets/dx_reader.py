"""
"""


import os
import numpy as np
import copy

from scipy.interpolate import RegularGridInterpolator


def read_dx_file(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()

    header_lines = []
    data_lines = []
    in_data = False
    for line in lines:
        if line.startswith("object 1 class gridpositions counts"):
            counts = list(map(int, line.strip().split()[-3:]))
        elif line.startswith("origin"):
            origin = list(map(float, line.strip().split()[1:]))
        elif line.startswith("delta"):
            if 'delta_x' not in locals():
                delta_x = list(map(float, line.strip().split()[1:]))
            elif 'delta_y' not in locals():
                delta_y = list(map(float, line.strip().split()[1:]))
            elif 'delta_z' not in locals():
                delta_z = list(map(float, line.strip().split()[1:]))
        elif line.startswith("object 3 class array"):
            in_data = True
            
        if not in_data or line.startswith("object 3 class array"):
            header_lines.append(line)
        else:
            data_lines.append(line)

    scalar_data = []
    for line in data_lines:  # skip array declaration line
        scalar_data.extend(map(float, line.strip().split()))

    return {
        'counts': counts,
        'origin': origin,
        'deltas': [delta_x, delta_y, delta_z],
        'scalars': scalar_data,
        'header': header_lines
    }


def write_dx(filename, origin, deltas, counts, data, title="density"):
    """
    Writes a 3D volumetric dataset to an OpenDX .dx file.

    Parameters:
        filename (str): Output file path.
        origin (tuple): Origin of the grid (x, y, z).
        deltas (list of 3 tuples): Delta vectors [(dx_x, dx_y, dx_z), ...].
        counts (tuple): Grid dimensions (nx, ny, nz).
        data (np.ndarray): 3D data array of shape (nx, ny, nz).
        title (str): Optional title in the header.
    """
    nx, ny, nz = counts
    assert data.shape == (nx, ny, nz), "Data shape does not match counts"

    with open(filename, 'w') as f:
        f.write(f"# {title}\n")
        f.write("object 1 class gridpositions counts {} {} {}\n".format(nx, ny, nz))
        f.write("origin {:.6f} {:.6f} {:.6f}\n".format(*origin))
        for delta in deltas:
            f.write("delta {:.6f} {:.6f} {:.6f}\n".format(*delta))
        f.write("object 2 class gridconnections counts {} {} {}\n".format(nx, ny, nz))
        f.write(f"object 3 class array type double rank 0 items {nx * ny * nz} data follows\n")

        flat_data = data.flatten(order='C')  # standard C-style flattening

        # Write 3 values per line
        for i in range(0, len(flat_data), 3):
            line = " ".join(f"{val:.6e}" for val in flat_data[i:i+3])
            f.write(line + "\n")


def apply_transformation_to_dx(dx_data, rotation_matrix, translation_vector):
    """
    """
    dx_data_copy = copy.deepcopy(dx_data)

    rot = np.array(rotation_matrix)
    trans = np.array(translation_vector)

    new_origin = rot @ np.array(dx_data_copy['origin']) + trans
    new_delta_x = rot @ np.array(dx_data_copy['deltas'][0])
    new_delta_y = rot @ np.array(dx_data_copy['deltas'][1])
    new_delta_z = rot @ np.array(dx_data_copy['deltas'][2])

    dx_data_copy['origin'] = new_origin.tolist()
    dx_data_copy['deltas'][0] = new_delta_x.tolist()
    dx_data_copy['deltas'][1] = new_delta_y.tolist()
    dx_data_copy['deltas'][2] = new_delta_z.tolist()

    return dx_data_copy


def get_dx_bounding_box(origin, deltas, counts):
    """
    Computes min and max corners of a .dx file in XYZ space.
    Returns:
        min_corner, max_corner: numpy arrays of shape (3,)
    """
    nx, ny, nz = counts
    grid_vectors = np.array([
        np.array(deltas[0]) * (nx - 1),
        np.array(deltas[1]) * (ny - 1),
        np.array(deltas[2]) * (nz - 1),
    ])
    
    corner_vectors = np.array([
        [0, 0, 0],
        grid_vectors[0],
        grid_vectors[1],
        grid_vectors[2],
        grid_vectors[0] + grid_vectors[1],
        grid_vectors[0] + grid_vectors[2],
        grid_vectors[1] + grid_vectors[2],
        grid_vectors[0] + grid_vectors[1] + grid_vectors[2],
    ])
    
    corners = np.array(origin) + corner_vectors
    return corners.min(axis=0), corners.max(axis=0)


def determine_common_grid(dx_files, grid_spacing=1.0):
    """
    Determines the common grid origin, unit deltas, and counts to cover all .dx files.
    All deltas are set to 1.0 in each axis (no rotation/skew).
    """
    min_corner = np.array([np.inf, np.inf, np.inf])
    max_corner = np.array([-np.inf, -np.inf, -np.inf])

    for file in dx_files:
        dx_data = read_dx_file(file)
        origin, deltas, counts = (dx_data['origin'],
                                  dx_data['deltas'],
                                  dx_data['counts'])
        
        file_min, file_max = get_dx_bounding_box(origin, deltas, counts)
        min_corner = np.minimum(min_corner, file_min)
        max_corner = np.maximum(max_corner, file_max)

    origin = min_corner
    deltas = [(grid_spacing, 0.0, 0.0),
              (0.0, grid_spacing, 0.0),
              (0.0, 0.0, grid_spacing)]

    counts = tuple(np.ceil((max_corner - min_corner) / grid_spacing).astype(int) + 1)
    return tuple(origin), deltas, counts


def interpolate_to_grid(origin, deltas, counts, data_array, new_origin, new_deltas, new_counts):
    """
    Efficient interpolation from an arbitrarily oriented regular grid to another.
    Assumes both grids are rectilinear (structured).
    """
    # Step 1: Build transformation matrix from index space (i,j,k) to physical space
    transform = np.stack(deltas, axis=1)  # Shape (3, 3)
    origin = np.array(origin)

    # Inverse transform: physical -> index space
    inv_transform = np.linalg.inv(transform)

    # Create the original grid indices
    nx, ny, nz = counts
    i = np.arange(nx)
    j = np.arange(ny)
    k = np.arange(nz)

    interpolator = RegularGridInterpolator(
        (i, j, k),
        data_array,
        bounds_error=False,
        fill_value=0.0,
    )

    # Step 2: Generate coordinates of the new grid in physical space
    ni, nj, nk = new_counts
    xi = np.arange(ni)
    yj = np.arange(nj)
    zk = np.arange(nk)

    XI, YJ, ZK = np.meshgrid(xi, yj, zk, indexing='ij')
    new_coords = (
        new_origin[0] + XI * new_deltas[0][0] + YJ * new_deltas[1][0] + ZK * new_deltas[2][0],
        new_origin[1] + XI * new_deltas[0][1] + YJ * new_deltas[1][1] + ZK * new_deltas[2][1],
        new_origin[2] + XI * new_deltas[0][2] + YJ * new_deltas[1][2] + ZK * new_deltas[2][2],
    )

    physical_coords = np.stack(new_coords, axis=-1)  # Shape (ni, nj, nk, 3)
    flat_coords = physical_coords.reshape(-1, 3)

    # Step 3: Convert new physical coordinates to original grid index space
    relative = flat_coords - origin  # Shift to origin
    ijk_coords = relative @ inv_transform.T  # Shape (N, 3)

    # Step 4: Interpolate
    interpolated = interpolator(ijk_coords).reshape(new_counts)
    return interpolated