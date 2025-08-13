"""
This script processes a collection of PDB files containing pocket coordinates, combines them into a 3D occupancy grid, and writes the resulting grid to a PDB file. The grid represents the spatial distribution of pocket atoms across multiple trajectories and dynamic IDs, allowing for the analysis of pocket location consensus.
Main functionalities:
- Defines a 3D grid covering a specified region of space, with configurable origin, dimensions, and spacing.
- Reads atomic coordinates from PDB files and maps them onto the grid, incrementing occupancy counts.
- Aggregates occupancy data across multiple PDB files, tracking unique dynamic and trajectory IDs.
- Writes the resulting occupancy grid to a PDB file, using grid center coordinates and applying occupancy thresholds.
- Extracts dynamic, trajectory, and pocket IDs from PDB filenames for tracking and aggregation.
Key variables:
- MIN_CORNER, MAX_CORNER: Define the spatial bounds of the grid.
- pockets_dir: Directory containing input PDB files.
- output_pdb: Output filename for the combined grid.
Usage:
- Place input PDB files in the specified directory, ensuring filenames follow the format "dyn{dynid}_traj{trajid}_pocket{pocketid}.pdb".
- Run the script to generate an output PDB file representing the combined occupancy grid.
- Output includes summary statistics about the processed data and grid configuration.
"""


import os
import numpy as np

# Constants.
MIN_CORNER = np.array([-60,-60,-75])
MAX_CORNER = np.array([65,70,70])

# Arguments.
pockets_dir = "/home/alex/Desktop/aligned_pdbs_test" # Directory containing input PDB files.
output_pdb = "output_grid.pdb" # Output filename for the combined grid.

# Functions.
def create_grid(origin, dims, spacing):
    """
    Creates a grid object with the specified origin, dimensions, and spacing.
    Parameters:
        origin (np.ndarray): The origin of the grid as a 3D coordinate (x, y, z).
        dims (np.ndarray): The dimensions of the grid as a 3D vector (width, height, depth).
        spacing (float): The spacing between grid points.
    Returns:
        dict: A dictionary representing the grid with keys 'origin', 'dims', 'spacing',
              'grid', and 'trajids'.
        'grid' is initialized to zeros, and 'trajids' is an empty list.
    """
    num_points = (np.array(dims)/spacing).round().astype(int)
    return {
        'origin': origin,
        'dims': dims,
        'spacing': spacing,
        'grid': np.zeros(num_points, dtype=int),
        'dynids': [],
        'trajids': [],
        'n_pockets_summed': 0
    }


def read_pocket_coordinates(pdb_file):
    """
    Reads atomic coordinates from a PDB file and returns them as a list of
    (x, y, z) tuples.
    Parameters:
        pdb_file (str): Path to the PDB file to be read.
    Returns:
        List[Tuple[float, float, float]]: A list of 3D coordinates (x, y, z)
        for each atom found in the file.
    """
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    coordinates = []
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            parts = line.split()
            x = float(parts[5])
            y = float(parts[6])
            z = float(parts[7])
            coordinates.append((x, y, z))

    return coordinates


def extract_grid(filepath, grid_obj):
    """
    Extracts a grid from a file and returns it.
    Parameters:
        filepath (str): Path to the file containing pocket coordinates.
        grid_obj (dict): The grid object to which the coordinates will be added.
    Returns:
        np.ndarray: A 3D numpy array representing the grid with occupancy counts.
    The grid is initialized to zeros and updated with occupancy counts based on the
    coordinates read from the file.
    The coordinates are mapped to the grid based on the grid's origin and spacing.
    If a coordinate is outside the grid bounds, it is ignored.
    The occupancy count for each grid point is incremented based on the number of
    coordinates that fall within that grid point.
    """
    grid = np.zeros(grid_obj['grid'].shape, dtype=int)

    pocket_coords = read_pocket_coordinates(filepath)
    for coord in pocket_coords:

        # Check if the coordinate is within the grid bounds.
        if coord[0] < grid_obj['origin'][0] or \
           coord[1] < grid_obj['origin'][1] or \
           coord[2] < grid_obj['origin'][2]:
            continue

        if coord[0] > grid_obj['origin'][0] + grid_obj['dims'][0] or \
           coord[1] > grid_obj['origin'][1] + grid_obj['dims'][1] or \
           coord[2] > grid_obj['origin'][2] + grid_obj['dims'][2]:
            continue

        # Map the coordinates to the grid.
        grid_x = int((coord[0] - grid_obj['origin'][0]) / grid_obj['spacing'])
        grid_y = int((coord[1] - grid_obj['origin'][1]) / grid_obj['spacing'])
        grid_z = int((coord[2] - grid_obj['origin'][2]) / grid_obj['spacing'])

        # Update the grid
        grid[grid_x, grid_y, grid_z] += 1
        
    return grid


def write_grid_to_pdb(grid_obj, output_pdb):
    """
    Writes the grid to a PDB file with proper PDB formatting.
    Uses grid center coordinates for more accurate representation.
    Parameters:
        grid_obj (dict): The grid object containing the grid data.
        output_pdb (str): Path to the output PDB file.
        min_threshold (float): Minimum occupancy threshold for writing atoms.
        max_threshold (float): Maximum occupancy threshold for writing atoms.
    Returns:
        None
    This function iterates through the grid and writes each atom to the PDB file
    if its occupancy is within the specified thresholds.
    The coordinates are calculated using the grid center, which provides a more
    accurate representation of the original coordinates.
    The PDB file is formatted according to the fixed-width columns required by the
    PDB format, including atom serial number, atom type, occupancy, and coordinates.
    The occupancy is normalized by the number of trajectories summed in the grid.
    """
    with open(output_pdb, 'w') as file:
        atom_serial = 1
        for x in range(grid_obj['grid'].shape[0]):
            for y in range(grid_obj['grid'].shape[1]):
                for z in range(grid_obj['grid'].shape[2]):
                    occupancy = grid_obj['grid'][x, y, z]
                    if occupancy == 0:
                        continue

                    perc_occupancy = occupancy/len(grid_obj['trajids'])
                    # if perc_occupancy < min_threshold or perc_occupancy > max_threshold:
                    #     continue

                    # Calculate coordinates using grid CENTER (not corner)
                    # This gives a more accurate representation of the original coordinates
                    x_coord = grid_obj['origin'][0] + (x + 0.5) * grid_obj['spacing']
                    y_coord = grid_obj['origin'][1] + (y + 0.5) * grid_obj['spacing']
                    z_coord = grid_obj['origin'][2] + (z + 0.5) * grid_obj['spacing']

                    # Format according to PDB fixed-width columns
                    # Columns: https://www.wwpdb.org/documentation/file-format
                    line = (
                        f"HETATM{atom_serial:5d}  C   PTH     1    "
                        f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
                        f"{1.00:6.2f}{perc_occupancy:6.3f}           C\n"
                    )
                    file.write(line)
                    atom_serial += 1


def extract_ids(file):
    """
    Extracts the IDs from a pocket filename.
    Parameters:
        file (str): The filename from which to extract IDs.
    Returns:
        tuple: A tuple containing the dynamic ID, trajectory ID, and pocket ID.
    The filename is expected to be in the format "dyn{dynid}_traj{trajid}_pocket{pocketid}.pdb".
    The IDs are extracted by splitting the filename and stripping the prefixes.
    If the filename does not match the expected format, it may raise an error.
    """
    basename = os.path.basename(file)
    dynid = basename.split("_")[0].strip("dyn")
    trajid = basename.split("_")[1].strip("traj")
    pocketid = basename.split("_")[2].strip("pocket")
    return dynid, trajid, pocketid


# Main
# Blank grid.
grid_obj = create_grid(MIN_CORNER, MAX_CORNER-MIN_CORNER, spacing=2)

# Iterate through all pocket PDB files in the directory.
for pdb in os.listdir(pockets_dir):
    pocket_file = os.path.join(pockets_dir, pdb)

    dynid, trajid, pocketid = extract_ids(pocket_file)

    if trajid not in grid_obj['trajids']:
        grid_obj['trajids'].append(trajid)

    if dynid not in grid_obj['dynids']:
        grid_obj['dynids'].append(dynid)

    # Extract grid from the PDB file and add it to common grid.
    grid = extract_grid(pocket_file, grid_obj)
    grid_obj['grid'] += grid
    grid_obj['n_pockets_summed'] += 1

# Output the result.
write_grid_to_pdb(grid_obj, "output_grid.pdb")

print("Grid written to output_grid.pdb")
print("Number of dynamic IDs:", len(grid_obj['dynids']))
print("Number of trajectories summed:", len(grid_obj['trajids']))
print("Number of pockets summed:", grid_obj['n_pockets_summed'])
print("Grid origin:", grid_obj['origin'])
print("Grid dimensions:", grid_obj['dims'])
print("Grid spacing:", grid_obj['spacing'])
print("Grid shape:", grid_obj['grid'].shape)
print("Grid data type:", grid_obj['grid'].dtype)
print("Dynamic IDs:", grid_obj['dynids'])
print("Trajectory IDs:", grid_obj['trajids'])
