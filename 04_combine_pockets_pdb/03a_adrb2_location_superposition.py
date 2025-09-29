"""
Modificated script specifically for ADRB2 (P07550).

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
uniprotid = "P07550"  # ADRB2

# Directory containing input PDB files.
pockets_dir = "/home/alex/sshfs_mountpoints/verde/Documents/pocket_tool/results/04_combine_pockets_pdb/00_aligning/aligned/aligned_pockets"
# Alignment CSV file path.
align_csv = "/home/alex/sshfs_mountpoints/verde/Documents/pocket_tool/results/04_combine_pockets_pdb/00_aligning/align.csv"
# Output directory.
output_dir = "/home/alex/sshfs_mountpoints/verde/Documents/pocket_tool/results/04_combine_pockets_pdb/01_joining"


# Functions.
def get_dynids_for_receptor(uniprotid, align_csv):
    """
    Reads the alignment CSV file and returns a list of dynamic IDs for the specified receptor.
    Parameters:
        uniprotid (str): The UniProt ID of the receptor to filter by.
        align_csv (str): Path to the alignment CSV file.
    Returns:
        List[str]: A list of dynamic IDs corresponding to the specified receptor.
    """
    dynids = []
    with open(align_csv, 'r') as file:
        lines = file.readlines()
    
    for line in lines[1:]:
        fields = line.strip().split(';')
        if fields[2].lower() == uniprotid.lower():
            dynids.append(fields[0])

    return dynids


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
        'n_pockets_summed': 0,
        'traj_contributors': {}  # Maps (x,y,z) -> set of trajectory IDs
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


def extract_grid(filepath, grid_obj, trajid):
    """
    Extracts a grid from a file and returns it along with trajectory information.
    Parameters:
        filepath (str): Path to the file containing pocket coordinates.
        grid_obj (dict): The grid object to which the coordinates will be added.
        trajid (str): The trajectory ID for tracking which trajectories contribute to each grid point.
    Returns:
        tuple: A tuple containing:
            - np.ndarray: A 3D numpy array representing the grid with binary occupancy (0 or 1).
            - dict: A dictionary mapping grid coordinates to sets of trajectory IDs.
    The grid is initialized to zeros and updated with binary occupancy based on the
    coordinates read from the file. Each grid point is marked as 1 if any atoms from
    this pocket occupy it, regardless of how many atoms are there.
    """
    grid = np.zeros(grid_obj['grid'].shape, dtype=int)
    traj_contributors = {}

    pocket_coords = read_pocket_coordinates(filepath)
    occupied_points = set()
    
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

        # Track unique grid points occupied by this pocket
        occupied_points.add((grid_x, grid_y, grid_z))
    
    # Mark occupied grid points and track trajectory contributors
    for grid_x, grid_y, grid_z in occupied_points:
        grid[grid_x, grid_y, grid_z] = 1
        coord_key = (grid_x, grid_y, grid_z)
        if coord_key not in traj_contributors:
            traj_contributors[coord_key] = set()
        traj_contributors[coord_key].add(trajid)
        
    return grid, traj_contributors


def write_grid_to_pdb(grid_obj, output_pdb):
    """
    Writes the grid to a PDB file with proper PDB formatting.
    Uses grid center coordinates for more accurate representation.
    Parameters:
        grid_obj (dict): The grid object containing the grid data.
        output_pdb (str): Path to the output PDB file.
    Returns:
        None
    This function iterates through the grid and writes each atom to the PDB file
    if its occupancy is greater than zero.
    The coordinates are calculated using the grid center, which provides a more
    accurate representation of the original coordinates.
    The PDB file is formatted according to the fixed-width columns required by the
    PDB format, including atom serial number, atom type, occupancy, and coordinates.
    The occupancy is calculated as the percentage of trajectories that have pockets
    contributing to each grid point, ensuring values between 0 and 1.
    """
    with open(output_pdb, 'w') as file:
        atom_serial = 1
        for x in range(grid_obj['grid'].shape[0]):
            for y in range(grid_obj['grid'].shape[1]):
                for z in range(grid_obj['grid'].shape[2]):
                    occupancy = grid_obj['grid'][x, y, z]
                    if occupancy == 0:
                        continue

                    # Calculate percentage based on number of trajectories that contribute to this grid point
                    coord_key = (x, y, z)
                    if coord_key in grid_obj['traj_contributors']:
                        n_contributing_trajs = len(grid_obj['traj_contributors'][coord_key])
                        perc_occupancy = n_contributing_trajs / len(grid_obj['trajids'])
                    else:
                        perc_occupancy = 0.0
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
dynids = get_dynids_for_receptor(uniprotid, align_csv)

# Blank grid.
grid_obj = create_grid(MIN_CORNER, MAX_CORNER-MIN_CORNER, spacing=2)

# Iterate through all pocket PDB files in the directory.
for pdb in os.listdir(pockets_dir):
    pocket_file = os.path.join(pockets_dir, pdb)

    dynid, trajid, pocketid = extract_ids(pocket_file)

    # Include only the dynamic IDs of the specified GPCR class.
    if dynid not in dynids:
        continue

    # Append dynid and trajid info to grid object.
    if trajid not in grid_obj['trajids']:
        grid_obj['trajids'].append(trajid)
    if dynid not in grid_obj['dynids']:
        grid_obj['dynids'].append(dynid)

    # Extract grid from the PDB file and add it to common grid.
    grid, traj_contributors = extract_grid(pocket_file, grid_obj, trajid)
    grid_obj['grid'] += grid
    grid_obj['n_pockets_summed'] += 1
    
    # Merge trajectory contributors
    for coord_key, contributors in traj_contributors.items():
        if coord_key not in grid_obj['traj_contributors']:
            grid_obj['traj_contributors'][coord_key] = set()
        grid_obj['traj_contributors'][coord_key].update(contributors)

# Output the result.
write_grid_to_pdb(
    grid_obj,
    os.path.join(output_dir, f"{uniprotid}_grid.pdb")
)

summary_file_path = os.path.join(
    output_dir, f"{uniprotid}_grid_summary.txt"
)
with open(summary_file_path, 'w') as summary_file:
    summary_file.write(f"Grid written to {uniprotid}_grid.pdb\n")
    summary_file.write(f"Number of dynamic IDs: {len(grid_obj['dynids'])}\n")
    summary_file.write(f"Number of trajectories summed: {len(grid_obj['trajids'])}\n")
    summary_file.write(f"Number of pockets summed: {grid_obj['n_pockets_summed']}\n")
    summary_file.write(f"Grid origin: {grid_obj['origin']}\n")
    summary_file.write(f"Grid dimensions: {grid_obj['dims']}\n")
    summary_file.write(f"Grid spacing: {grid_obj['spacing']}\n")
    summary_file.write(f"Grid shape: {grid_obj['grid'].shape}\n")
    summary_file.write(f"Grid data type: {grid_obj['grid'].dtype}\n")
    summary_file.write(f"Dynamic IDs: {', '.join(grid_obj['dynids'])}\n")
    summary_file.write(f"Trajectory IDs: {', '.join(grid_obj['trajids'])}\n")