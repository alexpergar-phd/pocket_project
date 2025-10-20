"""
For each pocket in each frame, extract its 360 degree location based on its x,y
 coordinates.
"""


import os
import math

import pandas as pd


# Arguments
pocket_dir = "/home/aperalta/Documents/pocket_tool/results/02_cryptic_pocket_detection/02_selecting_10percent/00_aligning/output_251020/aligned_pdbs/aligned_pockets/"
outdir = "/home/aperalta/Documents/pocket_tool/results/02_cryptic_pocket_detection/02_selecting_10percent/01_360location"


# Constants.
# The GPCR is like 50 A long, so middle section should be within +-8 A of the
# middle (Z=0). If above 8, we considered it in the EC region, below -8 in the
# IC.
Z_CUTOFF = 8.0
Z_MID = 0.0


# Functions.
def get_pdbs_in_directory(directory):
    """
    Get all PDB files in a directory.
    """
    pdb_files = []
    for file in os.listdir(directory):
        if file.endswith(".pdb"):
            pdb_files.append(os.path.join(directory, file))
    return pdb_files


def extract_ids(file):
    """
    Extracts the IDs from a pocket filename.
    """
    basename = os.path.basename(file)
    dynid = basename.split("_")[1]
    trajid = basename.split("_")[3]
    pocketid = basename.split("_")[5]
    return dynid, trajid, pocketid


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


def angle_from_origin(x, y):
    """
    Return the angle (in degrees, 0-360) of the vector (x, y) measured from the +X axis.
    Raises ValueError if the point is the origin (0,0).
    """
    if x == 0 and y == 0:
        raise ValueError("Angle is undefined for the origin (0,0).")
    ang = math.degrees(math.atan2(y, x))
    if ang < 0:
        ang += 360.0
    return ang


# Main.
results_dict = {
    'dynid': [],
    'trajid': [],
    'pocketid': [],
    'vertical_position': [],
    'angle_deg': [],
}

# Iterate over pocket PDB files.
for pocket_pdb in get_pdbs_in_directory(pocket_dir):
    pocket_coordinates = read_pocket_coordinates(pocket_pdb)
    dynid, trajid, pocketid = extract_ids(pocket_pdb)

    # For each coordinate, calculate the angle and store the results.
    for coord in pocket_coordinates:
        x, y, z = coord
        angle = round(angle_from_origin(x, y))

        if z > Z_MID + Z_CUTOFF:   vertical_position = "EC"
        elif z < Z_MID - Z_CUTOFF: vertical_position = "IC"
        else:                      vertical_position = "MID"

        results_dict['dynid'].append(dynid)
        results_dict['trajid'].append(trajid)
        results_dict['pocketid'].append(pocketid)
        results_dict['vertical_position'].append(vertical_position)
        results_dict['angle_deg'].append(angle)

# Save results to CSV.
df_results = pd.DataFrame(results_dict)
df_results.to_csv(os.path.join(outdir, "pocket_360_locations.csv"), index=False)

