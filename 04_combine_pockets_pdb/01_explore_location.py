"""
This script analyzes pocket structures in GPCRs by processing aligned pocket
PDB files and extracting geometric and spatial features.
Main functionalities:
- Reads pocket atom coordinates from PDB files.
- Extracts identifiers (dynid, trajid, pocketid) from filenames.
- Calculates the geometric center of pocket coordinates.
- Determines if the geometric center lies within a defined central cylinder
region of the GPCR.
- Classifies the z-location of the pocket center as extracellular loop (ECL),
above, below, or intracellular loop (ICL) based on z-coordinates.
- Extracts alpha carbon (CA) coordinates and residue IDs from a reference GPCR
structure.
- Defines residue ranges for each transmembrane helix (TM1-TM7, H8) and
extracts their coordinates.
- Calculates the minimum distance between pocket atoms and each TM helix to
determine contacts.
- Iterates over all aligned pocket PDBs, computes features, and prints a
summary table with identifiers, spatial classification, and TM contacts.
Constants:
- Paths to aligned pocket PDBs and reference structure.
- Cylinder geometry for central region.
- Z-coordinate thresholds for region classification.
- Distance threshold for TM contact.
Output:
- Prints a semicolon-separated table with columns: dynid, trajid, pocketid,
is_centered, z_location, tm_contacts.
"""


import os

import numpy as np
import fnmatch


# Paths
aligned_pockets_dir = "/home/alex/Desktop/aligned_pdbs_test"
ref_structure = "/home/alex/Documents/pocket_tool/data/ref_gpcr/data/a2a_6gdg_opm_rotated.pdb"


# Constants for cylinder that denotes the central region of the GPCR.
CYLINDER_START = (4, 1, -25)
CYLINDER_END = (0, 3, 20)
CYLINDER_RADIUS = 5.0

# Constants for z-coordinates that denote the ICL, center, and ECL regions.
Z_MIDDLE = -3
Z_TOP = 17
Z_BOTTOM = -20

# Minimum distance threshold for considering a contact with transmembrane
# helices.
MIN_DISTANCE_THRESHOLD = 5.0


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


def extract_ids(file):
    """
    Extracts the IDs from a pocket filename.
    """
    basename = os.path.basename(file)
    dynid = basename.split("_")[0].strip("dyn")
    trajid = basename.split("_")[1].strip("traj")
    pocketid = basename.split("_")[2].strip("pocket")
    return dynid, trajid, pocketid
    

def calculate_geometric_center(coordinates):
    """
    Calculates the geometric center of a set of coordinates.
    """
    coords_array = np.array(coordinates)
    center = np.mean(coords_array, axis=0)
    
    return tuple(center)


def point_in_cylinder(target_point, cylinder_start, cylinder_end, radius):
    """
    This formula is derived from the projection of a point onto the axis of a
    cylinder. It calculates the closest point on the cylinder axis to the
    target point and checks if the distance from the target point to this
    closest point is less than or equal to the cylinder radius.
    Parameters:
        target_point (tuple): The point to check (x, y, z).
        cylinder_start (tuple): The start point of the cylinder axis (x, y, z
        cylinder_end (tuple): The end point of the cylinder axis (x, y, z).
        radius (float): The radius of the cylinder.
    Returns:
        bool: True if the target point is within the cylinder, False otherwise.
    """
    P = np.array(target_point)
    A = np.array(cylinder_start)
    B = np.array(cylinder_end)

    AB = B - A
    AP = P - A

    t = np.dot(AP, AB) / np.dot(AB, AB)

    if t < 0 or t > 1:
        return False  # Outside height range

    Q = A + t * AB  # Closest point on axis
    distance = np.linalg.norm(P - Q)

    return distance <= radius


def get_z_location(z_top, z_middle, z_bottom, coordinates):
    """
    Determines the z-location of a pocket based on its coordinates.
    The z-location is classified into four categories:
    - "ecl" for extracellular loop (z >= z_top)
    - "above" for above the middle (z >= z_middle)
    - "below" for below the middle (z >= z_bottom)
    - "icl" for intracellular loop (z < z_bottom)
    Parameters:
        z_top (float): The z-coordinate threshold for the extracellular loop.
        z_middle (float): The z-coordinate threshold for the middle region.
        z_bottom (float): The z-coordinate threshold for the intracellular loop.
        coordinates (tuple): The (x, y, z) coordinates of the pocket.
    Returns:
        str: The z-location classification ("ecl", "above", "below", "icl
    """
    z_coord = coordinates[2]

    if z_coord >= z_top:
        return "ecl"
    elif z_coord >= z_middle:
        return "above"
    elif z_coord >= z_bottom:
        return "below"
    else:
        return "icl"


def read_receptor_ca_coordinates(pdb_file):
    """
    Reads the alpha carbon (CA) coordinates and residue IDs from a PDB file.
    Parameters:
        pdb_file (str): Path to the PDB file to be read.
    Returns:
        List[Tuple[Tuple[float, float, float], int]]: A list of tuples,
        where each tuple contains a 3D coordinate (x, y, z) and
        the corresponding residue ID.
    """
    with open(pdb_file, 'r') as file:
        lines = file.readlines()
    
    coordinates = []
    resids = []
    for line in lines:
        if line[13:15] == "CA":
            parts = line.split()
            x = float(parts[6])
            y = float(parts[7])
            z = float(parts[8])
            coordinates.append((x, y, z))
            resids.append(int(line[22:26]))

    return list(zip(coordinates, resids))

    
def min_distance_between_sets(set1, set2):
    """
    Calculates the minimum distance between two sets of coordinates.
    Parameters:
        set1 (List[Tuple[float, float, float]]): First set of coordinates
        set2 (List[Tuple[float, float, float]]): Second set of coordinates
    Returns:
        float: The minimum distance between any pair of coordinates from the
        two sets.
    """
    min_distance = float('inf')
    
    for coord1 in set1:
        for coord2 in set2:
            distance = np.linalg.norm(np.array(coord1) - np.array(coord2))
            if distance < min_distance:
                min_distance = distance
    
    return min_distance


def get_tm_contacts(pocket_coords, all_tm_coords, distance_threshold):
    """
    Determines which transmembrane helices (TM1-TM7, H8) are in contact with a
    given pocket based on a distance threshold.
    Parameters:
        pocket_coords (List[Tuple[float, float, float]]): Coordinates of the
        pocket.
        all_tm_coords (List[List[Tuple[float, float, float]]]): List of
        coordinates for each transmembrane helix (TM1-TM7, H8).
        distance_threshold (float): The distance threshold for considering a
        contact.
    Returns:
        List[str]: A list of transmembrane helix names that are in contact
        with the pocket. If no contacts are found, returns ["None"].
    """
    contact_with = []
    for tm_coords, tm_name in zip(
        all_tm_coords,
        ["tm1", "tm2", "tm3", "tm4", "tm5", "tm6", "tm7", "h8"]
    ):
        min_distance = min_distance_between_sets(pocket_coords, tm_coords)
        if min_distance < distance_threshold:
            contact_with.append(tm_name)
    if not contact_with:
        contact_with.append("None")
    
    return contact_with


def get_every_tm_coords(ref_structure):
    """
    Extracts the coordinates of all transmembrane helices (TM1-TM7, H8) from
    a reference GPCR structure.
    Parameters:
        ref_structure (str): Path to the reference GPCR structure PDB file.
    Returns:
        List[List[Tuple[float, float, float]]]: A list containing lists of
        coordinates for each transmembrane helix (TM1-TM7, H8).
    """
    h8_resids = range(298, 299+1)
    tm7_resids = range(267, 291+1)
    tm6_resids = range(225, 258+1)
    tm5_resids = range(174, 210+1)
    tm4_resids = range(117, 140+1)
    tm3_resids = range(74, 107+1)
    tm2_resids = range(40, 68+1)
    tm1_resids = range(8, 33+1)

    ref_coords_and_resids = read_receptor_ca_coordinates(ref_structure)
    tm1_coords = [coord for coord, resid in ref_coords_and_resids if resid in tm1_resids]
    tm2_coords = [coord for coord, resid in ref_coords_and_resids if resid in tm2_resids]
    tm3_coords = [coord for coord, resid in ref_coords_and_resids if resid in tm3_resids]
    tm4_coords = [coord for coord, resid in ref_coords_and_resids if resid in tm4_resids]
    tm5_coords = [coord for coord, resid in ref_coords_and_resids if resid in tm5_resids]
    tm6_coords = [coord for coord, resid in ref_coords_and_resids if resid in tm6_resids]
    tm7_coords = [coord for coord, resid in ref_coords_and_resids if resid in tm7_resids]
    h8_coords = [coord for coord, resid in ref_coords_and_resids if resid in h8_resids]

    return [tm1_coords, tm2_coords, tm3_coords, tm4_coords, tm5_coords, tm6_coords, tm7_coords, h8_coords]
        

def find(pattern, path):
    """
    Find all files matching a given pattern in a directory and its
    subdirectories.

    Args:
        pattern (str): The pattern to match files against.
        path (str): The directory path to search in.
    Returns:
        generator: A generator yielding paths of files that match the pattern.
    """
    for root, dirs, files in os.walk(path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                yield os.path.join(root, name)


# Main
# Read the reference structure and extract TM helix coordinates.
every_tm_coords = get_every_tm_coords(ref_structure)

# Print the header for the output table.
print("dynid;trajid;pocketid;is_centered;z_location;tm_contacts")

for pocket_pdb in find("*pocket*_tmaligned.pdb", aligned_pockets_dir):
    # Extract identifiers from the pocket PDB filename.
    dynid, trajid, pocketid = extract_ids(pocket_pdb)

    # Read the pocket coordinates and calculate the geometric center.
    pocket_coordinates = read_pocket_coordinates(pocket_pdb)
    geometric_center = calculate_geometric_center(pocket_coordinates)

    # Check if the geometric center is within the defined central cylinder
    # region of the GPCR.
    is_centered = point_in_cylinder(geometric_center, CYLINDER_START, CYLINDER_END, CYLINDER_RADIUS)

    # Determine the z-location of the pocket center.
    z_location = get_z_location(Z_TOP, Z_MIDDLE, Z_BOTTOM, geometric_center)

    # Get the transmembrane contacts for the pocket.
    tm_contacts = get_tm_contacts(pocket_coordinates, every_tm_coords, MIN_DISTANCE_THRESHOLD)

    # Print the results in a semicolon-separated format.
    print(f"{dynid};{trajid};{pocketid};{is_centered};{z_location};{'_'.join(tm_contacts)}")
