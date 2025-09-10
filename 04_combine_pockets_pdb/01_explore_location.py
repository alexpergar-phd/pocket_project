"""
Script to explore the location of pockets in relation to the protein structure.
It reads pocket PDB files, calculates their geometric centers, determines their
z-location, and identifies contacts with transmembrane helices. It also checks
if the pockets are centered within the protein core based on the positions of
the closest C-alpha atoms from the transmembrane helices.
"""


import os

import numpy as np
import fnmatch


# Paths
aligned_pockets_dir = "/home/aperalta/Documents/pocket_tool/results/04_combine_pockets_pdb/00_aligning/aligned/aligned_pockets"
ref_structure = "/home/aperalta/Documents/pocket_tool/data/ref_gpcr/data/classA_adrb2_2rh1_rotated.pdb"

# Constants for z-coordinates that denote the ICL, center, and ECL regions.
Z_MIDDLE = -2.5
Z_TOP = 17
Z_BOTTOM = -22  

# Minimum distance threshold for considering a contact with transmembrane
# helices.
MIN_DISTANCE_THRESHOLD = 5.0

# TM residues for the ADRB2 receptor (PDB ID: 2RH1).
TM1_RESIDS = list(range(26, 60))
TM2_RESIDS = list(range(37, 68))
TM3_RESIDS = list(range(102, 137))
TM4_RESIDS = list(range(146, 172))
TM5_RESIDS = list(range(196, 237))
TM6_RESIDS = list(range(262, 299))
TM7_RESIDS = list(range(304, 328))
H8_RESIDS = list(range(329, 343))
every_tm_coords = [TM1_RESIDS, TM2_RESIDS, TM3_RESIDS, TM4_RESIDS, TM5_RESIDS, TM6_RESIDS, TM7_RESIDS, H8_RESIDS]


# Functions.
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
    Calculates the geometric center of a list of 3D coordinates.
    Parameters:
        coordinates (List[Tuple[float, float, float]]): List of 3D coordinates.
    Returns:
        List[float, float, float]: The geometric center (x, y, z).
    """
    x_sum = sum(coord[0] for coord in coordinates)
    y_sum = sum(coord[1] for coord in coordinates)
    z_sum = sum(coord[2] for coord in coordinates)

    n = len(coordinates)
    return [x_sum / n, y_sum / n, z_sum / n]


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


def read_ca_coordinates_per_helix(pdb_file):
    """
    Reads C-alpha coordinates for each helix from a PDB file.
    Parameters:
        pdb_file (str): Path to the PDB file to be read.
    Returns:
        Dict[str, List[Tuple[float, float, float]]]: A dictionary where keys
        are helix names and values are lists of C-alpha coordinates (x, y, z).
    """
    with open(pdb_file, 'r') as file:
        lines = file.readlines()

    ca_coordinates = {}
    for line in lines:
        line = line.strip()
        if not (line.startswith("ATOM") and line[13:15] == "CA"):
            continue

        resid = int(line[22:26].strip())
        x = float(line[30:38].strip())
        y = float(line[38:46].strip())
        z = float(line[46:54].strip())
        coordinates = [x, y, z]

        if resid in TM1_RESIDS:
            ca_coordinates.setdefault("TM1", []).append(coordinates)
        elif resid in TM2_RESIDS:
            ca_coordinates.setdefault("TM2", []).append(coordinates)
        elif resid in TM3_RESIDS:
            ca_coordinates.setdefault("TM3", []).append(coordinates)
        elif resid in TM4_RESIDS:
            ca_coordinates.setdefault("TM4", []).append(coordinates)
        elif resid in TM5_RESIDS:
            ca_coordinates.setdefault("TM5", []).append(coordinates)
        elif resid in TM6_RESIDS:
            ca_coordinates.setdefault("TM6", []).append(coordinates)
        elif resid in TM7_RESIDS:
            ca_coordinates.setdefault("TM7", []).append(coordinates)

    return ca_coordinates


def get_closest_ca_per_helix(target_coord, ca_coordinates):
    """
    Finds the closest C-alpha coordinate for each helix to a target coordinate.
    Parameters:
        target_coord (Tuple[float, float, float]): The target coordinate (x, y, z).
        ca_coordinates (Dict[str, List[Tuple[float, float, float]]]): A dictionary where keys are helix names
            and values are lists of C-alpha coordinates (x, y, z).
    Returns:
        List[Tuple[float, float, float]]: A list of closest C-alpha coordinates"""
    closest_coords = []
    for helix, coords in ca_coordinates.items():
        closest = min(coords, key=lambda c: ((c[0] - target_coord[0]) ** 2 + (c[1] - target_coord[1]) ** 2 + (c[2] - target_coord[2]) ** 2) ** 0.5)
        closest_coords.append(closest)
    return closest_coords


def calculate_2point_vector(coord1, coord2):
    """
    Calculates the vector between two 3D coordinates.
    Parameters:
        coord1 (Tuple[float, float, float]): The first coordinate (x, y, z).
        coord2 (Tuple[float, float, float]): The second coordinate (x, y, z).
    Returns:
        List[float, float, float]: The vector from coord1 to coord2.
    """
    return [coord2[0] - coord1[0], coord2[1] - coord1[1], coord2[2] - coord1[2]]


def check_if_pocket_is_centered(pocket_center, closest_cas, interior_center):
    """
    Checks if a pocket is centered within the protein core based on the dot products
    of vectors from the pocket center and interior center to the closest C-alpha atoms.
    Parameters:
        pocket_center (Tuple[float, float, float]): The coordinates of the pocket center (x, y, z).
        closest_cas (List[Tuple[float, float, float]]): A list of coordinates of the closest C-alpha atoms (x, y, z).
        interior_center (Tuple[float, float, float]): The coordinates of the interior center (x, y, z).
    Returns:
        bool: True if the pocket is centered, False otherwise.
    """
    dot_products = []
    for closest_ca in closest_cas:
        ca2center = calculate_2point_vector(pocket_center, closest_ca)
        ca2pth = calculate_2point_vector(interior_center, closest_ca)
        dot_products.append(np.dot(ca2center[0:2], ca2pth[0:2]))  # Only use x and y components for the dot product

    return all(dp > 0 for dp in dot_products)


# Main.
# Read the C-alpha coordinates per helix from the reference structure.
ca_coordinates = read_ca_coordinates_per_helix(ref_structure)

# Remove TM1 and TM4 from consideration as they are far from the core.
del ca_coordinates["TM4"]
del ca_coordinates["TM1"]

# Print the header for the output table.
print("dynid;trajid;pocketid;is_centered;z_location;tm_contacts")

for pocket_pdb in find("*pocket*_tmaligned.pdb", aligned_pockets_dir):
    # Extract identifiers from the pocket PDB filename.
    dynid, trajid, pocketid = extract_ids(pocket_pdb)

    # Read the pocket coordinates and calculate the geometric center.
    pocket_coordinates = read_pocket_coordinates(pocket_pdb)
    pocket_center = calculate_geometric_center(pocket_coordinates)

    # The the CA coordinates closest to the pocket center, one per helix
    # (making kind of a "ring"). Then calculate its center.
    closest_cas = get_closest_ca_per_helix(pocket_center, ca_coordinates)
    interior_center = calculate_geometric_center(closest_cas)

    # Check if the pocket is centered within the protein core.
    is_centered = check_if_pocket_is_centered(pocket_center, 
                                              closest_cas, interior_center)

    # Determine the z-location of the pocket center.
    z_location = get_z_location(Z_TOP, Z_MIDDLE, Z_BOTTOM, pocket_center)

    # Get the transmembrane contacts for the pocket.
    tm_contacts = get_tm_contacts(pocket_coordinates, every_tm_coords, MIN_DISTANCE_THRESHOLD)

    # Print the results in a semicolon-separated format.
    print(f"{dynid};{trajid};{pocketid};{is_centered};{z_location};{'_'.join(tm_contacts)}")
