"""
This script processes GPCRmd data to align protein structures to a reference GPCR structure 
and transform density files. It performs the following steps:
1. Retrieves a list of dynIDs with precomputed pockets from the GPCRmd database.
2. Reads GPCRmd data from a JSON file and extracts relevant information for each dynID.
3. Aligns protein structures to a reference GPCR structure using Chimera, generating 
    rotation matrices and translation vectors.
4. Transforms density files (DX format) using the computed transformation matrices.
5. Saves the results, including aligned PDB files, transformed DX files, and metadata, 
    to specified directories and a CSV file.
Global Variables:
- ROOT_GPCRMD: Path to the root directory containing GPCRmd data.
- TMP_DIR: Path to the temporary directory for storing intermediate files.
- RESULTS_DIR: Path to the directory where results will be saved.
- TM_RESIDS_REF: Residue selection string for the reference GPCR structure.
- COMPL_INFO_PATH: Path to the JSON file containing GPCRmd data.
Execution:
- The script processes dynIDs within a specified range (from_index to to_index) provided as 
  command-line arguments.
- It aligns protein structures, transforms density files, and saves results to the output 
  directories and a CSV file.
"""

# Imports.
import os
import sys
import json
import uuid
import subprocess
import requests

import numpy as np

from dx_reader import read_dx_file, write_dx, apply_transformation_to_dx


# Paths to run locally.
# ROOT_GPCRMD = "/files_gpcrmd"
# TMP_DIR = "/home/alex/Desktop"
# REFERENCE_GPCR = "/home/alex/Desktop/static/reference_gpcr/a2a_6gdg_opm_rotated.pdb"
# CHIMERA = "/home/alex/.local/UCSF-Chimera64-1.17.3/bin/chimera"
# RESULTS_DIR = '/home/alex/Desktop/dx_transformation/00_aligning_and_transforming'

# Paths to run in cluster.
ROOT_GPCRMD = "/files_gpcrmd"
TMP_DIR = "/home/aperalta/Desktop"
REFERENCE_GPCR = "/home/aperalta/combine_pockets/a2a_6gdg_opm_rotated.pdb"
CHIMERA = "/soft/system/software/Chimera/1.16/bin/chimera"
RESULTS_DIR = '/home/aperalta/combine_pockets/results'

# Constants
TM_RESIDS_REF = "1-34,39-69,73-108,117-142,173-213,219-259,266-291" 
# ^-- Obtained by extracting it from GPCRdb GPCR "aa2ar_human".

COMPL_INFO_PATH = os.path.join(ROOT_GPCRMD, "Precomputed/compl_info.json")


# Arguments
from_index = int(sys.argv[1])
to_index = int(sys.argv[2])


# Functions.
def get_target_dynids():
    """
    This is the list of dynIDs that have pockets generated. It is obtained
    from the GPCRmd database.
    """
    return [10, 100, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008,
    1009, 101, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019,
    102, 1020, 1021, 1022, 1023, 1024, 1025, 1026, 1027, 1028, 1029, 103,
    1030, 1031, 1032, 1033, 104, 1049, 105, 106, 107, 108, 109, 11, 110,
    1105, 1106, 1107, 1108, 1109, 111, 1110, 1111, 112, 113, 114, 115, 116,
    117, 118, 119, 12, 120, 121, 122, 123, 124, 125, 126, 1269, 127, 1270,
    1271, 1272, 1273, 1274, 1275, 128, 129, 13, 130, 131, 132, 133, 134, 135,
    136, 137, 138, 139, 14, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
    15, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 16, 160, 161, 162,
    163, 164, 165, 166, 167, 168, 169, 17, 170, 171, 172, 173, 174, 175, 176,
    177, 178, 179, 18, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 19,
    190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 20, 200, 201, 202, 203,
    204, 205, 21, 22, 23, 234, 237, 238, 239, 24, 240, 241, 242, 243, 244,
    245, 247, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 4,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 6, 60, 61, 62, 63, 64, 65, 66, 67, 68, 687, 688, 689, 69, 690,
    691, 692, 693, 694, 695, 696, 697, 698, 699, 7, 70, 700, 702, 703, 704,
    705, 706, 707, 708, 709, 71, 710, 711, 712, 713, 714, 715, 716, 717, 718,
    719, 72, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 73, 730, 731,
    732, 733, 734, 735, 736, 737, 738, 739, 74, 740, 741, 742, 743, 744, 745,
    746, 748, 749, 75, 750, 751, 752, 753, 754, 755, 756, 757, 758, 759, 76,
    760, 761, 762, 763, 764, 765, 766, 767, 768, 769, 77, 770, 771, 772, 773,
    774, 775, 777, 778, 779, 78, 781, 782, 783, 784, 785, 786, 787, 788, 789,
    79, 790, 791, 792, 793, 794, 795, 796, 797, 799, 8, 80, 800, 801, 802,
    803, 804, 805, 806, 807, 808, 809, 81, 810, 811, 812, 813, 814, 815, 816,
    817, 818, 819, 82, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829, 83,
    830, 831, 832, 833, 834, 835, 836, 837, 838, 839, 84, 840, 841, 842, 843,
    844, 845, 846, 847, 848, 849, 85, 850, 851, 852, 853, 854, 855, 856, 857,
    858, 859, 86, 860, 861, 862, 863, 864, 865, 866, 869, 87, 870, 871, 872,
    873, 874, 875, 876, 877, 878, 879, 88, 880, 881, 884, 885, 886, 887, 888,
    889, 89, 890, 891, 892, 893, 894, 895, 896, 897, 898, 899, 90, 900, 901,
    902, 903, 904, 905, 906, 908, 909, 91, 910, 911, 912, 913, 914, 915, 916,
    917, 918, 919, 92, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 93,
    930, 931, 932, 933, 934, 935, 936, 937, 938, 939, 94, 940, 941, 943, 944,
    945, 946, 947, 949, 95, 950, 951, 952, 953, 954, 955, 956, 957, 958, 959,
    96, 960, 961, 962, 963, 964, 965, 966, 967, 968, 97, 970, 971, 973, 975,
    976, 977, 978, 979, 98, 980, 981, 982, 983, 984, 985, 986, 987, 988, 989,
    99, 990, 991, 992, 993, 994, 995, 996, 997, 998, 999]


def read_json_into_dict(json_path):
    """
    Reads a JSON file and returns the data as a dictionary.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def request_gpcrmd(dynid):
    """
    Requests the GPCRmd database for a specific dynid and returns the data.
    This function is not used in the current script but is provided for 
    completeness.
    """
    url = f"https://www.gpcrmd.org/api/search_dyn/info/{dynid}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"[ERROR] Error fetching data for dynid {dynid}: {response.status_code}")
        return None


def align_and_get_rotation_maxtrix(target_pdb, tm_resids_target_sel):
    """
    Aligns a target protein structure to a reference structure and retrieves the 
    rotation matrix and translation vector for the alignment.
    This function uses Chimera to perform the alignment and generate the transformation 
    matrix. Temporary files are created for the Chimera script and the transformation 
    matrix, which are cleaned up after execution.
    Args:
        target_pdb (str): Path to the target PDB file to be aligned.
        tm_resids_target_sel (str): Residue selection string for the target structure 
            to be used in the alignment.
    Returns:
        tuple: A tuple containing:
            - rotation_matrix (list of list of float): The 3x3 rotation matrix.
            - translation_vector (list of float): The translation vector.
    Raises:
        OSError: If there is an issue executing the Chimera command or accessing 
            temporary files.
    Note:
        This function assumes the existence of the following global variables:
        - REFERENCE_GPCR: Path to the reference GPCR structure.
        - TM_RESIDS_REF: Residue selection string for the reference structure.
        - TMP_DIR: Directory for storing temporary files.
        - CHIMERA: Path to the Chimera executable.
    """
    transmatrix_path = f'{TMP_DIR}/transmatrix_{str(uuid.uuid4())}.dat'
    aligned_pdb_basename = os.path.basename(target_pdb).replace(".pdb", f"_TM_aligned_to_ref.pdb")
    aligned_pdb_path = f'{RESULTS_DIR}/aligned_pdbs/{aligned_pdb_basename}'
    chimera_script = (
        f'open {REFERENCE_GPCR}; open {target_pdb}; '
        f'mm #1:{tm_resids_target_sel} #0:{TM_RESIDS_REF};'
        f'matrixget {transmatrix_path};'
        f'write relative 0 #1 {aligned_pdb_path};')

    # Writing and executing the chimera align script.
    tmp_script = os.path.join(
        TMP_DIR, f'tmp_chimera_script_{str(uuid.uuid4())}')
    with open(tmp_script, "w") as fh:
        print(chimera_script, file=fh)

    result = subprocess.run(
        [CHIMERA, '--nogui', f'cmd:{tmp_script}'], 
        stdout=subprocess.PIPE,  # Capture standard output
        text=True                # Decode output as string
    )

    terminal_output = result.stdout
    for line in terminal_output.split("\n"):
        if line.startswith("RMSD between"):
            rmsd = line.split("pairs: ")[1].strip(")")

    # Read the transmatrix file.
    rotation_matrix, translation_vector = read_transmatrix(transmatrix_path)
    
    # Clean up the temporary files.
    os.remove(tmp_script)
    os.remove(transmatrix_path)

    return rotation_matrix, translation_vector, rmsd, aligned_pdb_path


def read_transmatrix(filepath):
    """
    Reads a 3x4 transformation matrix from a file and separates it into a 
    rotation matrix and a translation vector.
    The function expects the file to contain a section labeled "Model 0.0" 
    with three lines, each containing four floating-point numbers. The first 
    three numbers in each line represent the rotation matrix, and the fourth 
    number represents the translation vector.
    Args:
        filepath (str): The path to the file containing the transformation matrix.
    Returns:
        tuple: A tuple containing:
            - rotation_matrix (list of list of float): A 3x3 rotation matrix.
            - translation_vector (list of float): A 3-element translation vector.
    Raises:
        ValueError: If the "Model 0.0" section does not contain a valid 3x4 matrix.
    """
    rotation_matrix = []
    translation_vector = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    in_model_0 = False
    matrix_lines = []

    for line in lines:
        line = line.strip()
        if line.startswith("Model"):
            if line == "Model 0.0":
                in_model_0 = True
            else:
                in_model_0 = False
        elif in_model_0 and line:
            matrix_lines.append([float(x) for x in line.split()])

    if len(matrix_lines) != 3:
        raise ValueError("Model 0.0 does not contain a valid 3x4 matrix.")

    for row in matrix_lines:
        rotation_matrix.append(row[:3])
        translation_vector.append(row[3])

    return rotation_matrix, translation_vector


def get_tm_resids(entry_data):
    """
    Extracts and returns a list of residue IDs corresponding to transmembrane (TM) regions
    from the given entry data.
    The function processes the 'gpcr_pdb' dictionary within the entry data, identifying
    residues associated with Ballesteros-Weinstein (BW) numbering that belong to TM regions
    (BW numbers starting with digits 1 through 7 followed by 'x').
    Args:
        entry_data (dict): A dictionary containing GPCR-related data. It is expected to have
                           a 'gpcr_pdb' key, which maps BW numbers to strings in the format
                           "resid-chain-resname".
    Returns:
        list: A list of integers representing the residue IDs of TM regions.
    """
    tm_resids = []
    for bw, resid_chain_resname in entry_data.get('gpcr_pdb', {}).items():
        if bw[0] in "1234567" and bw[1] == "x":
            resid = resid_chain_resname.split("-")[0]
            tm_resids.append(int(resid))
    return tm_resids


def convert_to_ranges(resids):
    """
    Converts a list of residue IDs into a range format (e.g., 33-94,102-137).
    This is for a correct functioning of Chimera's mm command.

    Args:
        resids (list of int): List of residue IDs.

    Returns:
        str: A string with ranges of residue IDs.
    """
    resids = list(map(int, resids))  # Ensure all are integers
    resids = sorted(set(resids))  # Sort and remove duplicates
    ranges = []
    start = resids[0]
    end = resids[0]

    for i in range(1, len(resids)):
        if resids[i] == end + 1:
            # Extend the range
            end = resids[i]
        else:
            # Close the current range and start a new one
            if start == end:
                ranges.append(f"{start}")
            else:
                ranges.append(f"{start}-{end}")
            start = resids[i]
            end = resids[i]

    # Add the final range
    if start == end:
        ranges.append(f"{start}")
    else:
        ranges.append(f"{start}-{end}")

    return ",".join(ranges)


### MAIN ###
# Get the target dynids.
target_dynids = get_target_dynids()

# Read the GPCRmd data into a dictionary.
compl_info = read_json_into_dict(COMPL_INFO_PATH)

# Create the folders for the results.
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "aligned_pdbs"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "dx_files"), exist_ok=True)

# Output file paths.
csv_filepath = os.path.join(RESULTS_DIR, f"results_{str(from_index)}_{str(to_index)}.csv")
log_filepath = os.path.join(RESULTS_DIR, f"log_{str(from_index)}_{str(to_index)}.txt")
log_fh = open(log_filepath, 'w')

# 1. GETTING FILE PATHS, SELECTIONS, ETC.
print("[*] Getting file paths, making request and taking selections...", file=log_fh)
dynids_results = []
for index, dynid in enumerate(target_dynids[from_index:to_index]):
    print(f"[*] Processing dynID {dynid} ({index + 1}/{len(target_dynids)})...", file=log_fh)

    # Check if the dynid is in the compl_info dictionary.
    entry_data = compl_info.get(f'dyn{str(dynid)}', None)
    if entry_data is None:
        print(f"[ERROR] Entry for dynID {dynid} not found in compl_info.", file=log_fh)
        continue

    # Get the PDB path for the structure.
    pdb_path_noroot = entry_data.get('struc_f', None)[1:] # Remove the first "/"
    pdb_path = os.path.join(ROOT_GPCRMD, pdb_path_noroot)

    # Get the TM residues and convert it to a chimera selection (ranges).
    tm_resids = get_tm_resids(entry_data)
    tm_resids_sel = convert_to_ranges(tm_resids)

    # Get the trajectory file paths and the density of pockets.
    # Save everything to a list.
    trajid_results = []
    for trajfile in entry_data.get('traj_fnames', None):
        trajid = trajfile.split("_")[0]
        
        pockets_freq_file = os.path.join(
            ROOT_GPCRMD,
            "Precomputed/MDpocket/"
            f"{str(trajid)}_trj_{str(dynid)}.xtc_mdpocket/"
            "mdpocket/mdpout_freq.dx"
        )

        trajid_results.append({
            'trajid': trajid,
            'traj_path': trajfile,
            'pockets_freq_file': pockets_freq_file
        })

    # Get additional info from GPCRdb for the CSV file.
    request_result = request_gpcrmd(dynid)
    if request_result == []:
        print(f"[ERROR] No data found for dynID {dynid} in GPCRmd.", file=log_fh)
        continue
    if request_result == None:
        print(f"[ERROR] Error fetching data for dynid {dynid}. Try again later.", file=log_fh)
        continue
    gpcrmd_data = request_result[0]

    # Save dynid data to the dynids_results list.
    dynids_results.append({
        'dynid': dynid,
        'pdb_path': pdb_path,
        'is_apoform': True if entry_data['lig_lname'] or entry_data['lig_sname'] else False,
        'tm_resids_sel': tm_resids_sel,
        'gpcrmd_data': gpcrmd_data,
        'trajid_results': trajid_results
    })

# 2. ALIGNING AND TRANSFORMING THE DENSITY FILES.
print("[*] Aligning and transforming the density files...", file=log_fh)

for index, dynid_result in enumerate(dynids_results):
    print(f"[*] Processing dynID {dynid} ({index + 1}/{len(target_dynids)})...", file=log_fh)
    dynid = dynid_result['dynid']
    pdb_path = dynid_result['pdb_path']
    tm_resids_sel = dynid_result['tm_resids_sel']
    is_apoform = dynid_result['is_apoform']
    gpcrmd_data = dynid_result['gpcrmd_data']

    # Align the first frame of the trajectory to the reference GPCR.
    try:
        rotation_matrix, translation_vector, rmsd, aligned_pdb_path = \
            align_and_get_rotation_maxtrix(pdb_path, tm_resids_sel)
    except FileNotFoundError as e:
        print(f"[ERROR] PDB file for dynid {dynid} not found", file=log_fh)

    # For each trajectory, transform the DX file.
    for trajid_result in dynid_result['trajid_results']:
        trajid = trajid_result['trajid']
        pockets_freq_file = trajid_result['pockets_freq_file']
        
        # Read the dx file.
        dx_data = read_dx_file(pockets_freq_file)

        # Transform the dx file.
        dx_data_transformed = apply_transformation_to_dx(
            dx_data, rotation_matrix, translation_vector)
        
        # Write the transformed dx file.
        new_df_file_basename = os.path.basename(
            pockets_freq_file.replace(".dx", f"_{dynid}_{trajid}_aligned.dx"))
        new_dx_file_path = os.path.join(RESULTS_DIR,
                                        "dx_files", new_df_file_basename)

        write_dx(
            filename=new_dx_file_path,
            origin=dx_data_transformed['origin'],
            deltas=dx_data_transformed['deltas'],
            counts=dx_data_transformed['counts'],
            data=np.array(dx_data_transformed['scalars']).reshape(dx_data_transformed['counts']),
            title="DX file transformed to aligned reference GPCR")

        # Save the results to the CSV file.
        csv_results = {}
        csv_results["dynid"] = dynid
        csv_results["trajid"] = trajid
        csv_results["uniprotid"] = gpcrmd_data['uniprot']
        csv_results["uniprot_name"] = gpcrmd_data['uprot_entry']
        csv_results["protein_name"] = gpcrmd_data['protname']
        csv_results["pdb_code"] = gpcrmd_data['pdb_namechain']
        csv_results["class_name"] = gpcrmd_data['class_name']
        try:
            csv_results["state"] = gpcrmd_data['state'][0]['name']
        except IndexError as e:
            print(f"[WARNING] No state found for dynID {dynid}: {e}", file=log_fh)
            csv_results["state"] = None
        csv_results["species"] = gpcrmd_data['species']
        csv_results["family_slug"] = gpcrmd_data['fam_slug']
        csv_results["family_name"] = gpcrmd_data['fam_name']
        csv_results["model_name"] = gpcrmd_data['modelname']
        csv_results["is_apoform"] = is_apoform
        csv_results["rmsd_to_ref"] = rmsd
        csv_results["tm_resids_sel"] = tm_resids_sel
        csv_results["pdb_path"] = pdb_path
        csv_results["aligned_pdb_path"] = aligned_pdb_path
        csv_results["pockets_freq_file"] = pockets_freq_file
        csv_results["new_dx_file_path"] = new_dx_file_path
        csv_results["rotation_matrix"] = rotation_matrix
        csv_results["translation_vector"] = translation_vector

        # Print the values in "csv_results" to the console separated by commas
        with open(csv_filepath, 'a') as f:
            # Check if the file is empty. If it is, write the header.
            if os.stat(csv_filepath).st_size == 0:
                f.write(";".join(csv_results.keys()) + "\n")
            # Write the values.
            f.write(";".join([ str(value) for key, value in csv_results.items() ]) + "\n")

log_fh.close()