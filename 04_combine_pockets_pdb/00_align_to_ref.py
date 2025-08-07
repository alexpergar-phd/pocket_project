"""
This script aligns GPCR structures and their associated pocket files to a
reference GPCR structure using UCSF Chimera.
It processes a range of dynIDs, retrieves metadata from a precomputed JSON file 
and the GPCRmd API, and outputs alignment results and metadata to CSV and log
files.
Main Steps:
1. Loads a list of target dynIDs and reads GPCR metadata from a JSON file.
2. For each dynID in the specified range:
    - Retrieves structure and trajectory file paths.
    - Extracts transmembrane (TM) residue IDs and converts them to Chimera selection ranges.
    - Finds pocket PDB files for each trajectory.
    - Requests additional metadata from the GPCRmd API.
3. For each trajectory and its pockets:
    - Aligns the GPCR structure and pockets to the reference using Chimera's 'mm' command.
    - Saves aligned structures and pockets.
    - Extracts RMSD values from Chimera output.
    - Writes results and metadata to a CSV file.
4. Logs progress and errors to a log file.
Arguments:
- from_index (int): Start index for dynID processing (from sys.argv[1]).
- to_index (int): End index for dynID processing (from sys.argv[2]).
Outputs:
- CSV file with alignment results and metadata.
- Log file with progress and error messages.
- Aligned receptor and pocket PDB files in results directory.
Requirements:
- UCSF Chimera installed and accessible via CHIMERA path.
- GPCRmd data and precomputed JSON file available at specified paths.
"""

# Imports.
import os
import sys
import json
import uuid
import subprocess
import requests
import fnmatch


# Paths to run locally.
# ROOT_GPCRMD = "/files_gpcrmd"
# TMP_DIR = "/home/alex/Desktop"
# REFERENCE_GPCR = "/home/alex/Desktop/static/reference_gpcr/a2a_6gdg_opm_rotated.pdb"
# CHIMERA = "/home/alex/.local/UCSF-Chimera64-1.17.3/bin/chimera"
# RESULTS_DIR = '/home/alex/Desktop/dx_transformation/00_aligning_and_transforming'

# Paths to run in cluster.
ROOT_GPCRMD = "/files_gpcrmd"
TMP_DIR = "/home/aperalta/Desktop"
REFERENCE_GPCR = "/home/aperalta/combine_pockets/static/a2a_6gdg_opm_rotated.pdb"
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


def chimera_align(target_pdb, pocket_pdbs, tm_resids_target_sel, dynid, trajid):
    """
    Aligns the target GPCR PDB and its associated pocket PDBs to a reference GPCR structure
    using UCSF Chimera's 'mm' command.
    Args:
        target_pdb (str): Path to the target GPCR PDB file.
        pocket_pdbs (list of str): List of paths to pocket PDB files.
        tm_resids_target_sel (str): Chimera selection string for transmembrane residues in
                                    the target GPCR.
        dynid (int): Dynamic ID of the GPCR structure.
        trajid (str): Trajectory ID for the GPCR structure.
    Returns:
        str: RMSD value between the aligned target GPCR and the reference GPCR.
    """
    # Construct aligned PDB path.
    aligned_pdb_basename = os.path.basename(target_pdb).replace(".pdb", "_tmaligned.pdb")
    aligned_pdb_path = f'{RESULTS_DIR}/aligned/aligned_receptors/{aligned_pdb_basename}'

    # Construct each of the aligned pocket PDB paths.
    pocket_ids = [os.path.basename(pocket_pdb).split("_")[0] for pocket_pdb in pocket_pdbs]
    aligned_pockets_paths = []
    for pocket_id in pocket_ids:
        aligned_pocket_basename = f'dyn{dynid}_traj{trajid}_pocket{pocket_id}_tmaligned.pdb'
        aligned_pocket_path = f'{RESULTS_DIR}/aligned/aligned_pockets/{aligned_pocket_basename}'
        aligned_pockets_paths.append(aligned_pocket_path)

    # Create the Chimera script to align the target GPCR and all the pockets.
    open_pockets_script = ""
    copy_and_write_script = ""
    for index, (pocket_pdb, aligned_pocket_path) in enumerate(zip(pocket_pdbs, aligned_pockets_paths)):
        open_pockets_script += f'open {pocket_pdb};\n'
        copy_and_write_script += f'matrixcopy #1 #{str(index+2)};\n'  # Apply same transformation to #X model
        copy_and_write_script += f'write relative 0 #{str(index+2)} {aligned_pocket_path};\n'

    chimera_script = (
        f'open {REFERENCE_GPCR}; '                    # Model #0
        f'open {target_pdb}; '                        # Model #1
        + open_pockets_script                         # Open all pocket PDBs
        + f'mm #1:{tm_resids_target_sel} #0:{TM_RESIDS_REF}; '  # Align model #1 to #0
        + f'select #1 & protein; write selected relative 0 #1 {aligned_pdb_path};' # Save aligned target GPCR
        + copy_and_write_script                       # Apply same transformation to all pockets and save them
    )

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

    # Extract the RMSD value from the Chimera output.
    terminal_output = result.stdout
    for line in terminal_output.split("\n"):
        if line.startswith("RMSD between"):
            rmsd = line.split("pairs: ")[1].strip(")")
    
    # Clean up the temporary files.
    os.remove(tmp_script)

    return rmsd


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


### MAIN ###
# Get the target dynids.
target_dynids = get_target_dynids()

# Read the GPCRmd data into a dictionary.
compl_info = read_json_into_dict(COMPL_INFO_PATH)

# Create the folders for the results.
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "aligned"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "aligned", "aligned_receptors"), exist_ok=True)
os.makedirs(os.path.join(RESULTS_DIR, "aligned", "aligned_pockets"), exist_ok=True)

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

    # Get the trajectory file paths and the different pocket files.
    # Save everything to a list.
    trajid_results = []
    for trajfile in entry_data.get('traj_fnames', None):
        trajid = trajfile.split("_")[0]
        
        mdpocket_dir = os.path.join(
            ROOT_GPCRMD,
            "Precomputed/MDpocket",
            f"{str(trajid)}_trj_{str(dynid)}.xtc_mdpocket"
        )
        
        pocket_pdbs = list(find("*_coordinates_DBSCAN.pdb", mdpocket_dir))
        trajid_results.append({
            'trajid': trajid,
            'traj_path': trajfile,
            'pockets_pdbs': pocket_pdbs
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

    # For each trajectory...
    for trajid_result in dynid_result['trajid_results']:
        trajid = trajid_result['trajid']
        pockets_pdbs = trajid_result['pockets_pdbs']
        
        # Align the target PDB to the reference GPCR and save the aligned PDB.
        try:
            rmsd = chimera_align(
                target_pdb=pdb_path,
                pocket_pdbs=pockets_pdbs,
                tm_resids_target_sel=tm_resids_sel,
                dynid=dynid,
                trajid=trajid
            )
        except FileNotFoundError as e:
            print(f"[ERROR] PDB file for dynid {dynid} not found: {e}", file=log_fh)
            continue
        except UnboundLocalError as e:
            print(f"[ERROR] There was a problem with {dynid}: {e}", file=log_fh)
            continue


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

        # Print the values in "csv_results" to the console separated by commas
        with open(csv_filepath, 'a') as f:
            # Check if the file is empty. If it is, write the header.
            if os.stat(csv_filepath).st_size == 0:
                f.write(";".join(csv_results.keys()) + "\n")
            # Write the values.
            f.write(";".join([ str(value) for key, value in csv_results.items() ]) + "\n")

log_fh.close()

