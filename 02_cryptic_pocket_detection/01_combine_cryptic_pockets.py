"""
This script takes the dataframe of cryptic pockets (dynIDs, trajIDs and
pocketIDs), aligns the structures to a reference GPCR backbone using Chimera,
and saves the aligned pockets and receptors in designated directories.
"""


import os
import json
import pandas as pd
import uuid
import subprocess


#~~ Constants ~~#
input_csv_path = "/home/aperalta/Documents/pocket_tool/results/02_cryptic_pocket_detection/00_detecting_based_on_volume/output_251022/cryptic_pockets_filtered_10perc.csv"
results_dir = "/home/aperalta/Documents/pocket_tool/results/02_cryptic_pocket_detection/02_selecting_10percent/00_aligning/output_251022"


#~~ Constants ~~#
ROOT_GPCRMD = "/home/aperalta/sshfs_mountpoints/gpcrmd/media/computation/files"
TMP_DIR = "/home/aperalta/Desktop"
REFERENCE_GPCR = "/home/aperalta/Documents/pocket_tool/data/ref_gpcr/data/classA_adrb2_2rh1_rotated.pdb"
CHIMERA = "/soft/system/software/Chimera/1.16/bin/chimera"
COMPL_INFO_PATH = "/home/aperalta/sshfs_mountpoints/gpcrmd/media/files/Precomputed/compl_info.json"

TM_RESIDS_REF = "26-61,66-97,102-137,146-172,196-237,262-299,304-328" # For 2RH1


#~~ Functions ~~#
def get_target_ids(csv_path):
    """
    Extracts unique dynamic IDs, trajectory IDs, and pocket IDs from a CSV
    file.
    """
    df = pd.read_csv(csv_path, sep=";")
    dyn_ids = df["dyn_id"]
    traj_ids = df["traj_id"]
    pocket_ids = df["pocket_id"]

    return zip(dyn_ids, traj_ids, pocket_ids)


def read_json_into_dict(json_path):
    """
    Reads a JSON file and returns the data as a dictionary.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


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


def generate_dirs():
    """
    Generates necessary directories for storing results.
    """
    os.makedirs(f'{results_dir}/aligned_pdbs/', exist_ok=True)
    os.makedirs(f'{results_dir}/aligned_pdbs/aligned_pockets', exist_ok=True)
    os.makedirs(f'{results_dir}/aligned_pdbs/aligned_receptors', exist_ok=True)


def chimera_align(target_pdb, pocket_pdb, tm_resids_target_sel, ids):
    """
    """
    aligned_pdb_basename = os.path.basename(target_pdb).replace(".pdb", "_tmaligned.pdb")
    aligned_pdb_path = f'{results_dir}/aligned_pdbs/aligned_receptors/{aligned_pdb_basename}'

    # aligned_pocket_basename = os.path.basename(pocket_pdb).replace(".pdb", "_tmaligned.pdb")
    aligned_pocket_basename = f'dyn_{ids[0]}_traj_{ids[1]}_pocket_{ids[2]}_tmaligned.pdb'
    aligned_pocket_path = f'{results_dir}/aligned_pdbs/aligned_pockets/{aligned_pocket_basename}'

    chimera_script = (
        f'open {REFERENCE_GPCR}; '                    # Model #0
        f'open {target_pdb}; '                        # Model #1
        f'open {pocket_pdb}; '                        # Model #2
        f'mm #1:{tm_resids_target_sel} #0:{TM_RESIDS_REF}; '  # Align model #1 to #0
        f'matrixcopy #1 #2; '                         # Apply same transformation to #2
        f'select #1 & protein; write selected relative 0 #1 {aligned_pdb_path};' # Save aligned target GPCR
        f'write relative 0 #2 {aligned_pocket_path};' # Save aligned pocket
    )

    # Writing and executing the chimera align script.
    tmp_script = os.path.join(
        TMP_DIR, f'tmp_chimera_script_{str(uuid.uuid4())}')
    with open(tmp_script, "w") as fh:
        print(chimera_script, file=fh)

    subprocess.run(
        [CHIMERA, '--nogui', f'cmd:{tmp_script}'], 
        stdout=subprocess.PIPE,  # Capture standard output
        text=True                # Decode output as string
    )

    # Clean up the temporary files.
    os.remove(tmp_script)


#~~ Main ~~#
generate_dirs()

compl_info = read_json_into_dict(COMPL_INFO_PATH)

for dyn_id, traj_id, pocket_id in get_target_ids(input_csv_path):
    print(f"[PROGRESS] Processing dynID {dyn_id}, trajectory {traj_id}, pocket {pocket_id} ...")
    entry_data = compl_info.get(f'dyn{str(dyn_id)}', None)
    if entry_data is None:
        print(f"[ERROR] Entry for dynID {dyn_id} not found in compl_info.")
        continue

    # Get the PDB path for the structure.
    pdb_path_noroot = entry_data.get('struc_f', None)[1:] # Remove the first "/"
    pdb_path = os.path.join(ROOT_GPCRMD, pdb_path_noroot)

    # Get the TM residues and convert it to a chimera selection (ranges).
    tm_resids = get_tm_resids(entry_data)
    tm_resids_sel = convert_to_ranges(tm_resids)

    # Get the pocket file path.
    pocket_pdb_file = os.path.join(
        ROOT_GPCRMD,
        "Precomputed/MDpocket/"
        f"{str(traj_id)}_trj_{str(dyn_id)}.xtc_mdpocket/"
        f"DBSCANclustering_coordinates_pdb_{str(traj_id)}_trj_{str(dyn_id)}.xtc/"
        f"{str(pocket_id)}_coordinates_DBSCAN.pdb"
    )

    # Align the structures using Chimera.
    chimera_align(
        target_pdb=pdb_path,
        pocket_pdb=pocket_pdb_file,
        tm_resids_target_sel=tm_resids_sel,
        ids=[dyn_id, traj_id, pocket_id]
    )
