"""
Script to read the pocket descriptor files and generate a CSV file with the
following columns:
- frame
- dyn_id
- traj_id
- pocket_id
- pock_volume
- pock_pol_asa
- pock_apol_asa
"""


import os
import fnmatch
import re

import pandas as pd


### PATHS ###
PROJECT_ROOT = "/home/aperalta/Documents/pocket_tool"
pocket_dir = "/home/aperalta/sshfs_mountpoints/mdpocket_oversized"


### FUNCTIONS ###
def read_dynid_trajids():
    trajid_dict = {}
    trajids_path = os.path.join(PROJECT_ROOT, "data/working_sims/data/dynids_trajids.csv")
    with open(trajids_path, 'r') as file:
        next(file) # Skip header line
        for line in file:
            dynid, trajid = line.strip().split(';')
            if dynid in trajid_dict:
                trajid_dict[dynid].append(trajid)
            else:
                trajid_dict[dynid] = [trajid]
    return trajid_dict


def read_descriptors_file(file_path):
    """
    Read a tab-separated file and return a dictionary where each key is a
    column name and each value is a list of column values.

    Args:
        file_path (str): Path to the tab-separated file.

    Returns:
        dict: A dictionary with column names as keys and lists of values as
        values.
    """
    values = {}
    fh = open(file_path, 'r')

    # Read the header line and initialize the values dictionary
    header = fh.readline()
    for field in header.split():
        values[field] = []
    
    # Read the rest of the file and populate the values dictionary
    for line in fh:
        fields = line.split()
        for i, field in enumerate(header.split()):
            values[field].append(fields[i])

    fh.close()
    return values


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


def extract_ids(path):
    """
    Extracts trajectory ID, dynID, and pocket ID from a given file path.
    
    Parameters:
        path (str): The full file path.
        
    Returns:
        dict: A dictionary with keys 'trajectory_id', 'dyn_id', and 'pocket_id'.
    """
    # Match descriptorPockets_<traj_id>_trj_<dyn_id>.xtc
    trajectory_match = re.search(r'descriptorPockets_(\d+)_trj_(\d+)\.xtc', path)
    # Match pocket_num_<pocket_id>_descriptors.txt
    pocket_match = re.search(r'pocket_num_(\d+)_descriptors\.txt', path)

    if trajectory_match and pocket_match:
        traj_id = trajectory_match.group(1)
        dyn_id = trajectory_match.group(2)
        pocket_id = pocket_match.group(1)

        return (traj_id, dyn_id, pocket_id)
    else:
        print(f"Could not extract all required IDs from the path: {path}.")
        return None, None, None


### Main ###
df_dict = {
    'frame': [],
    'dyn_id': [],
    'traj_id': [],
    'pocket_id': [],
    'pock_volume': [],
    'pock_pol_asa': [],
    'pock_apol_asa': []}

dynids_trajids = read_dynid_trajids()

for index, file in enumerate(find("pocket_num_*_descriptors.txt", pocket_dir)):

    # if index > 200:
    #     break

    try:
        # Extract IDs from the file path
        traj_id, dyn_id, pocket_id = extract_ids(file)
        if traj_id == None:
            continue

        # Skip if not in our set
        if dyn_id not in dynids_trajids.keys():
            continue

        if traj_id not in dynids_trajids[dyn_id]:
            continue

        # if not (dyn_id == "8" and traj_id == "10170" and pocket_id == "8"):
        #     continue

        print(f"[*] Processing file {file} (index {index})")

        # Get the volumes and smooth them
        values_dict = read_descriptors_file(file)
        for frame in range(len(values_dict['pock_volume'])):
            volume = float(values_dict['pock_volume'][frame])
            pock_pol_asa = float(values_dict['pock_pol_asa'][frame])
            pock_apol_asa = float(values_dict['pock_apol_asa'][frame])

            # Append the data to the dictionary
            df_dict['frame'].append(frame)
            df_dict['dyn_id'].append(dyn_id)
            df_dict['traj_id'].append(traj_id)
            df_dict['pocket_id'].append(pocket_id)
            df_dict['pock_volume'].append(volume)
            df_dict['pock_pol_asa'].append(pock_pol_asa)
            df_dict['pock_apol_asa'].append(pock_apol_asa)

    except Exception as e:
        print(f"Error processing file {file} (index {index}): {e}")
        continue

df = pd.DataFrame(df_dict)

df.to_csv('pocket_descriptors.csv', index=False)