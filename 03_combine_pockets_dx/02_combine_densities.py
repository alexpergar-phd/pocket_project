"""
"""


import os
import json

import numpy as np
import pandas as pd

from dx_reader import read_dx_file, write_dx


density_dir = "/home/aperalta/combine_pockets/results/dx_files_interpoled"
outdir = "/home/aperalta/combine_pockets/results/dx_files_averaged"

GPCRDB_SLUG_JSON = "/home/aperalta/combine_pockets/gpcrdb_slugs.json"
ALIGNING_CSV = "/home/aperalta/combine_pockets/results/aligning.csv"


def get_dynids_dx_files(input_dir, dynids):
    """
    Get the list of .dx files in the directory.
    """
    dx_files = []
    for file in os.listdir(input_dir):
        # Excluding.
        if not file.endswith(".dx"):
            continue
        if not file.endswith("_interpolated.dx"):
            continue
        if int(file.split("_")[2]) not in dynids:
            continue

        # Append if satifies the conditions.
        dx_files.append(os.path.join(input_dir, file))

    return dx_files


def get_common_grid(dx_files):
    """
    Get the common grid for all .dx files.
    """
    # Read the first file to get the origin, deltas and counts.
    df_data = read_dx_file(dx_files[0])

    origin, deltas, counts = (
        df_data['origin'], df_data['deltas'], df_data['counts'])
    data_array = np.array(df_data['scalars']).reshape(counts)
    zeroes_array = np.zeros_like(data_array)

    return origin, deltas, counts, zeroes_array


def calculate_average_densities(dx_files, counts, zeroes_array):
    """
    Combine the densities from all .dx files into a single array.
    """
    sum_array = zeroes_array.copy()

    for index, dx_file in enumerate(dx_files):
        print(f"\t[*] Processing file {index + 1}/{len(dx_files)}: {os.path.basename(dx_file)}")

        # Read the .dx file and extract the data.
        df_data = read_dx_file(dx_file)
        formated_scalars = np.array(df_data['scalars']).reshape(counts)
        data_array = formated_scalars

        # Sum into the common array.
        sum_array += data_array

    average_array = sum_array / len(dx_files)
    return average_array


def load_slug_name_dict(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return { entry["slug"]: entry["name"].replace(" ", "_") for entry in data }


def get_every_choice_column(df, column_name):
    """
    Get the list of every choice in a column.
    """
    choices = []
    for index, row in df.iterrows():
        if row[column_name] not in choices:
            if str(row[column_name]) == "nan":
                continue
            choices.append(row[column_name])
    return choices


def get_every_slug_by_levels(every_slug):
    """
    Get the list of every slug by levels.
    """
    gpcr_classes, gpcr_families, gpcr_subtypes = set(), set(), set()
    for slug in every_slug: 
        gpcr_class = slug.split("_")[0]
        gpcr_family = "_".join(slug.split("_")[0:2])
        gpcr_subtype = "_".join(slug.split("_")[0:3])
        gpcr_classes.add(gpcr_class)
        gpcr_families.add(gpcr_family)
        gpcr_subtypes.add(gpcr_subtype)
    return list(gpcr_classes), list(gpcr_families), list(gpcr_subtypes)
    

def get_dynids_that_satisfy_conditions(df, column_name, choice):
    """
    """
    dynids = []
    for index, row in df.iterrows():
        if str(row[column_name]).startswith(choice):
            dynids.append(int(row["dynid"]))
    return dynids


def get_combinations(list1, list2):
    """
    Get all possible combinations of elements between two lists.
    """
    combinations = []
    for item1 in list1:
        for item2 in list2:
            combinations.append((item1, item2))
    return combinations


def get_dynids_for_group_and_state(df, grouping_type, group_name, state):
    """
    """
    grouping_dynids = get_dynids_that_satisfy_conditions(df, grouping_type, group_name)
    if state == 'all':
        dynids = list(set(grouping_dynids))
    else:
        state_dynids = get_dynids_that_satisfy_conditions(df, "state", state)
        dynids = list(set(grouping_dynids) & set(state_dynids))
    return dynids


def generate_density_file_main(density_dir, outdir, group_name, state, dynids):
    """
    """
    if len(dynids) == 0:
        print(f"\t[*] No dynids found for {group_name} and state {state}.")
        return 

    # Compute the average density.
    dx_files = get_dynids_dx_files(density_dir, dynids)
    origin, deltas, counts, zeroes_array = get_common_grid(dx_files)
    average_array = calculate_average_densities(dx_files, counts, zeroes_array)

    # Write the .dx file.
    write_dx(
        filename=os.path.join(outdir, f"{group_name}-{state.lower()}.dx"),
        origin=origin,
        deltas=deltas,
        counts=counts,
        data=average_array,
        title=f"{group_name} - {state} - n_files: {len(dx_files)} - n_dynids:{len(dynids)} --> {dynids}"
    )


## Main ##
slug_2_name = load_slug_name_dict(GPCRDB_SLUG_JSON)

# Open CSV file to write RMSD to reference values.
out_csv_file = os.path.join(outdir, "rmsd_to_ref.csv")
out_csv_fh = open(out_csv_file, "w")
out_csv_fh.write("group_name;state;dynids;n_trajectories;avg_rmsd_to_ref\n")

# Read the CSV file.
df = pd.read_csv(ALIGNING_CSV, sep=";")
df = df[df['species'] == "Homo sapiens"] # Keep only human proteins.

# Get the list of every choice in the columns.
every_uniprot_name = get_every_choice_column(df, "uniprot_name")
every_slug = get_every_choice_column(df, "family_slug")
every_gpcr_class, every_gpcr_family, every_gpcr_subtype = get_every_slug_by_levels(every_slug)
every_slug_by_level = every_gpcr_class + every_gpcr_family + every_gpcr_subtype
every_state = get_every_choice_column(df, "state")
every_state.append("all")

# Make every slug density averages.
for slug, state in get_combinations(every_slug_by_level, every_state):
    try:
        group_name = slug_2_name[slug]
    except KeyError:
        print(f'[WARNING] The slug {slug} does not exist. Skipped.')
        continue

    print(f"[*] Processing {group_name} and state {state}...")

    # Determine the group type.
    match len(slug.split("_")):
        case 1:
            classification_type = 'by_class'
        case 2:
            classification_type = 'by_family'
        case 3:
            classification_type = 'by_subtype'
        case _:
            raise Exception

    # Getting the dynids.
    dynids = get_dynids_for_group_and_state(df, "family_slug", slug, state)

    # Calculate the average rmsd_to_ref for these dynids.
    dynids_df = df[df['dynid'].isin(dynids)]
    avg_rmsd_to_ref = round(dynids_df['rmsd_to_ref'].mean(),3)
    out_csv_fh.write(f"{group_name};{state};{dynids};{len(dynids_df)};{avg_rmsd_to_ref}\n")
    out_csv_fh.flush()

    # Generate the average density file.
    final_output_dir = os.path.join(outdir, classification_type)
    os.makedirs(final_output_dir, exist_ok=True)
    generate_density_file_main(
        density_dir=density_dir,
        outdir=final_output_dir,
        group_name=group_name,
        state=state,
        dynids=dynids
    )

# Make every receptor density averages.
final_output_dir = os.path.join(outdir, "by_receptor")
os.makedirs(final_output_dir, exist_ok=True)
for uniprot_name, state in get_combinations(every_uniprot_name, every_state):
    print(f"[*] Processing {uniprot_name} and state {state}...")

    # Getting the dynids.
    dynids = get_dynids_for_group_and_state(df, "uniprot_name", uniprot_name, state)

    # Calculate the average rmsd_to_ref for these dynids.
    dynids_df = df[df['dynid'].isin(dynids)]
    avg_rmsd_to_ref = round(dynids_df['rmsd_to_ref'].mean(),3)
    out_csv_fh.write(f"{uniprot_name};{state};{dynids};{len(dynids_df)};{avg_rmsd_to_ref}\n")
    out_csv_fh.flush()

    # Generate the average density file.
    generate_density_file_main(
        density_dir=density_dir,
        outdir=final_output_dir,
        group_name=uniprot_name,
        state=state,
        dynids=dynids
    )

out_csv_fh.close()

