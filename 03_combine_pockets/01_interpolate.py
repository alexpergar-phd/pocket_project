"""
This script processes a set of .dx files by interpolating their data to a common grid.
The script performs the following steps:
1. Reads a directory containing .dx files.
2. Determines a common grid that can be used for all the .dx files.
3. Interpolates the data from each .dx file to the common grid.
4. Writes the interpolated data to new .dx files in the specified output directory.

Output:
- For each input .dx file, a new .dx file is created with the interpolated data.
    The new files are saved in the output directory with "_interpolated" appended to their filenames.
"""


import os

import numpy as np

from dx_reader import read_dx_file, write_dx, determine_common_grid, interpolate_to_grid, manual_common_grid


# Get the directory containing the .dx files and the output directory.
density_dir = "/home/aperalta/combine_pockets/results/dx_files"
outdir = "/home/aperalta/combine_pockets/results/dx_files_interpoled"

# Get the list of .dx files in the directory.
dx_files = []
for file in os.listdir(density_dir):
    if file.endswith(".dx"):
        dx_files.append(os.path.join(density_dir, file))

# Determine the common grid for all .dx files.
# common_origin, common_deltas, common_counts = determine_common_grid(dx_files)
common_origin, common_deltas, common_counts = manual_common_grid()


# Interpolate each .dx file to the common grid.
for dx_file in dx_files:
    
    # Read the .dx file and extract the data.
    dx_data = read_dx_file(dx_file)
    origin, deltas, counts, data_array = (
        dx_data['origin'],
        dx_data['deltas'],
        dx_data['counts'],
        np.array(dx_data['scalars']).reshape(dx_data['counts']))

    # Interpolate the data to the common grid.
    interpolated_data = interpolate_to_grid(
        origin, deltas, counts, data_array, common_origin, common_deltas, common_counts)
    
    # Write the interpolated data to a new .dx file.
    write_dx(
        filename=os.path.join(outdir, os.path.basename(dx_file.replace('.dx', '_interpolated.dx'))),
        origin=common_origin,
        deltas=common_deltas,
        counts=common_counts,
        data=interpolated_data,
        title="DX file interpolated to the common grid",
    )

