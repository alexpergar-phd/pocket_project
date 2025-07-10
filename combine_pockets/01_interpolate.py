"""
"""


import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from dx_reader import read_dx_file, write_dx, get_dx_bounding_box, determine_common_grid, interpolate_to_grid


density_dir = "/home/alex/Desktop/freq_results/dx_files"
outdir = "/home/alex/Desktop/freq_results"

combined_grid = None
dx_files = []
for file in os.listdir(density_dir):
    if file.endswith(".dx"):
        dx_files.append(os.path.join(density_dir, file))

common_origin, common_deltas, common_counts = determine_common_grid(dx_files)

print(common_origin, common_deltas, common_counts)

for dx_file in dx_files:
    print(dx_file)
    dx_data = read_dx_file(dx_file)
    origin, deltas, counts, data_array = (dx_data['origin'],
                                          dx_data['deltas'],
                                          dx_data['counts'],
                                          np.array(dx_data['scalars']).reshape(dx_data['counts']))

    interpolated_data = interpolate_to_grid(origin, deltas, counts, data_array, common_origin, common_deltas, common_counts)
    write_dx(
        os.path.join(outdir, os.path.basename(dx_file.replace('.dx', '_interpolated.dx'))),
        common_origin,
        common_deltas,
        common_counts,
        interpolated_data
    )

