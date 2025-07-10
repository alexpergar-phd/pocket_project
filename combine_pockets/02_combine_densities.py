"""
"""


import os
import numpy as np

from dx_reader import read_dx_file, write_dx


density_dir = "/home/alex/Desktop/freq_results/interpolated"
outdir = "/home/alex/Desktop/freq_results"

combined_grid = None

dx_files = [os.path.join(density_dir, file)
    for file in os.listdir(density_dir) if file.endswith(".dx")]

sum_array, origin, deltas, counts = (None, None, None, None)
for dx_file in dx_files:
    df_data = read_dx_file(dx_file)
    origin, deltas, counts, data_array = (df_data['origin'],
                                          df_data['deltas'],
                                          df_data['counts'],
                                          np.array(df_data['scalars']).reshape(df_data['counts']))

    if sum_array is None:
        sum_array = np.zeros_like(data_array)
    if origin is None:
        origin = df_data['origin']
    if deltas is None:
        deltas = df_data['deltas']
    if counts is None:
        counts = df_data['counts']

    sum_array += data_array

average_array = sum_array / len(dx_files)

write_dx(
    os.path.join(outdir, os.path.basename('combined.dx')),
    origin,
    deltas,
    counts,
    average_array
)
    