"""
"""


import os
import numpy as np
import csv


from dx_reader import read_dx_file, get_dx_bounding_box


density_dir = "/home/aperalta/combine_pockets/results/dx_files"


def generate_dx_bounding_box_report(dx_files, output_csv="dx_bounding_box_report.csv"):
    file_boxes = []

    min_corner = np.array([np.inf, np.inf, np.inf])
    max_corner = np.array([-np.inf, -np.inf, -np.inf])

    # First pass: collect bounding boxes and determine global min/max
    for file in dx_files:
        dx_data = read_dx_file(file)
        origin, deltas, counts = dx_data['origin'], dx_data['deltas'], dx_data['counts']

        file_min, file_max = get_dx_bounding_box(origin, deltas, counts)
        file_boxes.append((file, file_min, file_max))

        min_corner = np.minimum(min_corner, file_min)
        max_corner = np.maximum(max_corner, file_max)

    # Write CSV
    with open(output_csv, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "File",
            "MinX", "MinY", "MinZ",
            "MaxX", "MaxY", "MaxZ",
            "ExtentX", "ExtentY", "ExtentZ",
            "TouchesGlobalMin", "TouchesGlobalMax"
        ])

        for file, fmin, fmax in file_boxes:
            touches_min = any(np.isclose(fmin, min_corner, atol=1e-6))
            touches_max = any(np.isclose(fmax, max_corner, atol=1e-6))
            extent = fmax - fmin
            writer.writerow([
                file,
                *fmin,
                *fmax,
                *extent,
                touches_min,
                touches_max
            ])

    print(f"CSV report written to: {output_csv}")


# Get the list of .dx files in the directory.
dx_files = []
for file in os.listdir(density_dir):
    if file.endswith(".dx"):
        dx_files.append(os.path.join(density_dir, file))

# Generate the bounding box report
generate_dx_bounding_box_report(dx_files, output_csv="dx_bounding_box_report.csv")