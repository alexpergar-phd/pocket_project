"""
Compares MD simulations to a novel structure by aligning to the binding pocket and calculating RMSD.
Saves results to a CSV file.
"""

import sys
from os.path import join as pjoin

import pandas as pd
import MDAnalysis as mda
from MDAnalysis.analysis import align


#-- Arguments --#
outpath = sys.argv[1]


#-- Constants --#
# Residues within 5A of ligand.
POCKET_RESIDS = [179, 176, 220, 302, 122, 121, 117, 101, 299, 298, 295, 191, 192, 193, 213, 272, 276, 279, 217, 118, 180, 175]

# Paths
NOVEL_PDB_PATH = "/home/aperalta/Documents/pocket_tool/results/09_novel_structure_analysis/gpr32_novel_clean.pdb"

MD_BASE_PATH = "/home/aperalta/Documents/pocket_tool/data/af_sims/gpr32_o75388_sims"
SIMULATIONS_PATHS = [
    ["active", "rep_1", pjoin(MD_BASE_PATH, "active/structure.psf"), pjoin(MD_BASE_PATH, "active/wrapped_aligned_x1.xtc")],
    ["active", "rep_2", pjoin(MD_BASE_PATH, "active/structure.psf"), pjoin(MD_BASE_PATH, "active/wrapped_aligned_x2.xtc")],
    ["active", "rep_3", pjoin(MD_BASE_PATH, "active/structure.psf"), pjoin(MD_BASE_PATH, "active/wrapped_aligned_x3.xtc")],
    ["inactive", "rep_1", pjoin(MD_BASE_PATH, "inactive/structure.psf"), pjoin(MD_BASE_PATH, "inactive/wrapped_aligned_x1.xtc")],
    ["inactive", "rep_2", pjoin(MD_BASE_PATH, "inactive/structure.psf"), pjoin(MD_BASE_PATH, "inactive/wrapped_aligned_x2.xtc")],
    ["inactive", "rep_3", pjoin(MD_BASE_PATH, "inactive/structure.psf"), pjoin(MD_BASE_PATH, "inactive/wrapped_aligned_x3.xtc")]
]


#-- Functions --#
def align_sim_to_pocket(u_to_align, u_ref, u_ref_chainID, resids_to_align):
    """
    Aligns the whole simulation to the binding pocket.

    Args:
        u_to_align (MDAnalysis.Universe): The simulation universe to align.
        u_ref (MDAnalysis.Universe): The reference universe (PDB) to align to.
        u_ref_chainID (str): The chain ID in the reference universe to use.
        resids_to_align (list): List of residue IDs to use for alignment.

    Returns:
        list: A list of tuples containing (frame number, RMSD value).
    """
    selection_str = ('protein and not backbone and not name H* and resid ' + ' '.join(map(str, resids_to_align)))
    results_alignment = align.AlignTraj(
        u_to_align,
        u_ref,
        select={
            'mobile': selection_str,
            'reference': selection_str + ' and chainID ' + u_ref_chainID
        },
        match_atoms=False, # If true, fails because of atom weights missmatch.
        in_memory=True
    ).run()

    return list(zip(range(0, len(results_alignment.rmsd)), results_alignment.rmsd))


#-- Main --#
csv_dict = {
    'state': [],
    'replicate': [],
    'frame': [],
    'rmsd': []
}

# Compare each simulation to the PDB structure
for simulation in SIMULATIONS_PATHS:
    state, replicate, psf, xtc = simulation

    print(f"[*] Loading simulation: {state} - {replicate}")
    md_u = mda.Universe(psf, xtc, in_memory=True)
    pdb_u = mda.Universe(NOVEL_PDB_PATH, in_memory=True)

    print(f"\t[*] Aligning simulation to pocket")
    align_sim_to_pocket(md_u, pdb_u, 'X', POCKET_RESIDS)

    for frame, rmsd in align_sim_to_pocket(md_u, pdb_u, 'X', POCKET_RESIDS):
        csv_dict['state'].append(state)
        csv_dict['replicate'].append(replicate)
        csv_dict['frame'].append(frame)
        csv_dict['rmsd'].append(rmsd)


# Save results to CSV
pd.DataFrame(csv_dict).to_csv(
    pjoin(outpath, "rmsd_comparison_novel_pocket.csv"),
    index=False)