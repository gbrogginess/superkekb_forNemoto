""""
================================================================================
Merge Dynamic Aperture htcondor job outputs
================================================================================
Belle II Simulation & Modelling Group

Authors:    G. Broggi
Email:      TBD
Date:       2026-09-08
================================================================================

Usage:
    python 002_merge_da.py <OUT_DIR>

<OUT_DIR> is the same output directory used by 002_da_study.py, i.e. the
folder that contains one "Job.<jobID>" subfolder per htcondor job (each
holding an Outputdata/da_xy.npz with that job's phase(s)).
"""

################################################################################
# Required Packages
################################################################################
import glob
import os
import sys
import numpy as np

sys.path.insert(0, "/eos/user/g/gbroggi/helpers/")
from _fold_da import fold_and_phase_average_da

########################################
# Check command line arguments
########################################
if len(sys.argv) != 2:
    print("Usage: python 002_merge_da.py <OUT_DIR>")
    sys.exit(1)

OUT_DIR = sys.argv[1]

################################################################################
# Collect job outputs
################################################################################
job_dirs = sorted(
    glob.glob(os.path.join(OUT_DIR, "Job.*")),
    key = lambda p: int(p.rsplit(".", 1)[-1]))

if not job_dirs:
    print(f"No 'Job.*' folders found in {OUT_DIR}")
    sys.exit(1)

phase_list       = []
at_turn_list     = []
x_norm_init_list = []
y_norm_init_list = []
z_norm_init_list = []
gemitt_x = gemitt_y = gemitt_z = None

for job_dir in job_dirs:
    da_xy_fpath = os.path.join(job_dir, "Outputdata", "da_xy.npz")
    if not os.path.isfile(da_xy_fpath):
        print(f"Skipping {job_dir}: no da_xy.npz found")
        continue

    data = np.load(da_xy_fpath)

    phase_list.append(data["phase"])
    at_turn_list.append(data["at_turn"])
    x_norm_init_list.append(data["x_norm_init"])
    y_norm_init_list.append(data["y_norm_init"])
    z_norm_init_list.append(data["z_norm_init"])

    gemitt_x = float(data["gemitt_x"])
    gemitt_y = float(data["gemitt_y"])
    gemitt_z = float(data["gemitt_z"])

if not phase_list:
    print(f"No usable da_xy.npz files found under {OUT_DIR}")
    sys.exit(1)

########################################
# Concatenate along the phase axis
########################################
phase       = np.concatenate(phase_list, axis=0)
at_turn     = np.concatenate(at_turn_list, axis=0)
x_norm_init = np.concatenate(x_norm_init_list, axis=0)
y_norm_init = np.concatenate(y_norm_init_list, axis=0)
z_norm_init = np.concatenate(z_norm_init_list, axis=0)

########################################
# Sort by phase value (in case jobs finished out of order)
########################################
order       = np.argsort(phase)
phase       = phase[order]
at_turn     = at_turn[order]
x_norm_init = x_norm_init[order]
y_norm_init = y_norm_init[order]
z_norm_init = z_norm_init[order]

da_xy = {
    "phase":       phase,
    "at_turn":     at_turn,
    "x_norm_init": x_norm_init,
    "y_norm_init": y_norm_init,
    "z_norm_init": z_norm_init,
    "gemitt_x":    gemitt_x,
    "gemitt_y":    gemitt_y,
    "gemitt_z":    gemitt_z,
}

################################################################################
# Fold and phase-average
################################################################################
da_xy_folded = fold_and_phase_average_da(
    da_dict = da_xy,
    mode    = "xy")

################################################################################
# Save merged outputs
################################################################################
merged_dir = os.path.join(OUT_DIR, "merged")
os.makedirs(merged_dir, exist_ok=True)

########################################
# Raw (all phases, unfolded) DA data
########################################
raw_fpath = os.path.join(merged_dir, "da_xy_merged.npz")
np.savez(
    raw_fpath,
    phase       = da_xy["phase"],
    at_turn     = da_xy["at_turn"],
    x_norm_init = da_xy["x_norm_init"],
    y_norm_init = da_xy["y_norm_init"],
    z_norm_init = da_xy["z_norm_init"],
    gemitt_x    = da_xy["gemitt_x"],
    gemitt_y    = da_xy["gemitt_y"],
    gemitt_z    = da_xy["gemitt_z"])

########################################
# Folded, phase-averaged DA data (ready for plotting)
########################################
folded_fpath = os.path.join(merged_dir, "da_xy_folded.npz")
np.savez(
    folded_fpath,
    surviving   = da_xy_folded["surviving"],
    x_norm_init = da_xy_folded["x_norm_init"],
    y_norm_init = da_xy_folded["y_norm_init"],
    z_norm_init = da_xy_folded["z_norm_init"],
    gemitt_x    = da_xy_folded["gemitt_x"],
    gemitt_y    = da_xy_folded["gemitt_y"],
    gemitt_z    = da_xy_folded["gemitt_z"])

print(f"Merged {len(job_dirs)} job folder(s), {phase.shape[0]} phase(s) total")
print(f"Raw merged DA saved to:    {raw_fpath}")
print(f"Folded merged DA saved to: {folded_fpath}")