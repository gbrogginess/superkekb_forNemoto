""""
================================================================================
Merge FMA htcondor job outputs
================================================================================
Belle II Simulation & Modelling Group

Authors:    G. Broggi
Email:      TBD
Date:       2026-XX-XX
================================================================================

Usage:
    python 003_merge_fma.py <OUT_DIR>

<OUT_DIR> is the same output directory used by 003_fma.py, i.e. the folder
that contains one "Job.<jobID>" subfolder per htcondor job (each job having
tracked one phase and one chunk of the (norm_a, norm_b) amplitude grid).
"""

################################################################################
# Required Packages
################################################################################
import glob
import json
import os
import sys
import numpy as np

########################################
# Check command line arguments
########################################
if len(sys.argv) != 2:
    print("Usage: python 003_merge_fma.py <OUT_DIR>")
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

########################################
# Per-particle fields to concatenate across jobs
########################################
FIELDS = [
    "particle_id", "phase",
    "x_norm_init", "y_norm_init", "z_norm_init",
    "Jx", "Jy", "Jz", "Ax", "Ay", "Az",
    "x0", "px0", "y0", "py0", "zeta0", "delta0",
    "lost", "loss_turn",
    "tunes_x", "tunes_y", "tunes_z",
    "diff_coeff", "diff_x", "diff_y", "diff_z"]

records = {field: [] for field in FIELDS}

q0_x = q0_y = None
scan_mode = beambeam_on = radiation_model = resonance_order = None
norm_x_array = norm_y_array = norm_z_array = phases = None

for job_dir in job_dirs:
    fma_fpath    = os.path.join(job_dir, "Outputdata", "fma.npz")
    params_fpath = os.path.join(job_dir, "Outputdata", "params.json")

    if not os.path.isfile(fma_fpath):
        print(f"Skipping {job_dir}: no fma.npz found")
        continue

    data = np.load(fma_fpath)
    for field in FIELDS:
        records[field].append(data[field])

    if q0_x is None:
        q0_x = float(data["q0_x"])
        q0_y = float(data["q0_y"])

    if scan_mode is None and os.path.isfile(params_fpath):
        with open(params_fpath, "r", encoding="utf-8") as f:
            job_params = json.load(f)
        scan_mode       = job_params["SCAN_MODE"]
        beambeam_on     = job_params["BEAMBEAM_ON"]
        radiation_model = job_params["RADIATION_MODEL"]
        resonance_order = job_params["RESONANCE_ORDER"]
        norm_x_array    = np.array(job_params["NORM_X_ARRAY"])
        norm_y_array    = np.array(job_params["NORM_Y_ARRAY"])
        norm_z_array    = np.array(job_params["NORM_Z_ARRAY"])
        phases          = np.array(job_params["PHASES"])

if not records["phase"]:
    print(f"No usable fma.npz files found under {OUT_DIR}")
    sys.exit(1)

########################################
# Concatenate across jobs
########################################
for field in FIELDS:
    records[field] = np.concatenate(records[field])

########################################
# Sort by (phase, particle_id) for a tidy, reproducible ordering
########################################
order = np.lexsort((records["particle_id"], records["phase"]))
for field in FIELDS:
    records[field] = records[field][order]

################################################################################
# Injection tune per phase (tune of the particle closest to zero amplitude)
################################################################################
unique_phases = np.unique(records["phase"])
q_inj_x       = np.full(unique_phases.shape, np.nan)
q_inj_y       = np.full(unique_phases.shape, np.nan)

for i, p in enumerate(unique_phases):
    mask = records["phase"] == p
    amp2 = records["x_norm_init"][mask]**2 + records["y_norm_init"][mask]**2
    closest = np.argmin(amp2)
    q_inj_x[i] = records["tunes_x"][mask][closest]
    q_inj_y[i] = records["tunes_y"][mask][closest]

q0_x_full = np.full(records["phase"].shape, q0_x if q0_x is not None else np.nan, dtype=float)
q0_y_full = np.full(records["phase"].shape, q0_y if q0_y is not None else np.nan, dtype=float)

################################################################################
# Save merged outputs
################################################################################
merged_dir = os.path.join(OUT_DIR, "merged")
os.makedirs(merged_dir, exist_ok=True)

merged_fpath = os.path.join(merged_dir, "fma_merged.npz")
np.savez(
    merged_fpath,
    **records,
    q0_x            = q0_x_full,
    q0_y            = q0_y_full,
    phases          = phases if phases is not None else unique_phases,
    q_inj_x         = q_inj_x,
    q_inj_y         = q_inj_y,
    norm_x_array    = norm_x_array if norm_x_array is not None else np.array([]),
    norm_y_array    = norm_y_array if norm_y_array is not None else np.array([]),
    norm_z_array    = norm_z_array if norm_z_array is not None else np.array([]),
    max_order       = resonance_order if resonance_order is not None else -1,
    scan_mode       = scan_mode if scan_mode is not None else "unknown",
    beambeam_on     = bool(beambeam_on) if beambeam_on is not None else False,
    radiation_model = radiation_model if radiation_model is not None else "none")

print(f"Merged {len(job_dirs)} job folder(s), {records['phase'].shape[0]} particles total "
      f"across {unique_phases.shape[0]} phase(s)")
print(f"Merged FMA data saved to: {merged_fpath}")