""""
================================================================================
Dynamic Aperture Study (phase scan)
================================================================================
Belle II Simulation & Modelling Group

Authors:    T. Nemoto, J.P.T. Salvesen, G. Broggi
Email:      TBD
Date:       2026-09-08
================================================================================
"""

################################################################################
# Required Packages
################################################################################
import json
import os
import sys
import numpy as np
import xtrack as xt
import xobjects as xo

sys.path.insert(0, "/eos/user/g/gbroggi/helpers/")
from _track_da import track_da

########################################
# Check command line arguments
########################################
if len(sys.argv) != 2:
    print("Usage: python 002_da_study.py <jobID>")
    sys.exit(1)

jobID = int(sys.argv[1])

################################################################################
# User Parameters
################################################################################

########################################
# Working directory
########################################
WORK_DIR                 = "/eos/user/g/gbroggi/htcondor_outputs/"

########################################
# Output directories
########################################
OUT_DIR = "002_da_study"

# Create output directory
output_dir = os.path.join(WORK_DIR, OUT_DIR)
os.makedirs(output_dir, exist_ok=True)

# Create job-specific subdirectory
job_dir = os.path.join(output_dir, f"Job.{jobID}")
os.makedirs(job_dir, exist_ok=True)

# Create logs subdirectory inside job directory
logs_dir = os.path.join(job_dir, "logs")
os.makedirs(logs_dir, exist_ok=True)

# Create Outputdata subdirectory inside job directory
outputdata_dir = os.path.join(job_dir, "Outputdata")
os.makedirs(outputdata_dir, exist_ok=True)

########################################
# Lattice
########################################
ENV_FILEPATH = "/eos/user/g/gbroggi/superkekb_forNemoto/sler_1802_60_1_aper.json"
LINE_NAME    = "line"

########################################
# Tracking parameters
########################################
ELE_START = "injectio"
N_TURNS   = 1000

########################################
# Multithreading
########################################
CONTEXT = xo.ContextCpu(omp_num_threads = "auto")

########################################
# Betatron phases to scan
########################################
# NOTE on parallelisation:
# track_da() loops over "phases" independently -- each phase is tracked on
# its own full (x,y) amplitude grid, so splitting the phase scan across jobs
# is exact and safe. One job tracks one phase: queue exactly len(PHASES)
# jobs (arguments = $(ProcId)) so every phase gets its own htcondor job.
PHASES = np.array([0, 1 * np.pi / 4, np.pi / 2, 3 * np.pi / 4])

if jobID >= len(PHASES):
    print(f"jobID {jobID} has no assigned phase (only {len(PHASES)} phases defined).")
    sys.exit(1)

JOB_PHASES = PHASES[[jobID]]

########################################
# Emittances used to normalise the DA scan
########################################
GEMITT_X = 4.00E-09      # This is placeholder
GEMITT_Y = 1.00E-12      # This is placeholder
GEMITT_Z = 3.00E-06      # This is placeholder

########################################
# Amplitude grid (normalised coordinates)
########################################
N_X = 81
N_Y = 81

X_DA_RANGE = 30
Y_DA_RANGE = 250

da_x_scan = np.linspace(-X_DA_RANGE, X_DA_RANGE, N_X)
da_y_scan = np.linspace(-Y_DA_RANGE, Y_DA_RANGE, N_Y)

params = {
    "ENV_FILEPATH": ENV_FILEPATH,
    "LINE_NAME": LINE_NAME,
    "ELE_START": ELE_START,
    "N_TURNS": N_TURNS,
    "CONTEXT": CONTEXT.__class__.__name__,
    "PHASES": PHASES.tolist(),
    "JOB_PHASES": JOB_PHASES.tolist(),
    "GEMITT_X": GEMITT_X,
    "GEMITT_Y": GEMITT_Y,
    "GEMITT_Z": GEMITT_Z,
    "N_X": N_X,
    "N_Y": N_Y,
    "X_DA_RANGE": X_DA_RANGE,
    "Y_DA_RANGE": Y_DA_RANGE,
}

################################################################################
# Lattice setup
################################################################################

########################################
# Load environment
########################################
env  = xt.load(ENV_FILEPATH)
line = env[LINE_NAME]

################################################################################
# Dynamic Aperture tracking (phase scan, xy mode)
################################################################################
print(f"\nJob {jobID}: tracking DA for phase(s) {JOB_PHASES}")

da_xy = track_da(
    line             = line,
    norm_a_array     = da_x_scan,
    norm_b_array     = da_y_scan,
    phases           = JOB_PHASES,
    n_turns          = N_TURNS,
    gemitt_x         = GEMITT_X,
    gemitt_y         = GEMITT_Y,
    gemitt_z         = GEMITT_Z,
    mode             = "xy",
    tracking_context = CONTEXT)

################################################################################
# Save outputs
################################################################################

########################################
# DA data (raw, this job's phase(s) only)
########################################
da_xy_fpath = os.path.join(outputdata_dir, "da_xy.npz")
np.savez(
    da_xy_fpath,
    phase       = da_xy["phase"],
    at_turn     = da_xy["at_turn"],
    x_norm_init = da_xy["x_norm_init"],
    y_norm_init = da_xy["y_norm_init"],
    z_norm_init = da_xy["z_norm_init"],
    gemitt_x    = da_xy["gemitt_x"],
    gemitt_y    = da_xy["gemitt_y"],
    gemitt_z    = da_xy["gemitt_z"])

########################################
# Params (for provenance)
########################################
params_fpath = os.path.join(outputdata_dir, "params.json")
with open(params_fpath, "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)

print(f"Job {jobID}: done, saved to {da_xy_fpath}")