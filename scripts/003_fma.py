""""
================================================================================
# FMA study with naff_harmonics
================================================================================
Belle II Simulation & Modelling Group

Authors:    J.P.T. Salvesen, T. Nemoto, G. Broggi
Email:      TBD
Date:       2026-XX-XX
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
import xfields as xf

from scipy.constants import e as qelectron

sys.path.insert(0, "/eos/user/g/gbroggi/helpers/")
from _fma_helpers import generate_particle_grid, run_fma

########################################
# Check command line arguments
########################################
if len(sys.argv) != 2:
    print("Usage: python 003_fma.py <jobID>")
    sys.exit(1)

jobID = int(sys.argv[1])

################################################################################
# User Parameters
################################################################################

########################################
# Working directory
########################################
WORK_DIR = "/eos/user/g/gbroggi/htcondor_outputs/"

########################################
# Output directories
########################################
OUT_DIR = "003_fma"

# Create output directory
output_dir = os.path.join(WORK_DIR, OUT_DIR)
os.makedirs(output_dir, exist_ok=True)

# Create job-specific subdirectory
job_dir = os.path.join(output_dir, f"Job.{jobID}")
os.makedirs(job_dir, exist_ok=True)

# Create Outputdata subdirectory inside job directory
outputdata_dir = os.path.join(job_dir, "Outputdata")
os.makedirs(outputdata_dir, exist_ok=True)

########################################
# Lattice
########################################
ENV_FILEPATH = "/eos/user/g/gbroggi/superkekb_forNemoto/lattices/sler_1802_60_1_aper.json"
LINE_NAME    = "line"

GEMITT_X = 4.00E-09      # This is placeholder
GEMITT_Y = 1.00E-12      # This is placeholder
GEMITT_Z = 3.00E-06      # This is placeholder

########################################
# FMA parameters
########################################
NORM_X_ARRAY = np.linspace(0, 30, 61)
NORM_Y_ARRAY = np.linspace(0, 300, 61)
NORM_Z_ARRAY = np.linspace(0, 20, 41)

SCAN_MODE = "xy"      # "xy", "zx", "zy"

N_TURNS   = 2**10
PHASES    = np.array([0, 1, 2, 3]) * np.pi / 4
N_WINDOWS = 10

########################################
# Configuration
########################################
BEAMBEAM_ON                = True
RADIATION_MODEL            = "mean"
FREEZE_LONGITUDINAL        = False
REDUCED_XSUITE_INTEGRATION = False
RESONANCE_ORDER            = 10      # only used by downstream Q-plane plotting

########################################
# Beam-Beam Parameters
########################################
GEMITT_X_HER = 4.50E-9
GEMITT_Y_HER = 25.0E-12
GEMITT_Z_HER = 3.20E-6

BETX_HER = 60E-3
BETY_HER = 1E-3
BETS_HER = 8.00

HALF_XING_RAD = 41.5E-3

CURRENT_HER = 1.000
CHARGE_HER  = -1
N_BUNCHES   = 2300

N_SLICES = 301
BB_SCALE = 1.0

########################################
# Multithreading
########################################
TWISS_CONTEXT = xo.ContextCpu(omp_num_threads = 0)
CONTEXT       = xo.ContextCpu(omp_num_threads = "auto")

def compute_twiss6d(line):
    line.discard_tracker()
    line.build_tracker(_context = TWISS_CONTEXT)
    return line.twiss6d()

########################################
# htcondor job: (phase, particle chunk) mapping
########################################
# Every particle in the (norm_a, norm_b) grid is tracked and NAFF-analysed
# completely independently of every other particle (unlike the DA study,
# there is no adaptive frontier/skip logic here). This makes it safe to
# split BOTH by phase AND by chunks of the amplitude grid within a phase:
#   jobID = phase_index * N_CHUNKS_PER_PHASE + chunk_index
# queue len(PHASES) * N_CHUNKS_PER_PHASE jobs (arguments = $(ProcId)) and
# 003_merge_fma.py reassembles the full grid afterwards. Increase
# N_CHUNKS_PER_PHASE for more parallelism, decrease it (down to 1) for
# fewer, larger jobs.
N_CHUNKS_PER_PHASE = 5

N_JOBS = len(PHASES) * N_CHUNKS_PER_PHASE
if jobID >= N_JOBS:
    print(f"jobID {jobID} is out of range (only {N_JOBS} jobs defined: "
          f"{len(PHASES)} phases x {N_CHUNKS_PER_PHASE} chunks).")
    sys.exit(1)

PHASE_IDX = jobID // N_CHUNKS_PER_PHASE
CHUNK_IDX = jobID % N_CHUNKS_PER_PHASE

JOB_PHASE = PHASES[PHASE_IDX]

if SCAN_MODE == "xy":
    n_particles_total = len(NORM_X_ARRAY) * len(NORM_Y_ARRAY)
elif SCAN_MODE == "zx":
    n_particles_total = len(NORM_Z_ARRAY) * len(NORM_X_ARRAY)
elif SCAN_MODE == "zy":
    n_particles_total = len(NORM_Z_ARRAY) * len(NORM_Y_ARRAY)
else:
    raise ValueError(f"Invalid SCAN_MODE: {SCAN_MODE}. Must be one of 'xy', 'zx', or 'zy'.")

chunk_ids = np.array_split(np.arange(n_particles_total), N_CHUNKS_PER_PHASE)[CHUNK_IDX]

params = {
    "ENV_FILEPATH": ENV_FILEPATH,
    "LINE_NAME": LINE_NAME,
    "GEMITT_X": GEMITT_X,
    "GEMITT_Y": GEMITT_Y,
    "GEMITT_Z": GEMITT_Z,
    "NORM_X_ARRAY": NORM_X_ARRAY.tolist(),
    "NORM_Y_ARRAY": NORM_Y_ARRAY.tolist(),
    "NORM_Z_ARRAY": NORM_Z_ARRAY.tolist(),
    "SCAN_MODE": SCAN_MODE,
    "N_TURNS": N_TURNS,
    "PHASES": PHASES.tolist(),
    "N_WINDOWS": N_WINDOWS,
    "BEAMBEAM_ON": BEAMBEAM_ON,
    "RADIATION_MODEL": RADIATION_MODEL,
    "FREEZE_LONGITUDINAL": FREEZE_LONGITUDINAL,
    "REDUCED_XSUITE_INTEGRATION": REDUCED_XSUITE_INTEGRATION,
    "RESONANCE_ORDER": RESONANCE_ORDER,
    "CONTEXT": CONTEXT.__class__.__name__,
    "N_CHUNKS_PER_PHASE": N_CHUNKS_PER_PHASE,
    "PHASE_IDX": int(PHASE_IDX),
    "CHUNK_IDX": int(CHUNK_IDX),
    "JOB_PHASE": float(JOB_PHASE),
    "N_PARTICLES_TOTAL": int(n_particles_total),
}

################################################################################
# Lattice setup
################################################################################

########################################
# Load environment
########################################
env  = xt.load(ENV_FILEPATH)
line = env[LINE_NAME]

########################################
# Configure models and integrators
########################################
if REDUCED_XSUITE_INTEGRATION:
    tt          = line.get_table()
    tt_drift    = tt.rows[tt.element_type == "Drift"]
    tt_bend     = tt.rows[tt.element_type == "Bend"]
    tt_quad     = tt.rows[tt.element_type == "Quadrupole"]
    tt_sext     = tt.rows[tt.element_type == "Sextupole"]
    tt_oct      = tt.rows[tt.element_type == "Octupole"]
    tt_mult     = tt.rows[tt.element_type == "Multipole"]
    tt_sol      = tt.rows[tt.element_type == "UniformSolenoid"]
    tt_cavi     = tt.rows[tt.element_type == "Cavity"]

    line.set(
        tt_drift,
        model               = "exact")
    line.set(
        tt_bend,
        model               = "bend-kick-bend",
        integrator          = "uniform",
        num_multipole_kicks = 3)
    line.set(
        tt_quad,
        model               = "mat-kick-mat",
        integrator          = "yoshida4",
        num_multipole_kicks = 7)
    line.set(
        tt_sext,
        model               = "mat-kick-mat",
        integrator          = "yoshida4",
        num_multipole_kicks = 7)
    line.set(
        tt_oct,
        model               = "mat-kick-mat",
        integrator          = "yoshida4",
        num_multipole_kicks = 7)
    line.set(
        tt_mult,
        model               = "mat-kick-mat",
        integrator          = "yoshida4",
        num_multipole_kicks = 7)
    line.set(
        tt_sol,
        num_multipole_kicks = 3)
    line.set(
        tt_cavi,
        model               = "drift-kick-drift-exact",
        integrator          = "yoshida4",
        absolute_time       = False)

########################################
# Radiation model
########################################
line.discard_tracker()
line.build_tracker(_context = TWISS_CONTEXT)
if RADIATION_MODEL is not None:
    line.configure_radiation(model = RADIATION_MODEL)

########################################
# Twiss
########################################
tw = compute_twiss6d(line)

################################################################################
# Install Beam-beam
################################################################################
if BEAMBEAM_ON:
    ########################################
    # Compute beam size
    ########################################
    sigma_x_her  = np.sqrt(GEMITT_X_HER * BETX_HER)
    sigma_px_her = np.sqrt(GEMITT_X_HER / BETX_HER)
    sigma_y_her  = np.sqrt(GEMITT_Y_HER * BETY_HER)
    sigma_py_her = np.sqrt(GEMITT_Y_HER / BETY_HER)
    sigma_z_her  = np.sqrt(GEMITT_Z_HER * BETS_HER)

    ########################################
    # Calculate Bunch Intensities
    ########################################
    FREV_SKEKB          = 1 / tw.T_rev0
    bunch_intensity_her = (CURRENT_HER) / (qelectron * FREV_SKEKB * N_BUNCHES)

    ########################################
    # Create Slicer
    ########################################
    slicer = xf.TempSlicer(
        n_slices = N_SLICES,
        sigma_z  = sigma_z_her,
        mode     = "shatilov")

    ########################################
    # Install Element
    ########################################
    line.insert(
        what = "bbeam",
        obj  = xf.BeamBeamBiGaussian3D(
            other_beam_q0     = CHARGE_HER,
            phi               = HALF_XING_RAD,
            alpha             = 0,
            config_for_update = None,
            # decide between round or elliptical kick formula
            min_sigma_diff    = 1E-28,
            # slice intensity [num. real particles] n_slices inferred from length of this
            slices_other_beam_num_particles = slicer.bin_weights * bunch_intensity_her,
            # unboosted strong beam moments
            slices_other_beam_zeta_center   = slicer.bin_centers,
            slices_other_beam_Sigma_11      = N_SLICES * [sigma_x_her**2],
            slices_other_beam_Sigma_22      = N_SLICES * [sigma_px_her**2],
            slices_other_beam_Sigma_33      = N_SLICES * [sigma_y_her**2],
            slices_other_beam_Sigma_44      = N_SLICES * [sigma_py_her**2],
            # only if B on
            slices_other_beam_zeta_bin_width_star_beamstrahlung = \
                slicer.bin_widths_beamstrahlung / \
                np.cos(HALF_XING_RAD),  # boosted dz
            # has to be set
            slices_other_beam_Sigma_12 = N_SLICES * [0],
            slices_other_beam_Sigma_34 = N_SLICES * [0],
            flag_luminosity            = True),
        at     = 0,
        from_  = "ip.0",
        anchor = "start")

    ########################################
    # Attach Beam-beam Strength Scale
    ########################################
    env.vars["beambeam_scale"]   = 1.0
    line["bbeam"].scale_strength = env.vars["beambeam_scale"]

    ########################################
    # Twiss with element installed but off
    ########################################
    env.vars["beambeam_scale"] = 0.0
    tw = compute_twiss6d(line)

    ########################################
    # Attach nominal Strength Scale
    ########################################
    env.vars["beambeam_scale"] = BB_SCALE

########################################
# Beam Sizes
########################################
beamsizes = tw.get_beam_covariance(
    gemitt_x    = GEMITT_X,
    gemitt_y    = GEMITT_Y,
    gemitt_zeta = GEMITT_Z)

################################################################################
# Build particle grid chunk for this job
################################################################################
print(f"\nJob {jobID}: phase index {PHASE_IDX} (phase = {JOB_PHASE:.4f}), "
      f"chunk {CHUNK_IDX + 1}/{N_CHUNKS_PER_PHASE} "
      f"({len(chunk_ids)}/{n_particles_total} particles)")

particles = generate_particle_grid(
    line          = line,
    phase         = JOB_PHASE,
    beamsizes     = beamsizes,
    gemitt_x      = GEMITT_X,
    gemitt_y      = GEMITT_Y,
    gemitt_zeta   = GEMITT_Z,
    scan_mode     = SCAN_MODE,
    norm_x_array  = NORM_X_ARRAY,
    norm_y_array  = NORM_Y_ARRAY,
    norm_z_array  = NORM_Z_ARRAY,
    select_ids    = chunk_ids,
    zero_tol      = 1E-10)

n_particles = len(particles.state)
particle_id = chunk_ids
x0          = particles.x.copy()
px0         = particles.px.copy()
y0          = particles.y.copy()
py0         = particles.py.copy()
zeta0       = particles.zeta.copy()
delta0      = particles.delta.copy()

################################################################################
# Track
################################################################################
line.discard_tracker()
line.build_tracker(_context = CONTEXT)

line.track(
    particles            = particles,
    num_turns            = N_TURNS,
    freeze_longitudinal  = FREEZE_LONGITUDINAL,
    turn_by_turn_monitor = True,
    with_progress        = 1)

last_track = line.record_last_track

particles.sort(interleave_lost_particles = True)
lost_mask  = particles.state <= 0
loss_turns = particles.at_turn.copy()

line.discard_tracker()
line.build_tracker(_context = TWISS_CONTEXT)

################################################################################
# Compute normalised coordinates turn-by-turn
################################################################################
x_norm, px_norm, y_norm, py_norm, zeta_norm, pzeta_norm = [], [], [], [], [], []
for turn in range(N_TURNS):
    fake_particles = line.build_particles(
        x     = last_track.x[:, turn],
        px    = last_track.px[:, turn],
        y     = last_track.y[:, turn],
        py    = last_track.py[:, turn],
        zeta  = last_track.zeta[:, turn],
        delta = last_track.delta[:, turn])
    # TODO: NEED TO PASS EMITTANCES
    norm_coords = tw.get_normalized_coordinates(fake_particles)
    x_norm.append(norm_coords.x_norm)
    px_norm.append(norm_coords.px_norm)
    y_norm.append(norm_coords.y_norm)
    py_norm.append(norm_coords.py_norm)
    zeta_norm.append(norm_coords.zeta_norm)
    pzeta_norm.append(norm_coords.pzeta_norm)

x_norm     = np.array(x_norm)
px_norm    = np.array(px_norm)
y_norm     = np.array(y_norm)
py_norm    = np.array(py_norm)
zeta_norm  = np.array(zeta_norm)
pzeta_norm = np.array(pzeta_norm)

norm_tracking_records = xt.Table({
    # Name is required
    "name"       : np.arange(x_norm.shape[0]),
    # Normalized Coordinates (we need shape n_turns per row)
    "x_norm"     : x_norm,
    "px_norm"    : px_norm,
    "y_norm"     : y_norm,
    "py_norm"    : py_norm,
    "zeta_norm"  : zeta_norm,
    "pzeta_norm" : pzeta_norm})

################################################################################
# Run FMA
################################################################################
tunes_x, tunes_y, tunes_z, diff_coeff, diff_x, diff_y, diff_z = run_fma(
    norm_tracking_records = norm_tracking_records,
    n_windows             = N_WINDOWS,
    scan_mode             = SCAN_MODE,
    nominal_tune_x        = tw.qx % 1,
    nominal_tune_y        = tw.qy % 1,
    valid_mask            = ~lost_mask)

assert tunes_x.shape == (n_particles,)
assert tunes_y.shape == (n_particles,)
assert tunes_z.shape == (n_particles,)
assert diff_coeff.shape == (n_particles,)
assert diff_x.shape == (n_particles,)
assert diff_y.shape == (n_particles,)
assert diff_z.shape == (n_particles,)

################################################################################
# Calculate Initial Actions
################################################################################
ax = np.sqrt(norm_tracking_records.x_norm[0, :]**2    + norm_tracking_records.px_norm[0, :]**2)
ay = np.sqrt(norm_tracking_records.y_norm[0, :]**2    + norm_tracking_records.py_norm[0, :]**2)
az = np.sqrt(norm_tracking_records.zeta_norm[0, :]**2 + norm_tracking_records.pzeta_norm[0, :]**2)

assert ax.shape == (n_particles,)
assert ay.shape == (n_particles,)
assert az.shape == (n_particles,)

Jx = 0.5 * ax**2
Jy = 0.5 * ay**2
Jz = 0.5 * az**2

Ax = np.sqrt(2 * Jx)
Ay = np.sqrt(2 * Jy)
Az = np.sqrt(2 * Jz)

assert Jx.shape == (n_particles,)
assert Jy.shape == (n_particles,)
assert Jz.shape == (n_particles,)
assert Ax.shape == (n_particles,)
assert Ay.shape == (n_particles,)
assert Az.shape == (n_particles,)
assert lost_mask.shape == (n_particles,)
assert loss_turns.shape == (n_particles,)

################################################################################
# Save outputs
################################################################################

########################################
# FMA data (this job's phase and chunk only)
########################################
fma_fpath = os.path.join(outputdata_dir, "fma.npz")
np.savez(
    fma_fpath,
    particle_id  = particle_id,
    phase        = np.full(n_particles, JOB_PHASE),
    x_norm_init  = norm_tracking_records.x_norm[0, :],
    y_norm_init  = norm_tracking_records.y_norm[0, :],
    z_norm_init  = norm_tracking_records.zeta_norm[0, :],
    Jx = Jx, Jy = Jy, Jz = Jz,
    Ax = Ax, Ay = Ay, Az = Az,
    x0 = x0, px0 = px0, y0 = y0, py0 = py0, zeta0 = zeta0, delta0 = delta0,
    lost       = lost_mask,
    loss_turn  = loss_turns,
    tunes_x    = tunes_x,
    tunes_y    = tunes_y,
    tunes_z    = tunes_z,
    diff_coeff = diff_coeff,
    diff_x     = diff_x,
    diff_y     = diff_y,
    diff_z     = diff_z,
    q0_x       = tw.qx % 1,
    q0_y       = tw.qy % 1)

########################################
# Params (for provenance)
########################################
params_fpath = os.path.join(outputdata_dir, "params.json")
with open(params_fpath, "w", encoding="utf-8") as f:
    json.dump(params, f, indent=2)

print(f"Job {jobID}: done, saved to {fma_fpath}")