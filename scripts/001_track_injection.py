""""
================================================================================
Track an injected beam
================================================================================
Belle II Simulation & Modelling Group

Authors:    T. Nemoto, J.P.T. Salvesen
Email:      TBD
Date:       2026-07-31
================================================================================
"""

################################################################################
# Required Packages
################################################################################
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt
import xfields as xf
import xcoll as xc

from scipy.constants import e as qelectron

from _build_particles import prepare_injection_beam

########################################
# Check command line arguments
########################################
if len(sys.argv) != 2:
    print("Usage: python 001_injection_efficiency.py <jobID>")
    sys.exit(1)

jobID = int(sys.argv[1])

################################################################################
# User Parameters
################################################################################

########################################
# Working directory
########################################
WORK_DIR                 = "/eos/user/t/tnemoto/"

########################################
# Output directories
########################################
OUT_DIR = "001_injection_efficiency_output"

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
ENV_FILEPATH            = "/eos/user/t/tnemoto/xsuite/injection/config/lattices/sler_1802_60_09_aper.json"
LINE_NAME               = "line" 

########################################
# Switching configurations
########################################
GEANT4_SCATTERING_ON    = True
BEAMBEAM_ON             = True
RADIATION_MODEL         = "mean"
TRANSVERSE_ONLY         = False

########################################
# Tracking Parameters
########################################
N_TURNS                 = int(2E3) # number of turns to simulate

########################################
# Multithreading
########################################
CONTEXT                 = xo.ContextCpu(omp_num_threads = "auto")

########################################
# CollDB
########################################
if GEANT4_SCATTERING_ON:
    COLLDB_FILEPATH = "/eos/user/t/tnemoto/xsuite/injection/config/colldbs/colldb_ler_2026b_0616_DIFPOS.json"
else:
    COLLDB_FILEPATH = "/eos/user/t/tnemoto/xsuite/injection/config/colldbs/colldb_ler_2025c_14112025_ba.json"

COLLDB_ADJUSTMENTS  = {}
# Example of adjustment
# COLLDB_ADJUSTMENTS  = {
#     "pmd05v1":  {
#         "jaw":  [0.02000, -0.02000]
#     }
# }

########################################
# Particles
########################################
PARTICLE_FILEPATH       = "/eos/user/t/tnemoto/xsuite/injection/config/particles/ler_yoshimoto.dat"
TARGET_NEMITT_X         = 207.2E-6             #2026ab average
TARGET_NEMITT_Y         = 47.3E-6              #2026ab average
BUNCH_POPULATION        = 1.8100376E+10        # 2.9nC per shot
N_PARTICLES             = int(5E3)             # up to 50k

########################################
# Injection Parameters
########################################
# TODO: TBD on sign conventions  
# these sould be plus
INJECTION_OFFSET_X      = 12.80E-3        #2026ab b_y* = 0.9mm
INJECTION_OFFSET_PX     = 1.7741E-3
INJECTION_OFFSET_Y      = 0.0 
INJECTION_OFFSET_PY     = 0.0
INJECTION_OFFSET_ZETA   = 0.0
INJECTION_OFFSET_DELTA  = 0.0

########################################
# Injection Errors
########################################
INJECTION_ERROR_X       = 0.0
INJECTION_ERROR_PX      = 0.0
INJECTION_ERROR_Y       = 0.0
INJECTION_ERROR_PY      = 0.0
INJECTION_ERROR_ZETA    = 0.0
INJECTION_ERROR_DELTA   = 0.0

########################################
# Radiation
########################################
RNG_SEED                = jobID

########################################
# Start element
########################################
ELE_START               = "injectio"

########################################
# Beam-Beam Parameters
########################################
GEMITT_X_HER            = 4.50E-9
GEMITT_Y_HER            = 25.0E-12
GEMITT_Z_HER            = 3.20E-6

BETX_HER                = 60E-3
BETY_HER                = 1E-3
BETS_HER                = 8.00

HALF_XING_RAD           = 41.5E-3

CURRENT_HER             = 1.000
CHARGE_HER              = -1
N_BUNCHES               = 2300

N_SLICES                = 301
BB_SCALE                = 1.0

params = {
    "ENV_FILEPATH": ENV_FILEPATH,
    "LINE_NAME": LINE_NAME,
    "GEANT4_SCATTERING_ON": GEANT4_SCATTERING_ON,
    "N_TURNS": N_TURNS,
    "CONTEXT": CONTEXT.__class__.__name__,
    "COLLDB_FILEPATH": COLLDB_FILEPATH,
    "COLLDB_ADJUSTMENTS": COLLDB_ADJUSTMENTS,
    "PARTICLE_FILEPATH": PARTICLE_FILEPATH,
    "TARGET_NEMITT_X": TARGET_NEMITT_X,
    "TARGET_NEMITT_Y": TARGET_NEMITT_Y,
    "BUNCH_POPULATION": BUNCH_POPULATION,
    "N_PARTICLES": N_PARTICLES,
    "INJECTION_OFFSET_X": INJECTION_OFFSET_X,
    "INJECTION_OFFSET_PX": INJECTION_OFFSET_PX,
    "INJECTION_OFFSET_Y": INJECTION_OFFSET_Y,
    "INJECTION_OFFSET_PY": INJECTION_OFFSET_PY,
    "INJECTION_OFFSET_ZETA": INJECTION_OFFSET_ZETA,
    "INJECTION_OFFSET_DELTA": INJECTION_OFFSET_DELTA,
    "INJECTION_ERROR_X": INJECTION_ERROR_X,
    "INJECTION_ERROR_PX": INJECTION_ERROR_PX,
    "INJECTION_ERROR_Y": INJECTION_ERROR_Y,
    "INJECTION_ERROR_PY": INJECTION_ERROR_PY,
    "INJECTION_ERROR_ZETA": INJECTION_ERROR_ZETA,
    "INJECTION_ERROR_DELTA": INJECTION_ERROR_DELTA,
    "RADIATION_MODEL": RADIATION_MODEL,
    "RNG_SEED": RNG_SEED,
}

################################################################################
# Lattice setup
################################################################################

########################################
# Load environment
########################################
env     = xt.load(ENV_FILEPATH)

########################################
# Select line
########################################
line    = env.lines["line"]

########################################
# Configure radiation
########################################
line.configure_radiation(model =  RADIATION_MODEL)

########################################
# Twiss
########################################
tw      = line.twiss6d()

################################################################################
# Install Beam-beam
################################################################################
if BEAMBEAM_ON:
    ########################################
    # Compute beam size
    ########################################
    sigma_x_her     = np.sqrt(GEMITT_X_HER * BETX_HER)
    sigma_px_her    = np.sqrt(GEMITT_X_HER / BETX_HER)
    sigma_y_her     = np.sqrt(GEMITT_Y_HER * BETY_HER)
    sigma_py_her    = np.sqrt(GEMITT_Y_HER / BETY_HER)
    sigma_z_her     = np.sqrt(GEMITT_Z_HER * BETS_HER)

    ########################################
    # Calculate Bunch Intensities
    ########################################
    FREV_SKEKB          = 1 / tw.T_rev0
    bunch_intensity_her = (CURRENT_HER) / (qelectron * FREV_SKEKB * N_BUNCHES)

    ########################################
    # Create Slicer
    ########################################
    slicer  = xf.TempSlicer(
        n_slices    = N_SLICES,
        sigma_z     = sigma_z_her,
        mode        = "shatilov")

    ########################################
    # Install Element
    ########################################
    line.insert(
        what    = "bbeam",
        obj     = xf.BeamBeamBiGaussian3D(
            other_beam_q0       = CHARGE_HER,
            phi                 = HALF_XING_RAD,
            alpha               = 0,
            config_for_update   = None,
            # decide between round or elliptical kick formula
            min_sigma_diff      = 1E-28,
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
                np.cos(HALF_XING_RAD),  # boosted dz
            # has to be set
            slices_other_beam_Sigma_12      = N_SLICES * [0],
            slices_other_beam_Sigma_34      = N_SLICES * [0],
            flag_luminosity                 = True),
        at      = 0,
        from_   = "ip.0",
        anchor  = "start")

    ########################################
    # Attach Beam-beam Strength Scale
    ########################################
    env.vars["beambeam_scale"]  = 1.0
    line["bbeam"].scale_strength = env.vars["beambeam_scale"]

    ########################################
    # Twiss with element installed but off
    ########################################
    env.vars["beambeam_scale"]  = 0.0
    tw      = line.twiss6d()

    ########################################
    # Attach nominal Strength Scale
    ########################################
    env.vars["beambeam_scale"]  = BB_SCALE

################################################################################
# Collimator setup
################################################################################

########################################
# Create Collimator Database
########################################
# We pass absolute positions, so these emittances are placeholder
colldb  = xc.CollimatorDatabase.from_json(
    COLLDB_FILEPATH,
    nemitt_x    = 1.0,
    nemitt_y    = 1.0)

########################################
# Perform collimator adjustments
########################################
for collimator_name, adjustments in COLLDB_ADJUSTMENTS.items():
    for key, value in adjustments.items():
        colldb[collimator_name][key] = value

########################################
# Create Collimator Apertures
########################################
coll_aperts = [
    xt.LimitEllipse(a = 0.045, b = 0.045),
    xt.LimitEllipse(a = 0.045, b = 0.045)]

########################################
# Install Collimator Apertures
########################################
# colldb.install_fluka_collimators
if GEANT4_SCATTERING_ON:
    colldb.install_geant4_collimators(
        line        = line,
        apertures   = coll_aperts,
        verbose     = True)
else:
    colldb.install_black_absorbers(
        line        = line,
        apertures   = coll_aperts,
        verbose     = True)

########################################
# Twiss with collimators
########################################
tw      = line.twiss6d()

########################################
# Assign optics to collimators
########################################
line.xcoll.collimators.assign_optics(twiss = tw)

################################################################################
# Prepare Injection Beam
################################################################################

########################################
# Generate particles from Yoshimoto beam
########################################
particles  = prepare_injection_beam(
    line                    = line,
    twiss                   = tw,
    input_file              = PARTICLE_FILEPATH,
    ele_start               = ELE_START,
    n_part                  = N_PARTICLES,
    capacity                = N_PARTICLES * 4,
    target_nemitt_x         = TARGET_NEMITT_X,
    target_nemitt_y         = TARGET_NEMITT_Y,
    initially_centre_bunch  = True,
    injection_x_offset      = INJECTION_OFFSET_X,
    injection_y_offset      = INJECTION_OFFSET_Y,
    injection_px_offset     = INJECTION_OFFSET_PX,
    injection_py_offset     = INJECTION_OFFSET_PY,
    injection_zeta_offset   = INJECTION_OFFSET_ZETA,
    injection_delta_offset  = INJECTION_OFFSET_DELTA,
    injection_x_error       = INJECTION_ERROR_X,
    injection_y_error       = INJECTION_ERROR_Y,
    injection_px_error      = INJECTION_ERROR_PX,
    injection_py_error      = INJECTION_ERROR_PY,
    injection_zeta_error    = INJECTION_ERROR_ZETA,
    injection_delta_error   = INJECTION_ERROR_DELTA)

########################################
# Fix at element and at s
########################################
particles.s[particles.state == 1]           = tw["s", ELE_START]
particles.at_element[particles.state == 1]  = line.element_names.index(ELE_START)

########################################
# Zero longitudinal coordinates
########################################
if TRANSVERSE_ONLY:
    particles.zeta[particles.state == 1]    = 0
    particles.delta[particles.state == 1]   = 0

################################################################################
# Setup for tracking
################################################################################

########################################
# Create interaction record
########################################
impacts = xc.InteractionRecord(line = line)

########################################
# Start GEANT4 engine
########################################
if GEANT4_SCATTERING_ON:
    xc.geant4.engine.seed = RNG_SEED
    xc.geant4.engine.start(line = line)

########################################
# Enable scattering
########################################
line.xcoll.scattering.enable()

########################################
# Configure radiation
########################################
line.configure_radiation(model = RADIATION_MODEL)

########################################
# Build tracker
########################################
line.discard_tracker()
line.build_tracker(_context = CONTEXT)

################################################################################
# Track
################################################################################
print(f"\nTrack particles from {ELE_START}")

line.track(
    particles       = particles,
    ele_start       = ELE_START,
    ele_stop        = ELE_START,
    num_turns       = N_TURNS,
    with_progress   = 1,
    time            = True)

print(f"\nTracking {N_TURNS} turns done in: {line.time_last_track} s\n")

################################################################################
# Disable collimation
################################################################################

########################################
# Disable scattering
########################################
line.xcoll.scattering.disable()

########################################
# Stop GEANT4 engine
########################################
if GEANT4_SCATTERING_ON:
    xc.geant4.engine.stop()

########################################
# Close interaction record
########################################
impacts.stop()

################################################################################
# Loss Interpolation
################################################################################

########################################
# Switch bend fringe model for backtracking
########################################
# full fringe map is not invertible for backtracking
line.configure_bend_model(edge = "linear")
line.configure_quadrupole_model(edge = "suppressed")

########################################
# Create LossMap
########################################
# 1cm interpolation
LossMap = xc.LossMap(
    line                = line,
    part                = particles,
    line_is_reversed    = False,
    interpolation       = 0.01,
    weights             = None,
    weight_function     = None)

################################################################################
# Save outputs
################################################################################

########################################
# Impacts
########################################
impacts_fpath = os.path.join(outputdata_dir, 'impacts.parquet')
df_impacts  = impacts.to_pandas()
df_impacts.to_parquet(impacts_fpath)
    
########################################
# LossMap
########################################
lossmap_fpath = os.path.join(outputdata_dir, 'lossmap.json')
LossMap.to_json(lossmap_fpath)
with open(lossmap_fpath, "r", encoding="utf-8") as f:
    lossmap_json = json.load(f)

lossmap_json["params"] = params
with open(lossmap_fpath, "w", encoding="utf-8") as f:
    json.dump(lossmap_json, f, indent=2)

########################################
# Lost particles
########################################
lost_particles_fpath = os.path.join(outputdata_dir, 'particles.parquet')
# -999999 is empty capacity, 0 is alive
lost_particles  = particles.filter(
    (particles.state > -999999999) & (particles.state <= 0))

at_element_names    = np.take(line.element_names, lost_particles.at_element)

df = pd.DataFrame({
    "s":                   lost_particles.s,
    "state":               lost_particles.state,
    "particle_id":         lost_particles.particle_id,
    "parent_particle_id":  lost_particles.parent_particle_id,
    "pdg_id":              lost_particles.pdg_id,
    "at_element":          at_element_names,
    "at_turn":             lost_particles.at_turn,
    "x":                   lost_particles.x,
    "px":                  lost_particles.px,
    "y":                   lost_particles.y,
    "py":                  lost_particles.py,
    "zeta":                lost_particles.zeta,
    "delta":               lost_particles.delta,
    "weight":              lost_particles.weight,
}).astype({
    "s":                   np.float64,
    "state":               np.int16,
    "particle_id":         np.int32,
    "parent_particle_id":  np.int32,
    "pdg_id":              np.int32,
    "at_turn":             np.int32,
    "at_element":          "category",
    "x":                   np.float32,
    "px":                  np.float32,
    "y":                   np.float32,
    "py":                  np.float32,
    "zeta":                np.float32,
    "delta":               np.float32,
    "weight":              np.float32})

df.attrs["max_particle_id"] = int(max(particles.particle_id))

df.to_parquet(lost_particles_fpath, index=False, compression="snappy")