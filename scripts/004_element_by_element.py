""""
================================================================================
Track a Single Injected Particle Element-by-Element
================================================================================
Belle II Simulation & Modelling Group

Authors:    T. Nemoto, J.P.T. Salvesen, G. Broggi
Email:      TBD
Date:       2026-09-17
================================================================================
"""

################################################################################
# Required Packages
################################################################################
import numpy as np
import matplotlib.pyplot as plt
import xobjects as xo
import xtrack as xt

################################################################################
# User Parameters
################################################################################

########################################
# Lattice
########################################
ENV_FILEPATH            = "/eos/user/g/gbroggi/superkekb_forNemoto/lattices/sler_1802_60_09_aper.json"
LINE_NAME               = "line"

########################################
# Switching configurations
########################################
RADIATION_MODEL         = "mean"        # None, "mean", or "quantum"

# twiss() cannot be run with a stochastic radiation model, so we twiss with
# "mean" instead and only switch to the real RADIATION_MODEL right before
# the EBE tracking loop.
TWISS_RADIATION_MODEL   = "mean" if RADIATION_MODEL == "quantum" else RADIATION_MODEL

########################################
# Injection Parameters
########################################
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
# Start element
########################################
ELE_START               = "injectio"

########################################
# Tracking parameters
########################################
N_TURNS                 = 100

########################################
# Multithreading
########################################
CONTEXT                 = xo.ContextCpu()

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
line    = env.lines[LINE_NAME]

########################################
# Cycle line
########################################
# Places ELE_START at index 0 / s = 0, so build_particles() and the
# per-turn EBE tracking below both naturally start and stop there.
line.cycle(ELE_START)

########################################
# Configure radiation
########################################
line.configure_radiation(model = TWISS_RADIATION_MODEL)

########################################
# Build tracker
########################################
line.discard_tracker()
line.build_tracker(_context = CONTEXT)

########################################
# Twiss
########################################
tw      = line.twiss6d()

################################################################################
# Collect Aperture Information
################################################################################
tt      = line.get_table()
tt_aper = tt.rows.match(element_type = "LimitEllipse")

s_aper = tt_aper.s
x_aper, y_aper   = [], []
x_shift, y_shift = [], []

for nn in tt_aper.name:
    x_aper.append(line[nn].a)
    x_shift.append(line[nn].shift_x)
    y_aper.append(line[nn].b)
    y_shift.append(line[nn].shift_y)

x_aper = np.array(x_aper)
y_aper = np.array(y_aper)

################################################################################
# Track Injected Particle one turn EBE
################################################################################

########################################
# Build particle
########################################
test_part = line.build_particles(
    x       = INJECTION_OFFSET_X + INJECTION_ERROR_X,
    px      = INJECTION_OFFSET_PX + INJECTION_ERROR_PX,
    y       = INJECTION_OFFSET_Y + INJECTION_ERROR_Y,
    py      = INJECTION_OFFSET_PY + INJECTION_ERROR_PY,
    zeta    = INJECTION_OFFSET_ZETA + INJECTION_ERROR_ZETA,
    delta   = INJECTION_OFFSET_DELTA + INJECTION_ERROR_DELTA)

########################################
# Fix longitudinal closed orbit
########################################
zeta_co  = tw["zeta", ELE_START]
delta_co = tw["delta", ELE_START]

mask_active = test_part.state == 1
test_part.zeta[mask_active] += zeta_co
delta_temp = test_part.delta.copy()
delta_temp[mask_active] += delta_co
test_part.update_delta(delta_temp)

########################################
# Configure radiation for tracking
########################################
line.configure_radiation(model = RADIATION_MODEL)
line.discard_tracker()
line.build_tracker(_context = CONTEXT)

########################################
# Build storage
########################################
ebe_records = {
    "s"   :     [],
    "x"   :     [],
    "px"  :     [],
    "y"   :     [],
    "py"  :     [],
    "zeta":     [],
    "delta":    []}

########################################
# Track EBE
########################################
for i in range(N_TURNS):
    line.track(test_part, turn_by_turn_monitor = "ONE_TURN_EBE")

    ebe_records["s"].append(line.record_last_track.s[0])
    ebe_records["x"].append(line.record_last_track.x[0])
    ebe_records["px"].append(line.record_last_track.px[0])
    ebe_records["y"].append(line.record_last_track.y[0])
    ebe_records["py"].append(line.record_last_track.py[0])
    ebe_records["zeta"].append(line.record_last_track.zeta[0])
    ebe_records["delta"].append(line.record_last_track.delta[0])

########################################
# Convert to numpy arrays
########################################
for key in ebe_records.keys():
    ebe_records[key] = np.array(ebe_records[key])

########################################
# Merge to continuous arrays
########################################
# s needs to be continuous, so we need to add the twiss s at the beginning
for i in range(N_TURNS):
    ebe_records["s"][i] += i * tw.s[-1]

ebe_records["s"]     = np.concatenate(ebe_records["s"])
ebe_records["x"]     = np.concatenate(ebe_records["x"])
ebe_records["px"]    = np.concatenate(ebe_records["px"])
ebe_records["y"]     = np.concatenate(ebe_records["y"])
ebe_records["py"]    = np.concatenate(ebe_records["py"])
ebe_records["zeta"]  = np.concatenate(ebe_records["zeta"])
ebe_records["delta"] = np.concatenate(ebe_records["delta"])

################################################################################
# Plot
################################################################################

########################################
# Orbit
########################################
fig, axs    = plt.subplots(2, 1, figsize = (8, 4), sharex = True)

axs[0].plot(tw.s, tw.x, label = "Twiss Orbit")
axs[0].plot(ebe_records["s"], ebe_records["x"], label = "EBE Orbit")
axs[0].plot(tt_aper.s, x_aper + x_shift, c = "k", label = "Aperture")
axs[0].plot(tt_aper.s, -x_aper + x_shift, c = "k")

axs[1].plot(tw.s, tw.y, label = "Twiss Orbit")
axs[1].plot(ebe_records["s"], ebe_records["y"], label = "EBE Orbit")
axs[1].plot(tt_aper.s, y_aper + y_shift, c = "k", label = "Aperture")
axs[1].plot(tt_aper.s, -y_aper + y_shift, c = "k")

axs[1].set_xlabel("s [m]")
axs[0].set_ylabel("x [m]")
axs[0].legend()
axs[1].set_ylabel("y [m]")
axs[1].legend()

########################################
# Momentum
########################################
fig, axs    = plt.subplots(2, 1, figsize = (8, 4), sharex = True)

axs[0].plot(ebe_records["s"], ebe_records["px"])
axs[1].plot(ebe_records["s"], ebe_records["py"])

axs[1].set_xlabel("s [m]")
axs[0].set_ylabel("px [1]")
axs[1].set_ylabel("py [1]")

########################################
# Zeta and Delta
########################################
fig, axs    = plt.subplots(2, 1, figsize = (8, 4), sharex = True)

axs[0].plot(ebe_records["s"], ebe_records["zeta"])
axs[1].plot(ebe_records["s"], ebe_records["delta"])

axs[1].set_xlabel("s [m]")
axs[0].set_ylabel("zeta [m]")
axs[1].set_ylabel("delta [1]")

########################################
# Show plots
########################################
plt.show()
print("done")
