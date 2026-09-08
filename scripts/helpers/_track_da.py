"""
Dynamic Aperture XY [Oide Method]
=============================================
Author(s):  John P T Salvesen
Email:      john.salvesen@cern.ch
Date:       18-11-2025
"""

################################################################################
# Required Modules
################################################################################
import xtrack as xt
import xobjects as xo
import numpy as np
import time
import matplotlib.pyplot as plt

################################################################################
# User Parameters
################################################################################

################################################################################
# Helper Functions
################################################################################
def print_heading(text, mode = "section"):
    assert mode in ["section", "subsection"]
    if mode == "section":
        print("\n" + "#" * 80 + "\n" + text + "\n" + "#" * 80)
    elif mode == "subsection":
        print("\n" + "#" * 40 + "\n" + text + "\n" + "#" * 40)

################################################################################
# User Parameters
################################################################################
# TODO: Still not 100% happy with this
# Can end up with the case where a particle isn't tracked
# It has no dead neighbours, but then the particle beside it is, and found to be dead
# Then the original particle is never tracked, even though it should be
# This is the downside of doing in rings

# Possible to flip from not checking death but checking alive?
# Assert a minimum number of alive neighbours?
# When checking 8 I think that means 3 alive neighbours minimum?
    # But this fails at the cornners where it means 5 alive neighbours minimum
# When checking 4 this check isn't very good

################################################################################
# Track Frontier Functions
################################################################################
def rectangular_ring_ids(shape: tuple) -> np.ndarray:
    """
    Return an (H,W) array giving the rectangular ring index of each cell.
    Ring 0 is the outer border; increases by 1 as you move inward.
    """
    H, W    = shape
    ii, jj  = np.indices((H, W))
    return np.minimum.reduce([ii, jj, H - 1 - ii, W - 1 - jj])

def neighbour_check_4(mask: np.ndarray) -> np.ndarray:
    """
    4-neighborhood dilation WITHOUT wraparound.

    #####
    ##x##
    #xox#
    ##x##
    #####

    """
    out              = np.zeros_like(mask, dtype = bool)
    out[1: , :]     |= mask[:-1, :]     # up
    out[:-1, :]     |= mask[1: , :]     # down
    out[: , 1:]     |= mask[: , :-1]    # left
    out[: , :-1]    |= mask[: , 1:]     # right
    return out

def neighbour_check_8(mask: np.ndarray) -> np.ndarray:
    """
    8-neighborhood dilation WITHOUT wraparound.

    #####
    #xxx#
    #xox#
    #xxx#
    #####

    """
    out              = np.zeros_like(mask, dtype = bool)
    out[1: , :]     |= mask[:-1, :]     # up
    out[:-1, :]     |= mask[1: , :]     # down
    out[: , 1:]     |= mask[: , :-1]    # left
    out[: , :-1]    |= mask[: , 1:]     # right
    out[1: , 1:]    |= mask[:-1, :-1]   # up-left
    out[1: , :-1]   |= mask[:-1, 1:]    # up-right
    out[:-1, 1:]    |= mask[1: , :-1]   # down-left
    out[:-1, :-1]   |= mask[1: , 1:]    # down-right
    return out

def track_frontiers(
        line:                   xt.Line,
        norm_a_grid:            np.ndarray,
        master_particles:       xt.Particles,
        n_turns:                int,
        track_time:             bool            = True,
        neighbourhood:          int             = 8,
        with_progress:          bool | int      = 10,
        _diagnostic_plotting:   bool            = False) -> np.ndarray:
    """
    Iteratively choose indices to track ring-by-ring.

    Parameters
    ----------
    line : xt.Line
        The line to track.
    norm_a_grid : (H, W) float
        The grid of normalized amplitudes in the horizontal plane.
    master_particles : xt.Particles
        The master particles to track.
    n_turns : int
        The number of turns to track.
    track_time : bool, optional
        Whether to track and print elapsed time. Default is True.
    neighbourhood : {4, 8}, optional
        The neighborhood definition for frontier growth. Default is 8.
    _diagnostic_plotting : bool, optional
        Whether to produce diagnostic plots at each ring. Default is False.

    Returns
    -------
    at_turn_grid : (H, W) float
        The grid of at-turn values after tracking.
    """
    ########################################
    # Assertions
    ########################################
    assert neighbourhood in (4, 8), "neighbourhood must be 4 or 8"

    ########################################
    # Elapsed Time Tracking
    ########################################
    if track_time:
        start_time  = time.time()

    ########################################
    # Build Rings
    ########################################
    grid_shape  = norm_a_grid.shape
    ring_id     = rectangular_ring_ids(grid_shape)
    max_ring    = int(ring_id.max())

    ########################################
    # Output: At turn grid
    ########################################
    at_turn_grid    = np.full(grid_shape, np.nan, dtype = float)

    ########################################
    # Dead particles mask to drive frontier
    ########################################
    died_mask       = np.zeros(grid_shape, dtype = bool)

    ########################################
    # Diagnostic Plotting
    ########################################
    if _diagnostic_plotting:
        fig1, ax1 = plt.subplots(figsize = (6,6))
        ax1.scatter(
            master_particles.x,
            master_particles.y,
            c   = "m")
        ax1.set_xlabel("x [m]")
        ax1.set_ylabel("y [m]")

    ########################################
    # Iterate over rings
    ########################################
    for k in range(0, max_ring + 1):
        ring_mask = (ring_id == k)

        ########################################
        # Get the indices to track
        ########################################
        if k == 0:
            # Track the entire outer ring
            to_track_mask = ring_mask
        else:
            # Track only cells that touch a *dead cell* from outer ring
            if neighbourhood == 8:
                frontier        = neighbour_check_8(died_mask)
            else:
                frontier        = neighbour_check_4(died_mask)
            to_track_mask   = frontier & ring_mask
        idx = np.flatnonzero(to_track_mask)

        ########################################
        # Exit if no indices to track
        ########################################
        if idx.size == 0:
            # No need to continue if nothing to track
            continue

        ########################################
        # Select Particles Here
        ########################################
        selection_mask  = np.isin(master_particles.particle_id, idx)
        particles       = master_particles.filter(selection_mask)

        ########################################
        # Track here
        ########################################
        line.track(
            particles       = particles,
            num_turns       = n_turns,
            with_progress   = with_progress)                    # type: ignore
        
        ########################################
        # Check for survival
        ########################################
        particles.sort(interleave_lost_particles = True)
        pids           = particles.particle_id

        at_turn         = particles.at_turn
        died_this_test      = pids[at_turn != n_turns]
        survived_this_test  = pids[at_turn == n_turns]
        
        ########################################
        # Diagnostic Plotting
        ########################################
        if _diagnostic_plotting:
            ax1.scatter(                                        # type: ignore
                master_particles.x[died_this_test],
                master_particles.y[died_this_test],
                c   = "r")
            ax1.scatter(                                        # type: ignore
                master_particles.x[survived_this_test],
                master_particles.y[survived_this_test],
                c   = "g")

            fig2, ax2 = plt.subplots(figsize = (6,6))
            ax2.scatter(
                master_particles.x,
                master_particles.y,
                c   = "m")
            ax2.scatter(
                master_particles.x[died_this_test],
                master_particles.y[died_this_test],
                c   = "r")
            ax2.scatter(
                master_particles.x[survived_this_test],
                master_particles.y[survived_this_test],
                c   = "g")
            ax2.set_xlabel("x [m]")
            ax2.set_ylabel("y [m]")

        ########################################
        # Update at_turn grid
        ########################################
        at_turn_here   = particles.at_turn.astype(float)
        m_flat         = at_turn_grid.ravel()
        m_flat[pids]   = at_turn_here

        ########################################
        # Update died frontier
        ########################################
        died_here       = (particles.at_turn != n_turns)
        dm_flat         = died_mask.ravel()
        dm_flat[pids]   = died_here

        ########################################
        # Propagate only with failures
        ########################################
        if not died_here.any():
            break

    ########################################
    # Elapsed Time Tracking
    ########################################
    if track_time:
        end_time  = time.time()
        elapsed   = end_time - start_time                                       # type: ignore
        print(f"Tracking completed in {elapsed:.2f} seconds.")

    if _diagnostic_plotting:
        plt.show()

    return at_turn_grid

################################################################################
# Main Function
################################################################################
def track_da(
        line:               xt.Line,
        norm_a_array:       np.ndarray,
        norm_b_array:       np.ndarray,
        phases:             np.ndarray | list,
        n_turns:            int,
        gemitt_x:           float,
        gemitt_y:           float,
        gemitt_z:           float,
        mode:               str,
        tracking_context:   xo.ContextCpu):
    """
    Track Dynamic Aperture in specified mode.

    Parameters
    ----------
    line : xt.Line
        The line to track.
    norm_a_array : 1D float array
        The normalized amplitude array for plane A.
    norm_b_array : 1D float array
        The normalized amplitude array for plane B.
    phases : 1D float array or list
        The betatron phases to consider.
    n_turns : int
        The number of turns to track.
    gemitt_x : float
        The geometric emittance in the x plane [m].
    gemitt_y : float
        The geometric emittance in the y plane [m].
    gemitt_z : float
        The geometric emittance in the z plane [m].
    mode : {"xy", "zx", "zy", "ma"}
        The tracking mode.
    tracking_context : xo.ContextCpu
        The xobjects context to use for tracking.
    """

    ########################################
    # Assertions
    ########################################
    assert mode in ["xy", "zx", "zy", "ma"]
    assert line.particle_ref is not None, ("line.particle_ref is None. "
        "Please set line.particle_ref before calling track_da().")

    ############################################################################
    # Setup
    ############################################################################
    print_heading(f"Tracking DA in mode {mode}", mode = "section")

    ########################################
    # Twiss
    ########################################
    print_heading("Configuring Mean Radiation", mode = "subsection")
    line.discard_tracker()
    line.build_tracker(_context = xo.context_default)
    line.configure_radiation(model = "mean")

    ########################################
    # Twiss
    ########################################
    print_heading("Calculating Twiss with Radiation", mode = "subsection")
    tw  = line.twiss(eneloss_and_damping = True)

    ########################################
    # Calculate Normalised Emittances
    ########################################
    print_heading("Calculating Normalised Emittances", mode = "subsection")
    nemitt_x    = gemitt_x * line.particle_ref.beta0 * line.particle_ref.gamma0
    nemitt_y    = gemitt_y * line.particle_ref.beta0 * line.particle_ref.gamma0
    sigma_z     = np.sqrt(gemitt_z * tw.bets0)
    sigma_delta = np.sqrt(gemitt_z / tw.bets0)

    ########################################
    # Create Output Dictionaries
    ########################################
    dynamic_aperture    = {
        "phase":            [],
        "at_turn":          [],
        "x_norm_init":      [],
        "y_norm_init":      [],
        "z_norm_init":      [],
        "gemitt_x":         gemitt_x,
        "gemitt_y":         gemitt_y,
        "gemitt_z":         gemitt_z}

    ############################################################################
    # Build Particles
    ############################################################################
    print_heading("Building Particles for DA Tracking", mode = "subsection")

    ########################################
    # Create Mesh
    ########################################
    NORM_A_GRID, NORM_B_GRID    = np.meshgrid(norm_a_array, norm_b_array)

    ########################################
    # Build Particles
    ########################################
    particles_all_phases    = []
    for phase in phases:

        if mode == "xy":
            particles_all_phases.append(line.build_particles(
                nemitt_x    = nemitt_x,
                nemitt_y    = nemitt_y,
                x_norm      = NORM_A_GRID.flatten() * np.cos(phase),
                px_norm     = NORM_A_GRID.flatten() * np.sin(phase),
                y_norm      = NORM_B_GRID.flatten() * np.cos(phase),
                py_norm     = NORM_B_GRID.flatten() * np.sin(phase),
                zeta        = tw.zeta[0],
                delta       = tw.delta[0]))
        elif mode == "zx":
            particles_all_phases.append(line.build_particles(
                nemitt_x    = nemitt_x,
                nemitt_y    = nemitt_y,
                x_norm      = NORM_B_GRID.flatten() * np.cos(phase),
                px_norm     = NORM_B_GRID.flatten() * np.sin(phase),
                y_norm      = 0,
                py_norm     = 0,
                zeta        = NORM_A_GRID.flatten() * np.cos(phase) * \
                    sigma_z + tw.zeta[0],
                delta       = NORM_A_GRID.flatten() * np.sin(phase) * \
                    sigma_delta + tw.delta[0]))
        elif mode == "zy":
            particles_all_phases.append(line.build_particles(
                nemitt_x    = nemitt_x,
                nemitt_y    = nemitt_y,
                x_norm      = 0,
                px_norm     = 0,
                y_norm      = NORM_B_GRID.flatten() * np.cos(phase),
                py_norm     = NORM_B_GRID.flatten() * np.sin(phase),
                zeta        = NORM_A_GRID.flatten() * np.cos(phase) * \
                    sigma_z + tw.zeta[0],
                delta       = NORM_A_GRID.flatten() * np.sin(phase) * \
                    sigma_delta + tw.delta[0]))
        elif mode == "ma":
            particles_all_phases.append(line.build_particles(
                nemitt_x    = nemitt_x,
                nemitt_y    = nemitt_y,
                x_norm      = NORM_B_GRID.flatten() * np.cos(phase),
                px_norm     = NORM_B_GRID.flatten() * np.sin(phase),
                y_norm      = NORM_B_GRID.flatten() * np.cos(phase),
                py_norm     = NORM_B_GRID.flatten() * np.sin(phase),
                zeta        = NORM_A_GRID.flatten() * np.cos(phase) * \
                    sigma_z + tw.zeta[0],
                delta       = NORM_A_GRID.flatten() * np.sin(phase) * \
                    sigma_delta + tw.delta[0]))

    ################################################################################
    # Dynamic Aperture Tracking
    ################################################################################

    ########################################
    # Build Trackers
    ########################################
    print_heading("Building DA Trackers", mode = "subsection")
    line.discard_tracker()
    line.build_tracker(_context = tracking_context)
    
    ########################################
    # Track HER
    ########################################
    print_heading("Tracking", mode = "subsection")

    for phase, particles in zip(phases, particles_all_phases):
        print("Tracking DA for phase:", phase)

        at_turn_grid    = track_frontiers(
            line                    = line,
            norm_a_grid             = NORM_A_GRID,
            master_particles        = particles,
            n_turns                 = n_turns,
            track_time              = True,
            neighbourhood           = 8,
            with_progress           = 10,
            _diagnostic_plotting    = False)

        ########################################
        # Append Data
        ########################################
        dynamic_aperture["phase"].append(phase)
        dynamic_aperture["at_turn"].append(at_turn_grid)

        if mode == "xy":
            dynamic_aperture["x_norm_init"].append(NORM_A_GRID)
            dynamic_aperture["y_norm_init"].append(NORM_B_GRID)
            dynamic_aperture["z_norm_init"].append(np.zeros_like(NORM_A_GRID))
        elif mode == "zx":
            dynamic_aperture["x_norm_init"].append(NORM_B_GRID)
            dynamic_aperture["y_norm_init"].append(np.zeros_like(NORM_A_GRID))
            dynamic_aperture["z_norm_init"].append(NORM_A_GRID)
        elif mode == "zy":
            dynamic_aperture["x_norm_init"].append(np.zeros_like(NORM_A_GRID))
            dynamic_aperture["y_norm_init"].append(NORM_B_GRID)
            dynamic_aperture["z_norm_init"].append(NORM_A_GRID)
        elif mode == "ma":
            dynamic_aperture["x_norm_init"].append(NORM_B_GRID)
            dynamic_aperture["y_norm_init"].append(NORM_B_GRID)
            dynamic_aperture["z_norm_init"].append(NORM_A_GRID)

    # TODO: Work out how to do this properly
    # It requires filling the NaNs with the max turn number first
    # But doing so loses information about which particles were never tracked
    # ########################################
    # # Mean over phases
    # ########################################
    # if len(list(phases)) > 1:
    #     at_turn_mean    = np.mean(
    #         np.array(dynamic_aperture["at_turn"]),
    #         axis    = 0)
    # else:
    #     at_turn_mean    = np.array(dynamic_aperture["at_turn"])

    ########################################
    # Convert Types
    ########################################
    dynamic_aperture    = {
        "phase":            np.array(dynamic_aperture["phase"]),
        "at_turn":          np.array(dynamic_aperture["at_turn"]),
        # "at_turn_mean":     at_turn_mean,
        "x_norm_init":      np.array(dynamic_aperture["x_norm_init"]),
        "y_norm_init":      np.array(dynamic_aperture["y_norm_init"]),
        "z_norm_init":      np.array(dynamic_aperture["z_norm_init"]),
        "gemitt_x":         gemitt_x,
        "gemitt_y":         gemitt_y,
        "gemitt_z":         gemitt_z}
    
    return dynamic_aperture
    
