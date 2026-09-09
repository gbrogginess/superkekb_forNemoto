"""
FMA Helpers [NAFF-based tune and diffusion extraction]
=============================================
Author(s):  J.P.T. Salvesen, T. Nemoto, G. Broggi
Email:      TBD
Date:       2026-XX-XX
"""

################################################################################
# Required Packages
################################################################################
import numpy as np
import nafflib as naff

################################################################################
# Remove Zeros
################################################################################
def remove_zeros(values, zero_tol, sigma = 1E-6):
    return np.where(
        np.abs(values * sigma) < zero_tol,
        zero_tol * (-1)**(np.signbit(values)) / sigma,
        values)

################################################################################
# Build Grid
################################################################################
def generate_particle_grid(
        line,
        phase,
        beamsizes,
        gemitt_x,
        gemitt_y,
        gemitt_zeta,
        scan_mode,
        norm_x_array,
        norm_y_array,
        norm_z_array,
        select_ids = None,
        zero_tol   = 1E-10):
    """
    Build particles for tracking.

    select_ids : array of int, optional
        Indices into the flattened (norm_a, norm_b) grid to build particles
        for. Used to track only a chunk of the full grid (htcondor split).
        If None, the full grid is built.
    """

    assert scan_mode in ("xy", "zx", "zy")
    nemitt_x    = gemitt_x * line.particle_ref.beta0 * line.particle_ref.gamma0
    nemitt_y    = gemitt_y * line.particle_ref.beta0 * line.particle_ref.gamma0
    nemitt_zeta = gemitt_zeta * line.particle_ref.beta0 * line.particle_ref.gamma0

    if scan_mode == "xy":
        norm_x_grid, norm_y_grid = np.meshgrid(norm_x_array, norm_y_array, indexing = "ij")
        norm_x = norm_x_grid.flatten()
        norm_y = norm_y_grid.flatten()

        if select_ids is not None:
            norm_x = norm_x[select_ids]
            norm_y = norm_y[select_ids]

        norm_x_values  = norm_x * np.cos(phase)
        norm_px_values = norm_x * np.sin(phase)
        norm_y_values  = norm_y * np.cos(phase)
        norm_py_values = norm_y * np.sin(phase)

        norm_x_values  = remove_zeros(norm_x_values, zero_tol, sigma = beamsizes.sigma_x[0])
        norm_px_values = remove_zeros(norm_px_values, zero_tol, sigma = beamsizes.sigma_px[0])
        norm_y_values  = remove_zeros(norm_y_values, zero_tol, sigma = beamsizes.sigma_y[0])
        norm_py_values = remove_zeros(norm_py_values, zero_tol, sigma = beamsizes.sigma_py[0])

        particles = line.build_particles(
            x_norm   = norm_x_values,
            px_norm  = norm_px_values,
            y_norm   = norm_y_values,
            py_norm  = norm_py_values,
            nemitt_x = nemitt_x,
            nemitt_y = nemitt_y)

    elif scan_mode == "zx":
        norm_z_grid, norm_x_grid = np.meshgrid(norm_z_array, norm_x_array, indexing = "ij")
        norm_z = norm_z_grid.flatten()
        norm_x = norm_x_grid.flatten()

        if select_ids is not None:
            norm_z = norm_z[select_ids]
            norm_x = norm_x[select_ids]

        norm_z_values  = norm_z * np.cos(phase)
        norm_pz_values = norm_z * np.sin(phase)
        norm_x_values  = norm_x * np.cos(phase)
        norm_px_values = norm_x * np.sin(phase)

        norm_z_values  = remove_zeros(norm_z_values, zero_tol, sigma = beamsizes.sigma_zeta[0])
        norm_pz_values = remove_zeros(norm_pz_values, zero_tol, sigma = beamsizes.sigma_pzeta[0])
        norm_x_values  = remove_zeros(norm_x_values, zero_tol, sigma = beamsizes.sigma_x[0])
        norm_px_values = remove_zeros(norm_px_values, zero_tol, sigma = beamsizes.sigma_px[0])

        particles = line.build_particles(
            x_norm     = norm_x_values,
            px_norm    = norm_px_values,
            zeta_norm  = norm_z_values,
            pzeta_norm = norm_pz_values,
            nemitt_x   = nemitt_x,
            nemitt_y   = nemitt_y,
            nemitt_z   = nemitt_zeta)

    elif scan_mode == "zy":
        norm_z_grid, norm_y_grid = np.meshgrid(norm_z_array, norm_y_array, indexing = "ij")
        norm_z = norm_z_grid.flatten()
        norm_y = norm_y_grid.flatten()

        if select_ids is not None:
            norm_z = norm_z[select_ids]
            norm_y = norm_y[select_ids]

        norm_z_values  = norm_z * np.cos(phase)
        norm_pz_values = norm_z * np.sin(phase)
        norm_y_values  = norm_y * np.cos(phase)
        norm_py_values = norm_y * np.sin(phase)

        norm_z_values  = remove_zeros(norm_z_values, zero_tol, sigma = beamsizes.sigma_zeta[0])
        norm_pz_values = remove_zeros(norm_pz_values, zero_tol, sigma = beamsizes.sigma_pzeta[0])
        norm_y_values  = remove_zeros(norm_y_values, zero_tol, sigma = beamsizes.sigma_y[0])
        norm_py_values = remove_zeros(norm_py_values, zero_tol, sigma = beamsizes.sigma_py[0])

        particles = line.build_particles(
            y_norm     = norm_y_values,
            py_norm    = norm_py_values,
            zeta_norm  = norm_z_values,
            pzeta_norm = norm_pz_values,
            nemitt_x   = nemitt_x,
            nemitt_y   = nemitt_y,
            nemitt_z   = nemitt_zeta)

    else:
        raise ValueError(f"Invalid scan_mode: {scan_mode}. Must be one of 'xy', 'zx', or 'zy'.")

    return particles

################################################################################
# Tunes from NAFF
################################################################################
def tunes_from_naff(
        a_norm_array,
        pa_norm_array,
        nominal_tune          = None,
        strongest_peak_factor = 4.0,
        max_candidates        = 100):
    """
    Compute tunes for each particle using NAFFLIB, correctly removing DC offsets.

    Parameters
    ----------
    a_norm_array : np.ndarray
        Array of normalized amplitudes, shape (n_particles, n_turns).
    pa_norm_array : np.ndarray
        Array of normalized momenta, shape (n_particles, n_turns).

    Returns
    -------
    tunes : np.ndarray
        Array of tunes for each particle, shape (n_particles,).
    """

    n_particles = a_norm_array.shape[0]

    a_dc_offset  = np.mean(a_norm_array, axis = 1)
    pa_dc_offset = np.mean(pa_norm_array, axis = 1)

    assert len(a_dc_offset) == n_particles
    assert len(pa_dc_offset) == n_particles

    amplitudes, frequencies = naff.multiparticle_harmonics(
        x  = (a_norm_array.T - a_dc_offset).T,
        px = (pa_norm_array.T - pa_dc_offset).T)

    amplitudes  = np.asarray(amplitudes)
    frequencies = np.asarray(frequencies)

    if amplitudes.ndim == 1:
        amplitudes = amplitudes[:, np.newaxis]
    if frequencies.ndim == 1:
        frequencies = frequencies[:, np.newaxis]

    def tune_distance(values, target):
        delta = np.abs(values - target)
        return np.minimum(delta, 1 - delta)

    tunes = np.empty(n_particles, dtype = float)

    for i in range(n_particles):
        amp_i  = amplitudes[i].ravel()
        freq_i = frequencies[i].ravel()

        valid = np.isfinite(amp_i) & np.isfinite(freq_i)
        if not np.any(valid):
            tunes[i] = np.nan
            continue

        amp_i   = amp_i[valid]
        freq_i  = freq_i[valid]
        amp_mag = np.abs(amp_i)

        strongest_idx  = np.argmax(amp_mag)
        strongest_freq = freq_i[strongest_idx]
        strongest_amp  = amp_mag[strongest_idx]

        if nominal_tune is None:
            tunes[i] = strongest_freq % 1
            continue

        nearby_mask = tune_distance(freq_i, nominal_tune) < 0.2
        if np.any(nearby_mask):
            freq_i  = freq_i[nearby_mask]
            amp_mag = amp_mag[nearby_mask]

        candidate_count = min(max_candidates, len(amp_mag))
        if candidate_count == 0:
            tunes[i] = strongest_freq % 1
            continue

        strongest_candidates = np.argsort(amp_mag)[-candidate_count:]
        candidate_freqs      = freq_i[strongest_candidates]
        candidate_amps       = amp_mag[strongest_candidates]

        nominal_index = np.argmin(tune_distance(candidate_freqs, nominal_tune))
        nominal_freq  = candidate_freqs[nominal_index]
        nominal_amp   = candidate_amps[nominal_index]

        # Choose the strongest peak unless it is far from the nominal tune,
        # in which case fall back to the strongest peak near the nominal tune
        if strongest_amp >= strongest_peak_factor * nominal_amp:
            chosen_freq = strongest_freq
        else:
            chosen_freq = nominal_freq

        tunes[i] = chosen_freq % 1

    assert len(tunes) == n_particles

    return tunes

################################################################################
# Compute FMA
################################################################################
def run_fma(
        norm_tracking_records,
        n_windows,
        scan_mode,
        nominal_tune_x,
        nominal_tune_y,
        valid_mask = None):
    """
    Extract per-particle tunes and a tune-diffusion coefficient from
    turn-by-turn normalised tracking data.

    scan_mode : {"xy", "zx", "zy"}
        Which two planes diff_coeff combines. Must match the scan_mode used
        to build the particle grid.
    nominal_tune_x, nominal_tune_y : float
        Machine working point (e.g. tw.qx % 1, tw.qy % 1). Passed on to NAFF
        to pick the right spectral peak when several are present.
    """

    n_turns     = norm_tracking_records.x_norm.shape[0]
    n_particles = norm_tracking_records.x_norm.shape[1]

    if valid_mask is None:
        valid_mask = np.ones(n_particles, dtype = bool)
    else:
        valid_mask = np.asarray(valid_mask, dtype = bool)
        if valid_mask.shape[0] != n_particles:
            raise ValueError(f"valid_mask has length {valid_mask.shape[0]}, expected {n_particles}")

    # Overall tunes, only for particles that are still alive
    tunes_x = np.full(n_particles, np.nan, dtype = float)
    tunes_y = np.full(n_particles, np.nan, dtype = float)
    tunes_z = np.full(n_particles, np.nan, dtype = float)

    if np.any(valid_mask):
        tunes_x[valid_mask] = tunes_from_naff(
            a_norm_array  = norm_tracking_records.x_norm[:, valid_mask].T,
            pa_norm_array = norm_tracking_records.px_norm[:, valid_mask].T,
            nominal_tune  = nominal_tune_x)
        tunes_y[valid_mask] = tunes_from_naff(
            a_norm_array  = norm_tracking_records.y_norm[:, valid_mask].T,
            pa_norm_array = norm_tracking_records.py_norm[:, valid_mask].T,
            nominal_tune  = nominal_tune_y)
        tunes_z[valid_mask] = tunes_from_naff(
            a_norm_array  = norm_tracking_records.zeta_norm[:, valid_mask].T,
            pa_norm_array = norm_tracking_records.pzeta_norm[:, valid_mask].T)

    # Tunes over sliding half-length windows, to get the tune diffusion
    tunes_x_windows = []
    tunes_y_windows = []
    tunes_z_windows = []

    for i in range(n_windows + 1):
        lower_index = int(i * (n_turns / 2) / n_windows)
        upper_index = int(0.5 * n_turns + i * (n_turns / 2) / n_windows)
        print(f"Window {i}: {lower_index} - {upper_index}")

        n_turns_window = upper_index - lower_index
        assert norm_tracking_records.x_norm[lower_index:upper_index, :].shape[0] == n_turns_window
        assert norm_tracking_records.px_norm[lower_index:upper_index, :].shape[0] == n_turns_window
        assert norm_tracking_records.y_norm[lower_index:upper_index, :].shape[0] == n_turns_window
        assert norm_tracking_records.py_norm[lower_index:upper_index, :].shape[0] == n_turns_window
        assert norm_tracking_records.zeta_norm[lower_index:upper_index, :].shape[0] == n_turns_window
        assert norm_tracking_records.pzeta_norm[lower_index:upper_index, :].shape[0] == n_turns_window

        tunes_x_window = np.full(n_particles, np.nan, dtype = float)
        tunes_y_window = np.full(n_particles, np.nan, dtype = float)
        tunes_z_window = np.full(n_particles, np.nan, dtype = float)

        if np.any(valid_mask):
            tunes_x_window[valid_mask] = tunes_from_naff(
                a_norm_array  = norm_tracking_records.x_norm[lower_index:upper_index, valid_mask].T,
                pa_norm_array = norm_tracking_records.px_norm[lower_index:upper_index, valid_mask].T,
                nominal_tune  = nominal_tune_x)
            tunes_y_window[valid_mask] = tunes_from_naff(
                a_norm_array  = norm_tracking_records.y_norm[lower_index:upper_index, valid_mask].T,
                pa_norm_array = norm_tracking_records.py_norm[lower_index:upper_index, valid_mask].T,
                nominal_tune  = nominal_tune_y)
            tunes_z_window[valid_mask] = tunes_from_naff(
                a_norm_array  = norm_tracking_records.zeta_norm[lower_index:upper_index, valid_mask].T,
                pa_norm_array = norm_tracking_records.pzeta_norm[lower_index:upper_index, valid_mask].T)

        tunes_x_windows.append(tunes_x_window)
        tunes_y_windows.append(tunes_y_window)
        tunes_z_windows.append(tunes_z_window)

    tunes_x_windows = np.array(tunes_x_windows)
    tunes_y_windows = np.array(tunes_y_windows)
    tunes_z_windows = np.array(tunes_z_windows)

    change_x = np.full(n_particles, np.nan, dtype = float)
    change_y = np.full(n_particles, np.nan, dtype = float)
    change_z = np.full(n_particles, np.nan, dtype = float)

    if np.any(valid_mask):
        change_x[valid_mask] = np.std(tunes_x_windows[:, valid_mask], axis = 0)
        change_y[valid_mask] = np.std(tunes_y_windows[:, valid_mask], axis = 0)
        change_z[valid_mask] = np.std(tunes_z_windows[:, valid_mask], axis = 0)

    # Get rid of exactly 0 values for the log
    change_x = np.where(np.isclose(change_x, 0), 1E-12, change_x)
    change_y = np.where(np.isclose(change_y, 0), 1E-12, change_y)
    change_z = np.where(np.isclose(change_z, 0), 1E-12, change_z)

    diff_coeff = np.full(n_particles, np.nan, dtype = float)
    diff_x     = np.full(n_particles, np.nan, dtype = float)
    diff_y     = np.full(n_particles, np.nan, dtype = float)
    diff_z     = np.full(n_particles, np.nan, dtype = float)

    if np.any(valid_mask):
        if scan_mode == "xy":
            diff_coeff[valid_mask] = 0.5 * np.log10(change_x[valid_mask]**2 + change_y[valid_mask]**2)
        elif scan_mode == "zx":
            diff_coeff[valid_mask] = 0.5 * np.log10(change_x[valid_mask]**2 + change_z[valid_mask]**2)
        elif scan_mode == "zy":
            diff_coeff[valid_mask] = 0.5 * np.log10(change_y[valid_mask]**2 + change_z[valid_mask]**2)
        else:
            raise ValueError(f"Invalid scan_mode: {scan_mode}. Must be one of 'xy', 'zx', or 'zy'.")
        diff_x[valid_mask] = 0.5 * np.log10(change_x[valid_mask]**2)
        diff_y[valid_mask] = 0.5 * np.log10(change_y[valid_mask]**2)
        diff_z[valid_mask] = 0.5 * np.log10(change_z[valid_mask]**2)

    # Put tunes on [0, 1] interval for valid particles only
    tunes_x = np.where(valid_mask, tunes_x % 1, np.nan)
    tunes_y = np.where(valid_mask, tunes_y % 1, np.nan)
    tunes_z = np.where(valid_mask, tunes_z % 1, np.nan)

    return tunes_x, tunes_y, tunes_z, diff_coeff, diff_x, diff_y, diff_z