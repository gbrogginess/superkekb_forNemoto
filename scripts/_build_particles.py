""""
================================================================================
Particle Generation
================================================================================
Belle II Simulation & Modelling Group

Authors:    T. Nemoto, G. Nigrelli, J.P.T. Salvesen
Email:      TBD
Date:       2026-07-30
================================================================================
"""

################################################################################
# Required Packages
################################################################################
import os
import numpy as np
import xtrack as xt

################################################################################
# Helper Functions
################################################################################

########################################
# Read Yoshimoto"s .dat file and convert to dictionary
########################################
def read_dat_to_dict(filename, n_turns=1, n_particles=None, seed=None):
    """
    Reads a .dat file with a header like:
    x[m]   xp[rad]   y[m]   yp[rad]   z[m]   dpp[]   flag   @ S=...

    
    Parameters
    ----------
    filename : str
        Path to the .dat file.
    n_turns : int
        Number of turns to consider for reshaping the data.
    n_particles : int or None
        Number of particles to select. If None, all particles are used.
    seed : int or None
        Random seed for reproducibility when selecting particles.

    Returns
    -------
    dic_coordinates : dict
        A dictionary containing the particle coordinates in the format:
        (x, px, y, py, zeta, delta) with shape (n_particles, n_turns).
    """

    ########################################
    # Initial parse
    ########################################
    data_rows = []
    with open(filename, "r") as f:
        # First line is header columns: coorinates and then "@" followed by S value
        header_line     = f.readline()
        header_parts    = header_line.split("@")[0].split()
        
        # Map header names to indices
        col_index = {name: i for i, name in enumerate(header_parts)}
        
        # Now read the rest of the file
        for line in f:
            line = line.strip()
            if not line or line.startswith("@"):
                continue
            values = [float(v) for v in line.split()]
            data_rows.append(values)

    ########################################
    # Convert to numpy array
    ########################################
    data = np.array(data_rows, dtype=float)  # shape (total_rows, n_cols)
    total_rows = data.shape[0]

    ########################################
    # Determine particle count
    ########################################
    # File must have shape n_particles * n_turns (therefore divisible by n_turns)
    if total_rows % n_turns != 0:
        raise ValueError(
            f"Row count {total_rows} is not divisible by n_turns={n_turns}")
    n_particles_total = total_rows // n_turns

    # If unspecified, use all particles
    if n_particles is None:
        n_particles = n_particles_total

    # Check if requested number of particles exceeds available
    if n_particles > n_particles_total:
        raise ValueError("Requested more particles than available")

    # Reshape data to (n_particles_total, n_turns, n_cols)
    reshaped = data.reshape((n_particles_total, n_turns, -1))

    ########################################
    # Randomly select particles if needed
    ########################################
    if n_particles < n_particles_total:
        rng         = np.random.default_rng(seed)
        chosen_idx  = rng.choice(
            a       = n_particles_total,
            size    = n_particles,
            replace = False)
        reshaped    = reshaped[chosen_idx]
    else:
        reshaped = reshaped[:n_particles]

    ########################################
    # Build output dictionary based on header mapping
    ########################################
    dic_coordinates = {
        "x":     reshaped[:, :, col_index["x[m]"]],
        "px":    reshaped[:, :, col_index["xp[rad]"]],
        "y":     reshaped[:, :, col_index["y[m]"]],
        "py":    reshaped[:, :, col_index["yp[rad]"]],
        "zeta":  reshaped[:, :, col_index["z[m]"]],
        "delta": reshaped[:, :, col_index["dpp[]"]]}
    return dic_coordinates


########################################
# Compute sigma matrices and emittances from particle distributions
########################################
def sigma_matrix_and_emittances_from_tracking(
        x:      np.ndarray,
        px:     np.ndarray,
        y:      np.ndarray,
        py:     np.ndarray,
        zeta:   np.ndarray,
        delta:  np.ndarray):
    """
    Computes the covariance matrix and emittances from particle distribution data.
    
    Parameters
    ----------
    x, px, y, py, zeta, delta : np.ndarray
        1D arrays of particle coordinates in phase space.
    
    Returns
    -------
    cov_matrix : np.ndarray
        The 6x6 covariance matrix of the particle distribution.
    emittances : np.ndarray
        The 6D emittances corresponding to the phase space coordinates.
    """

    ########################################
    # Assert inputs are 1D arrays with equal lengths
    ########################################
    assert x.ndim == px.ndim == y.ndim == py.ndim == zeta.ndim == delta.ndim == 1, \
        "All inputs must be 1D arrays"
    assert len(x) == len(px) == len(y) == len(py) == len(zeta) == len(delta), \
        "All inputs must have the same length"

    ########################################
    # Build the antisimmetric matrix for the 6D phase space
    ########################################
    antisimmetric_matrix        = np.zeros((6,6))
    antisimmetric_matrix[0,1]   = 1
    antisimmetric_matrix[1,0]   = -1
    antisimmetric_matrix[2,3]   = 1
    antisimmetric_matrix[3,2]   = -1
    antisimmetric_matrix[4,5]   = 1
    antisimmetric_matrix[5,4]   = -1

    ########################################
    # Data cleaning
    ########################################
    # Convert to 2D array for matrix operations
    data = np.column_stack((x, px, y, py, zeta, delta))  # shape: (n_particles, 6)

    # Remove rows with NaN values
    data = data[~np.isnan(data).any(axis=1)]

    ########################################
    # Compute covariance matrix
    ########################################
    # Compute covariance matrix only if data remains
    if data.shape[0] > 1:
        cov_matrix  = np.cov(data, rowvar=False)
        # Adjust the covariance matrix by multiplying by (n-1)/n to replace Bessel"s correction
        cov_matrix  *= (data.shape[0] - 1) / data.shape[0]
    else:
        cov_matrix  = np.full((6, 6), np.nan)  # If no valid data, fill with NaNs

    ########################################
    # Compute emittances from the covariance matrix
    ########################################
    # Initialize empty arrays to store the result
    sigma_dot_antisimmetric = np.full((6, 6), np.nan)
    emittances              = np.full((6), np.nan)

    # Ensure sufficient statistics for accurate calculation 
    if np.isnan(cov_matrix[:, :]).any() or len(x[~np.isnan(x)]) < 119:
        print("Not enough particles (poor statistics) for acurate calculation of the sigma matrix and emittances therefore, are set to Nan !")
        return (cov_matrix, emittances)
    
    sigma_dot_antisimmetric[:, :]   = np.dot(cov_matrix[:, :], antisimmetric_matrix)
    eigenvalues, eigenvectors       = np.linalg.eig(sigma_dot_antisimmetric[:, :])
    
    max_indices = np.argmax(np.abs(eigenvectors), axis=0)
    
    # restor of the corrext order (x,y,z) of the emittances 
    for jj in range(3):
        
        if 2*jj in max_indices:
            indice = np.where(max_indices == 2*jj)[0][0]
            emittances[2*jj] = np.abs(1j*eigenvalues[indice])
        elif 2*jj+1 in max_indices:
            indice = np.where(max_indices == 2*jj+1)[0][0]
            emittances[2*jj] = np.abs(1j*eigenvalues[indice])

        if 2*jj+1 in max_indices:
            indice = np.where(max_indices == 2*jj+1)[0][0]
            emittances[2*jj+1] = np.abs(1j*eigenvalues[indice])
        elif 2*jj in max_indices:
            indice = np.where(max_indices == 2*jj)[0][0]
            emittances[2*jj+1] = np.abs(1j*eigenvalues[indice])
    
    return (cov_matrix, emittances)

########################################
# Compute relativistic beta and gamma
########################################
def calculate_electron_beta_gamma(p0c: float):
    """
    Calculate beta and gamma for an electron given its momentum in eV/c.

    Parameters
    ----------
    p0c: float
        Momentum of the electron in eV/c.

    Returns
    -------
    beta: float
        The velocity of the electron divided by the speed of light (v/c).
    gamma: float
        The Lorentz factor of the electron.
    """
    gamma   = np.sqrt(1.0 + (p0c / xt.ELECTRON_MASS_EV)**2)
    beta    = p0c / (gamma * xt.ELECTRON_MASS_EV)
    return beta, gamma

################################################################################
# Coordinate Transformation Functions
################################################################################

########################################
# Compute normalized coordinates from physical coordinates
########################################
def compute_normalized_coordinates(
        twiss,
        x,
        px,
        y,
        py,
        nemitt_x,
        nemitt_y,
        p0c,
        monitor_name = "0"):
    """
    Function to manualy calculate the normalized coordinates from the physical coordinates

    Parameters
    ----------
    twiss : xtrack.TwissTable
        The Twiss parameters of the line.
    x, y, px, py : np.ndarray
        Physical coordinates of the particles.
    nemitt_x, nemitt_y : float
        Normalized emittances in the x and y planes.
    p0c : float
        Momentum of the particle in eV/c.
    monitor_name : str, optional
        Name of the monitor element to use for Twiss parameters. Default is "0".

    Returns
    -------
    x_norm, y_norm, px_norm, py_norm : np.ndarray
        Normalized coordinates of the particles.
    """

    ########################################
    # Compute geometric emittances from normalized emittances
    ########################################
    beta, gamma = calculate_electron_beta_gamma(p0c)
    gemitt_x    =  nemitt_x / (beta * gamma) 
    gemitt_y    =  nemitt_y / (beta * gamma) 

    ########################################
    # Load optical functions
    ########################################
    if monitor_name == "0":
        monitor_name = "ip.0"

    betx = twiss["betx", monitor_name]
    bety = twiss["bety", monitor_name]
    alfx = twiss["alfx", monitor_name]
    alfy = twiss["alfy", monitor_name]

    ########################################
    # Compute normalized coordinates
    ########################################
    x_norm  = x / np.sqrt(betx * gemitt_x)
    px_norm = (alfx * x + betx* px) / np.sqrt(betx * gemitt_x)
    y_norm  = y / np.sqrt(bety * gemitt_y)
    py_norm = (alfy * y + bety* py) / np.sqrt(bety * gemitt_y)

    return x_norm, y_norm, px_norm, py_norm

########################################
# Compute physical coordinates from normalized coordinates
########################################
def compute_physical_coordinates(
        twiss,
        x_norm,
        px_norm,
        y_norm,
        py_norm,
        nemitt_x, 
        nemitt_y,
        p0c,
        monitor_name = "0"):
    """
    Function to manualy calculate the physical coordinates from the normalized coordinates

    Parameters
    ----------
    twiss : xtrack.TwissTable
        The Twiss parameters of the line.
    x_norm, y_norm, px_norm, py_norm : np.ndarray
        Normalized coordinates of the particles.
    nemitt_x, nemitt_y : float
        Normalized emittances in the x and y planes.
    p0c : float
        Momentum of the particle in eV/c.
    monitor_name : str, optional
        Name of the monitor element to use for Twiss parameters. Default is "0".

    Returns
    -------
    x, y, px, py : np.ndarray
        Physical coordinates of the particles.
    """

    ########################################
    # Compute geometric emittances from normalized emittances
    ########################################
    beta, gamma = calculate_electron_beta_gamma(p0c)
    gemitt_x    =  nemitt_x / (beta * gamma) 
    gemitt_y    =  nemitt_y / (beta * gamma) 

    ########################################
    # Load optical functions
    ########################################
    if monitor_name == "0":
        monitor_name = "ip.0"

    betx = twiss["betx", monitor_name]
    bety = twiss["bety", monitor_name]
    alfx = twiss["alfx", monitor_name]
    alfy = twiss["alfy", monitor_name]

    ########################################
    # Compute physical coordinates
    ########################################
    x   = x_norm * np.sqrt(betx * gemitt_x)
    px  = (px_norm - alfx * x_norm) * np.sqrt(gemitt_x / betx)
    y   = y_norm * np.sqrt(bety * gemitt_y)
    py  = (py_norm - alfy * y_norm) * np.sqrt(gemitt_y / bety)

    return x, y, px, py

################################################################################
# Create injection beam from file and prepare for tracking
################################################################################
def prepare_injection_beam(
        line,
        twiss,
        input_file,
        ele_start,
        n_part,
        capacity,
        turn_number             = 0,
        target_nemitt_x         = None,
        target_nemitt_y         = None,
        initially_centre_bunch  = True,
        injection_x_offset      = 0.0,
        injection_px_offset     = 0.0,
        injection_y_offset      = 0.0,
        injection_py_offset     = 0.0,
        injection_zeta_offset   = 0.0,
        injection_delta_offset  = 0.0,
        injection_x_error       = 0.0,
        injection_px_error      = 0.0,
        injection_y_error       = 0.0,
        injection_py_error      = 0.0,
        injection_zeta_error    = 0.0,
        injection_delta_error   = 0.0,
        twiss_off_momentum      = None,
        _context                = None):
    """
    Prepare an Xpart particles object for injection tracking based on a given input file.

    Parameters
    ----------
    line : xtrack.Line
        The accelerator line object.
    twiss : xtrack.TwissTable
        The Twiss parameters of the line.
    input_file : str
        Path to the input .dat file containing particle coordinates.
    ele_start : str
        Name of the element where tracking starts.
    n_part : int
        Number of particles to consider from the input file.
    capacity : int
        Capacity for the Xpart particles object.
    target_nemitt_x : float
        Target normalized emittance in the x-plane (m·rad).
    target_nemitt_y : float
        Target normalized emittance in the y-plane (m·rad).
    injection_x_offset : float
        Offset in the x-plane injection position (m).
    injection_px_offset : float
        Offset in the x-plane injection angle (rad).
    injection_y_offset : float
        Offset in the y-plane injection position (m).
    injection_py_offset : float
        Offset in the y-plane injection angle (rad).
    injection_zeta_offset : float
        Offset in the zeta-plane injection position (m).
    injection_delta_offset : float
        Offset in the delta-plane injection position.
    injection_x_error : float
        Error in the x-plane injection position (m).
    injection_y_error : float
        Error in the y-plane injection position (m).
    injection_px_error : float
        Error in the x-plane injection angle (rad).
    injection_py_error : float
        Error in the y-plane injection angle (rad).
    injection_zeta_error : float
        Error in the zeta-plane injection position (m).
    injection_delta_error : float
        Error in the delta-plane injection position.
    twiss_off_momentum : xtrack.TwissTable or None
        Twiss parameters for off-momentum particles, required for SI files.
    _context : xobjects.Context or None
        The context for the particles object. If None, the line"s context is used.

    Returns
    -------
    part : xpart.Particles
        The prepared particles object ready for tracking.
    """

    ########################################
    # Ensure the input file exists in correct style
    ########################################
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"File {input_file} not found")

    if not input_file.lower().endswith(".dat"):
        raise ValueError(f"File {input_file} must be a .dat file")
    base_name = os.path.basename(input_file).lower()
    base_name = os.path.splitext(base_name)[0]  # Remove extension

    if not (base_name.startswith("ler") or base_name.startswith("her")):
        raise ValueError(f'''File {input_file} must start with "LER" or "HER"''')

    ########################################
    # Ensure charge is correct for LER/HER
    ########################################
    if base_name.startswith("ler"):
        print(f"Processing LER file: {input_file}")
        if line.particle_ref.charge == -1:
            raise ValueError(f"LER injection beam must have positive charge")
        dic_coordinates = read_dat_to_dict(input_file, n_turns = 1, n_particles=n_part)
    elif base_name.startswith("her"):
        print(f"Processing HER file: {input_file}")
        if line.particle_ref.charge == +1:
            raise ValueError(f"HER injection beam must have negative charge")
        dic_coordinates = read_dat_to_dict(input_file, n_turns = 1, n_particles=n_part)

    ########################################
    # Extract coordinates from dictionary
    ########################################
    # In the dictionary, the shape is (n_particles, n_turns), so we squeeze to get 1D arrays
    x       = np.squeeze(dic_coordinates["x"][:, turn_number])
    y       = np.squeeze(dic_coordinates["y"][:, turn_number])
    px      = np.squeeze(dic_coordinates["px"][:, turn_number])
    py      = np.squeeze(dic_coordinates["py"][:, turn_number])
    zeta    = np.squeeze(dic_coordinates["zeta"][:, turn_number])
    delta   = np.squeeze(dic_coordinates["delta"][:, turn_number])

    ########################################
    # Compute emittances
    ########################################
    cov_matrices, emittances    = sigma_matrix_and_emittances_from_tracking(
        x       = x,
        y       = y,
        px      = px,
        py      = py,
        zeta    = zeta,
        delta   = delta)
    emit_x = emittances[0]
    emit_y = emittances[2]

    ########################################
    # Rescale emittances
    ########################################
    beta, gamma     = calculate_electron_beta_gamma(line.particle_ref.p0c)
    nemitt_initial  = np.array([emit_x * beta * gamma, emit_y * beta * gamma])

    if target_nemitt_x is None:
        target_nemitt_x = nemitt_initial[0]
    if target_nemitt_y is None:
        target_nemitt_y = nemitt_initial[1]
    nemitt_target   = np.array([target_nemitt_x, target_nemitt_y])

    # --- Normalize and renormalize coordinates ---
    x_norm, y_norm, px_norm, py_norm    = compute_normalized_coordinates(
        twiss           = twiss,
        x               = x,
        px              = px,
        y               = y,
        py              = py,
        nemitt_x        = nemitt_initial[0],
        nemitt_y        = nemitt_initial[1],
        p0c             = line.particle_ref.p0c,
        monitor_name    = ele_start)

    x, y, px, py = compute_physical_coordinates(
        twiss           = twiss,
        x_norm          = x_norm,
        px_norm         = px_norm,
        y_norm          = y_norm,
        py_norm         = py_norm,
        nemitt_x        = nemitt_target[0],
        nemitt_y        = nemitt_target[1],
        p0c             = line.particle_ref.p0c,
        monitor_name    = ele_start)

    ########################################
    # Initial bunch centering
    ########################################
    if initially_centre_bunch:
        x       -= np.mean(x)
        px      -= np.mean(px)
        y       -= np.mean(y)
        py      -= np.mean(py)
        zeta    -= np.mean(zeta)
        delta   -= np.mean(delta)
    
    ########################################
    # Injection offsets and angles
    ########################################
    # TODO: There are some sign reversals here that need clarifying
    if "SI" not in base_name:
        x       += injection_x_offset
        px      += injection_px_offset
        y       += injection_y_offset
        py      += injection_py_offset
        zeta    += injection_zeta_offset
        delta   += injection_delta_offset
    else:
        assert twiss_off_momentum is not None, \
            "twiss_off_momentum must be provided for SI files"

        x       += twiss_off_momentum["x", ele_start] - injection_x_offset
        px      += twiss_off_momentum["px", ele_start] - injection_px_offset
        y       += twiss_off_momentum["y", ele_start] - injection_y_offset
        py      += twiss_off_momentum["py", ele_start] - injection_py_offset
        zeta    += twiss_off_momentum["zeta", ele_start] - injection_zeta_offset
        delta   += twiss_off_momentum["delta", ele_start] - injection_delta_offset

    ########################################
    # Injection errors
    ########################################
    x       += injection_x_error
    px      += injection_px_error
    y       += injection_y_error
    py      += injection_py_error
    zeta    += injection_zeta_error
    delta   += injection_delta_error

    ########################################
    # Build particles
    ########################################
    _context    = _context if _context is not None else line._context

    particles   = line.build_particles(
        _context   = _context,
        _capacity  = capacity,
        x          = x,
        px         = px,
        y          = y,
        py         = py,
        zeta       = zeta,
        delta      = delta)

    return particles
