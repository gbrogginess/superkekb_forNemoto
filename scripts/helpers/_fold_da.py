"""
Plot Dynamic Aperture [Oide Method]
=============================================
Author(s):  John P T Salvesen
Email:      john.salvesen@cern.ch
Date:       18-11-2025
"""

################################################################################
# Required Modules
################################################################################
import numpy as np
from collections import deque

################################################################################
# Survival Boundary Extraction
################################################################################
def compute_surviving_region(
        a_grid,
        b_grid, 
        values_grid,
        center_idxs):
    """
    Flood-fill the "good" region connected to the given center on the same grid.

    Parameters
    ----------
    a_grid, b_grid : 2D arrays (meshgrid), shape (H, W)
        Physical coordinate grids.
    values_grid : 2D array, shape (H, W)
        Boolean or numeric values.
    center :
        - (iy, ix) if center_is_index=True
        - (xc, yc) in physical coords if center_is_index=False
        - "zero" to pick nearest-to-zero on each axis
        - None -> use midpoint in physical coords
    center_is_index : bool or None
        If None, auto-detect: treat as indices if both ints and in-bounds.
    threshold : float or None
        If values_grid is numeric, cells >= threshold are "good".
        Ignored if values_grid is boolean.
    connectivity : {4, 8}
        Neighbor definition for connectivity.
    return_boundary_coords : bool
        If True, also return (x_poly, y_poly) boundary coordinates using
        matplotlib.contour if available.

    Returns
    -------
    component_mask : (H, W) bool
        Cells that are good and connected to the center.
    boundary_mask : (H, W) bool
        Cells in component that touch at least one non-component neighbor or edge.
    (optional) x_poly, y_poly : 1D arrays
        A boundary polyline in physical coordinates. Only if return_boundary_coords=True.
    """
    H, W    = values_grid.shape

    # Boolean "good" mask that all survived
    good    = values_grid == np.nanmax(values_grid)

    # 1D axes from meshgrid (assumed monotonic along each axis)
    x1d = a_grid[0, :]
    y1d = b_grid[:, 0]
    if not (np.all(np.diff(x1d) > 0) or np.all(np.diff(x1d) < 0)):
        raise ValueError("a_grid must be strictly monotonic along axis=1.")
    if not (np.all(np.diff(y1d) > 0) or np.all(np.diff(y1d) < 0)):
        raise ValueError("b_grid must be strictly monotonic along axis=0.")

    # Indices of the center
    ix0, iy0 = int(center_idxs[0]), int(center_idxs[1])

    if not (0 <= iy0 < H and 0 <= ix0 < W):
        raise ValueError("""Center indices out of bounds.""")
    if not good[iy0, ix0]:
        raise ValueError("""Center is not inside a "good" cell.""")

    # Flood fill (BFS) over good cells to get the connected component
    component   = np.zeros_like(good, dtype = bool)
    q           = deque()
    q.append((iy0, ix0))
    component[iy0, ix0] = True

    # Assume a connectivity of 4
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while q:
        iy, ix = q.popleft()
        for dy, dx in neighbors:
            jy, jx = iy + dy, ix + dx
            if 0 <= jy < H and 0 <= jx < W and (not component[jy, jx]) and good[jy, jx]:
                component[jy, jx] = True
                q.append((jy, jx))

    return component

################################################################################
# Convert to positive quadrant via folding
################################################################################
def fold_bool_along_axis(A: np.ndarray, axis: int, keep="positive") -> np.ndarray:
    """
    Fold a 2D boolean array along axis (0=y, 1=x) with AND across mirrors.
    Keeps the "positive" (>=0) or "negative" (<=0) half, including the center if n is odd.
    """
    A = np.asarray(A, dtype=bool)
    n = A.shape[axis]
    h = n // 2  # floor(n/2)

    # Indices to KEEP (include center when n is odd)
    if keep == "positive":
        idx_keep = np.arange(h, n)        # length = n - h
    elif keep == "negative":
        idx_keep = np.arange(0, n - h)    # length = n - h
    else:
        raise ValueError("""keep must be "positive" or "negative" """)

    # Mirror each kept index to its partner across the center
    idx_mirr = (n - 1) - idx_keep         # same length as idx_keep

    A_keep = np.take(A, idx_keep, axis = axis)
    A_mirr = np.take(A, idx_mirr, axis = axis)

    return A_keep & A_mirr

def fold_bool_quadrant(A: np.ndarray) -> np.ndarray:
    """
    Fold a 2D boolean array about both axes to the x>=0, y>=0 quadrant,
    using AND across all mirrors.
    """
    return fold_bool_along_axis(fold_bool_along_axis(A, axis = 0), axis = 1)

################################################################################
# Fold DA to positive quadrant
################################################################################
def fold_and_phase_average_da(
        da_dict:    dict,
        mode:       str):

    assert mode in ("xy", "zx", "zy"), """mode must be "xy", "zx", or "zy" """

    ########################################
    # Get key data
    ########################################
    if mode == "xy":
        NORM_A_GRID = da_dict["x_norm_init"]
        NORM_B_GRID = da_dict["y_norm_init"]
    elif mode == "zx":
        NORM_A_GRID = da_dict["z_norm_init"]
        NORM_B_GRID = da_dict["x_norm_init"]
    elif mode == "zy":
        NORM_A_GRID = da_dict["z_norm_init"]
        NORM_B_GRID = da_dict["y_norm_init"]

    PHASES          = da_dict["phase"]
    AT_TURN         = da_dict["at_turn"]

    assert AT_TURN.shape[0] == len(PHASES)
    assert NORM_A_GRID.shape == NORM_B_GRID.shape == AT_TURN.shape

    if len(NORM_A_GRID.shape) == 2:
        NORM_A_GRID = np.array(NORM_A_GRID)
        NORM_B_GRID = np.array(NORM_B_GRID)
    elif len(NORM_A_GRID.shape) == 3:
        NORM_A_GRID = np.array(NORM_A_GRID[0, :, :])
        NORM_B_GRID = np.array(NORM_B_GRID[0, :, :])
    else:
        raise ValueError("NORM_A_GRID and NORM_B_GRID must be 2D or 3D arrays.")

    ########################################
    # Fill the NaNs with the max turn number
    ########################################
    max_turns   = np.nanmax(AT_TURN)
    AT_TURN     = np.nan_to_num(AT_TURN, nan = max_turns)

    ########################################
    # Get mean survival across phases
    ########################################
    at_turn_phase_mean  = np.mean(AT_TURN, axis = 0)

    ########################################
    # Get surviving regions
    ########################################
    surviving_region    = compute_surviving_region(
        a_grid      = NORM_A_GRID,
        b_grid      = NORM_B_GRID,
        values_grid = at_turn_phase_mean,
        center_idxs = (
            np.argmin(np.abs(NORM_A_GRID[0,:])),
            np.argmin(np.abs(NORM_B_GRID[:,0]))))

    ########################################
    # Fold to positive quadrant
    ########################################
    mask_x              = NORM_A_GRID[0, :] >= 0
    mask_y              = NORM_B_GRID[:, 0] >= 0
    FOLDED_NORM_A_GRID  = NORM_A_GRID[np.ix_(mask_y, mask_x)]
    FOLDED_NORM_B_GRID  = NORM_B_GRID[np.ix_(mask_y, mask_x)]
    FOLDED_SURVIVING    = fold_bool_quadrant(surviving_region)

    ########################################
    # Make a dictionary to return
    ########################################
    if mode == "xy":
        folded_averaged_da  = {
            "surviving":        FOLDED_SURVIVING,
            "x_norm_init":      FOLDED_NORM_A_GRID,
            "y_norm_init":      FOLDED_NORM_B_GRID,
            "z_norm_init":      np.zeros_like(FOLDED_NORM_A_GRID),
            "gemitt_x":         da_dict["gemitt_x"],
            "gemitt_y":         da_dict["gemitt_y"],
            "gemitt_z":         da_dict["gemitt_z"]}
    elif mode == "zx":
        folded_averaged_da  = {
            "surviving":        FOLDED_SURVIVING,
            "x_norm_init":      FOLDED_NORM_B_GRID,
            "y_norm_init":      np.zeros_like(FOLDED_NORM_A_GRID),
            "z_norm_init":      FOLDED_NORM_A_GRID,
            "gemitt_x":         da_dict["gemitt_x"],
            "gemitt_y":         da_dict["gemitt_y"],
            "gemitt_z":         da_dict["gemitt_z"]}
    elif mode == "zy":
        folded_averaged_da  = {
            "surviving":        FOLDED_SURVIVING,
            "x_norm_init":      np.zeros_like(FOLDED_NORM_A_GRID),
            "y_norm_init":      FOLDED_NORM_B_GRID,
            "z_norm_init":      FOLDED_NORM_A_GRID,
            "gemitt_x":         da_dict["gemitt_x"],
            "gemitt_y":         da_dict["gemitt_y"],
            "gemitt_z":         da_dict["gemitt_z"]}
    
    return folded_averaged_da
