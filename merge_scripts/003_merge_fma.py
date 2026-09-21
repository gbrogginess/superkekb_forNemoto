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
n_chunks_per_phase = n_particles_total = None

seen_jobs   = {}   # jobID -> (PHASE_IDX, CHUNK_IDX)
missing_jobs = []

for job_dir in job_dirs:
    fma_fpath    = os.path.join(job_dir, "Outputdata", "fma.npz")
    params_fpath = os.path.join(job_dir, "Outputdata", "params.json")

    job_id = int(job_dir.rsplit(".", 1)[-1])

    if not os.path.isfile(fma_fpath):
        print(f"Skipping {job_dir}: no fma.npz found")
        missing_jobs.append(job_id)
        continue

    data = np.load(fma_fpath)
    for field in FIELDS:
        records[field].append(data[field])

    if not os.path.isfile(params_fpath):
        print(f"WARNING: {job_dir} has no params.json - cannot verify its settings")
        continue

    with open(params_fpath, "r", encoding="utf-8") as f:
        job_params = json.load(f)

    seen_jobs[job_id] = (job_params["PHASE_IDX"], job_params["CHUNK_IDX"])

    if q0_x is None:
        q0_x = float(data["q0_x"])
        q0_y = float(data["q0_y"])

    if scan_mode is None:
        # First job defines the reference configuration.
        n_turns_ref        = job_params["N_TURNS"]
        n_chunks_per_phase = job_params["N_CHUNKS_PER_PHASE"]
        n_particles_total  = job_params["N_PARTICLES_TOTAL"]
        scan_mode          = job_params["SCAN_MODE"]
        beambeam_on        = job_params["BEAMBEAM_ON"]
        radiation_model    = job_params["RADIATION_MODEL"]
        resonance_order    = job_params["RESONANCE_ORDER"]
        norm_x_array       = np.array(job_params["NORM_X_ARRAY"])
        norm_y_array       = np.array(job_params["NORM_Y_ARRAY"])
        norm_z_array       = np.array(job_params["NORM_Z_ARRAY"])
        phases             = np.array(job_params["PHASES"])
    else:
        # Every later job must agree with it, otherwise the merge would be
        # silently stitching together outputs of two different run.py versions.
        mismatches = [
            key for key, ref in (
                ("N_TURNS",            n_turns_ref),
                ("N_CHUNKS_PER_PHASE", n_chunks_per_phase),
                ("N_PARTICLES_TOTAL",  n_particles_total),
                ("SCAN_MODE",          scan_mode),
                ("BEAMBEAM_ON",        beambeam_on),
                ("RADIATION_MODEL",    radiation_model))
            if job_params[key] != ref]
        for key, ref in (("NORM_X_ARRAY", norm_x_array),
                         ("NORM_Y_ARRAY", norm_y_array),
                         ("NORM_Z_ARRAY", norm_z_array),
                         ("PHASES",       phases)):
            if not np.array_equal(np.array(job_params[key]), ref):
                mismatches.append(key)
        if mismatches:
            sys.exit(f"ERROR: {job_dir} was run with different settings than the "
                     f"first job: {', '.join(mismatches)}. Stale files from an "
                     f"earlier run? Re-run all jobs with one version of run.py.")

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
# Sanity checks
################################################################################
# These are the checks that would have caught a bad merge (missing chunk,
# duplicated job, stale Output/ from an earlier run, wrong reassembly order).
errors = []

########################################
# Every job present exactly once
########################################
if n_chunks_per_phase is not None and phases is not None:
    n_jobs_expected = len(phases) * n_chunks_per_phase
    expected_ids    = set(range(n_jobs_expected))
    found_ids       = set(seen_jobs)

    if missing_jobs:
        errors.append(f"jobs with no fma.npz: {sorted(missing_jobs)}")
    if expected_ids - found_ids - set(missing_jobs):
        errors.append(f"job folders absent entirely: "
                      f"{sorted(expected_ids - found_ids - set(missing_jobs))}")
    if found_ids - expected_ids:
        errors.append(f"unexpected extra job folders (stale output?): "
                      f"{sorted(found_ids - expected_ids)}")

    # (phase_idx, chunk_idx) must be a bijection onto the expected jobs
    assignments = sorted(seen_jobs.values())
    expected_assignments = sorted(
        (p, c) for p in range(len(phases)) for c in range(n_chunks_per_phase))
    if assignments != expected_assignments:
        duplicates = sorted({a for a in assignments if assignments.count(a) > 1})
        absent     = sorted(set(expected_assignments) - set(assignments))
        detail = []
        if duplicates:
            detail.append(f"duplicated {duplicates}")
        if absent:
            detail.append(f"missing {absent}")
        errors.append("(phase_idx, chunk_idx) coverage is wrong: "
                      + "; ".join(detail))

########################################
# Every grid point present exactly once per phase
########################################
unique_phases = np.unique(records["phase"])

if phases is not None and len(unique_phases) != len(phases):
    errors.append(f"found {len(unique_phases)} phases in the data, "
                  f"expected {len(phases)}")

if n_particles_total is not None:
    for p in unique_phases:
        pid = records["particle_id"][records["phase"] == p]
        if pid.size != n_particles_total:
            errors.append(f"phase {p / np.pi:.2f} pi has {pid.size} particles, "
                          f"expected {n_particles_total}")
        if not np.array_equal(np.sort(pid), np.arange(n_particles_total)):
            n_dup = pid.size - len(np.unique(pid))
            errors.append(f"phase {p / np.pi:.2f} pi does not cover the grid "
                          f"0..{n_particles_total - 1} exactly once "
                          f"({n_dup} duplicated particle_id)")

########################################
# Merged amplitudes must equal the intended grid
########################################
GRID_AXES = {"xy": ("norm_x_array", "norm_y_array", "Ax", "Ay"),
             "zx": ("norm_z_array", "norm_x_array", "Az", "Ax"),
             "zy": ("norm_z_array", "norm_y_array", "Az", "Ay")}

if scan_mode in GRID_AXES:
    key_a, key_b, amp_a_key, amp_b_key = GRID_AXES[scan_mode]
    axis_a = {"norm_x_array": norm_x_array, "norm_y_array": norm_y_array,
              "norm_z_array": norm_z_array}[key_a]
    axis_b = {"norm_x_array": norm_x_array, "norm_y_array": norm_y_array,
              "norm_z_array": norm_z_array}[key_b]

    if axis_a is not None and axis_b is not None:
        grid_a, grid_b = np.meshgrid(axis_a, axis_b, indexing="ij")
        grid_a, grid_b = grid_a.ravel(), grid_b.ravel()

        for p in unique_phases:
            m   = records["phase"] == p
            pid = records["particle_id"][m]
            if pid.size != grid_a.size:
                continue                      # already reported above
            srt = np.argsort(pid)
            # Tolerance covers the `remove_zeros` nudge applied at the origin.
            tol_a = max(1e-3, 1e-4 * (axis_a.max() - axis_a.min()))
            tol_b = max(1e-3, 1e-4 * (axis_b.max() - axis_b.min()))
            err_a = np.abs(records[amp_a_key][m][srt] - grid_a).max()
            err_b = np.abs(records[amp_b_key][m][srt] - grid_b).max()
            if err_a > tol_a or err_b > tol_b:
                errors.append(
                    f"phase {p / np.pi:.2f} pi: merged amplitudes do not match "
                    f"the intended grid (max |{amp_a_key}-grid| = {err_a:.3g}, "
                    f"max |{amp_b_key}-grid| = {err_b:.3g}) -- wrong reassembly "
                    f"order or mismatched chunking")

########################################
# Report losses and NaNs loudly (never drop them silently)
########################################
print()
for p in unique_phases:
    m       = records["phase"] == p
    n_lost  = int(records["lost"][m].sum())
    n_nan   = int((~records["lost"][m] & ~np.isfinite(records["diff_coeff"][m])).sum())
    print(f"phase {p / np.pi:.2f} pi: {m.sum()} points, {n_lost} lost "
          f"({100.0 * n_lost / max(m.sum(), 1):.1f}%), "
          f"{n_nan} surviving with NaN diff_coeff")
    if n_nan:
        print(f"  WARNING: {n_nan} surviving particles have NaN diffusion "
              f"(NAFF failure) - they will be flagged, not dropped, in plot.py")

if errors:
    print("\nMERGE SANITY CHECKS FAILED:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
print("\nAll merge sanity checks passed.")

################################################################################
# Injection tune per phase (tune of the particle closest to zero amplitude)
################################################################################
q_inj_x       = np.full(unique_phases.shape, np.nan)
q_inj_y       = np.full(unique_phases.shape, np.nan)

for i, p in enumerate(unique_phases):
    mask = records["phase"] == p
    # Use the (phase-independent) amplitudes. x_norm_init/y_norm_init are
    # A*cos(phase), so at phase = pi/2 they are all ~0 and argmin picked an
    # essentially arbitrary particle.
    amp2 = records["Ax"][mask]**2 + records["Ay"][mask]**2
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