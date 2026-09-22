""""
================================================================================
# Frequency mapping study plotting script (CLI)
================================================================================
Belle II Simulation & Modelling Group

Authors:    J.P.T. Salvesen, T. Nemoto, G. Broggi
Adapted for command-line / alias use (analogous to plot_lossmap.py) by request.

Usage:
    python 003_plot_fma.py <fma_records_file.npz> [options]

Examples:
    plot_fma fma_records_xy_no_bb_no_rad.npz
    plot_fma fma_records_bb.npz --qplane --diff
    plot_fma fma_records_bb.npz --order 6 --outdir plots/
================================================================================
"""

################################################################################
# Required Packages
################################################################################
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# Set Serif font for all plots (for better LaTeX rendering)
################################################################################
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
})

################################################################################
# Scan configuration
################################################################################
SCAN_CONFIG = {
    "xy": {
        "tune_keys": ("tunes_x", "tunes_y"),
        "tune_labels": (r"$Q_x$", r"$Q_y$"),
        "diff_keys": ("diff_x", "diff_y"),
        "diff_labels": (r"$\log(\Delta Q_x^2)$", r"$\log(\Delta Q_y^2)$"),
        "amp_keys": ("Ax", "Ay"),
        "action_keys": ("Jx", "Jy"),
        "action_labels": (r"$A_x\ [\sigma_x]$", r"$A_y\ [\sigma_y]$"),
        "range_keys": ("norm_x_array", "norm_y_array"),
        "q0_keys": ("q0_x", "q0_y"),
        "qinj_keys": ("q_inj_x", "q_inj_y"),
    },
    "zx": {
        "tune_keys": ("tunes_z", "tunes_x"),
        "tune_labels": (r"$Q_z$", r"$Q_x$"),
        "diff_keys": ("diff_z", "diff_x"),
        "diff_labels": (r"$\log(\Delta Q_z^2)$", r"$\log(\Delta Q_x^2)$"),
        "amp_keys": ("Az", "Ax"),
        "action_keys": ("Jz", "Jx"),
        "action_labels": (r"$A_z\ [\sigma_z]$", r"$A_x\ [\sigma_x]$"),
        "range_keys": ("norm_z_array", "norm_x_array"),
        "q0_keys": (None, "q0_x"),
        "qinj_keys": (None, "q_inj_x"),
    },
    "zy": {
        "tune_keys": ("tunes_z", "tunes_y"),
        "tune_labels": (r"$Q_z$", r"$Q_y$"),
        "diff_keys": ("diff_z", "diff_y"),
        "diff_labels": (r"$\log(\Delta Q_z^2)$", r"$\log(\Delta Q_y^2)$"),
        "amp_keys": ("Az", "Ay"),
        "action_keys": ("Jz", "Jy"),
        "action_labels": (r"$A_z\ [\sigma_z]$", r"$A_y\ [\sigma_y]$"),
        "range_keys": ("norm_z_array", "norm_y_array"),
        "q0_keys": (None, "q0_y"),
        "qinj_keys": (None, "q_inj_y"),
    },
}

DIFF_VMIN = -7
DIFF_VMAX = -3

# Colours for particles that carry no diffusion value
LOST_COLOR = "0.72"
NAN_COLOR  = "tab:red"

################################################################################
# Helper Functions
################################################################################

def get_scalar_string(npz_data, key, default=None):
    if key not in npz_data.files:
        return default
    value = npz_data[key]
    if np.ndim(value) == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return str(value)

def get_max_order(npz_data, override):
    if override is not None:
        return int(override)
    if "max_order" in npz_data.files:
        return int(np.array(npz_data["max_order"]).item())
    return 10

def has_key(npz_data, key):
    return key is not None and key in npz_data.files

def get_reference_value(npz_data, key, phase_index=None):
    if key is None or key not in npz_data.files:
        return None
    arr = np.asarray(npz_data[key])
    if arr.size == 0:
        return None
    if phase_index is not None and phase_index < arr.size:
        return float(arr[phase_index])
    return float(arr.flat[0])

def get_action_plane_data(npz_data, scan_mode, mask):
    """
    Return the two *amplitudes* (in sigma) for the action-plane axes.

    IMPORTANT: this must NOT use x_norm_init / y_norm_init / z_norm_init.
    Those are the signed normalised coordinates at turn 0, which by
    construction are A * cos(phase) -- so they shrink by cos(phase), collapse
    to ~0 at phase = pi/2 and go negative at phase = 3*pi/4. The amplitude
    A = sqrt(a_norm^2 + pa_norm^2) is the phase-independent quantity the
    grid was built from, and is what the axis labels claim to show.
    """
    cfg = SCAN_CONFIG[scan_mode]

    amp_key_a, amp_key_b = cfg["amp_keys"]
    if has_key(npz_data, amp_key_a) and has_key(npz_data, amp_key_b):
        axis_a = np.asarray(npz_data[amp_key_a][mask], dtype=float)
        axis_b = np.asarray(npz_data[amp_key_b][mask], dtype=float)
        return axis_a, axis_b, cfg["action_labels"]

    # Fallback for older files that only stored the actions J = A^2 / 2.
    action_key_a, action_key_b = cfg["action_keys"]
    if not has_key(npz_data, action_key_a) or not has_key(npz_data, action_key_b):
        raise KeyError(f"Missing amplitude keys {cfg['amp_keys']} and "
                       f"action keys {cfg['action_keys']}")

    j_a = np.clip(np.asarray(npz_data[action_key_a][mask], dtype=float), 0.0, None)
    j_b = np.clip(np.asarray(npz_data[action_key_b][mask], dtype=float), 0.0, None)
    return np.sqrt(2.0 * j_a), np.sqrt(2.0 * j_b), cfg["action_labels"]

def get_axis_ranges(npz_data, scan_mode):
    """Intended grid extent per axis, taken from the arrays merge.py stored."""
    cfg = SCAN_CONFIG[scan_mode]
    ranges = []
    for key in cfg["range_keys"]:
        if not has_key(npz_data, key):
            ranges.append(None)
            continue
        arr = np.asarray(npz_data[key], dtype=float)
        ranges.append(None if arr.size == 0 else (float(arr.min()), float(arr.max())))
    return tuple(ranges)

def apply_axis_ranges(ax, ranges, amp_a, amp_b, labels, context):
    """
    Pin the axes to the intended grid and assert the data really covers it.

    This is the check that would have caught the original bug immediately:
    the plotted quantity must span the grid it was generated from.
    """
    for i, (rng, amp, label) in enumerate(zip(ranges, (amp_a, amp_b), labels)):
        if rng is None or amp.size == 0:
            continue
        lo, hi = rng
        span = hi - lo
        pad = 0.02 * span if span > 0 else 1.0
        (ax.set_xlim if i == 0 else ax.set_ylim)(lo - pad, hi + pad)

        data_lo, data_hi = float(np.nanmin(amp)), float(np.nanmax(amp))
        tol = max(1e-6, 1e-3 * span)
        if data_lo < lo - tol or data_hi > hi + tol:
            print(f"WARNING [{context}]: {label} data range "
                  f"[{data_lo:.4g}, {data_hi:.4g}] falls outside the intended "
                  f"grid range [{lo:.4g}, {hi:.4g}]")
        elif span > 0 and (data_hi - data_lo) < 0.5 * span:
            print(f"WARNING [{context}]: {label} only covers "
                  f"[{data_lo:.4g}, {data_hi:.4g}] of the intended "
                  f"[{lo:.4g}, {hi:.4g}] -- less than half the grid")

def add_resonance_lines(ax, max_order):
    res_map = plt.colormaps["viridis"].resampled(max_order + 1)
    order_color = {order: res_map(order / (max_order + 1)) for order in range(max_order + 1)}

    for n_coef in range(-max_order, max_order + 1):
        for m_coef in range(-max_order, max_order + 1):
            if n_coef == 0 and m_coef == 0:
                continue
            if abs(n_coef) + abs(m_coef) > max_order:
                continue

            order = abs(n_coef) + abs(m_coef)
            lw = max(0.5, 1.2 - 0.12 * order)
            alpha = max(0.5, 1.0 - 0.10 * order)
            color = order_color[order]
            corners = [n_coef * q + m_coef * r for q in [0.0, 1.0] for r in [0.0, 1.0]]

            p_min = int(np.floor(min(corners)))
            p_max = int(np.ceil(max(corners)))
            for p_val in range(p_min, p_max + 1):
                points = []
                if m_coef != 0:
                    for qx in [0.0, 1.0]:
                        qy = (p_val - n_coef * qx) / m_coef
                        if 0.0 <= qy <= 1.0:
                            points.append((qx, qy))
                if n_coef != 0:
                    for qy in [0.0, 1.0]:
                        qx = (p_val - m_coef * qy) / n_coef
                        if 0.0 <= qx <= 1.0:
                            points.append((qx, qy))

                points = list({(round(x_val, 10), round(y_val, 10)) for x_val, y_val in points})
                if len(points) >= 2:
                    points.sort()
                    xs, ys = zip(*points)
                    ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, zorder=0)

################################################################################
# Plot Functions
################################################################################

def plot_q_plane(npz_data, scan_mode, max_order, outdir):
    cfg = SCAN_CONFIG[scan_mode]
    phase_all = npz_data["phase"]
    diff_all = npz_data["diff_coeff"]
    tune_a_all = npz_data[cfg["tune_keys"][0]]
    tune_b_all = npz_data[cfg["tune_keys"][1]]
    lost_all = npz_data["lost"] if "lost" in npz_data.files else np.zeros_like(phase_all, dtype=bool)

    unique_phases = np.sort(np.unique(phase_all))
    for phase_index, ph in enumerate(unique_phases):
        mask = phase_all == ph
        tune_a = tune_a_all[mask]
        tune_b = tune_b_all[mask]
        diff_phase = diff_all[mask]
        lost = lost_all[mask].astype(bool)
        survived = ~lost
        idx = np.argsort(diff_phase[survived])

        fig, ax = plt.subplots(figsize=(8, 6))
        sc = ax.scatter(
            tune_a[survived][idx], tune_b[survived][idx],
            c=diff_phase[survived][idx], cmap="plasma", s=5, alpha=0.6,
            vmin=DIFF_VMIN, vmax=DIFF_VMAX,
        )

        q0_a = get_reference_value(npz_data, cfg["q0_keys"][0], phase_index=phase_index)
        q0_b = get_reference_value(npz_data, cfg["q0_keys"][1], phase_index=phase_index)
        if q0_a is not None and q0_b is not None:
            ax.scatter(q0_a, q0_b, color="red", marker="*", s=50, alpha=0.6,
                       label=f"nominal tune = ({q0_a:.4f}, {q0_b:.4f})")

        qinj_a = get_reference_value(npz_data, cfg["qinj_keys"][0], phase_index=phase_index)
        qinj_b = get_reference_value(npz_data, cfg["qinj_keys"][1], phase_index=phase_index)
        if qinj_a is not None and qinj_b is not None:
            ax.scatter(qinj_a, qinj_b, color="orange", marker="*", s=50, alpha=0.6,
                       label=f"initial tune = ({qinj_a:.4f}, {qinj_b:.4f})")

        add_resonance_lines(ax, max_order)
        ax.set_xlabel(cfg["tune_labels"][0])
        ax.set_ylabel(cfg["tune_labels"][1])
        ax.set_title(f"Q-plane, phase = {ph / np.pi:.2f}")
        ax.grid(True, alpha=0.3)
        if (q0_a is not None and q0_b is not None) or (qinj_a is not None and qinj_b is not None):
            ax.legend(loc="best")

        fig.colorbar(sc, ax=ax, label=r"$D=\log_{10}(\sqrt{\Delta Q_x^2+\Delta Q_y^2})$")
        fig.tight_layout()
        out_path = outdir / f"qplane_phase_{ph / np.pi:.2f}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

def plot_diff_components(npz_data, scan_mode, outdir):
    cfg = SCAN_CONFIG[scan_mode]
    phase_all = npz_data["phase"]
    tune_a_all = npz_data[cfg["tune_keys"][0]]
    tune_b_all = npz_data[cfg["tune_keys"][1]]
    unique_phases = np.sort(np.unique(phase_all))

    for phase_index, ph in enumerate(unique_phases):
        mask = phase_all == ph
        tune_a = tune_a_all[mask]
        tune_b = tune_b_all[mask]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
        for ax, diff_key, diff_label in zip(axes, cfg["diff_keys"], cfg["diff_labels"]):
            if diff_key not in npz_data.files:
                ax.set_title(f"Missing key: {diff_key}")
                ax.axis("off")
                continue

            diff_value = npz_data[diff_key][mask]
            idx = np.argsort(diff_value)
            sc = ax.scatter(
                tune_a[idx], tune_b[idx], c=diff_value[idx], cmap="plasma",
                s=7, alpha=0.7, vmin=DIFF_VMIN, vmax=DIFF_VMAX,
            )

            q0_a = get_reference_value(npz_data, cfg["q0_keys"][0], phase_index=phase_index)
            q0_b = get_reference_value(npz_data, cfg["q0_keys"][1], phase_index=phase_index)
            if q0_a is not None and q0_b is not None:
                ax.scatter(q0_a, q0_b, color="red", marker="*", s=70, alpha=0.8, label="nominal tune")

            qinj_a = get_reference_value(npz_data, cfg["qinj_keys"][0], phase_index=phase_index)
            qinj_b = get_reference_value(npz_data, cfg["qinj_keys"][1], phase_index=phase_index)
            if qinj_a is not None and qinj_b is not None:
                ax.scatter(qinj_a, qinj_b, color="orange", marker="*", s=70, alpha=0.8, label="initial tune")

            ax.set_xlabel(cfg["tune_labels"][0])
            ax.set_ylabel(cfg["tune_labels"][1])
            ax.set_title(f"{diff_label}, phase = {ph / np.pi:.2f}")
            ax.grid(True, alpha=0.3)
            fig.colorbar(sc, ax=ax, label=diff_label)

        axes[0].legend(loc="best")
        fig.tight_layout()
        out_path = outdir / f"diff_components_phase_{ph / np.pi:.2f}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

def plot_action_plane(npz_data, scan_mode, outdir):
    phase_all = npz_data["phase"]
    diff_all = np.asarray(npz_data["diff_coeff"], dtype=float)
    lost_all = npz_data["lost"] if "lost" in npz_data.files else np.zeros_like(phase_all, dtype=bool)
    ranges = get_axis_ranges(npz_data, scan_mode)

    unique_phases = np.sort(np.unique(phase_all))
    for ph in unique_phases:
        mask = phase_all == ph
        try:
            amp_a, amp_b, action_labels = get_action_plane_data(npz_data, scan_mode, mask)
        except KeyError as exc:
            print(f"Skip action plane: {exc}")
            return

        diff_phase = diff_all[mask]
        lost = lost_all[mask].astype(bool)
        # A surviving particle with NaN diffusion is a NAFF failure, not a
        # loss: keep the two apart instead of lumping them together.
        nan_diff = ~lost & ~np.isfinite(diff_phase)
        survived = ~lost & np.isfinite(diff_phase)

        print(f"phase {ph / np.pi:.2f} pi: {mask.sum()} grid points, "
              f"{survived.sum()} plotted, {lost.sum()} lost "
              f"({100.0 * lost.mean():.1f}%), {nan_diff.sum()} NaN diffusion")

        idx = np.argsort(diff_phase[survived])

        fig, ax = plt.subplots(figsize=(5, 4))
    
        # Lost particles are drawn, not dropped. Dropping them is what made
        # the axes stop at the dynamic aperture instead of at the grid edge.
        if np.any(lost):
            ax.scatter(amp_a[lost], amp_b[lost], c=LOST_COLOR, s=6,
                       alpha=0.35, linewidths=0, zorder=0,
                       label="Lost")
        if np.any(nan_diff):
            ax.scatter(amp_a[nan_diff], amp_b[nan_diff], facecolors="none",
                       edgecolors=NAN_COLOR, s=10, linewidths=0.4, zorder=1,
                       label=f"NaN diffusion ({nan_diff.sum()})")

        sc = ax.scatter(
            amp_a[survived][idx], amp_b[survived][idx],
            c=diff_phase[survived][idx], cmap="plasma", s=8, alpha=0.7,
            vmin=DIFF_VMIN, vmax=DIFF_VMAX, zorder=2,
        )

        ax.set_xlabel(action_labels[0])
        ax.set_ylabel(action_labels[1])
        ax.set_title(f"phase = {ph / np.pi:.2f} pi")
        apply_axis_ranges(ax, ranges, amp_a, amp_b, action_labels,
                          f"action plane, phase {ph / np.pi:.2f} pi")
        ax.grid(True, alpha=0.3)
        if np.any(lost) or np.any(nan_diff):
            ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
        fig.colorbar(sc, ax=ax, label=r"$D=\log_{10}(\sqrt{\Delta Q_x^2+\Delta Q_y^2})$")
        fig.tight_layout()
        out_path = outdir / f"action_plane_phase_{ph / np.pi:.2f}.png"
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

def plot_all_phases(npz_data, scan_mode, outdir):
    """
    Combine the phases onto one grid: each grid point keeps the WORST (largest)
    diffusion over the phases that survived there. A grid point is treated as
    "lost" for this plot if it was lost at at least one phase -- it is then
    drawn grey and excluded from the worst-case colour scatter even where it
    did survive.

    Grid points are identified by the integer particle_id, i.e. the index into
    the flattened amplitude grid. The previous np.unique(..., axis=0) on the
    float coordinate pairs did not group anything: the amplitudes differ in
    the last bits between phases, so 12168 survivors produced 12157 "unique"
    pairs instead of the ~3900 real grid points.
    """
    phase_all = npz_data["phase"]
    all_mask = np.ones_like(phase_all, dtype=bool)

    try:
        amp_a_all, amp_b_all, action_labels = get_action_plane_data(npz_data, scan_mode, all_mask)
    except KeyError as exc:
        print(f"Skip all-phases plot: {exc}")
        return

    if "particle_id" not in npz_data.files:
        print("Skip all-phases plot: missing particle_id (needed to group grid points)")
        return
    pid_all = np.asarray(npz_data["particle_id"], dtype=np.int64)

    diff_all = np.asarray(npz_data["diff_coeff"], dtype=float)
    lost_all = npz_data["lost"] if "lost" in npz_data.files else np.zeros_like(phase_all, dtype=bool)
    lost_all = lost_all.astype(bool)
    usable = ~lost_all & np.isfinite(diff_all)

    n_points = len(np.unique(pid_all))
    print(f"all phases: {n_points} grid points, {usable.sum()}/{usable.size} "
          f"(phase, point) samples usable, {lost_all.sum()} lost, "
          f"{(~lost_all & ~np.isfinite(diff_all)).sum()} NaN diffusion")

    if not np.any(usable):
        print("Skip all-phases plot: no survived particles.")
        return

    # Worst-case D per grid point, via an exact integer groupby.
    unique_pid, inverse = np.unique(pid_all, return_inverse=True)
    worst = np.full(unique_pid.size, -np.inf)
    np.maximum.at(worst, inverse[usable], diff_all[usable])

    # A grid point counts as "lost" if it was lost at at least one phase, not
    # only if it was lost at every phase. Such points are drawn grey and
    # excluded from the worst-case colour scatter, even if they survived
    # (and have a finite diffusion value) at other phases.
    lost_any = np.zeros(unique_pid.size, dtype=bool)
    np.logical_or.at(lost_any, inverse, lost_all)

    # Representative amplitude per grid point (identical across phases up to
    # float noise, so any sample will do).
    amp_a = np.zeros(unique_pid.size)
    amp_b = np.zeros(unique_pid.size)
    amp_a[inverse] = amp_a_all
    amp_b[inverse] = amp_b_all

    has_data = ~lost_any & np.isfinite(worst)
    n_dead = int(lost_any.sum())
    if n_dead:
        print(f"all phases: {n_dead} grid points lost at at least one phase "
              f"(drawn in grey, not dropped)")

    fig, ax = plt.subplots(figsize=(5, 4))
    if n_dead:
        ax.scatter(amp_a[lost_any], amp_b[lost_any], c=LOST_COLOR, s=6,
                   alpha=0.35, linewidths=0, zorder=0,
                   label="Lost at least once")
    order = np.argsort(worst[has_data])
    sc = ax.scatter(
        amp_a[has_data][order], amp_b[has_data][order], c=worst[has_data][order],
        cmap="plasma", s=7, vmin=DIFF_VMIN, vmax=DIFF_VMAX, zorder=2,
    )
    ax.set_xlabel(action_labels[0])
    ax.set_ylabel(action_labels[1])
    ax.set_title("All phases (worst-case $D$)")
    apply_axis_ranges(ax, get_axis_ranges(npz_data, scan_mode), amp_a, amp_b,
                      action_labels, "all phases")
    ax.grid(True, alpha=0.3)
    if n_dead:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.8)
    fig.colorbar(sc, ax=ax, label=r"$D=\log_{10}(\sqrt{\Delta Q_x^2+\Delta Q_y^2})$")

    fig.tight_layout()
    out_path = outdir / "all_phases.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")

################################################################################
# CLI
################################################################################

def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot FMA .npz records (fma_records_*.npz)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("npz", help="Path to fma_records_*.npz")
    parser.add_argument("--order", type=int, default=None, help="Resonance order override")
    parser.add_argument("--qplane", action="store_true", help="Also plot per-phase Q-plane diagrams")
    parser.add_argument("--diff", action="store_true", help="Also plot diff_x/diff_y components side by side")
    parser.add_argument("--no-action", action="store_true", help="Skip per-phase action-plane plots")
    parser.add_argument("--no-all-phases", action="store_true", help="Skip the combined all-phases plot")
    parser.add_argument("--outdir", default=".", help="Directory to save PNGs into")
    return parser.parse_args()

def main():
    if len(sys.argv) < 2:
        print("Usage: python 003_plot_fma.py <fma_records_file.npz> [options]")
        print("       (run with -h for all options)")
        sys.exit(1)

    args = parse_args()
    npz_path = Path(args.npz).expanduser()
    if not npz_path.exists():
        print(f"Error: NPZ file not found: {npz_path}")
        sys.exit(1)

    outdir = Path(args.outdir).expanduser()
    outdir.mkdir(parents=True, exist_ok=True)

    with np.load(npz_path) as data:
        scan_mode = get_scalar_string(data, "scan_mode", default="xy")
        if scan_mode not in SCAN_CONFIG:
            print(f"Unknown scan_mode '{scan_mode}', fallback to 'xy'")
            scan_mode = "xy"

        max_order = get_max_order(data, args.order)
        print(f"Loaded: {npz_path}")
        print(f"scan_mode={scan_mode}, max_order={max_order}")

        if args.qplane:
            plot_q_plane(data, scan_mode, max_order, outdir)
        if args.diff:
            plot_diff_components(data, scan_mode, outdir)
        if not args.no_action:
            plot_action_plane(data, scan_mode, outdir)
        if not args.no_all_phases:
            plot_all_phases(data, scan_mode, outdir)


if __name__ == "__main__":
    main()
