#!/usr/bin/env python3
"""Dashboard creation and display helpers."""

from common import *
import math
import os
import subprocess


def _make_output_paths(results_dir, stem):
    """Return the single dashboard output path used by the minimal application."""
    results_dir.mkdir(parents=True, exist_ok=True)
    return {"dashboard_plot": results_dir / "{}_validation_dashboard.png".format(stem)}


def _subject_color_cycle(count):
    """Return stable Matplotlib subject colors."""
    base = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    return [base[i % len(base)] for i in range(max(0, int(count)))]


def _configure_matplotlib_gui_backend(requested_backend=None):
    """Select a GUI backend before pyplot is imported.

    The script still works without a GUI backend because the dashboard is always
    saved as a PNG. This function only improves the chance that Matplotlib can
    open a live window when display mode is set to ``auto`` or ``matplotlib``.
    """
    try:
        import matplotlib
    except Exception as exc:
        return False, None, str(exc)

    current = str(matplotlib.get_backend())
    candidates = [requested_backend] if requested_backend else ["TkAgg", "QtAgg", "Qt5Agg", "WXAgg"]
    last_error = None
    for backend in candidates:
        if not backend:
            continue
        try:
            matplotlib.use(backend, force=True)
            return True, str(matplotlib.get_backend()), None
        except Exception as exc:
            last_error = exc

    if "agg" not in current.lower():
        return True, current, None
    return False, current, str(last_error) if last_error else "no GUI backend available"


def _open_file_with_system_viewer(path):
    """Open a saved dashboard PNG with the operating-system image viewer."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
        return True, None
    except Exception as exc:
        return False, str(exc)


def _display_dashboard_window(path, gui_backend=None, display_mode="file"):
    """Display the dashboard using the requested mode.

    ``file`` is the most reliable mode on Windows because it uses the normal
    image viewer. ``matplotlib`` opens a Matplotlib window when a GUI backend is
    available. ``auto`` tries Matplotlib first and then falls back to the file
    viewer.
    """
    mode = str(display_mode or "file").lower()
    if mode == "none":
        return False

    if mode in {"auto", "matplotlib"}:
        ok, backend, err = _configure_matplotlib_gui_backend(gui_backend)
        if ok:
            try:
                import matplotlib.pyplot as plt
                image = plt.imread(str(path))
                fig, ax = plt.subplots(figsize=(14, 8))
                ax.imshow(image)
                ax.axis("off")
                fig.canvas.manager.set_window_title("mmVitals dashboard")
                plt.show(block=True)
                return True
            except Exception as exc:
                err = str(exc)
        if mode == "matplotlib":
            print("Dashboard saved, but Matplotlib display failed: {}".format(err))
            return False

    ok, err = _open_file_with_system_viewer(path)
    if not ok:
        print("Dashboard saved: {}".format(path))
        if err:
            print("Display failed: {}".format(err))
    return ok


def _cell_position(cell, subject, range_axis, has_metric_axis):
    """Convert one ROI cell from range/angle into top-down x/y coordinates."""
    angle_deg = float(cell.get("angle_reported_deg", cell.get("selected_angle_deg", 0.0)))
    range_bin = int(cell.get("range_bin", subject.get("selected_range_bin", 0)))
    distance = float(range_axis[range_bin]) if has_metric_axis and 0 <= range_bin < len(range_axis) else float(range_bin)
    angle_rad = math.radians(angle_deg)
    return distance * math.cos(angle_rad), distance * math.sin(angle_rad)


def _save_validation_dashboard_plot(path, subjects, ra, range_axis, has_metric_axis, show_window=False, gui_backend=None):
    """Create the only output artifact: the combined vital-sign dashboard."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print("Matplotlib not available; cannot create dashboard: {}".format(exc))
        return

    power_map = np.asarray(ra.get("power_map", []), dtype=np.float64)
    candidate_bins = np.asarray(ra.get("candidate_bins", []), dtype=int)
    angles_deg = np.asarray(ra.get("angle_grid_reported", []), dtype=np.float64)
    if power_map.size == 0 or candidate_bins.size == 0 or angles_deg.size == 0:
        return

    ranges = np.asarray(range_axis)[candidate_bins] if has_metric_axis else candidate_bins.astype(np.float64)
    angle_rad = np.deg2rad(angles_deg)
    rr, aa = np.meshgrid(ranges, angle_rad, indexing="ij")
    xx = rr * np.cos(aa)
    yy = rr * np.sin(aa)
    power_db = 10.0 * np.log10(power_map + 1e-30)
    power_db = power_db - np.nanmax(power_db)

    fig = plt.figure(figsize=(16, 9))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.25], height_ratios=[0.72, 0.28])
    ax_room = fig.add_subplot(grid[:, 0])
    ax_trend = fig.add_subplot(grid[0, 1])
    ax_boxes = fig.add_subplot(grid[1, 1])
    ax_boxes.axis("off")

    # Top-down room view.
    point_size = max(45.0, 4800.0 / max(1, power_map.size))
    scatter = ax_room.scatter(xx.ravel(), yy.ravel(), c=power_db.ravel(), cmap="viridis", s=point_size, marker="s", alpha=0.85, linewidths=0)
    cbar = fig.colorbar(scatter, ax=ax_room, pad=0.02, shrink=0.86)
    cbar.set_label("Relative motion power [dB]")

    max_range = float(np.max(ranges)) if ranges.size else 1.0
    for angle in [float(np.min(angles_deg)), 0.0, float(np.max(angles_deg))]:
        a = math.radians(angle)
        ax_room.plot([0.0, max_range * math.cos(a)], [0.0, max_range * math.sin(a)], linestyle=":" if abs(angle) > 1e-9 else "--", color="0.55", linewidth=1.0)
    ax_room.scatter([0.0], [0.0], s=110, marker="^", color="black", label="Radar")

    colors = _subject_color_cycle(len(subjects))
    handles = [Line2D([0], [0], marker="^", color="black", linestyle="None", markersize=8, label="Radar")]
    box_text = []

    for color, subject in zip(colors, subjects):
        index = int(subject.get("subject_index", 0))
        accepted = subject.get("roi_cells", []) or []
        rejected = subject.get("roi_rejected_cells", []) or []

        if accepted:
            pts = [_cell_position(cell, subject, range_axis, has_metric_axis) for cell in accepted]
            ax_room.scatter([p[0] for p in pts], [p[1] for p in pts], s=82, marker="o", facecolors="none", edgecolors=color, linewidths=2.0)
        if rejected:
            pts = [_cell_position(cell, subject, range_axis, has_metric_axis) for cell in rejected]
            ax_room.scatter([p[0] for p in pts], [p[1] for p in pts], s=42, marker="x", color=color, linewidths=1.2, alpha=0.75)

        angle = subject.get("selected_angle_deg")
        distance = subject.get("selected_range_m") if has_metric_axis else subject.get("selected_range_bin")
        if angle is not None and distance is not None:
            a = math.radians(float(angle))
            x = float(distance) * math.cos(a)
            y = float(distance) * math.sin(a)
            ax_room.scatter([x], [y], s=185, marker="*", color=color, edgecolors="black", linewidths=0.9, zorder=5)
            ax_room.annotate("S{:02d}".format(index), (x, y), xytext=(8, 8), textcoords="offset points", color=color, weight="bold")

        heart = float(subject.get("heart_rate_beats_per_min", float("nan")))
        breath = float(subject.get("breathing_rate_breaths_per_min", float("nan")))
        rng = subject.get("selected_range_m")
        ang = subject.get("selected_angle_deg")
        box_text.append((color, "Subject {:02d}\nHeart rate: {:.1f} bpm\nBreathing: {:.1f} /min\nRange: {:.2f} m, angle: {:.1f} deg".format(index, heart, breath, float(rng) if rng is not None else float("nan"), float(ang) if ang is not None else float("nan"))))
        handles.append(Line2D([0], [0], marker="*", color=color, markeredgecolor="black", linestyle="None", markersize=11, label="Subject {:02d}".format(index)))

        times = np.asarray(subject.get("rate_trend_time_s", []), dtype=np.float64)
        heart_trend = np.asarray(subject.get("rate_trend_heart_bpm", []), dtype=np.float64)
        breath_trend = np.asarray(subject.get("rate_trend_breath_bpm", []), dtype=np.float64)
        if times.size:
            ax_trend.plot(times, heart_trend, linestyle="-", color=color, linewidth=2.0, label="S{:02d} heart".format(index))
            ax_trend.plot(times, breath_trend, linestyle="--", color=color, linewidth=2.0, label="S{:02d} breath".format(index))

    handles.append(Line2D([0], [0], marker="o", color="black", markerfacecolor="none", linestyle="None", markersize=8, label="Used chest bins"))
    handles.append(Line2D([0], [0], marker="x", color="black", linestyle="None", markersize=8, label="Rejected bins"))

    ax_room.set_title("Room view and selected chest bins")
    ax_room.set_xlabel("Forward distance X [m]" if has_metric_axis else "Forward X")
    ax_room.set_ylabel("Lateral position Y [m]" if has_metric_axis else "Lateral Y")
    ax_room.grid(True, alpha=0.25)
    ax_room.set_aspect("equal", adjustable="box")
    ax_room.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)

    ax_trend.set_title("Breathing and heart-rate trend")
    ax_trend.set_xlabel("Time [s]")
    ax_trend.set_ylabel("Rate [per min]")
    ax_trend.grid(True, alpha=0.35)
    ax_trend.legend(loc="best", ncol=2)

    for i, (color, text) in enumerate(box_text):
        x = 0.02 + (i % 2) * 0.49
        y = 0.88 - (i // 2) * 0.48
        ax_boxes.text(x, y, text, transform=ax_boxes.transAxes, va="top", ha="left", fontsize=11,
                      bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor=color, linewidth=2.0))

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close(fig)

__all__ = [name for name in globals() if not name.startswith('__')]
