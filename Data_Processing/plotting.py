#!/usr/bin/env python3
# Auto-split from ADC_To_Vital_Signs.py.
# Keep Python 3.8 compatibility.

from common import *
import os
import subprocess


def _make_output_paths(results_dir, stem):
    results_dir.mkdir(parents=True, exist_ok=True)
    return {
        "summary_json": results_dir / "{}_vital_signs_summary.json".format(stem),
        "summary_txt": results_dir / "{}_vital_signs_summary.txt".format(stem),
        "disp_npy": results_dir / "{}_displacement_signal.npy".format(stem),
        "breath_npy": results_dir / "{}_breathing_signal.npy".format(stem),
        "heart_npy": results_dir / "{}_heart_signal.npy".format(stem),
        "csv": results_dir / "{}_vital_signs_timeseries.csv".format(stem),
        "plot": results_dir / "{}_vital_signs_plot.png".format(stem),
        "range_plot": results_dir / "{}_range_profile.png".format(stem),
        "angle_spectrum_npy": results_dir / "{}_angle_spectrum.npy".format(stem),
        "range_angle_plot": results_dir / "{}_range_angle_map.png".format(stem),
        "spectrum_plot": results_dir / "{}_vital_spectra.png".format(stem),
        "roi_plot": results_dir / "{}_roi_diagnostics.png".format(stem),
        "subject_comparison_plot": results_dir / "{}_subject_comparison.png".format(stem),
        "quality_plot": results_dir / "{}_quality_summary.png".format(stem),
        "dashboard_plot": results_dir / "{}_validation_dashboard.png".format(stem),
        "rate_trend_csv": results_dir / "{}_rate_trends.csv".format(stem),
    }

def _write_timeseries_csv(path, time_s, displacement_m, breathing, heart):
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_s",
                "displacement_m",
                "displacement_mm",
                "breathing_signal_m",
                "heart_signal_m",
            ]
        )
        for t, d, b, h in zip(time_s, displacement_m, breathing, heart):
            writer.writerow([float(t), float(d), float(d * 1000.0), float(b), float(h)])



def _write_rate_trend_csv(path, time_s, breathing_rate_bpm, heart_rate_bpm):
    with open(str(path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "breathing_rate_breaths_per_min", "heart_rate_beats_per_min"])
        for t, b, h in zip(time_s, breathing_rate_bpm, heart_rate_bpm):
            writer.writerow([float(t), float(b), float(h)])

def _save_plots(paths, time_s, displacement_m, breathing, heart, range_axis, range_profile, selected_bin, has_metric_axis, breath_bpm, heart_bpm):
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Matplotlib not available; skipping plots: {}".format(exc))
        return

    # Vital signs plot
    plt.figure(figsize=(12, 8))

    ax1 = plt.subplot(3, 1, 1)
    ax1.plot(time_s, displacement_m * 1000.0)
    ax1.set_ylabel("Displacement [mm]")
    ax1.set_title("Chest displacement from unwrapped phase")

    ax2 = plt.subplot(3, 1, 2)
    ax2.plot(time_s, breathing * 1000.0)
    ax2.set_ylabel("Breathing [mm]")
    ax2.set_title("Breathing band, estimate: {:.1f} breaths/min".format(breath_bpm))

    ax3 = plt.subplot(3, 1, 3)
    ax3.plot(time_s, heart * 1000.0)
    ax3.set_xlabel("Time [s]")
    ax3.set_ylabel("Heart band [mm]")
    ax3.set_title("Heart band, estimate: {:.1f} beats/min".format(heart_bpm))

    plt.tight_layout()
    plt.savefig(str(paths["plot"]), dpi=150)
    plt.close()

    # Range profile plot
    plt.figure(figsize=(10, 5))
    x = range_axis if has_metric_axis else np.arange(len(range_profile))
    plt.plot(x, 20.0 * np.log10(range_profile + 1e-12))
    selected_x = range_axis[selected_bin] if has_metric_axis else selected_bin
    plt.axvline(selected_x, linestyle="--")
    plt.xlabel("Range [m]" if has_metric_axis else "Range bin")
    plt.ylabel("Magnitude [dB]")
    plt.title("Mean range profile, selected bin {}".format(selected_bin))
    plt.tight_layout()
    plt.savefig(str(paths["range_plot"]), dpi=150)
    plt.close()

def _save_range_angle_map_plot(path, ra, range_axis, has_metric_axis, subjects=None):
    """Save a range-angle diagnostic plot with detected targets and ROI cells."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Matplotlib not available; skipping range-angle plot: {}".format(exc))
        return

    power_map = np.asarray(ra.get("power_map"), dtype=np.float64)
    if power_map.size == 0:
        return
    candidate_bins = np.asarray(ra.get("candidate_bins"), dtype=int)
    angles = np.asarray(ra.get("angle_grid_reported"), dtype=np.float64)
    if candidate_bins.size == 0 or angles.size == 0:
        return
    ranges = np.asarray(range_axis)[candidate_bins] if has_metric_axis else candidate_bins.astype(float)
    p_db = 10.0 * np.log10(power_map + 1e-30)
    p_db = p_db - np.nanmax(p_db)

    plt.figure(figsize=(11, 6))
    extent = [float(np.min(angles)), float(np.max(angles)), float(np.min(ranges)), float(np.max(ranges))]
    plt.imshow(p_db, origin="lower", aspect="auto", extent=extent, vmin=-30.0, vmax=0.0)
    plt.colorbar(label="Relative motion power [dB]")
    plt.xlabel("Azimuth angle [deg]")
    plt.ylabel("Range [m]" if has_metric_axis else "Range bin")
    plt.title("Range-angle motion map with detected subjects")

    if subjects:
        for s in subjects:
            ang = s.get("selected_angle_deg")
            rng = s.get("selected_range_m") if has_metric_axis else s.get("selected_range_bin")
            if ang is None or rng is None:
                continue
            plt.plot([float(ang)], [float(rng)], marker="x", markersize=10, mew=2)
            plt.text(float(ang), float(rng), " S{}".format(s.get("subject_index", "?")), va="bottom")
            if s.get("roi_enabled") and s.get("roi_angle_min_deg") is not None:
                amin = float(s.get("roi_angle_min_deg"))
                amax = float(s.get("roi_angle_max_deg"))
                rmin = s.get("roi_range_min_m") if has_metric_axis else min(s.get("roi_range_bins", [rng]))
                rmax = s.get("roi_range_max_m") if has_metric_axis else max(s.get("roi_range_bins", [rng]))
                if rmin is not None and rmax is not None:
                    xs = [amin, amax, amax, amin, amin]
                    ys = [float(rmin), float(rmin), float(rmax), float(rmax), float(rmin)]
                    plt.plot(xs, ys, linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()


def _save_vital_spectrum_plot(path, vitals, breath_low_hz, breath_high_hz, heart_low_hz, heart_high_hz):
    """Save breathing and heart spectral diagnostic plots for one subject."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Matplotlib not available; skipping spectrum plot: {}".format(exc))
        return

    bf = np.asarray(vitals.get("breathing_freqs_hz", []), dtype=np.float64)
    bp = np.asarray(vitals.get("breathing_power", []), dtype=np.float64)
    hf = np.asarray(vitals.get("heart_freqs_hz", []), dtype=np.float64)
    hp = np.asarray(vitals.get("heart_power", []), dtype=np.float64)
    if bf.size == 0 or bp.size == 0 or hf.size == 0 or hp.size == 0:
        return

    def _db(y):
        y = np.asarray(y, dtype=np.float64)
        finite = np.isfinite(y)
        ref = float(np.nanmax(y[finite])) if np.any(finite) else 1.0
        return 10.0 * np.log10((y + 1e-30) / (ref + 1e-30))

    plt.figure(figsize=(12, 7))

    ax1 = plt.subplot(2, 1, 1)
    mask_b = (bf >= float(breath_low_hz)) & (bf <= float(breath_high_hz))
    ax1.plot(bf[mask_b] * 60.0, _db(bp[mask_b]))
    raw_b = vitals.get("breathing_raw_bin_rate_breaths_per_min")
    est_b = vitals.get("breathing_rate_breaths_per_min")
    if raw_b is not None:
        ax1.axvline(float(raw_b), linestyle="--", linewidth=1, label="Raw FFT bin")
    if est_b is not None:
        ax1.axvline(float(est_b), linestyle=":", linewidth=2, label="Interpolated")
    ax1.set_title("Breathing spectrum")
    ax1.set_xlabel("Breathing rate [breaths/min]")
    ax1.set_ylabel("Relative power [dB]")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    ax2 = plt.subplot(2, 1, 2)
    mask_h = (hf >= float(heart_low_hz)) & (hf <= float(heart_high_hz))
    ax2.plot(hf[mask_h] * 60.0, _db(hp[mask_h]))
    raw_h = vitals.get("heart_raw_bin_rate_beats_per_min")
    est_h = vitals.get("heart_rate_beats_per_min")
    if raw_h is not None:
        ax2.axvline(float(raw_h), linestyle="--", linewidth=1, label="Raw FFT bin")
    if est_h is not None:
        ax2.axvline(float(est_h), linestyle=":", linewidth=2, label="Interpolated")
    ax2.set_title("Heart spectrum")
    ax2.set_xlabel("Heart rate [beats/min]")
    ax2.set_ylabel("Relative power [dB]")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()


def _save_roi_diagnostic_plot(path, subject, ra, range_axis, has_metric_axis):
    """Save accepted/rejected ROI cells for one subject on the range-angle map."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Matplotlib not available; skipping ROI diagnostic plot: {}".format(exc))
        return

    power_map = np.asarray(ra.get("power_map"), dtype=np.float64)
    candidate_bins = np.asarray(ra.get("candidate_bins"), dtype=int)
    angles = np.asarray(ra.get("angle_grid_reported"), dtype=np.float64)
    if power_map.size == 0 or candidate_bins.size == 0 or angles.size == 0:
        return

    ranges = np.asarray(range_axis)[candidate_bins] if has_metric_axis else candidate_bins.astype(float)
    p_db = 10.0 * np.log10(power_map + 1e-30)
    p_db = p_db - np.nanmax(p_db)

    plt.figure(figsize=(11, 6))
    extent = [float(np.min(angles)), float(np.max(angles)), float(np.min(ranges)), float(np.max(ranges))]
    plt.imshow(p_db, origin="lower", aspect="auto", extent=extent, vmin=-30.0, vmax=0.0)
    plt.colorbar(label="Relative motion power [dB]")
    plt.xlabel("Azimuth angle [deg]")
    plt.ylabel("Range [m]" if has_metric_axis else "Range bin")
    plt.title("Subject {:02d} chest ROI diagnostics".format(subject.get("subject_index", 0)))

    def _cell_xy(cell):
        a = float(cell.get("angle_reported_deg", cell.get("selected_angle_deg", 0.0)))
        rb = int(cell.get("range_bin", cell.get("selected_range_bin", 0)))
        r = float(range_axis[rb]) if has_metric_axis and rb < len(range_axis) else float(rb)
        return a, r

    accepted = subject.get("roi_cells", []) or []
    rejected = subject.get("roi_rejected_cells", []) or []
    if accepted:
        xy = np.array([_cell_xy(c) for c in accepted], dtype=np.float64)
        plt.scatter(xy[:, 0], xy[:, 1], marker="o", s=55, label="Accepted ROI cells")
    if rejected:
        xy = np.array([_cell_xy(c) for c in rejected], dtype=np.float64)
        plt.scatter(xy[:, 0], xy[:, 1], marker="x", s=65, label="Rejected ROI cells")

    ang = subject.get("selected_angle_deg")
    rng = subject.get("selected_range_m") if has_metric_axis else subject.get("selected_range_bin")
    if ang is not None and rng is not None:
        plt.scatter([float(ang)], [float(rng)], marker="*", s=160, label="Detected center")

    if subject.get("roi_enabled") and subject.get("roi_angle_min_deg") is not None:
        amin = float(subject.get("roi_angle_min_deg"))
        amax = float(subject.get("roi_angle_max_deg"))
        rmin = subject.get("roi_range_min_m") if has_metric_axis else min(subject.get("roi_range_bins", [rng]))
        rmax = subject.get("roi_range_max_m") if has_metric_axis else max(subject.get("roi_range_bins", [rng]))
        if rmin is not None and rmax is not None:
            plt.plot([amin, amax, amax, amin, amin], [float(rmin), float(rmin), float(rmax), float(rmax), float(rmin)], linestyle="--", linewidth=1, label="ROI boundary")

    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()


def _save_subject_comparison_plot(path, subjects):
    """Save a compact comparison plot for all detected subjects."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("Matplotlib not available; skipping subject comparison plot: {}".format(exc))
        return
    if not subjects:
        return

    labels = ["S{:02d}".format(int(s.get("subject_index", i + 1))) for i, s in enumerate(subjects)]
    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38
    breath = [float(s.get("breathing_rate_breaths_per_min", float("nan"))) for s in subjects]
    heart = [float(s.get("heart_rate_beats_per_min", float("nan"))) for s in subjects]
    det = [float(s.get("detection_quality_db", s.get("target_quality_db", float("nan")))) for s in subjects]
    vital = [float(s.get("vital_score_db", float("nan"))) for s in subjects]

    plt.figure(figsize=(12, 8))
    ax1 = plt.subplot(2, 1, 1)
    ax1.bar(x - width/2.0, breath, width, label="Breathing [1/min]")
    ax1.bar(x + width/2.0, heart, width, label="Heart [bpm]")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("Rate")
    ax1.set_title("Detected vital rates by subject")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="best")

    ax2 = plt.subplot(2, 1, 2)
    ax2.bar(x - width/2.0, det, width, label="Detection quality [dB]")
    ax2.bar(x + width/2.0, vital, width, label="Vital score [dB]")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Quality [dB]")
    ax2.set_title("Detection and vital-signal quality")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="best")

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close()




def _subject_color_cycle(n):
    try:
        import matplotlib.pyplot as plt
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    except Exception:
        colors = []
    if not colors:
        colors = ['C{}'.format(i) for i in range(max(1, n))]
    return [colors[i % len(colors)] for i in range(max(1, n))]


def _configure_matplotlib_gui_backend(requested_backend=None):
    """Try to configure an interactive Matplotlib backend before pyplot is imported.

    Return (ok, backend_name, error). This function is intentionally safe: it
    never raises for missing Tk/Qt/WX packages. If pyplot has already been
    imported with a non-GUI backend, Matplotlib may not be able to switch. For
    this reason the entry script calls this before any plotting function runs.
    """
    try:
        import matplotlib
    except Exception as exc:
        return False, None, exc

    current = str(matplotlib.get_backend())
    candidates = [str(requested_backend)] if requested_backend else ["TkAgg", "QtAgg", "Qt5Agg", "WXAgg"]

    if "agg" not in current.lower() and not requested_backend:
        return True, current, None

    last_error = None
    for candidate in candidates:
        try:
            matplotlib.use(candidate, force=True)
            return True, str(matplotlib.get_backend()), None
        except Exception as exc:
            last_error = exc
    return False, str(matplotlib.get_backend()), last_error


def _open_file_with_system_viewer(path):
    """Open a generated image with the OS default viewer. Returns (ok, error)."""
    path = Path(path)
    try:
        if sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
            return True, None
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return True, None
        subprocess.Popen(["xdg-open", str(path)])
        return True, None
    except Exception as exc:
        return False, exc


def _display_dashboard_window(path, gui_backend=None, display_mode="auto"):
    """Display the saved dashboard after processing has finished.

    display_mode:
      auto       Try Matplotlib GUI first, then OS file viewer.
      matplotlib Use only Matplotlib imshow.
      file       Use only OS default image viewer.
      none       Do not display.
    """
    path = Path(path)
    if not path.exists():
        print("Dashboard display skipped: file does not exist: {}".format(path))
        return False

    mode = str(display_mode or "auto").lower()
    if mode == "none":
        return False

    if mode in {"auto", "matplotlib"}:
        ok, backend_name, backend_error = _configure_matplotlib_gui_backend(gui_backend)
        try:
            import matplotlib
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg
            backend_name = str(matplotlib.get_backend())
            if ok and "agg" not in backend_name.lower():
                img = mpimg.imread(str(path))
                fig, ax = plt.subplots(figsize=(16, 8))
                ax.imshow(img)
                ax.axis("off")
                fig.canvas.manager.set_window_title("mmWave vital-sign validation dashboard")
                print("Dashboard GUI backend: {}".format(backend_name))
                plt.show(block=True)
                return True
            print("Dashboard Matplotlib GUI unavailable. Current backend: {}. Last error: {}".format(backend_name, backend_error))
        except Exception as exc:
            print("Dashboard Matplotlib display failed: {}".format(exc))

    if mode in {"auto", "file"}:
        ok, err = _open_file_with_system_viewer(path)
        if ok:
            print("Dashboard opened with the operating-system image viewer: {}".format(path))
            return True
        print("Dashboard file-viewer fallback failed: {}".format(err))

    print("Dashboard was saved here: {}".format(path))
    return False


def _save_validation_dashboard_plot(path, subjects, ra, range_axis, has_metric_axis, show_window=False, gui_backend=None):
    """Save one combined validation figure and optionally display it.

    Left: top-down room/scene plot using range+azimuth data. The background color shows
    relative motion power across the scanned area. Detected subjects are marked and the
    accepted/rejected chest bins used for the measurement are highlighted.

    Right: time-varying breathing and heart-rate estimates. Each subject keeps one
    color; breathing uses dashed lines and heart uses solid lines.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        print('Matplotlib not available; skipping validation dashboard plot: {}'.format(exc))
        return

    power_map = np.asarray(ra.get('power_map', []), dtype=np.float64)
    candidate_bins = np.asarray(ra.get('candidate_bins', []), dtype=int)
    angles_deg = np.asarray(ra.get('angle_grid_reported', []), dtype=np.float64)
    if power_map.size == 0 or candidate_bins.size == 0 or angles_deg.size == 0:
        return

    ranges = np.asarray(range_axis)[candidate_bins] if has_metric_axis else candidate_bins.astype(np.float64)
    ang_rad = np.deg2rad(angles_deg)
    rr, aa = np.meshgrid(ranges, ang_rad, indexing='ij')
    xx = rr * np.cos(aa)
    yy = rr * np.sin(aa)
    p_db = 10.0 * np.log10(power_map + 1e-30)
    p_db = p_db - np.nanmax(p_db)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Left panel: top-down room/scene plot.
    marker_size = max(60.0, 5200.0 / max(1, power_map.size))
    bg = ax1.scatter(xx.ravel(), yy.ravel(), c=p_db.ravel(), cmap='viridis',
                     s=marker_size, marker='s', alpha=0.85, linewidths=0)
    cbar = fig.colorbar(bg, ax=ax1, pad=0.02, shrink=0.85)
    cbar.set_label('Relative motion power [dB]')

    # Radar origin and field-of-view guides.
    max_range = float(np.max(ranges)) if ranges.size else 1.0
    min_ang = float(np.min(angles_deg))
    max_ang = float(np.max(angles_deg))
    ax1.scatter([0.0], [0.0], s=110, marker='^', color='black', label='Radar')
    for a_deg in [min_ang, 0.0, max_ang]:
        a = math.radians(a_deg)
        ax1.plot([0.0, max_range * math.cos(a)], [0.0, max_range * math.sin(a)],
                 linestyle=':' if abs(a_deg) > 1e-9 else '--', color='gray', linewidth=1.0, alpha=0.7)
    # Simple room boundary as data extent rectangle for orientation.
    x_min = float(np.min(xx))
    x_max = float(np.max(xx))
    y_min = float(np.min(yy))
    y_max = float(np.max(yy))
    pad_x = 0.08 * max(0.5, x_max - x_min)
    pad_y = 0.08 * max(0.5, y_max - y_min)
    ax1.plot([x_min - pad_x, x_max + pad_x, x_max + pad_x, x_min - pad_x, x_min - pad_x],
             [y_min - pad_y, y_min - pad_y, y_max + pad_y, y_max + pad_y, y_min - pad_y],
             color='0.55', linewidth=1.0, alpha=0.6)

    colors = _subject_color_cycle(len(subjects))
    legend_handles = [Line2D([0], [0], marker='^', color='black', linestyle='None', markersize=8, label='Radar')]

    for color, subject in zip(colors, subjects):
        accepted = subject.get('roi_cells', []) or []
        rejected = subject.get('roi_rejected_cells', []) or []

        acc_x, acc_y = [], []
        for cell in accepted:
            a_deg = float(cell.get('angle_reported_deg', cell.get('selected_angle_deg', 0.0)))
            rb = int(cell.get('range_bin', subject.get('selected_range_bin', 0)))
            r = float(range_axis[rb]) if has_metric_axis and 0 <= rb < len(range_axis) else float(rb)
            a = math.radians(a_deg)
            acc_x.append(r * math.cos(a))
            acc_y.append(r * math.sin(a))
        if acc_x:
            ax1.scatter(acc_x, acc_y, s=88, marker='o', facecolors='none', edgecolors=color, linewidths=2.0)

        rej_x, rej_y = [], []
        for cell in rejected:
            a_deg = float(cell.get('angle_reported_deg', cell.get('selected_angle_deg', 0.0)))
            rb = int(cell.get('range_bin', subject.get('selected_range_bin', 0)))
            r = float(range_axis[rb]) if has_metric_axis and 0 <= rb < len(range_axis) else float(rb)
            a = math.radians(a_deg)
            rej_x.append(r * math.cos(a))
            rej_y.append(r * math.sin(a))
        if rej_x:
            ax1.scatter(rej_x, rej_y, s=56, marker='x', color=color, linewidths=1.5, alpha=0.8)

        ang = subject.get('selected_angle_deg')
        rng = subject.get('selected_range_m') if has_metric_axis else subject.get('selected_range_bin')
        if ang is not None and rng is not None:
            a = math.radians(float(ang))
            x = float(rng) * math.cos(a)
            y = float(rng) * math.sin(a)
            ax1.scatter([x], [y], s=190, marker='*', color=color, edgecolors='k', linewidths=0.9, zorder=5)
            ax1.annotate('S{:02d}'.format(int(subject.get('subject_index', 0))), (x, y),
                         xytext=(8, 8), textcoords='offset points', color=color, fontsize=10, weight='bold')

        legend_handles.append(Line2D([0], [0], marker='*', color=color, markeredgecolor='k',
                                     linestyle='None', markersize=11, label='S{:02d} subject'.format(int(subject.get('subject_index', 0)))))

    legend_handles.append(Line2D([0], [0], marker='o', color='black', markerfacecolor='none',
                                 linestyle='None', markersize=8, label='Accepted chest bins'))
    legend_handles.append(Line2D([0], [0], marker='x', color='black', linestyle='None', markersize=8, label='Rejected bins'))

    ax1.set_title('Room view with detected subjects and selected chest bins')
    ax1.set_xlabel('Forward distance X [m]' if has_metric_axis else 'Forward X')
    ax1.set_ylabel('Lateral position Y [m]' if has_metric_axis else 'Lateral Y')
    ax1.grid(True, alpha=0.25)
    ax1.set_aspect('equal', adjustable='box')
    ax1.set_xlim(x_min - pad_x, x_max + pad_x)
    ax1.set_ylim(y_min - pad_y, y_max + pad_y)
    ax1.legend(handles=legend_handles, loc='upper right', fontsize=8, framealpha=0.95)

    # Right panel: time-varying heart and breathing rates.
    for color, subject in zip(colors, subjects):
        tt = np.asarray(subject.get('rate_trend_time_s', []), dtype=np.float64)
        bb = np.asarray(subject.get('rate_trend_breath_bpm', []), dtype=np.float64)
        hh = np.asarray(subject.get('rate_trend_heart_bpm', []), dtype=np.float64)
        if tt.size:
            ax2.plot(tt, hh, linestyle='-', color=color, linewidth=2.0, label='S{:02d} heart'.format(int(subject.get('subject_index', 0))))
            ax2.plot(tt, bb, linestyle='--', color=color, linewidth=2.0, label='S{:02d} breath'.format(int(subject.get('subject_index', 0))))

    ax2.set_title('Breathing and heart rate over time')
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('Rate [per min]')
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc='best', ncol=2)

    plt.tight_layout()
    plt.savefig(str(path), dpi=150)
    plt.close(fig)

def _json_default(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)

def _make_subject_output_paths(results_dir, stem, subject_index):
    subject_stem = "{}_subject_{:02d}".format(stem, int(subject_index))
    return _make_output_paths(results_dir, subject_stem)

__all__ = [name for name in globals() if not name.startswith('__')]
