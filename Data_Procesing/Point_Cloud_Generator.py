#!/usr/bin/env python3
"""
Point_Cloud_Generator.py

Generate a first-pass radar point cloud from parsed IWR1843/DCA1000 ADC data.

Expected input:
    Data_Procesing/Preliminary_Results/*_parsed.npz
created by Raw_Data_Parser.py.

Default output:
    Data_Procesing/Point_Cloud_Results/

Important limitation:
    If the capture used only one TX with 4 RX, this script can estimate range + azimuth
    only. It writes z=0. True 3D x/y/z point clouds require a 2D virtual antenna array
    from a valid TDM-MIMO capture and a correct antenna geometry/calibration.

Python: 3.8+
Dependencies: numpy, matplotlib optional for PNG plots
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np

C = 299792458.0


def db20(x, floor=1e-12):
    return 20.0 * np.log10(np.maximum(np.asarray(x), floor))


def db10(x, floor=1e-12):
    return 10.0 * np.log10(np.maximum(np.asarray(x), floor))


def find_project_root(start):
    """Find project root from current path or script path."""
    start = Path(start).resolve()
    if start.is_file():
        start = start.parent
    for p in [start] + list(start.parents):
        if (p / "Data_Procesing").exists() and (p / "ADC_Recorded_Data").exists():
            return p
    # If executed from Data_Procesing, parent is likely project root
    if start.name == "Data_Procesing":
        return start.parent
    return start


def find_latest_parsed_file(results_dir):
    candidates = sorted(Path(results_dir).glob("*_parsed.npz"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("No *_parsed.npz file found in: {0}".format(results_dir))
    return candidates[0]


def _as_plain_metadata(value):
    """Return metadata as a normal dict.

    Raw_Data_Parser.py has existed in a few versions. Some versions save
    metadata as a Python dict/object array, while others save it as a JSON
    string in a scalar numpy array. This helper accepts both formats.
    """
    if value is None:
        return {}

    # Unwrap scalar numpy arrays.
    if isinstance(value, np.ndarray):
        try:
            if value.shape == ():
                value = value.item()
        except Exception:
            pass

    # Most recent parser versions may store JSON text.
    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return {}

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return {}

    if isinstance(value, dict):
        metadata = dict(value)
    else:
        try:
            metadata = dict(value)
        except Exception:
            return {}

    # Flatten nested config values into the top level, but do not overwrite
    # existing top-level keys. This lets the rest of the code accept both
    # metadata["sample_rate_ksps"] and metadata["config"]["sample_rate_ksps"].
    cfg = metadata.get("config")
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if k not in metadata:
                metadata[k] = v

    return metadata


def load_parsed_npz(path):
    data = np.load(str(path), allow_pickle=True)

    if "cube_frames" in data and data["cube_frames"].size > 0:
        cube_frames = data["cube_frames"]
    elif "cube_chirps" in data:
        cube_chirps = data["cube_chirps"]
        # Fallback: treat whole chirp sequence as one frame
        cube_frames = cube_chirps[np.newaxis, ...]
    else:
        raise KeyError("Input npz must contain cube_frames or cube_chirps")

    metadata = {}
    if "metadata" in data:
        metadata = _as_plain_metadata(data["metadata"])

    return cube_frames, metadata


def get_meta_float(metadata, keys, default=None):
    for k in keys:
        if k in metadata and metadata[k] is not None:
            try:
                return float(metadata[k])
            except Exception:
                pass
    return default


def derive_axes(metadata, num_range_bins, num_doppler_bins, fallback_frame_period_s=None):
    """Derive range and Doppler axes from metadata when available."""
    sample_rate_ksps = get_meta_float(metadata, ["sample_rate_ksps", "dig_out_sample_rate_ksps", "digOutSampleRate", "dig_out_sample_rate", "digOutSampleRate_ksps"], None)
    slope_mhz_us = get_meta_float(metadata, ["freq_slope_mhz_us", "freq_slope_mhz_per_us", "slope_mhz_us", "slope_mhz_per_us", "freqSlopeConst"], None)
    start_freq_ghz = get_meta_float(metadata, ["start_freq_ghz", "start_frequency_ghz", "startFreqConst"], 77.0)
    frame_period_s = get_meta_float(metadata, ["frame_period_s", "periodicity_s", "frame_periodicity_s", "frame_periodicity_ms"], fallback_frame_period_s)
    if frame_period_s is not None and frame_period_s > 1.0:
        # Some parser metadata stores frame periodicity in milliseconds.
        frame_period_s = frame_period_s / 1000.0
    idle_time_us = get_meta_float(metadata, ["idle_time_us", "idleTimeConst"], None)
    ramp_end_time_us = get_meta_float(metadata, ["ramp_end_time_us", "rampEndTime"], None)

    if sample_rate_ksps is not None and slope_mhz_us is not None and slope_mhz_us != 0:
        fs_hz = sample_rate_ksps * 1e3
        slope_hz_s = slope_mhz_us * 1e12
        range_res = C * fs_hz / (2.0 * slope_hz_s * num_range_bins)
        range_axis = np.arange(num_range_bins) * range_res
    else:
        range_axis = np.arange(num_range_bins, dtype=float)

    wavelength = C / (start_freq_ghz * 1e9)

    # Doppler axis over chirps inside one frame. Prefer chirp period if available.
    if idle_time_us is not None and ramp_end_time_us is not None:
        chirp_period_s = (idle_time_us + ramp_end_time_us) * 1e-6
    elif frame_period_s is not None and num_doppler_bins > 0:
        chirp_period_s = frame_period_s / float(num_doppler_bins)
    else:
        chirp_period_s = None

    if chirp_period_s is not None and chirp_period_s > 0:
        doppler_freq = np.fft.fftshift(np.fft.fftfreq(num_doppler_bins, d=chirp_period_s))
        doppler_axis = doppler_freq * wavelength / 2.0
    else:
        doppler_axis = np.arange(num_doppler_bins, dtype=float) - num_doppler_bins / 2.0

    return range_axis, doppler_axis, wavelength


def hann_or_ones(n, enable=True):
    if enable:
        return np.hanning(n).astype(np.float32)
    return np.ones(n, dtype=np.float32)


def range_doppler_cube(frame_cube, n_range, n_doppler, remove_static=True, window=True):
    """
    frame_cube shape: [chirps, rx, samples]
    output shape: [doppler, rx, range]
    """
    chirps, rx, samples = frame_cube.shape
    x = frame_cube.astype(np.complex64, copy=False)

    # Remove DC offset per chirp/rx
    x = x - np.mean(x, axis=2, keepdims=True)

    wr = hann_or_ones(samples, window)[np.newaxis, np.newaxis, :]
    rd = np.fft.fft(x * wr, n=n_range, axis=2)

    # Keep positive range bins only; caller decides max range bins.
    if remove_static:
        # Remove mean over slow time to suppress static clutter.
        rd = rd - np.mean(rd, axis=0, keepdims=True)

    wd = hann_or_ones(chirps, window)[:, np.newaxis, np.newaxis]
    rd = np.fft.fftshift(np.fft.fft(rd * wd, n=n_doppler, axis=0), axes=0)
    return rd


def make_detection_map(rd_cube):
    """Non-coherent RX sum of range-Doppler power."""
    # rd_cube: [doppler, rx, range]
    return np.sum(np.abs(rd_cube) ** 2, axis=1)


def local_maxima_2d(power_db, threshold_db, max_points, guard_r=1, guard_d=1, min_range_bin=2, max_range_bin=None):
    """Simple local-max detector. Returns list of (doppler_bin, range_bin, power_db)."""
    nd, nr = power_db.shape
    if max_range_bin is None or max_range_bin > nr:
        max_range_bin = nr

    candidates = []
    for d in range(guard_d, nd - guard_d):
        for r in range(max(min_range_bin, guard_r), max_range_bin - guard_r):
            val = power_db[d, r]
            if val < threshold_db:
                continue
            patch = power_db[d - guard_d:d + guard_d + 1, r - guard_r:r + guard_r + 1]
            if val >= np.max(patch):
                candidates.append((d, r, float(val)))

    candidates.sort(key=lambda t: t[2], reverse=True)
    return candidates[:max_points]


def estimate_azimuth_from_snapshot(snapshot_rx, angle_fft_size=64, max_angle_deg=70.0):
    """
    Estimate azimuth angle from one complex RX vector using a ULA assumption.

    snapshot_rx: [num_rx]
    Assumption: RX spacing is lambda/2 along horizontal axis.
    Returns angle_deg and angle_spectrum_db.
    """
    num_rx = snapshot_rx.shape[0]
    if num_rx < 2:
        return 0.0, np.zeros(angle_fft_size)

    w = np.hanning(num_rx).astype(np.float32)
    spec = np.fft.fftshift(np.fft.fft(snapshot_rx * w, n=angle_fft_size))
    mag = np.abs(spec)

    # FFT spatial frequency bins normalized to cycles per element.
    u_bins = np.fft.fftshift(np.fft.fftfreq(angle_fft_size, d=0.5))  # d=lambda/2, u=sin(theta)
    valid = np.abs(u_bins) <= math.sin(math.radians(max_angle_deg))
    if not np.any(valid):
        idx = int(np.argmax(mag))
    else:
        valid_indices = np.where(valid)[0]
        idx = int(valid_indices[np.argmax(mag[valid_indices])])

    u = float(np.clip(u_bins[idx], -1.0, 1.0))
    angle_rad = math.asin(u)
    return math.degrees(angle_rad), db20(mag)


def spherical_to_cartesian(range_m, azimuth_deg, elevation_deg=0.0):
    """
    Coordinate convention:
        y = forward/range direction
        x = lateral left/right
        z = vertical
    """
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    x = range_m * math.cos(el) * math.sin(az)
    y = range_m * math.cos(el) * math.cos(az)
    z = range_m * math.sin(el)
    return x, y, z


def process_frame(frame_idx, frame_cube, metadata, args, range_axis, doppler_axis):
    chirps, rx, samples = frame_cube.shape
    rd_cube = range_doppler_cube(
        frame_cube,
        n_range=args.range_fft_size,
        n_doppler=args.doppler_fft_size or chirps,
        remove_static=not args.keep_static,
        window=not args.no_window,
    )
    power = make_detection_map(rd_cube)  # [doppler, range]

    # Use only positive range half by default.
    max_rbin = args.max_range_bin
    if max_rbin is None:
        max_rbin = args.range_fft_size // 2

    search_power = power[:, :max_rbin]
    noise_floor_db = float(np.median(db10(search_power)))
    threshold_db = noise_floor_db + args.threshold_db_above_median
    power_db = db10(power)

    detections = local_maxima_2d(
        power_db,
        threshold_db=threshold_db,
        max_points=args.max_points_per_frame,
        guard_r=args.guard_range_bins,
        guard_d=args.guard_doppler_bins,
        min_range_bin=args.min_range_bin,
        max_range_bin=max_rbin,
    )

    points = []
    for d_bin, r_bin, snr_db_like in detections:
        # Complex snapshot across RX at detected RD cell.
        snapshot = rd_cube[d_bin, :, r_bin]
        az_deg, _ = estimate_azimuth_from_snapshot(
            snapshot,
            angle_fft_size=args.angle_fft_size,
            max_angle_deg=args.max_angle_deg,
        )

        range_m = float(range_axis[r_bin]) if r_bin < len(range_axis) else float(r_bin)
        doppler_mps = float(doppler_axis[d_bin]) if d_bin < len(doppler_axis) else float(d_bin)
        x, y, z = spherical_to_cartesian(range_m, az_deg, 0.0)

        points.append({
            "frame": int(frame_idx),
            "range_bin": int(r_bin),
            "doppler_bin": int(d_bin),
            "range_m": range_m,
            "doppler_mps": doppler_mps,
            "azimuth_deg": float(az_deg),
            "elevation_deg": 0.0,
            "x_m": float(x),
            "y_m": float(y),
            "z_m": float(z),
            "power_db": float(snr_db_like),
            "noise_floor_db": float(noise_floor_db),
            "threshold_db": float(threshold_db),
        })

    return points, power


def write_points_csv(path, points):
    fieldnames = [
        "frame", "x_m", "y_m", "z_m", "range_m", "doppler_mps",
        "azimuth_deg", "elevation_deg", "range_bin", "doppler_bin",
        "power_db", "noise_floor_db", "threshold_db",
    ]
    with open(str(path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in points:
            writer.writerow(p)


def save_points_ply(path, points):
    """Save point cloud as ASCII PLY. Color is grayscale from power."""
    if points:
        powers = np.array([p["power_db"] for p in points], dtype=float)
        pmin, pmax = float(np.min(powers)), float(np.max(powers))
    else:
        pmin, pmax = 0.0, 1.0
    denom = max(pmax - pmin, 1e-9)

    with open(str(path), "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {0}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for p in points:
            gray = int(np.clip(255.0 * (p["power_db"] - pmin) / denom, 0, 255))
            f.write("{x:.6f} {y:.6f} {z:.6f} {r:d} {g:d} {b:d}\n".format(
                x=p["x_m"], y=p["y_m"], z=p["z_m"], r=gray, g=gray, b=gray
            ))


def plot_point_cloud_png(path, points, title="Radar point cloud"):
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception as exc:
        print("Matplotlib not available; skipping PNG plot: {0}".format(exc))
        return

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    if points:
        xs = [p["x_m"] for p in points]
        ys = [p["y_m"] for p in points]
        zs = [p["z_m"] for p in points]
        cs = [p["power_db"] for p in points]
        sc = ax.scatter(xs, ys, zs, c=cs, s=10)
        fig.colorbar(sc, ax=ax, label="Power [dB]")
    ax.set_xlabel("x lateral [m]")
    ax.set_ylabel("y forward [m]")
    ax.set_zlabel("z vertical [m]")
    ax.set_title(title)
    ax.view_init(elev=20, azim=-60)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate radar point clouds from parsed mmWave ADC data.")
    parser.add_argument("--project-root", default=None, help="Project root. Auto-detected by default.")
    parser.add_argument("--input", default=None, help="Parsed *_parsed.npz file. Auto-detected by default.")
    parser.add_argument("--results-dir", default=None, help="Output directory. Default: Data_Procesing/Point_Cloud_Results")
    parser.add_argument("--range-fft-size", type=int, default=256, help="Range FFT size. Default: 256")
    parser.add_argument("--doppler-fft-size", type=int, default=None, help="Doppler FFT size. Default: chirps per frame")
    parser.add_argument("--angle-fft-size", type=int, default=64, help="Azimuth FFT size. Default: 64")
    parser.add_argument("--max-angle-deg", type=float, default=70.0, help="Max azimuth search angle. Default: 70")
    parser.add_argument("--threshold-db-above-median", type=float, default=18.0, help="Detection threshold above median RD power. Default: 18 dB")
    parser.add_argument("--max-points-per-frame", type=int, default=50, help="Max detections per frame. Default: 50")
    parser.add_argument("--min-range-bin", type=int, default=3, help="Ignore near-DC range bins below this. Default: 3")
    parser.add_argument("--max-range-bin", type=int, default=None, help="Max range bin to search. Default: range_fft_size/2")
    parser.add_argument("--guard-range-bins", type=int, default=1, help="Local-max guard range bins. Default: 1")
    parser.add_argument("--guard-doppler-bins", type=int, default=1, help="Local-max guard doppler bins. Default: 1")
    parser.add_argument("--frame-start", type=int, default=0, help="First frame to process. Default: 0")
    parser.add_argument("--frame-count", type=int, default=None, help="Number of frames to process. Default: all")
    parser.add_argument("--keep-static", action="store_true", help="Do not remove static clutter before Doppler FFT.")
    parser.add_argument("--no-window", action="store_true", help="Disable Hann windows.")
    parser.add_argument("--make-png", action="store_true", help="Create a PNG preview of all points.")
    args = parser.parse_args(argv)

    script_path = Path(__file__).resolve()
    project_root = Path(args.project_root).resolve() if args.project_root else find_project_root(script_path)

    prelim_dir = project_root / "Data_Procesing" / "Preliminary_Results"
    input_path = Path(args.input).resolve() if args.input else find_latest_parsed_file(prelim_dir)
    results_dir = Path(args.results_dir).resolve() if args.results_dir else project_root / "Data_Procesing" / "Point_Cloud_Results"
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Project root: {0}".format(project_root))
    print("Input parsed file: {0}".format(input_path))
    print("Output directory: {0}".format(results_dir))

    cube_frames, metadata = load_parsed_npz(input_path)
    if cube_frames.ndim != 4:
        raise ValueError("cube_frames must have shape [frames, chirps, rx, samples], got {0}".format(cube_frames.shape))

    num_frames, chirps_per_frame, num_rx, num_samples = cube_frames.shape
    print("cube_frames shape: {0}".format(cube_frames.shape))

    if num_rx < 2:
        print("Warning: less than 2 RX channels. Azimuth estimation will be unavailable.")

    if args.range_fft_size < num_samples:
        print("Warning: range FFT size is smaller than ADC samples; increasing to {0}".format(num_samples))
        args.range_fft_size = int(num_samples)

    n_dop = args.doppler_fft_size or chirps_per_frame
    range_axis, doppler_axis, wavelength = derive_axes(metadata, args.range_fft_size, n_dop)

    # Warn about 3D limitation.
    print("Note: This script estimates range + azimuth. z is set to 0 unless you add 2D MIMO geometry handling.")
    print("Wavelength used: {0:.6e} m".format(wavelength))

    frame_start = max(0, args.frame_start)
    frame_end = num_frames if args.frame_count is None else min(num_frames, frame_start + args.frame_count)

    all_points = []
    rd_accum = None
    for frame_idx in range(frame_start, frame_end):
        points, power = process_frame(frame_idx, cube_frames[frame_idx], metadata, args, range_axis, doppler_axis)
        all_points.extend(points)
        if rd_accum is None:
            rd_accum = power.astype(np.float64)
        else:
            rd_accum += power
        print("Frame {0}: {1} points".format(frame_idx, len(points)))

    stem = input_path.stem.replace("_parsed", "")
    csv_path = results_dir / (stem + "_point_cloud.csv")
    npy_path = results_dir / (stem + "_point_cloud.npy")
    ply_path = results_dir / (stem + "_point_cloud.ply")
    summary_path = results_dir / (stem + "_point_cloud_summary.json")
    rd_path = results_dir / (stem + "_mean_range_doppler.npy")

    write_points_csv(csv_path, all_points)

    # Save numeric point array: frame, x, y, z, range, doppler, az, el, power
    arr = np.array([
        [p["frame"], p["x_m"], p["y_m"], p["z_m"], p["range_m"], p["doppler_mps"],
         p["azimuth_deg"], p["elevation_deg"], p["power_db"]]
        for p in all_points
    ], dtype=np.float64)
    np.save(str(npy_path), arr)
    save_points_ply(ply_path, all_points)
    if rd_accum is not None:
        np.save(str(rd_path), rd_accum / max(frame_end - frame_start, 1))

    if args.make_png:
        png_path = results_dir / (stem + "_point_cloud_preview.png")
        plot_point_cloud_png(png_path, all_points, title=stem + " point cloud")

    summary = {
        "input": str(input_path),
        "frames_processed": int(frame_end - frame_start),
        "cube_frames_shape": [int(x) for x in cube_frames.shape],
        "num_points": int(len(all_points)),
        "coordinate_convention": "x=lateral, y=forward, z=vertical; z=0 for single-TX/linear-RX processing",
        "range_fft_size": int(args.range_fft_size),
        "doppler_fft_size": int(n_dop),
        "angle_fft_size": int(args.angle_fft_size),
        "threshold_db_above_median": float(args.threshold_db_above_median),
        "outputs": {
            "csv": str(csv_path),
            "npy": str(npy_path),
            "ply": str(ply_path),
            "mean_range_doppler": str(rd_path),
        },
        "warnings": [
            "This is not a true 3D point cloud unless the capture used a calibrated 2D virtual antenna array.",
            "For the current one-TX capture, z is set to 0 and the cloud is range-azimuth only.",
            "Detection threshold is simple median-based thresholding, not a production CFAR implementation.",
        ],
    }
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved:")
    print("  {0}".format(csv_path))
    print("  {0}".format(npy_path))
    print("  {0}".format(ply_path))
    print("  {0}".format(summary_path))
    if args.make_png:
        print("  {0}".format(results_dir / (stem + "_point_cloud_preview.png")))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
