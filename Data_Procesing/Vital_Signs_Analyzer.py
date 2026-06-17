#!/usr/bin/env python3
"""
Vital_Signs_Analyzer.py

Phase-based breathing and heart-rate extraction for parsed mmWave Studio
DCA1000 captures.

Expected input:
    Data_Procesing/Preliminary_Results/*_parsed.npz

Expected parsed cube:
    cube_frames with shape:
        (frames, chirps_per_frame, rx, adc_samples)

Outputs:
    Data_Procesing/Vital_Signs_Results/
        *_vital_signs_summary.json
        *_vital_signs_summary.txt
        *_displacement_signal.npy
        *_breathing_signal.npy
        *_heart_signal.npy
        *_vital_signs_plot.png
        *_range_profile.png
        *_vital_signs_timeseries.csv

This is engineering/prototype code only and is not for medical diagnosis.
"""

from __future__ import print_function

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np


C = 299_792_458.0


def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _decode_metadata(value):
    """Load metadata saved as dict, JSON string, bytes, or numpy scalar."""
    if value is None:
        return {}

    try:
        if isinstance(value, np.ndarray):
            if value.shape == ():
                value = value.item()
            elif value.size == 1:
                value = value.reshape(-1)[0]
            else:
                return {}
    except Exception:
        pass

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8")
        except Exception:
            return {}

    if isinstance(value, str):
        try:
            loaded = json.loads(value)
            return loaded if isinstance(loaded, dict) else {}
        except Exception:
            return {}

    if isinstance(value, dict):
        return value

    return {}


def _flatten_metadata(metadata):
    """Flatten a nested metadata/config dictionary for easier key lookup."""
    out = {}

    def visit(prefix, obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = str(k)
                out[key] = v
                visit(prefix + "." + key if prefix else key, v)
        else:
            if prefix:
                out[prefix] = obj

    visit("", metadata)
    return out


def _meta_get_float(metadata, keys, default=None):
    flat = _flatten_metadata(metadata)
    for key in keys:
        if key in flat:
            val = _safe_float(flat[key], None)
            if val is not None:
                return val
    return default


def _find_latest_parsed_file(prelim_dir):
    files = sorted(prelim_dir.glob("*_parsed.npz"))
    if not files:
        raise FileNotFoundError("No *_parsed.npz file found in: {}".format(prelim_dir))
    return files[-1]


def _get_project_root_from_script():
    # Script normally lives in mmWave_Studio/Data_Procesing/
    return Path(__file__).resolve().parents[1]


def _load_parsed_npz(path):
    data = np.load(str(path), allow_pickle=True)

    if "cube_frames" in data:
        cube_frames = data["cube_frames"]
    elif "cube_chirps" in data:
        cube_chirps = data["cube_chirps"]
        raise ValueError(
            "Input contains cube_chirps but not cube_frames. "
            "For vital signs, run the parser so it outputs complete cube_frames."
        )
    else:
        raise KeyError("Parsed file does not contain cube_frames.")

    metadata = {}
    if "metadata" in data:
        metadata = _decode_metadata(data["metadata"])
    elif "config" in data:
        metadata = _decode_metadata(data["config"])

    return cube_frames, metadata


def _derive_range_axis(metadata, n_fft, num_adc_samples):
    sample_rate_ksps = _meta_get_float(
        metadata,
        [
            "sample_rate_ksps",
            "dig_out_sample_rate_ksps",
            "digOutSampleRate",
            "config.sample_rate_ksps",
            "config.dig_out_sample_rate_ksps",
        ],
        None,
    )
    slope_mhz_per_us = _meta_get_float(
        metadata,
        [
            "slope_mhz_per_us",
            "freq_slope_mhz_per_us",
            "freq_slope_mhz_us",
            "freqSlopeConst",
            "config.slope_mhz_per_us",
            "config.freq_slope_mhz_per_us",
        ],
        None,
    )

    if sample_rate_ksps is None or slope_mhz_per_us is None or slope_mhz_per_us == 0:
        # Fallback: bins only
        return np.arange(n_fft, dtype=np.float64), False

    fs_hz = sample_rate_ksps * 1e3
    slope_hz_s = slope_mhz_per_us * 1e12

    freqs = np.arange(n_fft, dtype=np.float64) * fs_hz / float(n_fft)
    ranges = C * freqs / (2.0 * slope_hz_s)
    return ranges, True


def _derive_wavelength(metadata):
    start_freq_ghz = _meta_get_float(
        metadata,
        [
            "start_freq_ghz",
            "startFreqConst",
            "config.start_freq_ghz",
        ],
        77.0,
    )
    return C / (start_freq_ghz * 1e9)


def _derive_frame_rate(metadata, fallback=20.0):
    frame_period_ms = _meta_get_float(
        metadata,
        [
            "frame_periodicity_ms",
            "periodicity_ms",
            "framePeriodicityMs",
            "config.frame_periodicity_ms",
        ],
        None,
    )
    if frame_period_ms is not None and frame_period_ms > 0:
        return 1000.0 / frame_period_ms

    vital_rate = _meta_get_float(
        metadata,
        [
            "vital_slow_time_rate_hz",
            "config.vital_slow_time_rate_hz",
        ],
        None,
    )
    if vital_rate is not None and vital_rate > 0:
        return vital_rate

    return fallback


def _hann(n):
    if n <= 1:
        return np.ones(n, dtype=np.float64)
    return np.hanning(n)


def _moving_average(x, n):
    if n <= 1:
        return x.copy()
    kernel = np.ones(int(n), dtype=np.float64) / float(n)
    return np.convolve(x, kernel, mode="same")


def _fft_bandpass(signal, fs, low_hz, high_hz):
    """Simple FFT-domain bandpass. Avoids scipy dependency."""
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 4:
        return np.zeros_like(x)

    x0 = x - np.nanmean(x)
    spec = np.fft.rfft(x0)
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spec[~mask] = 0.0
    return np.fft.irfft(spec, n=n)


def _estimate_rate_fft(signal, fs, low_hz, high_hz):
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 8:
        return float("nan"), float("nan"), np.array([]), np.array([])

    x = x - np.nanmean(x)
    x = x * _hann(n)

    # Zero padding improves plot resolution, not true measurement resolution.
    n_fft = int(2 ** math.ceil(math.log(max(n, 256), 2)))
    n_fft = max(n_fft, 4096)

    spec = np.fft.rfft(x, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    power = np.abs(spec) ** 2

    band = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(band):
        return float("nan"), float("nan"), freqs, power

    idxs = np.where(band)[0]
    best = idxs[np.argmax(power[idxs])]
    freq_hz = float(freqs[best])
    rate_per_min = 60.0 * freq_hz

    # Simple peak quality metric: best peak relative to median band power.
    med = float(np.median(power[idxs])) if idxs.size else 0.0
    quality_db = 10.0 * math.log10((float(power[best]) + 1e-30) / (med + 1e-30))

    return rate_per_min, quality_db, freqs, power


def _select_range_bin(range_profile, range_axis, has_metric_axis, min_range_m, max_range_m, min_bin, max_bin):
    rp = np.asarray(range_profile, dtype=np.float64).copy()

    # Remove DC/very near bins by default.
    if min_bin is not None and min_bin > 0:
        rp[: int(min_bin)] = -np.inf

    if max_bin is not None and max_bin < rp.size:
        rp[int(max_bin) :] = -np.inf

    if has_metric_axis:
        mask = (range_axis >= min_range_m) & (range_axis <= max_range_m)
        if np.any(mask):
            rp[~mask] = -np.inf

    if not np.any(np.isfinite(rp)):
        raise ValueError("No valid range bins after applying selection limits.")

    return int(np.nanargmax(rp))


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


def analyze(args):
    project_root = Path(args.project_root).resolve() if args.project_root else _get_project_root_from_script()
    prelim_dir = project_root / "Data_Procesing" / "Preliminary_Results"
    results_dir = project_root / "Data_Procesing" / "Vital_Signs_Results"

    parsed_path = Path(args.parsed).resolve() if args.parsed else _find_latest_parsed_file(prelim_dir)

    cube_frames, metadata = _load_parsed_npz(parsed_path)
    if cube_frames.ndim != 4:
        raise ValueError(
            "cube_frames must have 4 dimensions: (frames, chirps, rx, samples). Got shape {}".format(cube_frames.shape)
        )

    n_frames_total, n_chirps, n_rx, n_samples = cube_frames.shape

    frame_start = max(0, int(args.frame_start))
    if args.frame_count is None:
        frame_end = n_frames_total
    else:
        frame_end = min(n_frames_total, frame_start + int(args.frame_count))

    if frame_end <= frame_start:
        raise ValueError("Invalid frame range: start={}, end={}".format(frame_start, frame_end))

    cube = cube_frames[frame_start:frame_end]
    n_frames = cube.shape[0]

    fs_vital = float(args.fs) if args.fs else _derive_frame_rate(metadata, fallback=20.0)
    wavelength = _derive_wavelength(metadata)

    range_fft_size = int(args.range_fft_size)
    if range_fft_size < n_samples:
        range_fft_size = int(2 ** math.ceil(math.log(n_samples, 2)))

    range_axis, has_metric_axis = _derive_range_axis(metadata, range_fft_size, n_samples)

    # Average chirps within each frame for one slow-time sample per frame.
    # Shape after mean: (frames, rx, samples)
    frame_adc = cube.mean(axis=1)

    # Remove static ADC DC bias over samples for each frame/RX.
    frame_adc = frame_adc - frame_adc.mean(axis=2, keepdims=True)

    window = _hann(n_samples).astype(np.float32)
    range_fft = np.fft.fft(frame_adc * window[None, None, :], n=range_fft_size, axis=2)

    # Use only positive range bins.
    half = range_fft_size // 2
    range_fft_pos = range_fft[:, :, :half]
    range_axis_pos = range_axis[:half]

    # Mean range profile over frames and RX.
    range_profile = np.mean(np.abs(range_fft_pos), axis=(0, 1))

    selected_bin = _select_range_bin(
        range_profile,
        range_axis_pos,
        has_metric_axis,
        float(args.min_range_m),
        float(args.max_range_m),
        int(args.min_range_bin),
        args.max_range_bin,
    )

    # Extract complex slow-time signal at selected range bin.
    # Noncoherent option is safer across uncalibrated RX; use strongest RX by default.
    rx_power = np.mean(np.abs(range_fft_pos[:, :, selected_bin]) ** 2, axis=0)
    selected_rx = int(np.argmax(rx_power))

    if args.rx_mode == "strongest":
        slow_complex = range_fft_pos[:, selected_rx, selected_bin]
    elif args.rx_mode == "rx0":
        selected_rx = 0
        slow_complex = range_fft_pos[:, 0, selected_bin]
    elif args.rx_mode == "sum":
        # Coherent sum can help if RX phases are stable, but can also cancel if uncalibrated.
        slow_complex = np.sum(range_fft_pos[:, :, selected_bin], axis=1)
        selected_rx = -1
    else:
        raise ValueError("Unknown rx_mode: {}".format(args.rx_mode))

    phase = np.unwrap(np.angle(slow_complex))

    # Remove slow drift before conversion. Moving average window default ~2 s.
    drift_window = max(1, int(round(float(args.drift_window_s) * fs_vital)))
    phase_detrended = phase - _moving_average(phase, drift_window)

    displacement_m = phase_detrended * wavelength / (4.0 * math.pi)

    # Optional extra de-mean
    displacement_m = displacement_m - np.mean(displacement_m)

    breathing = _fft_bandpass(displacement_m, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz))
    heart = _fft_bandpass(displacement_m, fs_vital, float(args.heart_low_hz), float(args.heart_high_hz))

    breath_bpm, breath_quality_db, breath_freqs, breath_power = _estimate_rate_fft(
        breathing, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz)
    )
    heart_bpm, heart_quality_db, heart_freqs, heart_power = _estimate_rate_fft(
        heart, fs_vital, float(args.heart_low_hz), float(args.heart_high_hz)
    )

    time_s = np.arange(n_frames, dtype=np.float64) / fs_vital

    selected_range = float(range_axis_pos[selected_bin]) if has_metric_axis else None

    stem = parsed_path.stem.replace("_parsed", "")
    paths = _make_output_paths(results_dir, stem)

    np.save(str(paths["disp_npy"]), displacement_m)
    np.save(str(paths["breath_npy"]), breathing)
    np.save(str(paths["heart_npy"]), heart)
    _write_timeseries_csv(paths["csv"], time_s, displacement_m, breathing, heart)

    summary = {
        "input_parsed_file": str(parsed_path),
        "cube_frames_shape_total": list(cube_frames.shape),
        "analyzed_frame_start": frame_start,
        "analyzed_frame_end_exclusive": frame_end,
        "analyzed_frames": n_frames,
        "capture_duration_s": float(n_frames / fs_vital),
        "vital_sample_rate_hz": float(fs_vital),
        "range_fft_size": int(range_fft_size),
        "selected_range_bin": int(selected_bin),
        "selected_range_m": selected_range,
        "selected_rx": int(selected_rx),
        "rx_mode": args.rx_mode,
        "wavelength_m": float(wavelength),
        "breathing_rate_breaths_per_min": float(breath_bpm),
        "breathing_quality_db": float(breath_quality_db),
        "heart_rate_beats_per_min": float(heart_bpm),
        "heart_quality_db": float(heart_quality_db),
        "breathing_band_hz": [float(args.breath_low_hz), float(args.breath_high_hz)],
        "heart_band_hz": [float(args.heart_low_hz), float(args.heart_high_hz)],
        "note": "Engineering/prototype estimate only; not for medical diagnosis.",
    }

    with open(str(paths["summary_json"]), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(str(paths["summary_txt"]), "w", encoding="utf-8") as f:
        f.write("Vital signs analysis summary\n")
        f.write("============================\n")
        for k, v in summary.items():
            f.write("{}: {}\n".format(k, v))

    if args.make_plots:
        _save_plots(
            paths,
            time_s,
            displacement_m,
            breathing,
            heart,
            range_axis_pos,
            range_profile,
            selected_bin,
            has_metric_axis,
            breath_bpm,
            heart_bpm,
        )

    print("Vital signs analysis complete")
    print("============================")
    print("Input parsed file: {}".format(parsed_path))
    print("Analyzed frames:   {} to {} ({} frames)".format(frame_start, frame_end - 1, n_frames))
    print("Duration:          {:.2f} s".format(n_frames / fs_vital))
    print("Sample rate:       {:.3f} Hz".format(fs_vital))
    if selected_range is not None:
        print("Chest range bin:   {} ({:.3f} m)".format(selected_bin, selected_range))
    else:
        print("Chest range bin:   {}".format(selected_bin))
    print("Selected RX:       {}".format(selected_rx if selected_rx >= 0 else "coherent sum"))
    print("Breathing rate:    {:.2f} breaths/min  quality {:.1f} dB".format(breath_bpm, breath_quality_db))
    print("Heart rate:        {:.2f} beats/min     quality {:.1f} dB".format(heart_bpm, heart_quality_db))
    print("")
    print("Saved results:")
    for p in paths.values():
        if p.exists():
            print("  {}".format(p))

    return 0


def build_arg_parser():
    p = argparse.ArgumentParser(
        description="Extract breathing and heart-rate estimates from parsed mmWave ADC data."
    )
    p.add_argument("--project-root", default=None, help="Project root folder. Default: parent of Data_Procesing.")
    p.add_argument("--parsed", default=None, help="Path to *_parsed.npz. Default: latest in Preliminary_Results.")
    p.add_argument("--frame-start", type=int, default=0, help="First frame to analyze.")
    p.add_argument("--frame-count", type=int, default=None, help="Number of frames to analyze. Default: all available.")
    p.add_argument("--fs", type=float, default=None, help="Override vital-sign sample rate in Hz.")
    p.add_argument("--range-fft-size", type=int, default=512, help="Range FFT size.")
    p.add_argument("--min-range-m", type=float, default=0.4, help="Minimum chest search range in meters.")
    p.add_argument("--max-range-m", type=float, default=2.0, help="Maximum chest search range in meters.")
    p.add_argument("--min-range-bin", type=int, default=3, help="Ignore bins below this index.")
    p.add_argument("--max-range-bin", type=int, default=None, help="Ignore bins at/above this index.")
    p.add_argument("--rx-mode", default="strongest", choices=["strongest", "rx0", "sum"], help="RX selection mode.")
    p.add_argument("--drift-window-s", type=float, default=2.0, help="Moving-average drift removal window in seconds.")
    p.add_argument("--breath-low-hz", type=float, default=0.10, help="Breathing band lower cutoff.")
    p.add_argument("--breath-high-hz", type=float, default=0.60, help="Breathing band upper cutoff.")
    p.add_argument("--heart-low-hz", type=float, default=0.80, help="Heart band lower cutoff.")
    p.add_argument("--heart-high-hz", type=float, default=2.00, help="Heart band upper cutoff.")
    p.add_argument("--make-plots", action="store_true", help="Save PNG plots.")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    return analyze(args)


if __name__ == "__main__":
    raise SystemExit(main())
