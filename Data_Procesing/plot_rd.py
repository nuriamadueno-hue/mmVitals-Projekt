#!/usr/bin/env python3
"""Plot the latest range-Doppler map from Preliminary_Results."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "Preliminary_Results"


def latest_file(pattern: str) -> Path:
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {RESULTS_DIR}")
    return files[0]


def main() -> None:
    rd_path = latest_file("*_rd.npy")
    stem = rd_path.name[:-len("_rd.npy")]
    range_axis_path = RESULTS_DIR / f"{stem}_range_axis_m.npy"
    doppler_axis_path = RESULTS_DIR / f"{stem}_doppler_axis_mps.npy"

    rd = np.load(rd_path)
    rd_db = 20 * np.log10(rd + 1e-6)

    extent = None
    x_label = "Range bin"
    y_label = "Doppler bin"
    if range_axis_path.exists() and doppler_axis_path.exists():
        r = np.load(range_axis_path)
        d = np.load(doppler_axis_path)
        if len(r) == rd.shape[1] and len(d) == rd.shape[0]:
            extent = [float(r[0]), float(r[-1]), float(d[0]), float(d[-1])]
            x_label = "Range [m]"
            y_label = "Velocity [m/s]"

    plt.figure()
    plt.imshow(rd_db, aspect="auto", origin="lower", extent=extent)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.colorbar(label="Magnitude [dB]")
    plt.title(f"Range-Doppler Map: {stem}")
    out_png = RESULTS_DIR / f"{stem}_range_doppler.png"
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
