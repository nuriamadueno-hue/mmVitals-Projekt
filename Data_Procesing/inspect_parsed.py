#!/usr/bin/env python3
"""Inspect the latest parsed .npz result in Preliminary_Results."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "Preliminary_Results"


def latest_file(pattern: str) -> Path:
    files = sorted(RESULTS_DIR.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No files matching {pattern!r} found in {RESULTS_DIR}")
    return files[0]


def main() -> None:
    parsed_path = latest_file("*_parsed.npz")
    data = np.load(parsed_path, allow_pickle=False)
    cube = data["cube_chirps"]
    frames = data["cube_frames"]
    metadata = json.loads(str(data["metadata"]))

    print(f"Parsed file: {parsed_path}")
    print(f"cube_chirps shape: {cube.shape} = (chirps, RX, ADC samples)")
    print(f"cube_frames shape: {frames.shape} = (frames, chirps/frame, RX, ADC samples)")
    print(f"dtype: {cube.dtype}")
    print(f"first sample: {cube[0, 0, 0] if cube.size else 'empty'}")
    print("\nKey metadata:")
    for key in [
        "file_bytes",
        "words_per_chirp_for_file",
        "num_chirps_in_file",
        "num_complete_configured_frames",
        "num_partial_chirps",
    ]:
        print(f"  {key}: {metadata.get(key)}")


if __name__ == "__main__":
    main()
