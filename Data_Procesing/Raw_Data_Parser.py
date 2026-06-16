#!/usr/bin/env python3
"""
Project-layout aware parser for TI IWR1843 DCA1000 ADC data captured with mmWave Studio.

This version is adjusted for this project structure:

    mmWave_Studio/
        ADC_Recorded_Data/
            adc_data_Test.bin
            ... logs ...
        Data_Procesing/
            iwr1843_dca1000_parser.py
            Preliminary_Results/
        mmWave_Configuration/
            comfig_1.xml
        Resources/

Default behavior when you run this file with no arguments:

    1. Finds the project root from the script location.
    2. Reads the ADC .bin from ../ADC_Recorded_Data/.
    3. Reads the mmWave Studio XML config from ../mmWave_Configuration/.
    4. Pulls parsing inputs from the XML automatically.
    5. Writes parsed results into ./Preliminary_Results/.

The expected input .bin is the packet-reordered ADC-only file, for example
adc_data.bin or adc_data_Test.bin. It is not intended for adc_data_RAW_0.bin with
UDP packet metadata still present.

Output cube convention:

    cube_chirps.shape = (num_chirps, num_rx, num_adc_samples)
    cube_frames.shape = (num_complete_frames, chirps_per_frame, num_rx, num_adc_samples)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


C_MPS = 299_792_458.0
SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class RadarConfig:
    num_adc_samples: int = 256
    num_adc_bits: int = 16
    num_rx: int = 4
    num_lanes: int = 2
    is_complex: bool = True
    iq_swap: int = 0

    chirps_per_frame: int = 128
    num_frames_configured: int | None = None

    start_freq_ghz: float | None = 77.0
    freq_slope_mhz_per_us: float | None = None
    dig_out_sample_rate_ksps: float | None = None
    idle_time_us: float | None = None
    ramp_end_time_us: float | None = None
    adc_start_time_us: float | None = None
    frame_periodicity_ms: float | None = None

    tx_enabled_channel_count: int | None = None
    tx_enabled_in_chirp_count: int | None = None


@dataclass
class ProjectPaths:
    project_root: Path
    data_dir: Path
    config_dir: Path
    results_dir: Path


def _to_number(value: str) -> float | int | str:
    """Parse mmWave Studio XML values, including decimal-comma values like '64,55'."""
    value = value.strip().replace(",", ".")
    try:
        number = float(value)
    except ValueError:
        return value
    if math.isfinite(number) and abs(number - round(number)) < 1e-12:
        return int(round(number))
    return number


def _read_mmwave_studio_xml(xml_path: str | Path) -> dict[str, dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    sections: dict[str, dict[str, Any]] = {}
    for section in root:
        params: dict[str, Any] = {}
        for param in section.findall("param"):
            name = param.attrib.get("name")
            value = param.attrib.get("value")
            if name is not None and value is not None:
                params[name] = _to_number(value)
        sections[section.tag] = params
    return sections


def config_from_mmwave_studio_xml(xml_path: str | Path) -> RadarConfig:
    """Extract parsing and first-pass processing parameters from mmWave Studio XML."""
    sections = _read_mmwave_studio_xml(xml_path)

    channel = sections.get("apiname_channel_cfg", {})
    adc = sections.get("apiname_adc_cfg", {})
    lane = sections.get("apiname_lvdslane_cfg", {})
    profile = sections.get("apiname_profile_cfg", {})
    frame = sections.get("apiname_frame_cfg", {})
    chirp = sections.get("apiname_chirp_cfg", {})

    num_rx = sum(int(channel.get(f"rx{i}En", 0)) for i in range(4)) or 4
    tx_enabled_channel_count = sum(int(channel.get(f"tx{i}En", 0)) for i in range(3))

    # In mmWave Studio XML the chirp TX fields in this file are named tx1Enable,
    # tx2Enable, tx3Enable. This is useful for diagnostics and angle processing later.
    tx_enabled_in_chirp_count = sum(int(chirp.get(f"tx{i}Enable", 0)) for i in range(1, 4))

    bits_val = int(adc.get("bitsVal", 2))
    bits_map = {0: 12, 1: 14, 2: 16}
    num_adc_bits = bits_map.get(bits_val, 16)

    format_val = int(adc.get("formatVal", 1))
    # In mmWave Studio, formatVal=0 is real-only. formatVal=1/2 are complex modes.
    is_complex = format_val != 0

    num_lanes = sum(int(lane.get(f"lane{i}En", 0)) for i in range(1, 5)) or 2
    num_adc_samples = int(profile.get("numAdcSamples", 256))

    frame_start = int(frame.get("fchirpStartIdx", 0))
    frame_end = int(frame.get("fchirpEndIdx", 0))
    unique_chirps_per_loop = max(0, frame_end - frame_start + 1)
    loop_count = int(frame.get("loopCount", 128))
    chirps_per_frame = max(1, unique_chirps_per_loop * loop_count)

    return RadarConfig(
        num_adc_samples=num_adc_samples,
        num_adc_bits=num_adc_bits,
        num_rx=num_rx,
        num_lanes=num_lanes,
        is_complex=is_complex,
        iq_swap=int(adc.get("IQSwap", 0)),
        chirps_per_frame=chirps_per_frame,
        num_frames_configured=int(frame["frameCount"]) if "frameCount" in frame else None,
        start_freq_ghz=float(profile["startFreqConst"]) if "startFreqConst" in profile else None,
        freq_slope_mhz_per_us=float(profile["freqSlopeConst"]) if "freqSlopeConst" in profile else None,
        dig_out_sample_rate_ksps=float(profile["digOutSampleRate"]) if "digOutSampleRate" in profile else None,
        idle_time_us=float(profile["idleTimeConst"]) if "idleTimeConst" in profile else None,
        ramp_end_time_us=float(profile["rampEndTime"]) if "rampEndTime" in profile else None,
        adc_start_time_us=float(profile["adcStartTimeConst"]) if "adcStartTimeConst" in profile else None,
        frame_periodicity_ms=float(frame["periodicity"]) if "periodicity" in frame else None,
        tx_enabled_channel_count=tx_enabled_channel_count,
        tx_enabled_in_chirp_count=tx_enabled_in_chirp_count,
    )


def _apply_adc_bit_sign_extension(raw: np.ndarray, num_adc_bits: int) -> np.ndarray:
    """Match TI's MATLAB handling for 12/14-bit ADC samples stored in 16-bit words."""
    if num_adc_bits == 16:
        return raw
    raw32 = raw.astype(np.int32, copy=True)
    max_positive = 2 ** (num_adc_bits - 1) - 1
    raw32[raw32 > max_positive] -= 2 ** num_adc_bits
    return raw32.astype(np.int16)


def read_iwr1843_dca1000_bin(
    bin_path: str | Path,
    cfg: RadarConfig,
    *,
    allow_truncate: bool = False,
) -> dict[str, Any]:
    """
    Read an IWR1843/xWR16xx-style DCA1000 ADC-only .bin file.

    Complex data is reconstructed with the xWR16xx/IWR6843/IWR1843 2-lane DCA1000
    packing: two I words followed by two Q words:

        I0, I1, Q0, Q1, I2, I3, Q2, Q3, ...
    """
    bin_path = Path(bin_path)
    raw = np.fromfile(bin_path, dtype=np.int16)
    raw = _apply_adc_bit_sign_extension(raw, cfg.num_adc_bits)

    words_per_sample = 2 if cfg.is_complex else 1
    words_per_chirp = cfg.num_rx * cfg.num_adc_samples * words_per_sample

    if raw.size < words_per_chirp:
        raise ValueError(
            f"File is too small for one chirp: {raw.size} int16 words available, "
            f"{words_per_chirp} required for one chirp."
        )

    leftover_words = raw.size % words_per_chirp
    if leftover_words:
        msg = (
            "File size is not an integer number of chirps for the selected config. "
            f"raw.size={raw.size}, words_per_chirp={words_per_chirp}, "
            f"leftover_words={leftover_words}."
        )
        if not allow_truncate:
            raise ValueError(msg + " Re-run with --allow-truncate or check the config.")
        raw = raw[: raw.size - leftover_words]

    num_chirps = raw.size // words_per_chirp

    if cfg.is_complex:
        if raw.size % 4:
            raise ValueError(
                "Complex IWR1843 DCA1000 packing should be divisible by 4 int16 words "
                "because it is packed as I0, I1, Q0, Q1."
            )

        iq = np.empty(raw.size // 2, dtype=np.complex64)
        if int(cfg.iq_swap) == 0:
            iq[0::2] = raw[0::4].astype(np.float32) + 1j * raw[2::4].astype(np.float32)
            iq[1::2] = raw[1::4].astype(np.float32) + 1j * raw[3::4].astype(np.float32)
        else:
            # If Studio was configured for Q-first, swap I/Q during reconstruction.
            iq[0::2] = raw[2::4].astype(np.float32) + 1j * raw[0::4].astype(np.float32)
            iq[1::2] = raw[3::4].astype(np.float32) + 1j * raw[1::4].astype(np.float32)
        cube_chirps = iq.reshape(num_chirps, cfg.num_rx, cfg.num_adc_samples)
    else:
        cube_chirps = raw.reshape(num_chirps, cfg.num_rx, cfg.num_adc_samples)

    num_complete_frames = num_chirps // cfg.chirps_per_frame if cfg.chirps_per_frame else 0
    num_partial_chirps = num_chirps % cfg.chirps_per_frame if cfg.chirps_per_frame else num_chirps

    if num_complete_frames:
        usable_chirps = num_complete_frames * cfg.chirps_per_frame
        cube_frames = cube_chirps[:usable_chirps].reshape(
            num_complete_frames,
            cfg.chirps_per_frame,
            cfg.num_rx,
            cfg.num_adc_samples,
        )
        partial_chirps = cube_chirps[usable_chirps:]
    else:
        cube_frames = np.empty(
            (0, cfg.chirps_per_frame, cfg.num_rx, cfg.num_adc_samples),
            dtype=cube_chirps.dtype,
        )
        partial_chirps = cube_chirps

    metadata = {
        "bin_path": str(bin_path.resolve()),
        "file_bytes": bin_path.stat().st_size,
        "raw_int16_words": int(raw.size),
        "words_per_chirp": int(words_per_chirp),
        "num_chirps_in_file": int(num_chirps),
        "num_complete_configured_frames": int(num_complete_frames),
        "num_partial_chirps": int(num_partial_chirps),
        "cube_chirps_shape": [int(v) for v in cube_chirps.shape],
        "cube_frames_shape": [int(v) for v in cube_frames.shape],
        "config": asdict(cfg),
    }

    return {
        "raw_int16": raw,
        "cube_chirps": cube_chirps,
        "cube_frames": cube_frames,
        "partial_chirps": partial_chirps,
        "metadata": metadata,
    }


def range_axis_m(cfg: RadarConfig, n_fft: int | None = None) -> np.ndarray | None:
    if cfg.dig_out_sample_rate_ksps is None or cfg.freq_slope_mhz_per_us is None:
        return None
    n_fft = n_fft or cfg.num_adc_samples
    fs_hz = cfg.dig_out_sample_rate_ksps * 1e3
    slope_hz_per_s = cfg.freq_slope_mhz_per_us * 1e12
    return np.arange(n_fft) * fs_hz * C_MPS / (2.0 * slope_hz_per_s * n_fft)


def doppler_axis_mps(cfg: RadarConfig, num_chirps: int, n_fft: int | None = None) -> np.ndarray | None:
    if cfg.start_freq_ghz is None or cfg.idle_time_us is None or cfg.ramp_end_time_us is None:
        return None
    n_fft = n_fft or num_chirps
    wavelength_m = C_MPS / (cfg.start_freq_ghz * 1e9)
    chirp_period_s = (cfg.idle_time_us + cfg.ramp_end_time_us) * 1e-6
    return np.fft.fftshift(np.fft.fftfreq(n_fft, d=chirp_period_s)) * wavelength_m / 2.0


def make_range_doppler_map(
    cube_chirps: np.ndarray,
    *,
    n_range_fft: int | None = None,
    n_doppler_fft: int | None = None,
    remove_dc: bool = True,
) -> np.ndarray:
    """
    Build a simple RX-summed range-Doppler magnitude map.

    Input shape:  (num_chirps, num_rx, num_adc_samples)
    Output shape: (n_doppler_fft, n_range_fft)
    """
    x = cube_chirps.astype(np.complex64, copy=False)
    n_chirps, _, n_samples = x.shape
    n_range_fft = n_range_fft or n_samples
    n_doppler_fft = n_doppler_fft or n_chirps

    if remove_dc:
        x = x - np.mean(x, axis=2, keepdims=True)

    range_win = np.hanning(n_samples).astype(np.float32)
    doppler_win = np.hanning(n_chirps).astype(np.float32)

    range_fft_data = np.fft.fft(x * range_win[None, None, :], n=n_range_fft, axis=2)
    range_fft_data = range_fft_data * doppler_win[:, None, None]
    rd = np.fft.fftshift(np.fft.fft(range_fft_data, n=n_doppler_fft, axis=0), axes=0)
    return np.sum(np.abs(rd), axis=1)


def summarize_array(name: str, x: np.ndarray) -> dict[str, Any]:
    if np.iscomplexobj(x):
        mag = np.abs(x)
        return {
            f"{name}_shape": [int(v) for v in x.shape],
            f"{name}_dtype": str(x.dtype),
            f"{name}_abs_min": float(np.min(mag)),
            f"{name}_abs_max": float(np.max(mag)),
            f"{name}_abs_mean": float(np.mean(mag)),
            f"{name}_abs_std": float(np.std(mag)),
        }
    return {
        f"{name}_shape": [int(v) for v in x.shape],
        f"{name}_dtype": str(x.dtype),
        f"{name}_min": float(np.min(x)),
        f"{name}_max": float(np.max(x)),
        f"{name}_mean": float(np.mean(x)),
        f"{name}_std": float(np.std(x)),
    }


def _iter_matching_files(base: Path, patterns: Iterable[str], *, recursive: bool) -> list[Path]:
    files: list[Path] = []
    if not base.exists():
        return files
    for pattern in patterns:
        iterator = base.rglob(pattern) if recursive else base.glob(pattern)
        files.extend(path for path in iterator if path.is_file())
    seen: set[Path] = set()
    unique: list[Path] = []
    for file_path in files:
        resolved = file_path.resolve()
        if resolved not in seen:
            unique.append(file_path)
            seen.add(resolved)
    return unique


def _score_bin_file(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if "raw" in name:
        score += 100
    if "log" in name:
        score += 100
    if name.startswith("adc_data"):
        score -= 20
    # Prefer larger captures when there are multiple non-RAW files with similar names.
    size_score = -path.stat().st_size if path.exists() else 0
    return score, size_score, str(path)


def _score_xml_file(path: Path) -> tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if "config" in name or "comfig" in name:
        score -= 20
    if "studio" in name or "mmwave" in name:
        score -= 5
    return score, len(str(path)), str(path)


def _choose_one(candidates: list[Path], kind: str, score_func) -> Path:
    if not candidates:
        raise FileNotFoundError(f"No {kind} file found.")
    unique_by_resolved: dict[Path, Path] = {}
    for candidate in candidates:
        unique_by_resolved.setdefault(candidate.resolve(), candidate)
    ranked = sorted(unique_by_resolved.values(), key=score_func)
    best = ranked[0]
    if len(ranked) > 1 and score_func(ranked[0])[:1] == score_func(ranked[1])[:1]:
        # Do not fail here; choose the best and print the alternatives. This is convenient
        # when multiple older outputs or test captures are present in the project.
        print(f"Multiple possible {kind} files found. Using: {best}", file=sys.stderr)
        for alt in ranked[1:8]:
            print(f"  alternative: {alt}", file=sys.stderr)
    return best


def infer_project_paths(
    *,
    project_root_arg: str | None = None,
    data_dir_arg: str | None = None,
    config_dir_arg: str | None = None,
    results_dir_arg: str | None = None,
) -> ProjectPaths:
    if project_root_arg:
        project_root = Path(project_root_arg).expanduser().resolve()
    else:
        # Normal case: this script lives in mmWave_Studio/Data_Procesing/.
        candidate = SCRIPT_DIR.parent
        if (candidate / "ADC_Recorded_Data").exists() and (candidate / "mmWave_Configuration").exists():
            project_root = candidate.resolve()
        elif (Path.cwd() / "ADC_Recorded_Data").exists() and (Path.cwd() / "mmWave_Configuration").exists():
            project_root = Path.cwd().resolve()
        else:
            project_root = candidate.resolve()

    data_dir = Path(data_dir_arg).expanduser().resolve() if data_dir_arg else project_root / "ADC_Recorded_Data"
    config_dir = Path(config_dir_arg).expanduser().resolve() if config_dir_arg else project_root / "mmWave_Configuration"

    if results_dir_arg:
        results_dir = Path(results_dir_arg).expanduser().resolve()
    else:
        # Keep results beside the processing code, matching this project structure.
        if SCRIPT_DIR.name.lower() in {"data_procesing", "data_processing"}:
            results_dir = SCRIPT_DIR / "Preliminary_Results"
        else:
            results_dir = project_root / "Data_Procesing" / "Preliminary_Results"
    return ProjectPaths(
        project_root=project_root,
        data_dir=data_dir,
        config_dir=config_dir,
        results_dir=results_dir,
    )


def find_bin_file(data_dir: Path) -> Path:
    patterns = ["adc_data*.bin", "*.bin"]
    candidates = _iter_matching_files(data_dir, patterns, recursive=False)
    candidates = [p for p in candidates if "raw" not in p.name.lower() and "log" not in p.name.lower()]
    if not candidates:
        candidates = _iter_matching_files(data_dir, patterns, recursive=True)
        candidates = [p for p in candidates if "raw" not in p.name.lower() and "log" not in p.name.lower()]
    return _choose_one(candidates, ".bin", _score_bin_file)


def find_config_file(config_dir: Path, project_root: Path) -> Path:
    candidates = _iter_matching_files(config_dir, ["*.xml"], recursive=False)
    if not candidates:
        candidates = _iter_matching_files(config_dir, ["*.xml"], recursive=True)
    if not candidates:
        candidates = _iter_matching_files(project_root, ["*.xml"], recursive=True)
    return _choose_one(candidates, "XML config", _score_xml_file)


def expected_full_capture_bytes(cfg: RadarConfig) -> int | None:
    if cfg.num_frames_configured is None:
        return None
    return (
        cfg.num_frames_configured
        * cfg.chirps_per_frame
        * cfg.num_rx
        * cfg.num_adc_samples
        * (4 if cfg.is_complex else 2)
    )


def write_outputs(
    *,
    parsed: dict[str, Any],
    cfg: RadarConfig,
    bin_path: Path,
    config_path: Path,
    paths: ProjectPaths,
    n_range_fft: int | None,
    n_doppler_fft: int | None,
    skip_rd: bool,
) -> dict[str, Any]:
    paths.results_dir.mkdir(parents=True, exist_ok=True)

    stem = bin_path.stem
    parsed_npz = paths.results_dir / f"{stem}_parsed.npz"
    rd_npy = paths.results_dir / f"{stem}_rd.npy"
    range_axis_npy = paths.results_dir / f"{stem}_range_axis_m.npy"
    doppler_axis_npy = paths.results_dir / f"{stem}_doppler_axis_mps.npy"
    summary_json = paths.results_dir / f"{stem}_summary.json"
    summary_txt = paths.results_dir / f"{stem}_summary.txt"

    cube_chirps = parsed["cube_chirps"]
    cube_frames = parsed["cube_frames"]
    partial_chirps = parsed["partial_chirps"]
    metadata = parsed["metadata"]

    warnings: list[str] = []
    expected_bytes = expected_full_capture_bytes(cfg)
    if expected_bytes is not None and expected_bytes != metadata["file_bytes"]:
        warnings.append(
            "File size does not match the configured full capture size. "
            f"Configured={expected_bytes} bytes, actual={metadata['file_bytes']} bytes. "
            "The parser inferred the available chirp count from the actual file size."
        )
    if partial_chirps.shape[0]:
        warnings.append(
            f"{partial_chirps.shape[0]} chirps do not form a complete configured frame of "
            f"{cfg.chirps_per_frame} chirps."
        )
    if cfg.tx_enabled_channel_count and cfg.tx_enabled_in_chirp_count == 1:
        warnings.append(
            "Channel config enables more than one TX, but the chirp config uses one TX per chirp. "
            "This data should be treated as 1 TX x RX for angle processing unless you add more chirp configs."
        )

    np.savez_compressed(
        parsed_npz,
        cube_chirps=cube_chirps,
        cube_frames=cube_frames,
        partial_chirps=partial_chirps,
        metadata=json.dumps(metadata, indent=2),
    )

    rd_summary: dict[str, Any] | None = None
    if not skip_rd:
        doppler_block = cube_frames[0] if cube_frames.shape[0] else cube_chirps
        rd = make_range_doppler_map(
            doppler_block,
            n_range_fft=n_range_fft,
            n_doppler_fft=n_doppler_fft,
        )
        np.save(rd_npy, rd)
        rd_summary = summarize_array("range_doppler_map", rd)

        r_axis = range_axis_m(cfg, n_fft=rd.shape[1])
        d_axis = doppler_axis_mps(cfg, doppler_block.shape[0], n_fft=rd.shape[0])
        if r_axis is not None:
            np.save(range_axis_npy, r_axis)
        if d_axis is not None:
            np.save(doppler_axis_npy, d_axis)

    summary: dict[str, Any] = {
        "project_root": str(paths.project_root),
        "data_dir": str(paths.data_dir),
        "config_dir": str(paths.config_dir),
        "results_dir": str(paths.results_dir),
        "bin_file": str(bin_path),
        "config_file": str(config_path),
        "outputs": {
            "parsed_npz": str(parsed_npz),
            "range_doppler_npy": str(rd_npy) if not skip_rd else None,
            "range_axis_m_npy": str(range_axis_npy) if range_axis_npy.exists() else None,
            "doppler_axis_mps_npy": str(doppler_axis_npy) if doppler_axis_npy.exists() else None,
            "summary_json": str(summary_json),
            "summary_txt": str(summary_txt),
        },
        "metadata": metadata,
        "cube_chirps": summarize_array("cube_chirps", cube_chirps),
        "cube_frames_shape": [int(v) for v in cube_frames.shape],
        "partial_chirps_shape": [int(v) for v in partial_chirps.shape],
        "range_doppler": rd_summary,
        "warnings": warnings,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "IWR1843 DCA1000 processing summary",
        "=" * 40,
        f"Project root:   {paths.project_root}",
        f"Data folder:    {paths.data_dir}",
        f"Config folder:  {paths.config_dir}",
        f"Results folder: {paths.results_dir}",
        "",
        f"BIN file:       {bin_path}",
        f"Config file:    {config_path}",
        "",
        "Parsed cube:",
        f"  cube_chirps shape: {cube_chirps.shape} = (chirps, RX, ADC samples)",
        f"  cube_frames shape: {cube_frames.shape} = (frames, chirps/frame, RX, ADC samples)",
        f"  partial chirps:    {partial_chirps.shape[0]}",
        "",
        "Config used from XML:",
        f"  num_adc_samples:       {cfg.num_adc_samples}",
        f"  num_adc_bits:          {cfg.num_adc_bits}",
        f"  num_rx:                {cfg.num_rx}",
        f"  num_lanes:             {cfg.num_lanes}",
        f"  complex_iq:            {cfg.is_complex}",
        f"  iq_swap:               {cfg.iq_swap}",
        f"  chirps_per_frame:      {cfg.chirps_per_frame}",
        f"  configured_frames:     {cfg.num_frames_configured}",
        f"  start_freq_ghz:        {cfg.start_freq_ghz}",
        f"  slope_mhz_per_us:      {cfg.freq_slope_mhz_per_us}",
        f"  sample_rate_ksps:      {cfg.dig_out_sample_rate_ksps}",
        f"  idle_time_us:          {cfg.idle_time_us}",
        f"  ramp_end_time_us:      {cfg.ramp_end_time_us}",
        "",
        "Saved files:",
        f"  {parsed_npz}",
        f"  {summary_json}",
        f"  {summary_txt}",
    ]
    if not skip_rd:
        lines.append(f"  {rd_npy}")
        if range_axis_npy.exists():
            lines.append(f"  {range_axis_npy}")
        if doppler_axis_npy.exists():
            lines.append(f"  {doppler_axis_npy}")
    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"  - {warning}" for warning in warnings)

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    if warnings:
        print("\nWarnings were written to the summary files.", file=sys.stderr)
    return summary


def run_processing(args: argparse.Namespace) -> dict[str, Any]:
    paths = infer_project_paths(
        project_root_arg=args.project_root,
        data_dir_arg=args.data_dir,
        config_dir_arg=args.config_dir,
        results_dir_arg=args.results_dir,
    )

    bin_path = Path(args.bin).expanduser().resolve() if args.bin else find_bin_file(paths.data_dir).resolve()
    config_path = (
        Path(args.config).expanduser().resolve() if args.config else find_config_file(paths.config_dir, paths.project_root).resolve()
    )

    cfg = config_from_mmwave_studio_xml(config_path)
    parsed = read_iwr1843_dca1000_bin(bin_path, cfg, allow_truncate=args.allow_truncate)
    return write_outputs(
        parsed=parsed,
        cfg=cfg,
        bin_path=bin_path,
        config_path=config_path,
        paths=paths,
        n_range_fft=args.n_range_fft,
        n_doppler_fft=args.n_doppler_fft,
        skip_rd=args.skip_rd,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Parse IWR1843 DCA1000 data using this project's folder structure."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root. Default: inferred from script location, usually the mmWave_Studio folder.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Folder containing adc_data*.bin. Default: <project-root>/ADC_Recorded_Data.",
    )
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Folder containing the mmWave Studio XML. Default: <project-root>/mmWave_Configuration.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Output folder. Default: <project-root>/Data_Procesing/Preliminary_Results.",
    )
    parser.add_argument("--bin", default=None, help="Explicit ADC .bin file. Optional if the data folder has one file.")
    parser.add_argument("--config", default=None, help="Explicit XML config file. Optional if the config folder has one file.")
    parser.add_argument("--allow-truncate", action="store_true", help="Drop incomplete trailing int16 words.")
    parser.add_argument("--n-range-fft", type=int, default=None, help="Range FFT size. Default: numAdcSamples.")
    parser.add_argument("--n-doppler-fft", type=int, default=None, help="Doppler FFT size. Default: chirps used.")
    parser.add_argument("--skip-rd", action="store_true", help="Only parse ADC cube; do not create range-Doppler map.")

    args = parser.parse_args(argv)
    run_processing(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
