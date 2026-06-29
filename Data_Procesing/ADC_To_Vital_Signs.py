#!/usr/bin/env python3
"""
Raw_Data_Parser.py

Parser and first-pass processor for TI mmWave Studio + DCA1000 ADC data.

This version is designed for the project layout:

    mmWave_Studio/
        ADC_Recorded_Data/
            ADC_Data.bin
            ADC_Data_LogFile.txt
            ADC_Data_Raw_LogFile.csv
        mmWave_Configuration/
            Profile.csv
            optional mmWave Studio .xml
        Data_Procesing/
            Raw_Data_Parser.py
            Preliminary_Results/

Main improvements over the previous version:

* It does not depend on a stale root-level XML file.
* It prefers the log file next to the selected .bin, because that log records
  the configuration used when that .bin was captured.
* It reads Profile.csv from mmWave_Configuration when available.
* It validates file size against the selected configuration.
* It supports DCA1000 ADC-only files for:
    - IWR1843 / xWR16xx / IWR6843 style two-lane non-interleaved format
    - xWR12xx / xWR14xx style four-lane interleaved format
* It refuses unsupported captures instead of silently producing bad shapes.

Limitations:

* This script expects a packet-reordered ADC-only .bin file. Do not pass
  adc_data_RAW_0.bin unless packet reorder / zero-fill was already applied.
* TSW1400 is not supported by this script.
* Advanced-frame multi-subframe parsing is not fully implemented. The script
  uses normal FrameConfig when available and warns if AdvancedFrameConfig is seen.
* Angle estimation and vital-sign estimation are separate processing stages;
  this script only parses the raw ADC cube and creates a simple range-Doppler map.

Output cube convention:

    cube_chirps.shape = (num_chirps, num_rx, num_adc_samples)
    cube_frames.shape = (num_complete_frames, chirps_per_frame, num_rx, num_adc_samples)
"""

from __future__ import annotations
# Python 3.8 compatible: uses typing.Optional/Union/List/Dict/Tuple and no int.bit_count().

import argparse
import csv
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

C_MPS = 299_792_458.0
SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RadarConfig:
    capture_board: str = "DCA1000"
    device: Optional[str] = None
    parser_format: str = "auto"  # auto, dca1000_xwr16xx, dca1000_xwr14xx

    rx_mask: Optional[int] = None
    tx_channel_mask: Optional[int] = None
    num_rx: int = 4
    num_tx_channels_enabled: Optional[int] = None

    num_adc_samples: int = 256
    num_adc_bits: int = 16
    is_complex: bool = True
    iq_swap: int = 0
    ch_interleave: Optional[int] = None

    num_lanes: int = 2
    lane_mask: Optional[int] = None
    lane_format: Optional[int] = None
    lvds_msb_first: Optional[int] = None

    chirp_start_idx: int = 0
    chirp_end_idx: int = 0
    unique_chirps_per_loop: int = 1
    loop_count: int = 128
    chirps_per_frame: int = 128
    num_frames_configured: Optional[int] = None
    frame_periodicity_ms: Optional[float] = None

    chirp_tx_masks: Dict[int, int] = field(default_factory=dict)
    tx_enabled_in_chirp_count: Optional[int] = None

    start_freq_ghz: Optional[float] = None
    freq_slope_mhz_per_us: Optional[float] = None
    dig_out_sample_rate_ksps: Optional[float] = None
    idle_time_us: Optional[float] = None
    ramp_end_time_us: Optional[float] = None
    adc_start_time_us: Optional[float] = None
    tx_start_time_us: Optional[float] = None
    rx_gain_db: Optional[float] = None

    test_source_enabled: Optional[int] = None
    advanced_frame_seen: bool = False

    sources_used: List[str] = field(default_factory=list)
    config_warnings: List[str] = field(default_factory=list)


@dataclass
class ProjectPaths:
    project_root: Path
    data_dir: Path
    config_dir: Path
    results_dir: Path


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def _to_number(value: Optional[Union[str, int, float]]) -> Optional[Union[float, int, str]]:
    """Parse numbers from mmWave Studio files, including decimal commas."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip().replace(",", ".")
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return value
    if math.isfinite(number) and abs(number - round(number)) < 1e-12:
        return int(round(number))
    return number


def _as_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    number = _to_number(value)
    if number is None:
        return default
    try:
        return float(number)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    number = _to_number(value)
    if number is None:
        return default
    try:
        return int(round(float(number)))
    except (TypeError, ValueError):
        return default


def _popcount(mask: Optional[int]) -> int:
    """Return number of set bits in an integer mask.

    Do not use int.bit_count() here: some Windows installs used with
    mmWave Studio still run older Python versions where bit_count is
    unavailable. This implementation is compatible with Python 3.7+.
    """
    if mask is None:
        return 0
    value = int(mask)
    if value < 0:
        value = abs(value)
    return bin(value).count("1")


def _add_source(cfg: RadarConfig, source: str) -> None:
    if source not in cfg.sources_used:
        cfg.sources_used.append(source)


def _warn(cfg: RadarConfig, message: str) -> None:
    if message not in cfg.config_warnings:
        cfg.config_warnings.append(message)


# ---------------------------------------------------------------------------
# Project file discovery
# ---------------------------------------------------------------------------


def infer_project_paths(
    *,
    project_root_arg: Optional[str] = None,
    data_dir_arg: Optional[str] = None,
    config_dir_arg: Optional[str] = None,
    results_dir_arg: Optional[str] = None,
) -> ProjectPaths:
    if project_root_arg:
        project_root = Path(project_root_arg).expanduser().resolve()
    else:
        # Expected case: script is in mmWave_Studio/Data_Procesing.
        candidate = SCRIPT_DIR.parent
        if (candidate / "ADC_Recorded_Data").exists() or (candidate / "mmWave_Configuration").exists():
            project_root = candidate.resolve()
        elif (Path.cwd() / "ADC_Recorded_Data").exists() or (Path.cwd() / "mmWave_Configuration").exists():
            project_root = Path.cwd().resolve()
        else:
            project_root = candidate.resolve()

    data_dir = Path(data_dir_arg).expanduser().resolve() if data_dir_arg else project_root / "ADC_Recorded_Data"
    config_dir = Path(config_dir_arg).expanduser().resolve() if config_dir_arg else project_root / "mmWave_Configuration"

    if results_dir_arg:
        results_dir = Path(results_dir_arg).expanduser().resolve()
    else:
        if SCRIPT_DIR.name.lower() in {"data_procesing", "data_processing"}:
            results_dir = SCRIPT_DIR / "Preliminary_Results"
        else:
            results_dir = project_root / "Data_Procesing" / "Preliminary_Results"

    return ProjectPaths(project_root=project_root, data_dir=data_dir, config_dir=config_dir, results_dir=results_dir)


def _iter_matching_files(base: Path, patterns: Iterable[str], *, recursive: bool) -> List[Path]:
    if not base.exists():
        return []
    out: List[Path] = []
    for pattern in patterns:
        iterator = base.rglob(pattern) if recursive else base.glob(pattern)
        out.extend(p for p in iterator if p.is_file())
    seen: set[Path] = set()
    unique: List[Path] = []
    for path in out:
        resolved = path.resolve()
        if resolved not in seen:
            unique.append(path)
            seen.add(resolved)
    return unique


def _score_bin_file(path: Path) -> Tuple[int, int, str]:
    name = path.name.lower()
    score = 0
    if "raw" in name:
        score += 1000
    if "log" in name:
        score += 1000
    if name.startswith("adc_data") or name.startswith("adc"):
        score -= 50
    if path.parent.name.lower() == "adc_recorded_data":
        score -= 25
    # Prefer larger capture if otherwise similar.
    return score, -path.stat().st_size, str(path)


def find_bin_file(data_dir: Path) -> Path:
    patterns = ["*.bin"]
    candidates = _iter_matching_files(data_dir, patterns, recursive=False)
    candidates = [p for p in candidates if "raw" not in p.name.lower() and "log" not in p.name.lower()]
    if not candidates:
        candidates = _iter_matching_files(data_dir, patterns, recursive=True)
        candidates = [p for p in candidates if "raw" not in p.name.lower() and "log" not in p.name.lower()]
    if not candidates:
        raise FileNotFoundError(f"No ADC .bin file found in {data_dir}")
    ranked = sorted(candidates, key=_score_bin_file)
    if len(ranked) > 1:
        print(f"Multiple .bin files found. Using: {ranked[0]}", file=sys.stderr)
        for alt in ranked[1:8]:
            print(f"  alternative: {alt}", file=sys.stderr)
    return ranked[0]


def find_associated_log_file(bin_path: Path) -> Optional[Path]:
    """Find the mmWave Studio API log that belongs to a .bin file."""
    stem = bin_path.stem
    candidates = [
        bin_path.with_name(f"{stem}_LogFile.txt"),
        bin_path.with_name(f"{stem}_logfile.txt"),
        bin_path.with_name(f"{stem}_Logfile.txt"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    logs = list(bin_path.parent.glob("*_LogFile.txt")) + list(bin_path.parent.glob("*LogFile*.txt"))
    logs = [p for p in logs if "raw" not in p.name.lower()]
    if not logs:
        return None

    # Prefer matching prefix, otherwise closest modification time.
    lower_stem = stem.lower()
    prefix_matches = [p for p in logs if p.stem.lower().startswith(lower_stem)]
    if prefix_matches:
        return sorted(prefix_matches, key=lambda p: abs(p.stat().st_mtime - bin_path.stat().st_mtime))[0]
    return sorted(logs, key=lambda p: abs(p.stat().st_mtime - bin_path.stat().st_mtime))[0]


def find_associated_raw_log_file(bin_path: Path) -> Optional[Path]:
    stem = bin_path.stem
    candidates = [
        bin_path.with_name(f"{stem}_Raw_LogFile.csv"),
        bin_path.with_name(f"{stem}_raw_logfile.csv"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    logs = list(bin_path.parent.glob("*_Raw_LogFile.csv")) + list(bin_path.parent.glob("*Raw*Log*.csv"))
    if not logs:
        return None
    return sorted(logs, key=lambda p: abs(p.stat().st_mtime - bin_path.stat().st_mtime))[0]


def find_profile_csv(config_dir: Path, project_root: Path) -> Optional[Path]:
    candidates = list(config_dir.glob("*.csv")) if config_dir.exists() else []
    candidates = [p for p in candidates if "profile" in p.name.lower()]
    if candidates:
        return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]
    # Fall back to project-level search only if no config-dir profile is present.
    candidates = [p for p in project_root.rglob("*.csv") if "profile" in p.name.lower() and "result" not in str(p).lower()]
    if candidates:
        return sorted(candidates, key=lambda p: (len(str(p)), str(p)))[0]
    return None


def find_xml_config(config_dir: Path, project_root: Path, *, allow_project_root_fallback: bool) -> Optional[Path]:
    candidates = list(config_dir.glob("*.xml")) if config_dir.exists() else []
    if candidates:
        return sorted(candidates, key=lambda p: (0 if "config" in p.name.lower() or "comfig" in p.name.lower() else 1, str(p)))[0]
    if not allow_project_root_fallback:
        return None
    # Root-level XML is often a stale export. Use it only when no better source exists.
    candidates = [p for p in project_root.glob("*.xml") if p.is_file()]
    if candidates:
        return sorted(candidates, key=lambda p: (0 if "config" in p.name.lower() or "comfig" in p.name.lower() else 1, str(p)))[0]
    return None


# ---------------------------------------------------------------------------
# Config readers
# ---------------------------------------------------------------------------


def _read_mmwave_studio_xml(xml_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    root = ET.parse(xml_path).getroot()
    sections: Dict[str, Dict[str, Any]] = {}
    for section in root:
        params: Dict[str, Any] = {}
        for param in section.findall("param"):
            name = param.attrib.get("name")
            value = param.attrib.get("value")
            if name is not None:
                params[name] = _to_number(value)
        sections[section.tag] = params
    return sections


def apply_xml_config(cfg: RadarConfig, xml_path: Path) -> None:
    sections = _read_mmwave_studio_xml(xml_path)
    channel = sections.get("apiname_channel_cfg", {})
    adc = sections.get("apiname_adc_cfg", {})
    lane = sections.get("apiname_lvdslane_cfg", {})
    profile = sections.get("apiname_profile_cfg", {})
    frame = sections.get("apiname_frame_cfg", {})
    chirp = sections.get("apiname_chirp_cfg", {})
    adv = sections.get("apiname_advanceframe_cfg", {})

    rx_mask = 0
    for i in range(4):
        if _as_int(channel.get(f"rx{i}En"), 0):
            rx_mask |= 1 << i
    if rx_mask:
        cfg.rx_mask = rx_mask
        cfg.num_rx = _popcount(rx_mask)

    tx_mask = 0
    for i in range(3):
        if _as_int(channel.get(f"tx{i}En"), 0):
            tx_mask |= 1 << i
    if tx_mask:
        cfg.tx_channel_mask = tx_mask
        cfg.num_tx_channels_enabled = _popcount(tx_mask)

    bits_val = _as_int(adc.get("bitsVal"), None)
    if bits_val is not None:
        cfg.num_adc_bits = {0: 12, 1: 14, 2: 16}.get(bits_val, cfg.num_adc_bits)

    format_val = _as_int(adc.get("formatVal"), None)
    if format_val is not None:
        cfg.is_complex = format_val != 0
    if "IQSwap" in adc:
        cfg.iq_swap = _as_int(adc.get("IQSwap"), cfg.iq_swap) or 0

    lane_mask = 0
    for i in range(1, 5):
        if _as_int(lane.get(f"lane{i}En"), 0):
            lane_mask |= 1 << (i - 1)
    if lane_mask:
        cfg.lane_mask = lane_mask
        cfg.num_lanes = _popcount(lane_mask)
    if "laneFormat" in lane:
        cfg.lane_format = _as_int(lane.get("laneFormat"), cfg.lane_format)
    if "lvdsMsbFirst" in lane:
        cfg.lvds_msb_first = _as_int(lane.get("lvdsMsbFirst"), cfg.lvds_msb_first)

    if "numAdcSamples" in profile:
        cfg.num_adc_samples = _as_int(profile.get("numAdcSamples"), cfg.num_adc_samples) or cfg.num_adc_samples
    cfg.start_freq_ghz = _as_float(profile.get("startFreqConst"), cfg.start_freq_ghz)
    cfg.freq_slope_mhz_per_us = _as_float(profile.get("freqSlopeConst"), cfg.freq_slope_mhz_per_us)
    cfg.dig_out_sample_rate_ksps = _as_float(profile.get("digOutSampleRate"), cfg.dig_out_sample_rate_ksps)
    cfg.idle_time_us = _as_float(profile.get("idleTimeConst"), cfg.idle_time_us)
    cfg.ramp_end_time_us = _as_float(profile.get("rampEndTime"), cfg.ramp_end_time_us)
    cfg.adc_start_time_us = _as_float(profile.get("adcStartTimeConst"), cfg.adc_start_time_us)
    cfg.tx_start_time_us = _as_float(profile.get("txStartTime"), cfg.tx_start_time_us)
    cfg.rx_gain_db = _as_float(profile.get("rxGain"), cfg.rx_gain_db)

    if frame:
        cfg.chirp_start_idx = _as_int(frame.get("fchirpStartIdx"), cfg.chirp_start_idx) or 0
        cfg.chirp_end_idx = _as_int(frame.get("fchirpEndIdx"), cfg.chirp_end_idx) or cfg.chirp_start_idx
        cfg.unique_chirps_per_loop = max(1, cfg.chirp_end_idx - cfg.chirp_start_idx + 1)
        cfg.loop_count = _as_int(frame.get("loopCount"), cfg.loop_count) or cfg.loop_count
        cfg.chirps_per_frame = max(1, cfg.unique_chirps_per_loop * cfg.loop_count)
        if "frameCount" in frame:
            cfg.num_frames_configured = _as_int(frame.get("frameCount"), cfg.num_frames_configured)
        if "periodicity" in frame:
            cfg.frame_periodicity_ms = _as_float(frame.get("periodicity"), cfg.frame_periodicity_ms)

    if chirp:
        start = _as_int(chirp.get("chirpStartIdx"), 0) or 0
        end = _as_int(chirp.get("chirpEndIdx"), start) or start
        # XML labels are one-based in this export: tx1Enable maps to physical TX0.
        tx_mask_chirp = 0
        for one_based in range(1, 4):
            if _as_int(chirp.get(f"tx{one_based}Enable"), 0):
                tx_mask_chirp |= 1 << (one_based - 1)
        for idx in range(start, end + 1):
            cfg.chirp_tx_masks[idx] = tx_mask_chirp
        if tx_mask_chirp:
            cfg.tx_enabled_in_chirp_count = _popcount(tx_mask_chirp)

    if adv:
        cfg.advanced_frame_seen = True
        _warn(cfg, "AdvancedFrameConfig was found in XML. This script uses normal FrameConfig unless you add custom subframe handling.")

    _add_source(cfg, f"xml:{xml_path}")


def _decode_start_freq_ghz(start_freq_const: Union[int, float]) -> float:
    # mmWaveLink startFreqConst LSB: 3.6 GHz / 2^26
    return float(start_freq_const) * 3.6 / (2 ** 26)


def _decode_freq_slope_mhz_per_us(freq_slope_const: Union[int, float]) -> float:
    # mmWaveLink freqSlopeConst LSB for 77 GHz devices is about
    # 48.279 kHz/us = 0.048279 MHz/us.
    return float(freq_slope_const) * 3600.0 * 900.0 / (2 ** 26)


def parse_api_log(log_path: Path) -> Dict[str, List[List[Any]]]:
    commands: Dict[str, List[List[Any]]] = {}
    for line in log_path.read_text(errors="replace", encoding="utf-8").splitlines():
        if "API:" not in line:
            continue
        payload = line.split("API:", 1)[1].strip()
        parts = [p.strip() for p in payload.split(",")]
        if not parts or not parts[0]:
            continue
        cmd = parts[0]
        args: List[Any] = []
        for part in parts[1:]:
            if part == "":
                continue
            args.append(_to_number(part))
        commands.setdefault(cmd, []).append(args)
    return commands


def _last(commands: Dict[str, List[List[Any]]], name: str) -> Optional[List[Any]]:
    values = commands.get(name)
    if not values:
        return None
    return values[-1]


def apply_api_log_config(cfg: RadarConfig, log_path: Path) -> None:
    commands = parse_api_log(log_path)

    selected_capture = _last(commands, "select_capture_device")
    if selected_capture and selected_capture:
        cfg.capture_board = str(selected_capture[0])

    selected_chip = _last(commands, "select_chip_version")
    if selected_chip and selected_chip:
        cfg.device = str(selected_chip[0])

    channel = _last(commands, "ChannelConfig")
    if channel and len(channel) >= 2:
        tx_mask = _as_int(channel[0], None)
        rx_mask = _as_int(channel[1], None)
        if rx_mask is not None:
            cfg.rx_mask = rx_mask
            cfg.num_rx = _popcount(rx_mask) or cfg.num_rx
        if tx_mask is not None:
            cfg.tx_channel_mask = tx_mask
            cfg.num_tx_channels_enabled = _popcount(tx_mask)

    adc_out = _last(commands, "AdcOutConfig")
    if adc_out and len(adc_out) >= 2:
        bits_val = _as_int(adc_out[0], None)
        if bits_val is not None:
            cfg.num_adc_bits = {0: 12, 1: 14, 2: 16}.get(bits_val, cfg.num_adc_bits)
        fmt = _as_int(adc_out[1], None)
        if fmt is not None:
            cfg.is_complex = fmt != 0
        if len(adc_out) >= 3:
            cfg.iq_swap = _as_int(adc_out[2], cfg.iq_swap) or 0

    data_fmt = _last(commands, "DataFmtConfig")
    if data_fmt and len(data_fmt) >= 5:
        # The exact argument names differ slightly across Studio versions. Use this
        # only as a secondary source when AdcOutConfig/ChannelConfig are missing.
        if cfg.rx_mask is None:
            cfg.rx_mask = _as_int(data_fmt[0], None)
            cfg.num_rx = _popcount(cfg.rx_mask) or cfg.num_rx
        bits_val = _as_int(data_fmt[1], None)
        if bits_val is not None and bits_val in {0, 1, 2}:
            cfg.num_adc_bits = {0: 12, 1: 14, 2: 16}[bits_val]
        fmt = _as_int(data_fmt[2], None)
        if fmt is not None and fmt in {0, 1, 2}:
            cfg.is_complex = fmt != 0
        cfg.iq_swap = _as_int(data_fmt[3], cfg.iq_swap) or 0
        cfg.ch_interleave = _as_int(data_fmt[4], cfg.ch_interleave)

    lane_config = _last(commands, "LaneConfig")
    if lane_config and len(lane_config) >= 1:
        lane_mask = _as_int(lane_config[0], None)
        if lane_mask is not None:
            cfg.lane_mask = lane_mask
            cfg.num_lanes = _popcount(lane_mask) or cfg.num_lanes

    lvds_lane = _last(commands, "LvdsLaneConfig")
    if lvds_lane and len(lvds_lane) >= 1:
        cfg.lane_format = _as_int(lvds_lane[0], cfg.lane_format)
        if len(lvds_lane) >= 2:
            cfg.lvds_msb_first = _as_int(lvds_lane[1], cfg.lvds_msb_first)

    profile = _last(commands, "ProfileConfig")
    if profile and len(profile) >= 14:
        # API:ProfileConfig, profileId,startFreqConst,idleTimeConst,adcStartTimeConst,
        # rampEndTime,txOutPowerBackoffCode,txPhaseShifter,freqSlopeConst,
        # txStartTime,numAdcSamples,digOutSampleRate,hpf1,hpf2,rxGain,...
        cfg.start_freq_ghz = _decode_start_freq_ghz(float(profile[1]))
        cfg.idle_time_us = float(profile[2]) * 0.01
        cfg.adc_start_time_us = float(profile[3]) * 0.01
        cfg.ramp_end_time_us = float(profile[4]) * 0.01
        cfg.freq_slope_mhz_per_us = _decode_freq_slope_mhz_per_us(float(profile[7]))
        cfg.tx_start_time_us = float(profile[8]) * 0.01
        cfg.num_adc_samples = _as_int(profile[9], cfg.num_adc_samples) or cfg.num_adc_samples
        cfg.dig_out_sample_rate_ksps = _as_float(profile[10], cfg.dig_out_sample_rate_ksps)
        cfg.rx_gain_db = _as_float(profile[13], cfg.rx_gain_db)

    chirp_configs = commands.get("ChirpConfig", [])
    if chirp_configs:
        cfg.chirp_tx_masks.clear()
        for chirp in chirp_configs:
            if len(chirp) < 8:
                continue
            start = _as_int(chirp[0], 0) or 0
            end = _as_int(chirp[1], start) or start
            tx_mask = _as_int(chirp[7], 0) or 0
            for idx in range(start, end + 1):
                cfg.chirp_tx_masks[idx] = tx_mask
        if cfg.chirp_tx_masks:
            masks_in_frame = [cfg.chirp_tx_masks.get(i, 0) for i in range(min(cfg.chirp_tx_masks), max(cfg.chirp_tx_masks) + 1)]
            active_tx_bits = 0
            for mask in masks_in_frame:
                active_tx_bits |= mask
            cfg.tx_enabled_in_chirp_count = _popcount(active_tx_bits)

    frame = _last(commands, "FrameConfig")
    if frame and len(frame) >= 5:
        cfg.chirp_start_idx = _as_int(frame[0], cfg.chirp_start_idx) or 0
        cfg.chirp_end_idx = _as_int(frame[1], cfg.chirp_end_idx) or cfg.chirp_start_idx
        cfg.num_frames_configured = _as_int(frame[2], cfg.num_frames_configured)
        cfg.loop_count = _as_int(frame[3], cfg.loop_count) or cfg.loop_count
        cfg.unique_chirps_per_loop = max(1, cfg.chirp_end_idx - cfg.chirp_start_idx + 1)
        cfg.chirps_per_frame = max(1, cfg.unique_chirps_per_loop * cfg.loop_count)
        # Frame periodicity in mmWaveLink API logs uses 5 ns ticks.
        periodicity_ticks = _as_float(frame[4], None)
        if periodicity_ticks is not None:
            cfg.frame_periodicity_ms = periodicity_ticks * 5e-6

    if commands.get("AdvancedFrameConfig"):
        cfg.advanced_frame_seen = True
        _warn(cfg, "AdvancedFrameConfig was found in the API log. This script uses normal FrameConfig unless you add custom subframe handling.")

    test_source = _last(commands, "EnableTestSource")
    if test_source and len(test_source) >= 1:
        cfg.test_source_enabled = _as_int(test_source[0], None)

    _add_source(cfg, f"api_log:{log_path}")


def apply_profile_csv(cfg: RadarConfig, profile_csv: Path) -> None:
    text = profile_csv.read_text(errors="replace", encoding="utf-8-sig")
    # mmWave Studio profile export uses semicolon separation in this project.
    delimiter = ";" if ";" in text.splitlines()[0] else ","
    rows = list(csv.DictReader(text.splitlines(), delimiter=delimiter))
    if not rows:
        return

    target_profile_id = 0
    # Use first profile for now. This matches the common single-profile Studio export.
    row = rows[0]

    def get_by_substring(*needles: str) -> Optional[str]:
        for key, value in row.items():
            normalized = key.strip().lower().replace(" ", "")
            if all(needle.lower().replace(" ", "") in normalized for needle in needles):
                return value
        return None

    cfg.start_freq_ghz = _as_float(get_by_substring("start", "freq"), cfg.start_freq_ghz)
    cfg.freq_slope_mhz_per_us = _as_float(get_by_substring("frequency", "slope"), cfg.freq_slope_mhz_per_us)
    cfg.idle_time_us = _as_float(get_by_substring("idle", "time"), cfg.idle_time_us)
    cfg.tx_start_time_us = _as_float(get_by_substring("tx", "start", "time"), cfg.tx_start_time_us)
    cfg.adc_start_time_us = _as_float(get_by_substring("adc", "start", "time"), cfg.adc_start_time_us)
    cfg.num_adc_samples = _as_int(get_by_substring("adc", "samples"), cfg.num_adc_samples) or cfg.num_adc_samples
    cfg.dig_out_sample_rate_ksps = _as_float(get_by_substring("sample", "rate"), cfg.dig_out_sample_rate_ksps)
    cfg.ramp_end_time_us = _as_float(get_by_substring("ramp", "end", "time"), cfg.ramp_end_time_us)
    cfg.rx_gain_db = _as_float(get_by_substring("rx", "gain"), cfg.rx_gain_db)

    _add_source(cfg, f"profile_csv:{profile_csv}")


def apply_raw_log_diagnostics(cfg: RadarConfig, raw_log_csv: Path) -> Dict[str, Any]:
    diagnostics: Dict[str, Any] = {"raw_log_file": str(raw_log_csv)}
    text = raw_log_csv.read_text(errors="replace", encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip().strip(",")
        if not stripped:
            continue
        if "LVDS lane mode" in stripped:
            match = re.search(r"(\d+)\s*lane", stripped, flags=re.IGNORECASE)
            if match:
                raw_num_lanes = int(match.group(1))
                diagnostics["raw_log_num_lanes"] = raw_num_lanes
                if raw_num_lanes != cfg.num_lanes:
                    _warn(cfg, f"Raw capture log says {raw_num_lanes} LVDS lanes, config says {cfg.num_lanes} lanes.")
        elif "Out of sequence count" in stripped:
            match = re.search(r"-\s*(\d+)", stripped)
            if match:
                diagnostics["out_of_sequence_count"] = int(match.group(1))
        elif "Number of zero filled packets" in stripped:
            match = re.search(r"-\s*(\d+)", stripped)
            if match:
                diagnostics["zero_filled_packets"] = int(match.group(1))
        elif "Number of zero filled bytes" in stripped:
            match = re.search(r"-\s*(\d+)", stripped)
            if match:
                diagnostics["zero_filled_bytes"] = int(match.group(1))
        elif "Number of received packets" in stripped:
            match = re.search(r"-\s*(\d+)", stripped)
            if match:
                diagnostics["received_packets"] = int(match.group(1))
    _add_source(cfg, f"raw_log:{raw_log_csv}")
    return diagnostics


def build_config_from_available_sources(
    *,
    bin_path: Path,
    paths: ProjectPaths,
    explicit_config: Optional[Path] = None,
    explicit_log: Optional[Path] = None,
    explicit_profile_csv: Optional[Path] = None,
    explicit_raw_log: Optional[Path] = None,
    allow_root_xml_fallback: bool = False,
) -> Tuple[RadarConfig, Dict[str, Any]]:
    cfg = RadarConfig()
    diagnostics: Dict[str, Any] = {}

    api_log = explicit_log or find_associated_log_file(bin_path)
    raw_log = explicit_raw_log or find_associated_raw_log_file(bin_path)
    # If the caller explicitly provides an XML, treat it as the authoritative
    # profile source unless a Profile.csv is also explicitly provided. This avoids
    # accidentally overriding an explicit XML with an unrelated project-level CSV.
    if explicit_profile_csv is not None:
        profile_csv = explicit_profile_csv
    elif explicit_config is not None:
        profile_csv = None
    else:
        profile_csv = find_profile_csv(paths.config_dir, paths.project_root)

    # Prefer XML in mmWave_Configuration when present. Do not automatically use a
    # root-level XML if a matching API log or Profile.csv exists, because stale XML
    # files caused wrong axes and wrong expected frame counts in this project.
    xml_path = explicit_config
    if xml_path is None:
        xml_path = find_xml_config(
            paths.config_dir,
            paths.project_root,
            allow_project_root_fallback=allow_root_xml_fallback and api_log is None and profile_csv is None,
        )

    if xml_path is not None:
        apply_xml_config(cfg, xml_path)
        diagnostics["xml_config"] = str(xml_path)

    if api_log is not None:
        apply_api_log_config(cfg, api_log)
        diagnostics["api_log"] = str(api_log)

    if profile_csv is not None:
        # Profile.csv is human-readable and should match the selected Studio profile.
        # It overrides profile fields decoded from the API log.
        apply_profile_csv(cfg, profile_csv)
        diagnostics["profile_csv"] = str(profile_csv)

    if raw_log is not None:
        diagnostics["raw_capture_log"] = apply_raw_log_diagnostics(cfg, raw_log)

    infer_parser_format(cfg)
    validate_config(cfg)
    return cfg, diagnostics


def infer_parser_format(cfg: RadarConfig) -> None:
    if cfg.capture_board.upper() != "DCA1000":
        cfg.parser_format = "unsupported"
        _warn(cfg, f"Capture board {cfg.capture_board!r} is not supported by this parser. Only DCA1000 ADC-only files are supported.")
        return

    device = (cfg.device or "").lower()
    if any(token in device for token in ["1642", "1843", "6843", "16xx", "18xx"]):
        cfg.parser_format = "dca1000_xwr16xx"
    elif any(token in device for token in ["124", "144", "12xx", "14xx"]):
        cfg.parser_format = "dca1000_xwr14xx"
    else:
        # For IWR1843 through DCA1000, mmWave Studio often appears as AR1642.
        # If the capture has exactly two LVDS lanes, the xWR16xx parser is the safest default.
        if cfg.num_lanes == 2:
            cfg.parser_format = "dca1000_xwr16xx"
            _warn(cfg, "Device type was not recognized; using two-lane xWR16xx/IWR1843 DCA1000 format because num_lanes=2.")
        elif cfg.num_lanes == 4:
            cfg.parser_format = "dca1000_xwr14xx"
            _warn(cfg, "Device type was not recognized; using four-lane xWR12xx/xWR14xx DCA1000 format because num_lanes=4.")
        else:
            cfg.parser_format = "unsupported"
            _warn(cfg, f"Could not infer DCA1000 parser format from device={cfg.device!r}, num_lanes={cfg.num_lanes}.")


def validate_config(cfg: RadarConfig) -> None:
    if cfg.parser_format == "dca1000_xwr16xx":
        if cfg.num_lanes != 2:
            _warn(cfg, f"xWR16xx/IWR1843 DCA1000 format normally uses 2 LVDS lanes; config says {cfg.num_lanes}.")
        if cfg.num_rx not in {1, 2, 4}:
            _warn(cfg, f"xWR16xx/IWR1843 DCA1000 capture normally supports 1, 2, or 4 RX; config says {cfg.num_rx}.")
    elif cfg.parser_format == "dca1000_xwr14xx":
        if cfg.num_lanes != 4:
            _warn(cfg, f"xWR12xx/xWR14xx DCA1000 format normally uses 4 LVDS lanes; config says {cfg.num_lanes}.")

    if cfg.num_adc_samples <= 0:
        raise ValueError("num_adc_samples must be positive.")
    if cfg.num_rx <= 0:
        raise ValueError("num_rx must be positive.")
    if cfg.chirps_per_frame <= 0:
        raise ValueError("chirps_per_frame must be positive.")
    if cfg.num_adc_bits not in {12, 14, 16}:
        _warn(cfg, f"Unusual ADC bit depth: {cfg.num_adc_bits}. Sign extension may need verification.")


# ---------------------------------------------------------------------------
# ADC .bin parsing
# ---------------------------------------------------------------------------


def _apply_adc_bit_sign_extension(raw: np.ndarray, num_adc_bits: int) -> np.ndarray:
    """Match TI MATLAB handling for 12/14-bit ADC samples stored in 16-bit words."""
    if num_adc_bits == 16:
        return raw
    raw32 = raw.astype(np.int32, copy=True)
    max_positive = 2 ** (num_adc_bits - 1) - 1
    raw32[raw32 > max_positive] -= 2 ** num_adc_bits
    return raw32.astype(np.int16)


def _reconstruct_complex_xwr16xx(raw: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    if raw.size % 4:
        raise ValueError("xWR16xx/IWR1843 complex DCA1000 data must be divisible by 4 int16 words.")
    iq = np.empty(raw.size // 2, dtype=np.complex64)
    # TI format: I0, I1, Q0, Q1, I2, I3, Q2, Q3, ...
    # In normal Studio I-first mode, iq_swap=0 in the user's captures. If a capture
    # was configured Q-first, --force-iq-swap can be used or iq_swap from config can apply.
    if int(cfg.iq_swap) == 0:
        iq[0::2] = raw[0::4].astype(np.float32) + 1j * raw[2::4].astype(np.float32)
        iq[1::2] = raw[1::4].astype(np.float32) + 1j * raw[3::4].astype(np.float32)
    else:
        iq[0::2] = raw[2::4].astype(np.float32) + 1j * raw[0::4].astype(np.float32)
        iq[1::2] = raw[3::4].astype(np.float32) + 1j * raw[1::4].astype(np.float32)
    return iq


def _reconstruct_complex_xwr14xx(raw: np.ndarray, cfg: RadarConfig) -> np.ndarray:
    num_lanes = 4
    if raw.size % (num_lanes * 2):
        raise ValueError("xWR14xx complex DCA1000 data must be divisible by 8 int16 words.")
    data = raw.reshape(-1, num_lanes * 2).T
    if int(cfg.iq_swap) == 0:
        adc = data[0:4, :].astype(np.float32) + 1j * data[4:8, :].astype(np.float32)
    else:
        adc = data[4:8, :].astype(np.float32) + 1j * data[0:4, :].astype(np.float32)
    # Return rows as RX/lane and columns as sample index across all chirps,
    # matching TI's MATLAB retVal convention.
    return adc


def read_dca1000_adc_bin(
    bin_path: Union[str, Path],
    cfg: RadarConfig,
    *,
    allow_truncate: bool = False,
    force_iq_swap: Optional[int] = None,
) -> Dict[str, Any]:
    bin_path = Path(bin_path)
    if cfg.parser_format == "unsupported":
        raise ValueError("Unsupported capture format. See config_warnings in the summary.")

    if force_iq_swap is not None:
        cfg.iq_swap = int(force_iq_swap)

    raw = np.fromfile(bin_path, dtype=np.int16)
    raw = _apply_adc_bit_sign_extension(raw, cfg.num_adc_bits)

    words_per_sample = 2 if cfg.is_complex else 1
    words_per_chirp = cfg.num_rx * cfg.num_adc_samples * words_per_sample
    if cfg.parser_format == "dca1000_xwr14xx":
        # xWR14xx DCA1000 files always have four LVDS lanes. Disabled lanes may be zero-filled.
        parse_rx = 4
        words_per_chirp_for_file = parse_rx * cfg.num_adc_samples * words_per_sample
    else:
        parse_rx = cfg.num_rx
        words_per_chirp_for_file = words_per_chirp

    if raw.size < words_per_chirp_for_file:
        raise ValueError(
            f"File is too small for one chirp: {raw.size} int16 words available, "
            f"{words_per_chirp_for_file} required for one chirp."
        )

    leftover_words = raw.size % words_per_chirp_for_file
    if leftover_words:
        msg = (
            "File size is not an integer number of chirps for the selected config. "
            f"raw.size={raw.size}, words_per_chirp={words_per_chirp_for_file}, leftover_words={leftover_words}."
        )
        if not allow_truncate:
            raise ValueError(msg + " Re-run with --allow-truncate only if you intentionally want to drop trailing words.")
        raw = raw[: raw.size - leftover_words]

    num_chirps = raw.size // words_per_chirp_for_file

    if cfg.parser_format == "dca1000_xwr16xx":
        if cfg.is_complex:
            iq = _reconstruct_complex_xwr16xx(raw, cfg)
            cube_chirps = iq.reshape(num_chirps, cfg.num_rx, cfg.num_adc_samples)
        else:
            cube_chirps = raw.reshape(num_chirps, cfg.num_rx, cfg.num_adc_samples)

    elif cfg.parser_format == "dca1000_xwr14xx":
        if cfg.is_complex:
            adc_matrix = _reconstruct_complex_xwr14xx(raw, cfg)
            cube_all_lanes = adc_matrix.reshape(4, num_chirps, cfg.num_adc_samples).transpose(1, 0, 2)
        else:
            adc_matrix = raw.reshape(-1, 4).T
            cube_all_lanes = adc_matrix.reshape(4, num_chirps, cfg.num_adc_samples).transpose(1, 0, 2)
        # Keep the enabled RX rows in increasing RX order. If num_rx=4 this is a no-op.
        if cfg.rx_mask is not None:
            enabled = [i for i in range(4) if cfg.rx_mask & (1 << i)]
            cube_chirps = cube_all_lanes[:, enabled, :]
        else:
            cube_chirps = cube_all_lanes[:, : cfg.num_rx, :]
    else:
        raise ValueError(f"Unsupported parser_format: {cfg.parser_format}")

    num_complete_frames = num_chirps // cfg.chirps_per_frame
    num_partial_chirps = num_chirps % cfg.chirps_per_frame
    usable_chirps = num_complete_frames * cfg.chirps_per_frame
    if num_complete_frames:
        cube_frames = cube_chirps[:usable_chirps].reshape(
            num_complete_frames,
            cfg.chirps_per_frame,
            cube_chirps.shape[1],
            cfg.num_adc_samples,
        )
    else:
        cube_frames = np.empty((0, cfg.chirps_per_frame, cube_chirps.shape[1], cfg.num_adc_samples), dtype=cube_chirps.dtype)
    partial_chirps = cube_chirps[usable_chirps:]

    metadata = {
        "bin_path": str(bin_path.resolve()),
        "file_bytes": bin_path.stat().st_size,
        "raw_int16_words_after_optional_truncate": int(raw.size),
        "words_per_chirp_for_file": int(words_per_chirp_for_file),
        "words_per_chirp_enabled_rx": int(words_per_chirp),
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


# ---------------------------------------------------------------------------
# First-pass processing helpers
# ---------------------------------------------------------------------------


def range_axis_m(cfg: RadarConfig, n_fft: Optional[int] = None) -> Optional[np.ndarray]:
    if cfg.dig_out_sample_rate_ksps is None or cfg.freq_slope_mhz_per_us is None:
        return None
    n_fft = n_fft or cfg.num_adc_samples
    fs_hz = cfg.dig_out_sample_rate_ksps * 1e3
    slope_hz_per_s = cfg.freq_slope_mhz_per_us * 1e12
    return np.arange(n_fft) * fs_hz * C_MPS / (2.0 * slope_hz_per_s * n_fft)


def doppler_axis_mps(cfg: RadarConfig, num_chirps: int, n_fft: Optional[int] = None) -> Optional[np.ndarray]:
    if cfg.start_freq_ghz is None:
        return None
    n_fft = n_fft or num_chirps
    wavelength_m = C_MPS / (cfg.start_freq_ghz * 1e9)

    # Doppler FFT slow-time spacing is chirp-to-chirp, not frame-to-frame.
    if cfg.idle_time_us is not None and cfg.ramp_end_time_us is not None:
        chirp_period_s = (cfg.idle_time_us + cfg.ramp_end_time_us) * 1e-6
    else:
        return None
    return np.fft.fftshift(np.fft.fftfreq(n_fft, d=chirp_period_s)) * wavelength_m / 2.0


def vital_slow_time_rate_hz(cfg: RadarConfig) -> Optional[float]:
    """Frame-rate sampling frequency used later for vital-sign phase signals."""
    if cfg.frame_periodicity_ms is None or cfg.frame_periodicity_ms <= 0:
        return None
    return 1000.0 / cfg.frame_periodicity_ms


def make_range_doppler_map(
    cube_chirps: np.ndarray,
    *,
    n_range_fft: Optional[int] = None,
    n_doppler_fft: Optional[int] = None,
    remove_adc_dc: bool = True,
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

    if remove_adc_dc:
        x = x - np.mean(x, axis=2, keepdims=True)

    range_win = np.hanning(n_samples).astype(np.float32)
    doppler_win = np.hanning(n_chirps).astype(np.float32)

    range_fft_data = np.fft.fft(x * range_win[None, None, :], n=n_range_fft, axis=2)
    range_fft_data = range_fft_data * doppler_win[:, None, None]
    rd = np.fft.fftshift(np.fft.fft(range_fft_data, n=n_doppler_fft, axis=0), axes=0)
    return np.sum(np.abs(rd), axis=1)


def summarize_array(name: str, x: np.ndarray) -> Dict[str, Any]:
    if x.size == 0:
        return {f"{name}_shape": [int(v) for v in x.shape], f"{name}_dtype": str(x.dtype), f"{name}_empty": True}
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


def expected_full_capture_bytes(cfg: RadarConfig) -> Optional[int]:
    if cfg.num_frames_configured is None:
        return None
    return cfg.num_frames_configured * cfg.chirps_per_frame * cfg.num_rx * cfg.num_adc_samples * (4 if cfg.is_complex else 2)


def build_warnings(parsed: Dict[str, Any], cfg: RadarConfig) -> List[str]:
    warnings: List[str] = list(cfg.config_warnings)
    metadata = parsed["metadata"]
    expected_bytes = expected_full_capture_bytes(cfg)
    if expected_bytes is not None and expected_bytes != metadata["file_bytes"]:
        warnings.append(
            "File size does not match the configured full capture size. "
            f"Configured={expected_bytes} bytes, actual={metadata['file_bytes']} bytes. "
            "The parser inferred the available chirp count from the actual file size."
        )
    partial_count = int(parsed["partial_chirps"].shape[0])
    if partial_count:
        warnings.append(
            f"{partial_count} chirps do not form a complete configured frame of {cfg.chirps_per_frame} chirps."
        )
    if cfg.num_tx_channels_enabled and cfg.num_tx_channels_enabled > 1 and cfg.tx_enabled_in_chirp_count == 1:
        warnings.append(
            "Channel config enables more than one TX, but the chirp config uses one TX per chirp. "
            "Treat this as 1 TX x RX for this capture unless you add multiple chirp configs for TDM-MIMO."
        )
    if cfg.test_source_enabled:
        warnings.append("Test source appears to be enabled. Real vital-sign measurements require testSourceEn=0 / real RF capture.")
    return list(dict.fromkeys(warnings))


def write_outputs(
    *,
    parsed: Dict[str, Any],
    cfg: RadarConfig,
    bin_path: Path,
    paths: ProjectPaths,
    config_diagnostics: Dict[str, Any],
    n_range_fft: Optional[int],
    n_doppler_fft: Optional[int],
    skip_rd: bool,
) -> Dict[str, Any]:
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

    np.savez_compressed(
        parsed_npz,
        cube_chirps=cube_chirps,
        cube_frames=cube_frames,
        partial_chirps=partial_chirps,
        metadata=json.dumps(metadata, indent=2),
    )

    rd_summary: Optional[Dict[str, Any]] = None
    if not skip_rd:
        doppler_block = cube_frames[0] if cube_frames.shape[0] else cube_chirps
        rd = make_range_doppler_map(doppler_block, n_range_fft=n_range_fft, n_doppler_fft=n_doppler_fft)
        np.save(rd_npy, rd)
        rd_summary = summarize_array("range_doppler_map", rd)

        r_axis = range_axis_m(cfg, n_fft=rd.shape[1])
        d_axis = doppler_axis_mps(cfg, doppler_block.shape[0], n_fft=rd.shape[0])
        if r_axis is not None:
            np.save(range_axis_npy, r_axis)
        if d_axis is not None:
            np.save(doppler_axis_npy, d_axis)

    warnings = build_warnings(parsed, cfg)
    vfs = vital_slow_time_rate_hz(cfg)

    summary: Dict[str, Any] = {
        "project_root": str(paths.project_root),
        "data_dir": str(paths.data_dir),
        "config_dir": str(paths.config_dir),
        "results_dir": str(paths.results_dir),
        "bin_file": str(bin_path),
        "config_diagnostics": config_diagnostics,
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
        "vital_signs_later": {
            "slow_time_sample_rate_hz_from_frame_periodicity": vfs,
            "note": "For vital signs, use phase over frames at a selected range bin; do not use this chirp-rate Doppler axis as the breathing/heart-rate timebase.",
        },
        "warnings": warnings,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "mmWave Studio DCA1000 processing summary",
        "=" * 46,
        f"Project root:   {paths.project_root}",
        f"Data folder:    {paths.data_dir}",
        f"Config folder:  {paths.config_dir}",
        f"Results folder: {paths.results_dir}",
        "",
        f"BIN file:       {bin_path}",
        f"Config sources: {', '.join(cfg.sources_used) if cfg.sources_used else 'defaults only'}",
        "",
        "Parsed cube:",
        f"  cube_chirps shape: {cube_chirps.shape} = (chirps, RX, ADC samples)",
        f"  cube_frames shape: {cube_frames.shape} = (frames, chirps/frame, RX, ADC samples)",
        f"  partial chirps:    {partial_chirps.shape[0]}",
        "",
        "Config used:",
        f"  capture_board:          {cfg.capture_board}",
        f"  device:                 {cfg.device}",
        f"  parser_format:          {cfg.parser_format}",
        f"  rx_mask / num_rx:       {cfg.rx_mask} / {cfg.num_rx}",
        f"  tx_channel_mask:        {cfg.tx_channel_mask}",
        f"  chirp_tx_masks:         {cfg.chirp_tx_masks}",
        f"  num_adc_samples:        {cfg.num_adc_samples}",
        f"  num_adc_bits:           {cfg.num_adc_bits}",
        f"  complex_iq:             {cfg.is_complex}",
        f"  iq_swap:                {cfg.iq_swap}",
        f"  num_lanes:              {cfg.num_lanes}",
        f"  chirps_per_frame:       {cfg.chirps_per_frame}",
        f"  configured_frames:      {cfg.num_frames_configured}",
        f"  frame_periodicity_ms:   {cfg.frame_periodicity_ms}",
        f"  vital_slow_time_rate_hz:{vfs}",
        f"  start_freq_ghz:         {cfg.start_freq_ghz}",
        f"  slope_mhz_per_us:       {cfg.freq_slope_mhz_per_us}",
        f"  sample_rate_ksps:       {cfg.dig_out_sample_rate_ksps}",
        f"  idle_time_us:           {cfg.idle_time_us}",
        f"  ramp_end_time_us:       {cfg.ramp_end_time_us}",
        f"  adc_start_time_us:      {cfg.adc_start_time_us}",
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_processing(args: argparse.Namespace) -> Dict[str, Any]:
    paths = infer_project_paths(
        project_root_arg=args.project_root,
        data_dir_arg=args.data_dir,
        config_dir_arg=args.config_dir,
        results_dir_arg=args.results_dir,
    )

    bin_path = Path(args.bin).expanduser().resolve() if args.bin else find_bin_file(paths.data_dir).resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    log_path = Path(args.log).expanduser().resolve() if args.log else None
    profile_csv = Path(args.profile_csv).expanduser().resolve() if args.profile_csv else None
    raw_log = Path(args.raw_log).expanduser().resolve() if args.raw_log else None

    cfg, config_diagnostics = build_config_from_available_sources(
        bin_path=bin_path,
        paths=paths,
        explicit_config=config_path,
        explicit_log=log_path,
        explicit_profile_csv=profile_csv,
        explicit_raw_log=raw_log,
        allow_root_xml_fallback=args.allow_root_xml_fallback,
    )

    if args.parser_format:
        cfg.parser_format = args.parser_format
    if args.num_rx is not None:
        cfg.num_rx = args.num_rx
    if args.num_adc_samples is not None:
        cfg.num_adc_samples = args.num_adc_samples
    if args.chirps_per_frame is not None:
        cfg.chirps_per_frame = args.chirps_per_frame
    if args.real_only:
        cfg.is_complex = False
    if args.complex_iq:
        cfg.is_complex = True

    parsed = read_dca1000_adc_bin(bin_path, cfg, allow_truncate=args.allow_truncate, force_iq_swap=args.force_iq_swap)

    if args.dry_run:
        print(json.dumps({"config": asdict(cfg), "diagnostics": config_diagnostics, "metadata": parsed["metadata"]}, indent=2))
        return {"config": asdict(cfg), "diagnostics": config_diagnostics, "metadata": parsed["metadata"]}

    return write_outputs(
        parsed=parsed,
        cfg=cfg,
        bin_path=bin_path,
        paths=paths,
        config_diagnostics=config_diagnostics,
        n_range_fft=args.n_range_fft,
        n_doppler_fft=args.n_doppler_fft,
        skip_rd=args.skip_rd,
    )



# ===== Vital-sign helper functions from Vital_Signs_Analyzer.py =====
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


def _quadratic_peak_interpolation(power, peak_idx):
    """Return sub-bin peak offset using a quadratic fit around the FFT peak.

    The interpolation is performed on log power, which is usually more stable
    for narrow spectral peaks than linear power. The returned offset is in FFT
    bins and clipped to +/- 0.5 so a noisy side bin cannot move the estimate
    unrealistically far from the selected peak.
    """
    idx = int(peak_idx)
    if idx <= 0 or idx >= len(power) - 1:
        return 0.0

    y0 = math.log(float(power[idx - 1]) + 1e-30)
    y1 = math.log(float(power[idx]) + 1e-30)
    y2 = math.log(float(power[idx + 1]) + 1e-30)
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-30:
        return 0.0

    delta = 0.5 * (y0 - y2) / denom
    if not math.isfinite(delta):
        return 0.0
    return float(max(-0.5, min(0.5, delta)))


def _estimate_rate_fft(signal, fs, low_hz, high_hz, args=None):
    """Estimate a vital rate from the band-limited slow-time signal.

    The true physical resolution is still set by the analyzed duration,
    approximately fs / N. This function improves readout accuracy by using a
    zero-padded FFT and a quadratic interpolation around the spectral peak.
    """
    x = np.asarray(signal, dtype=np.float64)
    n = x.size
    if n < 8:
        return {
            "rate_per_min": float("nan"),
            "raw_bin_rate_per_min": float("nan"),
            "quality_db": float("nan"),
            "freqs_hz": np.array([]),
            "power": np.array([]),
            "n_samples": int(n),
            "n_fft": 0,
            "true_resolution_per_min": float("nan"),
            "padded_spacing_per_min": float("nan"),
            "interpolation_delta_bins": 0.0,
            "peak_bin_index": -1,
        }

    x = x - np.nanmean(x)
    x = x * _hann(n)

    pad_factor = 8.0
    min_fft = 4096
    if args is not None:
        pad_factor = max(1.0, float(getattr(args, "vital_fft_zeropad_factor", pad_factor)))
        min_fft = max(16, int(getattr(args, "vital_fft_min_size", min_fft)))

    requested = int(math.ceil(max(float(n), 256.0) * pad_factor))
    n_fft = int(2 ** math.ceil(math.log(max(requested, min_fft), 2)))

    spec = np.fft.rfft(x, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / fs)
    power = np.abs(spec) ** 2

    band = (freqs >= low_hz) & (freqs <= high_hz)
    true_resolution_per_min = 60.0 * float(fs) / float(n)
    padded_spacing_per_min = 60.0 * float(fs) / float(n_fft)

    if not np.any(band):
        return {
            "rate_per_min": float("nan"),
            "raw_bin_rate_per_min": float("nan"),
            "quality_db": float("nan"),
            "freqs_hz": freqs,
            "power": power,
            "n_samples": int(n),
            "n_fft": int(n_fft),
            "true_resolution_per_min": true_resolution_per_min,
            "padded_spacing_per_min": padded_spacing_per_min,
            "interpolation_delta_bins": 0.0,
            "peak_bin_index": -1,
        }

    idxs = np.where(band)[0]
    best = int(idxs[np.argmax(power[idxs])])
    raw_freq_hz = float(freqs[best])

    interpolate = True
    if args is not None:
        interpolate = bool(getattr(args, "vital_fft_interpolate", True))
    delta = _quadratic_peak_interpolation(power, best) if interpolate else 0.0
    interp_freq_hz = float((float(best) + delta) * float(fs) / float(n_fft))
    interp_freq_hz = max(float(low_hz), min(float(high_hz), interp_freq_hz))

    raw_rate_per_min = 60.0 * raw_freq_hz
    rate_per_min = 60.0 * interp_freq_hz

    # Simple peak quality metric: best peak relative to median band power.
    med = float(np.median(power[idxs])) if idxs.size else 0.0
    quality_db = 10.0 * math.log10((float(power[best]) + 1e-30) / (med + 1e-30))

    return {
        "rate_per_min": float(rate_per_min),
        "raw_bin_rate_per_min": float(raw_rate_per_min),
        "quality_db": float(quality_db),
        "freqs_hz": freqs,
        "power": power,
        "n_samples": int(n),
        "n_fft": int(n_fft),
        "true_resolution_per_min": float(true_resolution_per_min),
        "padded_spacing_per_min": float(padded_spacing_per_min),
        "interpolation_delta_bins": float(delta),
        "peak_bin_index": int(best),
    }


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



def _angle_grid_deg(min_deg, max_deg, step_deg):
    if step_deg <= 0:
        raise ValueError("angle step must be > 0")
    if max_deg < min_deg:
        raise ValueError("angle max must be >= angle min")
    count = int(math.floor((max_deg - min_deg) / step_deg)) + 1
    grid = min_deg + step_deg * np.arange(count, dtype=np.float64)
    if grid.size == 0 or grid[-1] < max_deg - 1e-9:
        grid = np.append(grid, float(max_deg))
    return grid


def _steering_vector_ula(num_rx, angle_deg, rx_spacing_lambda=0.5):
    """Return a simple azimuth steering vector for a uniform linear RX array.

    This assumes RX channels are ordered along the azimuth array and separated
    by rx_spacing_lambda wavelengths. For IWR1843/xWR16xx captures using only
    one TX, this gives a useful first-order azimuth AoA estimate from the 4 RX
    channels. If your hardware/channel order differs, use --invert-angle-sign
    or --rx-order to correct it.
    """
    idx = np.arange(int(num_rx), dtype=np.float64)
    theta = math.radians(float(angle_deg))
    phase = -2.0 * math.pi * float(rx_spacing_lambda) * np.sin(theta) * idx
    return np.exp(1j * phase).astype(np.complex128)


def _parse_rx_order(rx_order_text, num_rx):
    if rx_order_text is None or str(rx_order_text).strip() == "":
        return list(range(num_rx))
    parts = [x.strip() for x in str(rx_order_text).split(",") if x.strip() != ""]
    order = [int(x) for x in parts]
    if sorted(order) != list(range(num_rx)):
        raise ValueError("--rx-order must be a comma-separated permutation of 0..{}".format(num_rx - 1))
    return order


def _candidate_range_mask(range_axis, has_metric_axis, num_bins, min_range_m, max_range_m, min_bin, max_bin):
    mask = np.ones(int(num_bins), dtype=bool)
    if min_bin is not None and int(min_bin) > 0:
        mask[: int(min_bin)] = False
    if max_bin is not None and int(max_bin) < num_bins:
        mask[int(max_bin) :] = False
    if has_metric_axis:
        metric_mask = (range_axis[:num_bins] >= float(min_range_m)) & (range_axis[:num_bins] <= float(max_range_m))
        if np.any(metric_mask):
            mask &= metric_mask
    if not np.any(mask):
        raise ValueError("No valid range bins after applying range and bin limits.")
    return mask



def _compute_range_angle_map(range_fft_pos, range_axis_pos, has_metric_axis, args):
    """Compute a moving-target range-angle power map for multi-person detection.

    range_fft_pos shape: (frames, rx, positive_range_bins)
    Returns a dictionary with map power shaped (candidate_ranges, angles).
    """
    n_frames, n_rx, n_bins = range_fft_pos.shape
    rx_order = _parse_rx_order(args.rx_order, n_rx)
    x_all = range_fft_pos[:, rx_order, :]

    if args.invert_angle_sign:
        min_angle = -float(args.angle_max_deg)
        max_angle = -float(args.angle_min_deg)
    else:
        min_angle = float(args.angle_min_deg)
        max_angle = float(args.angle_max_deg)

    angle_grid_internal = _angle_grid_deg(min_angle, max_angle, float(args.angle_step_deg))
    angle_grid_reported = -angle_grid_internal if args.invert_angle_sign else angle_grid_internal

    range_mask = _candidate_range_mask(
        range_axis_pos,
        has_metric_axis,
        n_bins,
        float(args.min_range_m),
        float(args.max_range_m),
        int(args.min_range_bin),
        args.max_range_bin,
    )
    candidate_bins = np.where(range_mask)[0]

    # Mean non-static range profile for plotting and fallback diagnostics.
    range_profile = np.mean(np.abs(x_all), axis=(0, 1))

    x_scan = x_all[:, :, candidate_bins]
    if args.angle_remove_static_mean:
        # Use moving energy for detection, not static reflectors from furniture/walls.
        x_scan = x_scan - np.mean(x_scan, axis=0, keepdims=True)

    power_map = np.zeros((candidate_bins.size, angle_grid_internal.size), dtype=np.float64)
    for a_idx, angle in enumerate(angle_grid_internal):
        steer = _steering_vector_ula(n_rx, angle, float(args.rx_spacing_lambda))
        weights = np.conj(steer) / float(n_rx)
        y = np.einsum('frb,r->fb', x_scan, weights)
        power_map[:, a_idx] = np.mean(np.abs(y) ** 2, axis=0)

    return {
        "power_map": power_map,
        "candidate_bins": candidate_bins,
        "angle_grid_internal": angle_grid_internal,
        "angle_grid_reported": angle_grid_reported,
        "range_profile": range_profile,
        "rx_order": rx_order,
        "rx_spacing_lambda": float(args.rx_spacing_lambda),
    }


def _detect_range_angle_targets(ra, range_axis_pos, has_metric_axis, args):
    """Find locally separated subject candidates in a range-angle map."""
    power_map = np.asarray(ra["power_map"], dtype=np.float64)
    candidate_bins = np.asarray(ra["candidate_bins"], dtype=int)
    angle_grid_reported = np.asarray(ra["angle_grid_reported"], dtype=np.float64)
    angle_grid_internal = np.asarray(ra["angle_grid_internal"], dtype=np.float64)

    if power_map.size == 0:
        return []

    median_power = float(np.median(power_map))
    max_power = float(np.max(power_map))
    if not np.isfinite(max_power) or max_power <= 0:
        return []

    min_rel_db = float(args.target_min_relative_db)
    min_abs_quality_db = float(args.target_min_quality_db)
    max_targets = max(1, int(args.max_subjects))

    # Convert suppression distances from engineering units to index distances.
    if has_metric_axis and len(range_axis_pos) > 1:
        diffs = np.diff(range_axis_pos)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        bin_spacing_m = float(np.median(diffs)) if diffs.size else 0.1
    else:
        bin_spacing_m = 0.1
    suppress_range_bins = max(1, int(round(float(args.target_min_separation_m) / max(bin_spacing_m, 1e-9))))
    suppress_angle_bins = max(1, int(round(float(args.target_min_separation_deg) / max(float(args.angle_step_deg), 1e-9))))

    work = power_map.copy()
    targets = []
    for subject_idx in range(max_targets):
        flat_idx = int(np.argmax(work))
        peak = float(work.flat[flat_idx])
        if not np.isfinite(peak) or peak <= 0:
            break

        rel_db = 10.0 * math.log10((peak + 1e-30) / (max_power + 1e-30))
        quality_db = 10.0 * math.log10((peak + 1e-30) / (median_power + 1e-30))
        if subject_idx > 0 and rel_db < min_rel_db:
            break
        if quality_db < min_abs_quality_db:
            break

        r_local, a_idx = np.unravel_index(flat_idx, work.shape)
        rbin = int(candidate_bins[r_local])
        angle_reported = float(angle_grid_reported[a_idx])
        angle_internal = float(angle_grid_internal[a_idx])
        selected_range = float(range_axis_pos[rbin]) if has_metric_axis else None

        targets.append({
            "subject_index": int(subject_idx + 1),
            "selected_range_bin": int(rbin),
            "selected_range_m": selected_range,
            "selected_angle_deg": angle_reported,
            "internal_angle_deg": angle_internal,
            "target_power": peak,
            "target_relative_power_db": float(rel_db),
            "target_quality_db": float(quality_db),
        })

        # Non-maximum suppression around this target so adjacent bins/angles are not double-counted.
        r0 = max(0, r_local - suppress_range_bins)
        r1 = min(work.shape[0], r_local + suppress_range_bins + 1)
        a0 = max(0, a_idx - suppress_angle_bins)
        a1 = min(work.shape[1], a_idx + suppress_angle_bins + 1)
        if getattr(args, "target_suppress_same_range_all_angles", True):
            # With a 4-RX-only ULA, angular sidelobes can make one person appear as
            # several angle peaks at the same range. Suppress the full local range
            # interval by default. Use --allow-same-range-targets to disable this.
            work[r0:r1, :] = -np.inf
        else:
            work[r0:r1, a0:a1] = -np.inf

    return targets


def _beamform_slow_complex_for_target(range_fft_pos, target, rx_order, rx_spacing_lambda):
    n_rx = len(rx_order)
    rbin = int(target["selected_range_bin"])
    x_target = range_fft_pos[:, rx_order, rbin]
    steer = _steering_vector_ula(n_rx, float(target["internal_angle_deg"]), float(rx_spacing_lambda))
    return x_target @ (np.conj(steer) / float(n_rx))


def _phase_to_displacement(slow_complex, fs_vital, wavelength, args):
    phase = np.unwrap(np.angle(slow_complex))
    drift_window = max(1, int(round(float(args.drift_window_s) * fs_vital)))
    phase_detrended = phase - _moving_average(phase, drift_window)
    displacement_m = phase_detrended * wavelength / (4.0 * math.pi)
    displacement_m = displacement_m - np.mean(displacement_m)
    return displacement_m


def _extract_vitals_from_displacement(displacement_m, fs_vital, args):
    displacement_m = np.asarray(displacement_m, dtype=np.float64)
    displacement_m = displacement_m - np.nanmean(displacement_m)

    breathing = _fft_bandpass(displacement_m, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz))
    heart = _fft_bandpass(displacement_m, fs_vital, float(args.heart_low_hz), float(args.heart_high_hz))

    breath_est = _estimate_rate_fft(
        breathing, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz), args=args
    )
    heart_est = _estimate_rate_fft(
        heart, fs_vital, float(args.heart_low_hz), float(args.heart_high_hz), args=args
    )

    return {
        "displacement_m": displacement_m,
        "breathing": breathing,
        "heart": heart,
        "breathing_rate_breaths_per_min": float(breath_est["rate_per_min"]),
        "breathing_raw_bin_rate_breaths_per_min": float(breath_est["raw_bin_rate_per_min"]),
        "breathing_quality_db": float(breath_est["quality_db"]),
        "breathing_fft_n": int(breath_est["n_fft"]),
        "breathing_true_resolution_per_min": float(breath_est["true_resolution_per_min"]),
        "breathing_padded_spacing_per_min": float(breath_est["padded_spacing_per_min"]),
        "breathing_interpolation_delta_bins": float(breath_est["interpolation_delta_bins"]),
        "heart_rate_beats_per_min": float(heart_est["rate_per_min"]),
        "heart_raw_bin_rate_beats_per_min": float(heart_est["raw_bin_rate_per_min"]),
        "heart_quality_db": float(heart_est["quality_db"]),
        "heart_fft_n": int(heart_est["n_fft"]),
        "heart_true_resolution_per_min": float(heart_est["true_resolution_per_min"]),
        "heart_padded_spacing_per_min": float(heart_est["padded_spacing_per_min"]),
        "heart_interpolation_delta_bins": float(heart_est["interpolation_delta_bins"]),
    }


def _extract_vitals_from_slow_complex(slow_complex, fs_vital, wavelength, args):
    displacement_m = _phase_to_displacement(slow_complex, fs_vital, wavelength, args)
    return _extract_vitals_from_displacement(displacement_m, fs_vital, args)


def _range_bin_spacing_m(range_axis_pos, has_metric_axis):
    if has_metric_axis and len(range_axis_pos) > 1:
        diffs = np.diff(range_axis_pos)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return 0.1


def _make_subject_roi_cells(target, ra, range_axis_pos, has_metric_axis, args):
    """Return range/angle cells around the target that represent the chest ROI.

    The previous multi-subject version analyzed exactly one range-angle cell per
    subject. That can be too narrow: the chest reflection can span adjacent
    range bins and a few adjacent AoA bins. This ROI keeps cells around the
    detected peak and rejects weak local sidelobes using a local relative
    threshold.
    """
    power_map = np.asarray(ra["power_map"], dtype=np.float64)
    candidate_bins = np.asarray(ra["candidate_bins"], dtype=int)
    angles_internal = np.asarray(ra["angle_grid_internal"], dtype=np.float64)
    angles_reported = np.asarray(ra["angle_grid_reported"], dtype=np.float64)

    rbin = int(target["selected_range_bin"])
    a_internal = float(target["internal_angle_deg"])
    r_matches = np.where(candidate_bins == rbin)[0]
    if not r_matches.size:
        return []
    r_local_center = int(r_matches[0])
    a_center = int(np.argmin(np.abs(angles_internal - a_internal)))

    spacing_m = _range_bin_spacing_m(range_axis_pos, has_metric_axis)
    half_range_bins = max(0, int(round(float(args.chest_roi_range_m) / max(spacing_m, 1e-9))))
    half_range_bins = max(half_range_bins, int(args.chest_roi_min_range_bins))
    half_angle_bins = max(0, int(round(float(args.chest_roi_angle_deg) / max(float(args.angle_step_deg), 1e-9))))

    # Prefer the centre of the detected chest target while still allowing the
    # ROI to include adjacent bins. This prevents a stronger neighbouring edge
    # or multipath bin from dominating the averaged motion.
    sigma_range_bins = max(1.0, float(half_range_bins) * 0.65)
    sigma_angle_bins = max(1.0, float(half_angle_bins) * 0.65)

    r0 = max(0, r_local_center - half_range_bins)
    r1 = min(power_map.shape[0], r_local_center + half_range_bins + 1)
    a0 = max(0, a_center - half_angle_bins)
    a1 = min(power_map.shape[1], a_center + half_angle_bins + 1)

    local = power_map[r0:r1, a0:a1]
    if local.size == 0 or not np.any(np.isfinite(local)):
        return []

    local_peak = float(np.nanmax(local))
    min_power = local_peak * (10.0 ** (float(args.chest_roi_min_relative_db) / 10.0))

    cells = []
    for rr in range(r0, r1):
        for aa in range(a0, a1):
            pwr = float(power_map[rr, aa])
            if not np.isfinite(pwr) or pwr < min_power:
                continue
            dr = float(rr - r_local_center) / sigma_range_bins
            da = float(aa - a_center) / sigma_angle_bins
            distance_weight = math.exp(-0.5 * (dr * dr + da * da))
            cells.append({
                "range_local_index": int(rr),
                "angle_index": int(aa),
                "distance_weight": float(distance_weight),
                "range_bin": int(candidate_bins[rr]),
                "range_m": float(range_axis_pos[int(candidate_bins[rr])]) if has_metric_axis else None,
                "angle_internal_deg": float(angles_internal[aa]),
                "angle_reported_deg": float(angles_reported[aa]),
                "power": pwr,
            })

    # Keep strongest centre-weighted ROI cells first so the chest centre dominates over fringe cells.
    cells.sort(key=lambda c: c["power"] * c.get("distance_weight", 1.0), reverse=True)
    max_cells = max(1, int(args.chest_roi_max_cells))
    return cells[:max_cells]


def _beamform_slow_complex_for_cell(range_fft_pos, range_bin, angle_internal_deg, rx_order, rx_spacing_lambda):
    n_rx = len(rx_order)
    x_target = range_fft_pos[:, rx_order, int(range_bin)]
    steer = _steering_vector_ula(n_rx, float(angle_internal_deg), float(rx_spacing_lambda))
    return x_target @ (np.conj(steer) / float(n_rx))


def _extract_vitals_from_subject_roi(range_fft_pos, target, ra, range_axis_pos, has_metric_axis, fs_vital, wavelength, args):
    cells = _make_subject_roi_cells(target, ra, range_axis_pos, has_metric_axis, args)
    if not cells:
        slow_complex = _beamform_slow_complex_for_target(range_fft_pos, target, ra["rx_order"], ra["rx_spacing_lambda"])
        vitals = _extract_vitals_from_slow_complex(slow_complex, fs_vital, wavelength, args)
        return vitals, {
            "roi_enabled": False,
            "roi_cell_count": 1,
            "roi_range_bins": [int(target["selected_range_bin"])],
            "roi_angle_bins_deg": [float(target["selected_angle_deg"])],
            "roi_note": "fallback_single_cell",
        }

    seed_complex = _beamform_slow_complex_for_target(range_fft_pos, target, ra["rx_order"], ra["rx_spacing_lambda"])
    seed_disp = _phase_to_displacement(seed_complex, fs_vital, wavelength, args)
    seed_breath = _fft_bandpass(seed_disp, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz))

    displacements = []
    powers = []
    accepted_cells = []
    rejected_cells = []
    for cell in cells:
        slow_complex = _beamform_slow_complex_for_cell(
            range_fft_pos,
            cell["range_bin"],
            cell["angle_internal_deg"],
            ra["rx_order"],
            ra["rx_spacing_lambda"],
        )
        disp = _phase_to_displacement(slow_complex, fs_vital, wavelength, args)
        if not np.all(np.isfinite(disp)):
            continue

        breath = _fft_bandpass(disp, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz))
        denom = float(np.linalg.norm(seed_breath) * np.linalg.norm(breath))
        corr = float(np.dot(seed_breath, breath) / denom) if denom > 1e-30 else 0.0
        abs_corr = abs(corr)
        cell["breath_correlation_to_seed"] = corr

        is_seed_cell = (int(cell["range_bin"]) == int(target["selected_range_bin"]) and
                        abs(float(cell["angle_internal_deg"]) - float(target["internal_angle_deg"])) <= 0.5 * float(args.angle_step_deg) + 1e-9)
        if (not is_seed_cell) and abs_corr < float(args.chest_roi_min_breath_corr):
            rejected_cells.append(cell)
            continue

        if corr < 0.0:
            disp = -disp
        displacements.append(disp)
        powers.append(max(float(cell["power"]) * float(cell.get("distance_weight", 1.0)) * max(abs_corr, 0.25), 1e-30))
        accepted_cells.append(cell)

    if not displacements:
        slow_complex = _beamform_slow_complex_for_target(range_fft_pos, target, ra["rx_order"], ra["rx_spacing_lambda"])
        vitals = _extract_vitals_from_slow_complex(slow_complex, fs_vital, wavelength, args)
        return vitals, {"roi_enabled": False, "roi_cell_count": 1, "roi_note": "fallback_no_finite_cells"}

    disp_stack = np.vstack(displacements)

    weights = np.sqrt(np.asarray(powers, dtype=np.float64))
    weights = weights / max(float(np.sum(weights)), 1e-30)
    displacement_roi = np.sum(disp_stack * weights[:, None], axis=0)
    displacement_roi = displacement_roi - np.nanmean(displacement_roi)

    vitals = _extract_vitals_from_displacement(displacement_roi, fs_vital, args)

    unique_range_bins = sorted(set(int(c["range_bin"]) for c in accepted_cells))
    unique_angles = sorted(set(float(c["angle_reported_deg"]) for c in accepted_cells))
    range_vals = [float(range_axis_pos[b]) for b in unique_range_bins] if has_metric_axis else []
    roi_info = {
        "roi_enabled": True,
        "roi_cell_count": int(len(accepted_cells)),
        "roi_rejected_cell_count": int(len(rejected_cells)),
        "roi_range_bins": unique_range_bins,
        "roi_range_min_m": float(min(range_vals)) if range_vals else None,
        "roi_range_max_m": float(max(range_vals)) if range_vals else None,
        "roi_angle_min_deg": float(min(unique_angles)) if unique_angles else None,
        "roi_angle_max_deg": float(max(unique_angles)) if unique_angles else None,
        "roi_cell_min_relative_db": float(args.chest_roi_min_relative_db),
        "roi_cells": accepted_cells,
    }
    return vitals, roi_info


def _estimate_range_angle_target(range_fft_pos, range_axis_pos, has_metric_axis, args):
    """Backward-compatible single-target AoA selection."""
    ra = _compute_range_angle_map(range_fft_pos, range_axis_pos, has_metric_axis, args)

    if args.angle_range_mode == "selected_range":
        # Retain the older behavior: choose strongest range first, then scan angles only at that range.
        range_profile = ra["range_profile"]
        selected_bin = _select_range_bin(
            range_profile,
            range_axis_pos,
            has_metric_axis,
            float(args.min_range_m),
            float(args.max_range_m),
            int(args.min_range_bin),
            args.max_range_bin,
        )
        candidate_bins = np.asarray([selected_bin], dtype=int)
        old_candidates = ra["candidate_bins"]
        idx = np.where(old_candidates == selected_bin)[0]
        if idx.size:
            power_map = ra["power_map"][idx, :]
        else:
            # Fallback should rarely be used.
            power_map = ra["power_map"][:1, :]
        ra_single = dict(ra)
        ra_single["candidate_bins"] = candidate_bins
        ra_single["power_map"] = power_map
        targets = _detect_range_angle_targets(ra_single, range_axis_pos, has_metric_axis, args)
    elif args.angle_range_mode == "range_angle_peak":
        targets = _detect_range_angle_targets(ra, range_axis_pos, has_metric_axis, args)
    else:
        raise ValueError("Unknown angle_range_mode: {}".format(args.angle_range_mode))

    if not targets:
        raise ValueError("No range-angle target was detected. Try lowering --target-min-quality-db or widening the range limits.")

    target = targets[0]
    slow_complex = _beamform_slow_complex_for_target(range_fft_pos, target, ra["rx_order"], ra["rx_spacing_lambda"])

    # Spectrum at selected range for diagnostics.
    rbin = int(target["selected_range_bin"])
    r_local_matches = np.where(ra["candidate_bins"] == rbin)[0]
    angle_spectrum = None
    if r_local_matches.size:
        angle_spectrum = ra["power_map"][int(r_local_matches[0]), :]

    return {
        "slow_complex": slow_complex,
        "selected_bin": int(target["selected_range_bin"]),
        "selected_angle_deg": float(target["selected_angle_deg"]),
        "internal_angle_deg": float(target["internal_angle_deg"]),
        "angle_quality_db": float(target["target_quality_db"]),
        "angle_grid_deg": ra["angle_grid_reported"],
        "angle_spectrum_power": angle_spectrum,
        "range_profile": ra["range_profile"],
        "rx_order": ra["rx_order"],
        "rx_spacing_lambda": ra["rx_spacing_lambda"],
        "angle_range_mode": args.angle_range_mode,
    }


def _detect_multi_subjects(range_fft_pos, range_axis_pos, has_metric_axis, args):
    ra = _compute_range_angle_map(range_fft_pos, range_axis_pos, has_metric_axis, args)
    targets = _detect_range_angle_targets(ra, range_axis_pos, has_metric_axis, args)
    return targets, ra

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



# ---------------------------------------------------------------------------
# Combined ADC-to-vital-signs processing
# ---------------------------------------------------------------------------


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


def analyze_vital_signs_from_cube(cube_frames, metadata, args, *, input_bin_path, results_dir, stem):
    """Run the vital-sign algorithm directly from an in-memory parsed cube.

    Supports either one target or multiple range-angle separated subjects.
    cube_frames shape: (frames, chirps_per_frame, rx, adc_samples)
    metadata: parser metadata dictionary returned by read_dca1000_adc_bin(...)
    """
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

    # Average chirps within each frame: one slow-time sample per frame.
    frame_adc = cube.mean(axis=1)

    # Remove static ADC DC bias over samples for each frame/RX.
    frame_adc = frame_adc - frame_adc.mean(axis=2, keepdims=True)

    window = _hann(n_samples).astype(np.float32)
    range_fft = np.fft.fft(frame_adc * window[None, None, :], n=range_fft_size, axis=2)

    # Use only positive range bins.
    half = range_fft_size // 2
    range_fft_pos = range_fft[:, :, :half]
    range_axis_pos = range_axis[:half]

    range_profile = np.mean(np.abs(range_fft_pos), axis=(0, 1))
    time_s = np.arange(n_frames, dtype=np.float64) / fs_vital

    subjects = []
    angle_result = None
    out_paths_by_name = {}

    if args.angle_mode == "multi":
        targets, ra = _detect_multi_subjects(range_fft_pos, range_axis_pos, has_metric_axis, args)
        range_profile = ra["range_profile"]
        for target in targets:
            if bool(args.chest_roi_enable):
                vitals, roi_info = _extract_vitals_from_subject_roi(
                    range_fft_pos,
                    target,
                    ra,
                    range_axis_pos,
                    has_metric_axis,
                    fs_vital,
                    wavelength,
                    args,
                )
                rx_mode_text = "range_angle_roi_beamformed"
                angle_method_text = "4_rx_ula_range_angle_roi_beam_scan"
            else:
                slow_complex = _beamform_slow_complex_for_target(
                    range_fft_pos,
                    target,
                    ra["rx_order"],
                    ra["rx_spacing_lambda"],
                )
                vitals = _extract_vitals_from_slow_complex(slow_complex, fs_vital, wavelength, args)
                roi_info = {
                    "roi_enabled": False,
                    "roi_cell_count": 1,
                    "roi_range_bins": [int(target["selected_range_bin"])],
                    "roi_angle_bins_deg": [float(target["selected_angle_deg"])],
                }
                rx_mode_text = "range_angle_beamformed"
                angle_method_text = "4_rx_ula_range_angle_beam_scan"

            subject = dict(target)
            subject.update({
                "selected_rx": -2,
                "rx_mode": rx_mode_text,
                "angle_method": angle_method_text,
                "angle_remove_static_mean": bool(args.angle_remove_static_mean),
                "rx_order": ra["rx_order"],
                "rx_spacing_lambda": float(args.rx_spacing_lambda),
                "breathing_rate_breaths_per_min": vitals["breathing_rate_breaths_per_min"],
                "breathing_raw_bin_rate_breaths_per_min": vitals["breathing_raw_bin_rate_breaths_per_min"],
                "breathing_quality_db": vitals["breathing_quality_db"],
                "breathing_fft_n": vitals["breathing_fft_n"],
                "breathing_true_resolution_per_min": vitals["breathing_true_resolution_per_min"],
                "breathing_padded_spacing_per_min": vitals["breathing_padded_spacing_per_min"],
                "breathing_interpolation_delta_bins": vitals["breathing_interpolation_delta_bins"],
                "heart_rate_beats_per_min": vitals["heart_rate_beats_per_min"],
                "heart_raw_bin_rate_beats_per_min": vitals["heart_raw_bin_rate_beats_per_min"],
                "heart_quality_db": vitals["heart_quality_db"],
                "heart_fft_n": vitals["heart_fft_n"],
                "heart_true_resolution_per_min": vitals["heart_true_resolution_per_min"],
                "heart_padded_spacing_per_min": vitals["heart_padded_spacing_per_min"],
                "heart_interpolation_delta_bins": vitals["heart_interpolation_delta_bins"],
            })
            subject.update(roi_info)
            subjects.append(subject)

            if not args.no_save:
                spaths = _make_subject_output_paths(results_dir, stem, subject["subject_index"])
                np.save(str(spaths["disp_npy"]), vitals["displacement_m"])
                np.save(str(spaths["breath_npy"]), vitals["breathing"])
                np.save(str(spaths["heart_npy"]), vitals["heart"])
                _write_timeseries_csv(spaths["csv"], time_s, vitals["displacement_m"], vitals["breathing"], vitals["heart"])
                if args.make_plots:
                    _save_plots(
                        spaths,
                        time_s,
                        vitals["displacement_m"],
                        vitals["breathing"],
                        vitals["heart"],
                        range_axis_pos,
                        range_profile,
                        int(subject["selected_range_bin"]),
                        has_metric_axis,
                        vitals["breathing_rate_breaths_per_min"],
                        vitals["heart_rate_beats_per_min"],
                    )
                for key, value in spaths.items():
                    out_paths_by_name["subject_{:02d}_{}".format(subject["subject_index"], key)] = value

        paths = _make_output_paths(results_dir, stem)
        if not args.no_save:
            # Save global range-angle diagnostics.
            np.save(str(paths["angle_spectrum_npy"]), {
                "power_map": ra["power_map"],
                "candidate_bins": ra["candidate_bins"],
                "angle_grid_reported": ra["angle_grid_reported"],
                "rx_order": ra["rx_order"],
            })
            out_paths_by_name["range_angle_map"] = paths["angle_spectrum_npy"]

    else:
        selected_angle_deg = None
        angle_quality_db = None
        angle_method = "disabled"

        if args.angle_mode == "off":
            selected_bin = _select_range_bin(
                range_profile,
                range_axis_pos,
                has_metric_axis,
                float(args.min_range_m),
                float(args.max_range_m),
                int(args.min_range_bin),
                args.max_range_bin,
            )

            # Original behavior: extract complex slow-time signal at selected range bin.
            rx_power = np.mean(np.abs(range_fft_pos[:, :, selected_bin]) ** 2, axis=0)
            selected_rx = int(np.argmax(rx_power))

            if args.rx_mode == "strongest":
                slow_complex = range_fft_pos[:, selected_rx, selected_bin]
            elif args.rx_mode == "rx0":
                selected_rx = 0
                slow_complex = range_fft_pos[:, 0, selected_bin]
            elif args.rx_mode == "sum":
                slow_complex = np.sum(range_fft_pos[:, :, selected_bin], axis=1)
                selected_rx = -1
            else:
                raise ValueError("Unknown rx_mode: {}".format(args.rx_mode))
        elif args.angle_mode == "beamform":
            angle_result = _estimate_range_angle_target(
                range_fft_pos,
                range_axis_pos,
                has_metric_axis,
                args,
            )
            slow_complex = angle_result["slow_complex"]
            selected_bin = int(angle_result["selected_bin"])
            selected_rx = -2
            selected_angle_deg = float(angle_result["selected_angle_deg"])
            angle_quality_db = float(angle_result["angle_quality_db"])
            angle_method = "4_rx_ula_fft_beam_scan"
            range_profile = angle_result["range_profile"]
        else:
            raise ValueError("Unknown angle_mode: {}".format(args.angle_mode))

        vitals = _extract_vitals_from_slow_complex(slow_complex, fs_vital, wavelength, args)
        selected_range = float(range_axis_pos[selected_bin]) if has_metric_axis else None
        subjects.append({
            "subject_index": 1,
            "selected_range_bin": int(selected_bin),
            "selected_range_m": selected_range,
            "selected_angle_deg": selected_angle_deg,
            "target_quality_db": angle_quality_db,
            "angle_quality_db": angle_quality_db,
            "selected_rx": int(selected_rx),
            "rx_mode": args.rx_mode,
            "angle_method": angle_method,
            "breathing_rate_breaths_per_min": vitals["breathing_rate_breaths_per_min"],
            "breathing_raw_bin_rate_breaths_per_min": vitals["breathing_raw_bin_rate_breaths_per_min"],
            "breathing_quality_db": vitals["breathing_quality_db"],
            "breathing_fft_n": vitals["breathing_fft_n"],
            "breathing_true_resolution_per_min": vitals["breathing_true_resolution_per_min"],
            "breathing_padded_spacing_per_min": vitals["breathing_padded_spacing_per_min"],
            "breathing_interpolation_delta_bins": vitals["breathing_interpolation_delta_bins"],
            "heart_rate_beats_per_min": vitals["heart_rate_beats_per_min"],
            "heart_raw_bin_rate_beats_per_min": vitals["heart_raw_bin_rate_beats_per_min"],
            "heart_quality_db": vitals["heart_quality_db"],
            "heart_fft_n": vitals["heart_fft_n"],
            "heart_true_resolution_per_min": vitals["heart_true_resolution_per_min"],
            "heart_padded_spacing_per_min": vitals["heart_padded_spacing_per_min"],
            "heart_interpolation_delta_bins": vitals["heart_interpolation_delta_bins"],
        })

        paths = _make_output_paths(results_dir, stem)
        if not args.no_save:
            np.save(str(paths["disp_npy"]), vitals["displacement_m"])
            np.save(str(paths["breath_npy"]), vitals["breathing"])
            np.save(str(paths["heart_npy"]), vitals["heart"])
            if angle_result is not None and angle_result.get("angle_spectrum_power") is not None:
                angle_save = np.vstack([angle_result["angle_grid_deg"], angle_result["angle_spectrum_power"]]).T
                np.save(str(paths["angle_spectrum_npy"]), angle_save)
            _write_timeseries_csv(paths["csv"], time_s, vitals["displacement_m"], vitals["breathing"], vitals["heart"])
            if args.make_plots:
                _save_plots(
                    paths,
                    time_s,
                    vitals["displacement_m"],
                    vitals["breathing"],
                    vitals["heart"],
                    range_axis_pos,
                    range_profile,
                    int(selected_bin),
                    has_metric_axis,
                    vitals["breathing_rate_breaths_per_min"],
                    vitals["heart_rate_beats_per_min"],
                )
            out_paths_by_name.update(paths)

    results_dir.mkdir(parents=True, exist_ok=True)
    paths = _make_output_paths(results_dir, stem)

    primary = subjects[0] if subjects else {}
    summary = {
        "input_adc_bin_file": str(input_bin_path),
        "intermediate_parsed_file_used": False,
        "cube_frames_shape_total": list(cube_frames.shape),
        "analyzed_frame_start": frame_start,
        "analyzed_frame_end_exclusive": frame_end,
        "analyzed_frames": n_frames,
        "capture_duration_s": float(n_frames / fs_vital),
        "vital_sample_rate_hz": float(fs_vital),
        "range_fft_size": int(range_fft_size),
        "angle_mode": args.angle_mode,
        "angle_range_mode": args.angle_range_mode if args.angle_mode in {"beamform", "multi"} else None,
        "angle_remove_static_mean": bool(args.angle_remove_static_mean) if args.angle_mode in {"beamform", "multi"} else None,
        "rx_order": subjects[0].get("rx_order", list(range(n_rx))) if subjects else list(range(n_rx)),
        "rx_spacing_lambda": float(args.rx_spacing_lambda),
        "num_subjects_detected": int(len(subjects)),
        "subjects": subjects,
        # Backward-compatible top-level fields for existing checks.
        "selected_range_bin": primary.get("selected_range_bin"),
        "selected_range_m": primary.get("selected_range_m"),
        "selected_rx": int(primary.get("selected_rx", -999)) if primary else -999,
        "rx_mode": primary.get("rx_mode", args.rx_mode),
        "angle_method": primary.get("angle_method"),
        "selected_angle_deg": primary.get("selected_angle_deg"),
        "angle_quality_db": primary.get("angle_quality_db", primary.get("target_quality_db")),
        "wavelength_m": float(wavelength),
        "breathing_rate_breaths_per_min": primary.get("breathing_rate_breaths_per_min"),
        "breathing_quality_db": primary.get("breathing_quality_db"),
        "heart_rate_beats_per_min": primary.get("heart_rate_beats_per_min"),
        "heart_quality_db": primary.get("heart_quality_db"),
        "breathing_band_hz": [float(args.breath_low_hz), float(args.breath_high_hz)],
        "heart_band_hz": [float(args.heart_low_hz), float(args.heart_high_hz)],
        "vital_frequency_estimation": {
            "method": "zero_padded_fft_with_quadratic_peak_interpolation" if bool(args.vital_fft_interpolate) else "zero_padded_fft_raw_peak",
            "zeropad_factor": float(args.vital_fft_zeropad_factor),
            "minimum_fft_size": int(args.vital_fft_min_size),
            "true_resolution_per_min": float(60.0 * fs_vital / float(n_frames)),
            "padded_spacing_per_min": float(60.0 * fs_vital / float(subjects[0].get("breathing_fft_n", max(1, n_frames))) if subjects else float("nan")),
            "note": "True resolution is limited by capture duration. Zero padding and interpolation improve peak readout, not physical separation of very close rates.",
        },
        "target_detection": {
            "max_subjects": int(args.max_subjects),
            "target_min_relative_db": float(args.target_min_relative_db),
            "target_min_quality_db": float(args.target_min_quality_db),
            "target_min_separation_m": float(args.target_min_separation_m),
            "target_min_separation_deg": float(args.target_min_separation_deg),
            "chest_roi_enable": bool(args.chest_roi_enable),
            "chest_roi_range_m": float(args.chest_roi_range_m),
            "chest_roi_angle_deg": float(args.chest_roi_angle_deg),
            "chest_roi_min_relative_db": float(args.chest_roi_min_relative_db),
            "chest_roi_max_cells": int(args.chest_roi_max_cells),
            "chest_roi_min_breath_corr": float(args.chest_roi_min_breath_corr),
        },
        "parser_metadata": metadata,
        "note": "Engineering/prototype estimate only; not for medical diagnosis.",
    }

    if not args.no_save:
        with open(str(paths["summary_json"]), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=_json_default)
        with open(str(paths["summary_txt"]), "w", encoding="utf-8") as f:
            f.write("ADC-to-vital-signs multi-subject analysis summary\n")
            f.write("=================================================\n")
            for k, v in summary.items():
                if k == "parser_metadata":
                    f.write("parser_metadata: see JSON summary\n")
                elif k == "subjects":
                    f.write("subjects:\n")
                    for s in subjects:
                        f.write(
                            "  subject {idx}: range_bin={rbin}, range_m={rng}, angle_deg={ang}, breath_bpm={bbpm:.2f} raw={braw:.2f}, heart_bpm={hbpm:.2f} raw={hraw:.2f}\n".format(
                                idx=s.get("subject_index"),
                                rbin=s.get("selected_range_bin"),
                                rng=s.get("selected_range_m"),
                                ang=s.get("selected_angle_deg"),
                                bbpm=float(s.get("breathing_rate_breaths_per_min", float("nan"))),
                                braw=float(s.get("breathing_raw_bin_rate_breaths_per_min", float("nan"))),
                                hbpm=float(s.get("heart_rate_beats_per_min", float("nan"))),
                                hraw=float(s.get("heart_raw_bin_rate_beats_per_min", float("nan"))),
                            )
                        )
                        if s.get("breathing_true_resolution_per_min") is not None:
                            f.write("    fft: true_resolution={:.2f}/min, padded_spacing={:.3f}/min, n_fft={}\n".format(
                                float(s.get("breathing_true_resolution_per_min", float("nan"))),
                                float(s.get("breathing_padded_spacing_per_min", float("nan"))),
                                s.get("breathing_fft_n"),
                            ))
                        if s.get("roi_enabled"):
                            f.write("    roi: cells={cells}, range_bins={rbins}, angle={amin}..{amax} deg\n".format(
                                cells=s.get("roi_cell_count"),
                                rbins=s.get("roi_range_bins"),
                                amin=s.get("roi_angle_min_deg"),
                                amax=s.get("roi_angle_max_deg"),
                            ))
                else:
                    f.write("{}: {}\n".format(k, v))
        out_paths_by_name["summary_json"] = paths["summary_json"]
        out_paths_by_name["summary_txt"] = paths["summary_txt"]

    return summary, out_paths_by_name

def run_adc_to_vital_signs(args):
    paths = infer_project_paths(
        project_root_arg=args.project_root,
        data_dir_arg=args.data_dir,
        config_dir_arg=args.config_dir,
        results_dir_arg=args.results_dir,
    )

    bin_path = Path(args.bin).expanduser().resolve() if args.bin else find_bin_file(paths.data_dir).resolve()
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    log_path = Path(args.log).expanduser().resolve() if args.log else None
    profile_csv = Path(args.profile_csv).expanduser().resolve() if args.profile_csv else None
    raw_log = Path(args.raw_log).expanduser().resolve() if args.raw_log else None

    cfg, config_diagnostics = build_config_from_available_sources(
        bin_path=bin_path,
        paths=paths,
        explicit_config=config_path,
        explicit_log=log_path,
        explicit_profile_csv=profile_csv,
        explicit_raw_log=raw_log,
        allow_root_xml_fallback=args.allow_root_xml_fallback,
    )

    # Optional manual overrides, same intent as Raw_Data_Parser.py.
    if args.parser_format:
        cfg.parser_format = args.parser_format
    if args.num_rx is not None:
        cfg.num_rx = args.num_rx
    if args.num_adc_samples is not None:
        cfg.num_adc_samples = args.num_adc_samples
    if args.chirps_per_frame is not None:
        cfg.chirps_per_frame = args.chirps_per_frame
    if args.real_only:
        cfg.is_complex = False
    if args.complex_iq:
        cfg.is_complex = True

    parsed = read_dca1000_adc_bin(
        bin_path,
        cfg,
        allow_truncate=args.allow_truncate,
        force_iq_swap=args.force_iq_swap,
    )

    metadata = parsed["metadata"]
    metadata["config_diagnostics"] = config_diagnostics
    cube_frames = parsed["cube_frames"]

    results_dir = Path(args.vital_results_dir).expanduser().resolve() if args.vital_results_dir else paths.project_root / "Data_Procesing" / "Vital_Signs_Results"
    stem = args.output_stem if args.output_stem else bin_path.stem

    summary, out_paths = analyze_vital_signs_from_cube(
        cube_frames,
        metadata,
        args,
        input_bin_path=bin_path,
        results_dir=results_dir,
        stem=stem,
    )

    print("ADC-to-vital-signs processing complete")
    print("=====================================")
    print("Input ADC file:     {}".format(bin_path))
    print("Cube frames:        {}".format(tuple(cube_frames.shape)))
    print("Analyzed frames:    {}".format(summary["analyzed_frames"]))
    print("Duration:           {:.2f} s".format(summary["capture_duration_s"]))
    print("Sample rate:        {:.3f} Hz".format(summary["vital_sample_rate_hz"]))
    print("Angle mode:         {}".format(summary["angle_mode"]))
    print("Detected subjects:  {}".format(summary["num_subjects_detected"]))
    if summary.get("angle_remove_static_mean") is not None:
        print("AoA static removal: {}".format(summary["angle_remove_static_mean"]))
        print("RX order:           {}".format(summary["rx_order"]))
    print("")

    if summary["num_subjects_detected"]:
        print("Subject results")
        print("---------------")
        for subject in summary["subjects"]:
            rng = subject.get("selected_range_m")
            rng_text = "{:.3f} m".format(rng) if rng is not None else "bin units"
            ang = subject.get("selected_angle_deg")
            ang_text = "{:+.1f} deg".format(ang) if ang is not None else "n/a"
            print("Subject {:02d}:".format(subject["subject_index"]))
            print("  Range bin:        {} ({})".format(subject["selected_range_bin"], rng_text))
            print("  Angle bin:        {}".format(ang_text))
            if subject.get("roi_enabled"):
                rr = subject.get("roi_range_bins", [])
                rtxt = "{}".format(rr)
                if subject.get("roi_range_min_m") is not None:
                    rtxt += " ({:.3f}..{:.3f} m)".format(subject.get("roi_range_min_m"), subject.get("roi_range_max_m"))
                rejected = subject.get("roi_rejected_cell_count")
                rejected_txt = ", {} rejected".format(rejected) if rejected is not None else ""
                print("  Chest ROI:        {} cells{}; range bins {}, angle {:+.1f}..{:+.1f} deg".format(
                    subject.get("roi_cell_count"),
                    rejected_txt,
                    rtxt,
                    subject.get("roi_angle_min_deg"),
                    subject.get("roi_angle_max_deg"),
                ))
            if subject.get("target_relative_power_db") is not None:
                print("  Target power:     {:+.1f} dB rel, {:.1f} dB quality".format(subject["target_relative_power_db"], subject["target_quality_db"]))
            print("  Breathing rate:   {:.2f} breaths/min  quality {:.1f} dB".format(subject["breathing_rate_breaths_per_min"], subject["breathing_quality_db"]))
            if subject.get("breathing_raw_bin_rate_breaths_per_min") is not None:
                print("    raw FFT bin:    {:.2f} breaths/min; interp delta: {:+.3f} bins".format(
                    subject["breathing_raw_bin_rate_breaths_per_min"],
                    subject.get("breathing_interpolation_delta_bins", 0.0),
                ))
            print("  Heart rate:       {:.2f} beats/min     quality {:.1f} dB".format(subject["heart_rate_beats_per_min"], subject["heart_quality_db"]))
            if subject.get("heart_raw_bin_rate_beats_per_min") is not None:
                print("    raw FFT bin:    {:.2f} beats/min; interp delta: {:+.3f} bins".format(
                    subject["heart_raw_bin_rate_beats_per_min"],
                    subject.get("heart_interpolation_delta_bins", 0.0),
                ))
            if subject.get("breathing_true_resolution_per_min") is not None:
                print("  FFT resolution:   true {:.2f}/min; padded spacing {:.3f}/min; Nfft {}".format(
                    subject["breathing_true_resolution_per_min"],
                    subject["breathing_padded_spacing_per_min"],
                    subject.get("breathing_fft_n", "n/a"),
                ))
    else:
        print("No subjects detected. Try lowering --target-min-quality-db, --target-min-relative-db, or widening the range limits.")

    print("")
    if args.no_save:
        print("No result files saved because --no-save was used.")
    else:
        print("Saved result files:")
        for p in out_paths.values():
            try:
                if p.exists():
                    print("  {}".format(p))
            except AttributeError:
                pass

    return summary


def build_combined_arg_parser():
    p = argparse.ArgumentParser(
        description="Directly process TI mmWave Studio DCA1000 ADC .bin data into breathing and heart-rate estimates without parsed NPZ intermediates."
    )

    # Project / parser options.
    p.add_argument("--project-root", default=None, help="Project root. Default: inferred from script location.")
    p.add_argument("--data-dir", default=None, help="Folder containing ADC .bin files. Default: <project-root>/ADC_Recorded_Data.")
    p.add_argument("--config-dir", default=None, help="Folder containing Profile.csv or XML. Default: <project-root>/mmWave_Configuration.")
    p.add_argument("--results-dir", default=None, help="Parser results folder; only used for path inference compatibility.")
    p.add_argument("--vital-results-dir", default=None, help="Vital-sign result folder. Default: <project-root>/Data_Procesing/Vital_Signs_Results.")
    p.add_argument("--bin", default=None, help="ADC .bin file. Default: best match in data dir.")
    p.add_argument("--config", default=None, help="Explicit mmWave Studio XML config.")
    p.add_argument("--log", default=None, help="Explicit mmWave Studio API log file.")
    p.add_argument("--profile-csv", default=None, help="Explicit Profile.csv file.")
    p.add_argument("--raw-log", default=None, help="Explicit DCA1000 raw log CSV file.")
    p.add_argument("--allow-root-xml-fallback", action="store_true", help="Allow root-level XML config fallback if no log/Profile.csv is found.")
    p.add_argument("--allow-truncate", action="store_true", help="Drop trailing words if file size is not an integer number of chirps.")
    p.add_argument("--force-iq-swap", type=int, default=None, choices=[0, 1], help="Override IQ swap flag.")
    p.add_argument("--parser-format", default=None, choices=["dca1000_xwr16xx", "dca1000_xwr14xx"], help="Override parser format.")
    p.add_argument("--num-rx", type=int, default=None, help="Override number of RX channels.")
    p.add_argument("--num-adc-samples", type=int, default=None, help="Override ADC samples per chirp.")
    p.add_argument("--chirps-per-frame", type=int, default=None, help="Override chirps per frame.")
    p.add_argument("--real-only", action="store_true", help="Treat ADC data as real-only.")
    p.add_argument("--complex-iq", action="store_true", help="Treat ADC data as complex IQ.")

    # Vital-sign options.
    p.add_argument("--frame-start", type=int, default=0, help="First frame to analyze.")
    p.add_argument("--frame-count", type=int, default=None, help="Number of frames to analyze. Default: all available.")
    p.add_argument("--fs", type=float, default=None, help="Override vital-sign sample rate in Hz.")
    p.add_argument("--range-fft-size", type=int, default=512, help="Range FFT size.")
    p.add_argument("--min-range-m", type=float, default=0.4, help="Minimum chest search range in meters.")
    p.add_argument("--max-range-m", type=float, default=2.0, help="Maximum chest search range in meters.")
    p.add_argument("--min-range-bin", type=int, default=3, help="Ignore bins below this index.")
    p.add_argument("--max-range-bin", type=int, default=None, help="Ignore bins at/above this index.")
    p.add_argument("--rx-mode", default="strongest", choices=["strongest", "rx0", "sum"], help="RX selection mode used only when --angle-mode off.")
    p.add_argument("--angle-mode", default="multi", choices=["multi", "beamform", "off"], help="Use multi-subject range-angle beamforming, single-target beamforming, or old RX mode. Default: multi.")
    p.add_argument("--angle-range-mode", default="selected_range", choices=["selected_range", "range_angle_peak"], help="AoA target selection: scan angle at the strongest range bin, or search the strongest range-angle cell.")
    p.add_argument("--angle-min-deg", type=float, default=-60.0, help="Minimum azimuth angle to scan in degrees.")
    p.add_argument("--angle-max-deg", type=float, default=60.0, help="Maximum azimuth angle to scan in degrees.")
    p.add_argument("--angle-step-deg", type=float, default=1.0, help="Azimuth angle scan step in degrees.")
    p.add_argument("--rx-spacing-lambda", type=float, default=0.5, help="Assumed RX antenna spacing in wavelengths for the ULA AoA model.")
    p.add_argument("--rx-order", default=None, help="Optional RX channel order for AoA, for example 0,1,2,3. Use if hardware/channel order is reversed or remapped.")
    p.add_argument("--invert-angle-sign", action="store_true", help="Invert reported angle sign if left/right appears mirrored.")
    p.add_argument("--angle-keep-static", dest="angle_remove_static_mean", action="store_false", help="Do not remove the static mean for AoA detection. Default removes static mean so AoA follows the moving chest signal rather than room clutter.")
    p.set_defaults(angle_remove_static_mean=True)
    p.add_argument("--max-subjects", type=int, default=4, help="Maximum number of range-angle separated subjects to report in --angle-mode multi.")
    p.add_argument("--target-min-relative-db", type=float, default=-12.0, help="Reject additional targets more than this many dB below the strongest target. Example: -12 keeps targets within 12 dB.")
    p.add_argument("--target-min-quality-db", type=float, default=3.0, help="Minimum range-angle peak quality relative to the median range-angle map power.")
    p.add_argument("--target-min-separation-m", type=float, default=0.25, help="Minimum range separation used for non-maximum suppression between detected subjects.")
    p.add_argument("--target-min-separation-deg", type=float, default=15.0, help="Minimum angle separation used for non-maximum suppression between detected subjects.")
    p.add_argument("--allow-same-range-targets", dest="target_suppress_same_range_all_angles", action="store_false", help="Allow multiple detected subjects at the same range but different angles. Default suppresses same-range angular sidelobes.")
    p.set_defaults(target_suppress_same_range_all_angles=True)
    p.add_argument("--disable-chest-roi", dest="chest_roi_enable", action="store_false", help="Analyze only the single detected range-angle cell instead of a chest-sized ROI. Default uses ROI in multi mode.")
    p.set_defaults(chest_roi_enable=True)
    p.add_argument("--chest-roi-range-m", type=float, default=0.12, help="Half-width of the chest ROI in range around each detected subject. Default: 0.12 m.")
    p.add_argument("--chest-roi-min-range-bins", type=int, default=1, help="Minimum +/- range bins included in the chest ROI even if the meter width rounds smaller. Default: 1.")
    p.add_argument("--chest-roi-angle-deg", type=float, default=8.0, help="Half-width of the chest ROI in azimuth around each detected subject. Default: 8 deg.")
    p.add_argument("--chest-roi-min-relative-db", type=float, default=-10.0, help="Keep ROI cells within this dB level of the local ROI peak. Default: -10 dB.")
    p.add_argument("--chest-roi-max-cells", type=int, default=25, help="Maximum number of range-angle cells combined per subject ROI. Default: 25.")
    p.add_argument("--chest-roi-min-breath-corr", type=float, default=0.60, help="Reject ROI cells whose breathing-band motion is weakly correlated with the detected target seed. Default: 0.60.")
    p.add_argument("--drift-window-s", type=float, default=2.0, help="Moving-average drift removal window in seconds.")
    p.add_argument("--breath-low-hz", type=float, default=0.10, help="Breathing band lower cutoff.")
    p.add_argument("--breath-high-hz", type=float, default=0.60, help="Breathing band upper cutoff.")
    p.add_argument("--heart-low-hz", type=float, default=0.80, help="Heart band lower cutoff.")
    p.add_argument("--heart-high-hz", type=float, default=2.00, help="Heart band upper cutoff.")
    p.add_argument("--vital-fft-zeropad-factor", type=float, default=8.0, help="Zero-padding factor for breathing/heart spectra before peak picking. Default: 8.")
    p.add_argument("--vital-fft-min-size", type=int, default=8192, help="Minimum FFT length for breathing/heart spectra. Default: 8192.")
    p.add_argument("--disable-vital-fft-interpolation", dest="vital_fft_interpolate", action="store_false", help="Disable quadratic interpolation around the vital-rate FFT peak.")
    p.set_defaults(vital_fft_interpolate=True)
    p.add_argument("--make-plots", action="store_true", help="Save PNG plots.")
    p.add_argument("--no-save", action="store_true", help="Do not save any result files; print summary only.")
    p.add_argument("--output-stem", default=None, help="Output filename prefix. Default: ADC bin stem.")
    return p


def main(argv=None):
    args = build_combined_arg_parser().parse_args(argv)
    run_adc_to_vital_signs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
