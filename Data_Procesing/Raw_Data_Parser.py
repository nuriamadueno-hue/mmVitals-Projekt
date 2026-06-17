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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parse TI mmWave Studio DCA1000 ADC data for this project structure.")
    parser.add_argument("--project-root", default=None, help="Project root. Default: inferred from script location.")
    parser.add_argument("--data-dir", default=None, help="Folder containing ADC .bin files. Default: <project-root>/ADC_Recorded_Data.")
    parser.add_argument("--config-dir", default=None, help="Folder containing Profile.csv or XML. Default: <project-root>/mmWave_Configuration.")
    parser.add_argument("--results-dir", default=None, help="Output folder. Default: Data_Procesing/Preliminary_Results.")

    parser.add_argument("--bin", default=None, help="Explicit ADC .bin file.")
    parser.add_argument("--config", default=None, help="Explicit mmWave Studio XML file. Optional.")
    parser.add_argument("--log", default=None, help="Explicit mmWave Studio API LogFile.txt. Optional.")
    parser.add_argument("--profile-csv", default=None, help="Explicit Profile.csv. Optional.")
    parser.add_argument("--raw-log", default=None, help="Explicit DCA1000 Raw_LogFile.csv. Optional.")
    parser.add_argument("--allow-root-xml-fallback", action="store_true", help="Use a root-level XML if no log/Profile.csv is available. Off by default to avoid stale XML.")

    parser.add_argument("--parser-format", choices=["dca1000_xwr16xx", "dca1000_xwr14xx"], default=None, help="Override parser format.")
    parser.add_argument("--num-rx", type=int, default=None, help="Override RX count.")
    parser.add_argument("--num-adc-samples", type=int, default=None, help="Override ADC samples per chirp.")
    parser.add_argument("--chirps-per-frame", type=int, default=None, help="Override chirps per frame.")
    parser.add_argument("--real-only", action="store_true", help="Override ADC format to real only.")
    parser.add_argument("--complex-iq", action="store_true", help="Override ADC format to complex IQ.")
    parser.add_argument("--force-iq-swap", type=int, choices=[0, 1], default=None, help="Override IQ swap. 0=I-first, 1=Q-first.")

    parser.add_argument("--allow-truncate", action="store_true", help="Drop incomplete trailing int16 words. Use only for damaged/partial files.")
    parser.add_argument("--n-range-fft", type=int, default=None, help="Range FFT size. Default: num_adc_samples.")
    parser.add_argument("--n-doppler-fft", type=int, default=None, help="Doppler FFT size. Default: chirps used.")
    parser.add_argument("--skip-rd", action="store_true", help="Only parse ADC cube; do not create range-Doppler map.")
    parser.add_argument("--dry-run", action="store_true", help="Print inferred config and parsed dimensions without writing output files.")

    args = parser.parse_args(argv)
    run_processing(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
