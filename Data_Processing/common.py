#!/usr/bin/env python3
# Auto-split from ADC_To_Vital_Signs.py.
# Keep Python 3.8 compatibility.

from __future__ import annotations

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

# Export private helper names too because the split modules preserve the original
# internal function names and use star imports for Python 3.8 simplicity.
__all__ = [name for name in globals() if not name.startswith('__')]
C = C_MPS
__all__ = [name for name in globals() if not name.startswith('__')]
