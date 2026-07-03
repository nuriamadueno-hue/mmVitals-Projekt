#!/usr/bin/env python3
"""mmWave Studio configuration discovery and decoding."""
from common import *

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
        # Expected case: script is in mmWave_Studio/Data_Processing.
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
        if SCRIPT_DIR.name.lower() in {"data_processing", "data_processing"}:
            results_dir = SCRIPT_DIR / "Preliminary_Results"
        else:
            results_dir = project_root / "Data_Processing" / "Preliminary_Results"

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
    """Prefer normal ADC captures over small/log-like binary files."""
    name = path.name.lower()
    penalty = 0
    if "adc" not in name:
        penalty += 5
    if "test" in name:
        penalty += 2
    return penalty, -path.stat().st_size, name

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

__all__ = [name for name in globals() if not name.startswith('__')]
