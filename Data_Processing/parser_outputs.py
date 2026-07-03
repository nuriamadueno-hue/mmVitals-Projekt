#!/usr/bin/env python3
# Auto-split from ADC_To_Vital_Signs.py.
# Keep Python 3.8 compatibility.

from common import *
from adc_parser import *
from radar_config import *

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

__all__ = [name for name in globals() if not name.startswith('__')]
