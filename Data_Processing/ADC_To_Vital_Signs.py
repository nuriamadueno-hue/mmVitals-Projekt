#!/usr/bin/env python3
"""
Minimal mmWave vital-sign dashboard runner.

Run from the project root:
    python Data_Processing/ADC_To_Vital_Signs.py

The script processes the ADC capture, estimates subjects, breathing rate and heart
rate, and creates one output file only:
    Data_Processing/Vital_Signs_Results/adc_data_validation_dashboard.png
"""

from types import SimpleNamespace
from pathlib import Path
import argparse
import shutil

from common import *
from radar_config import *
from adc_parser import *
from vital_processing import analyze_vital_signs_from_cube
from plotting import _configure_matplotlib_gui_backend, _display_dashboard_window


def _default_args(cli):
    project_root = cli.project_root
    output_dir = cli.output_dir
    return SimpleNamespace(
        # Minimal user-facing options
        project_root=project_root,
        data_dir=cli.data_dir,
        config_dir=cli.config_dir,
        vital_results_dir=output_dir,
        bin=cli.bin,
        dashboard_display_mode=cli.display,
        max_subjects=cli.max_subjects,
        # Internal fixed defaults
        results_dir=None,
        config=None,
        log=None,
        profile_csv=None,
        raw_log=None,
        allow_root_xml_fallback=True,
        allow_truncate=True,
        force_iq_swap=None,
        parser_format=None,
        num_rx=None,
        num_adc_samples=None,
        chirps_per_frame=None,
        real_only=False,
        complex_iq=False,
        frame_start=0,
        frame_count=None,
        fs=None,
        range_fft_size=512,
        min_range_m=0.4,
        max_range_m=2.0,
        min_range_bin=3,
        max_range_bin=None,
        rx_mode="strongest",
        angle_mode="multi",
        angle_range_mode="selected_range",
        angle_min_deg=-60.0,
        angle_max_deg=60.0,
        angle_step_deg=1.0,
        rx_spacing_lambda=0.5,
        rx_order=None,
        invert_angle_sign=False,
        angle_remove_static_mean=True,
        target_min_relative_db=-12.0,
        target_min_quality_db=3.0,
        target_min_separation_m=0.25,
        target_min_separation_deg=15.0,
        target_suppress_same_range_all_angles=True,
        chest_roi_enable=True,
        chest_roi_range_m=0.12,
        chest_roi_min_range_bins=1,
        chest_roi_angle_deg=8.0,
        chest_roi_min_relative_db=-10.0,
        chest_roi_max_cells=25,
        chest_roi_min_breath_corr=0.60,
        drift_window_s=2.0,
        breath_low_hz=0.10,
        breath_high_hz=0.60,
        heart_low_hz=0.80,
        heart_high_hz=2.00,
        heart_peak_method="harmonic_aware",
        heart_harmonic_orders="2,3",
        heart_harmonic_reject_tolerance_bpm=4.0,
        heart_harmonic_soft_tolerance_bpm=6.0,
        heart_harmonic_penalty_db=8.0,
        heart_candidate_max_drop_db=10.0,
        heart_high_candidate_min_bpm=70.0,
        heart_high_candidate_bonus_db=0.0,
        heart_low_candidate_min_bpm=58.0,
        heart_low_candidate_penalty_db=6.0,
        filter_mode="auto",
        filter_order=4,
        vital_fft_zeropad_factor=8.0,
        vital_fft_min_size=8192,
        vital_fft_interpolate=True,
        reference_breath_rates=None,
        reference_heart_rates=None,
        reference_breath_subject_1=None,
        reference_heart_subject_1=None,
        reference_breath_subject_2=None,
        reference_heart_subject_2=None,
        rate_trend_window_s=30.0,
        rate_trend_step_s=2.0,
        rate_trend_median_s=6.0,
        rate_trend_max_jump_bpm_per_s=2.0,
        rate_trend_ema_alpha=0.35,
        rate_trend_full_capture_blend=0.0,
        rate_trend_candidate_max_drop_db=18.0,
        rate_trend_prior_penalty_db_per_bpm=0.20,
        rate_trend_continuity_penalty_db_per_bpm=0.55,
        rate_trend_harmonic_extra_penalty_db=12.0,
        show_dashboard=(cli.display != "none"),
        gui_backend=None,
        make_plots=True,
        no_save=False,
        dashboard_only=True,
        output_stem=None,
    )


def _clean_dashboard_output_dir(results_dir):
    results_dir.mkdir(parents=True, exist_ok=True)
    for path in results_dir.iterdir():
        if path.is_file():
            path.unlink()
        elif path.is_dir() and path.name == "__pycache__":
            shutil.rmtree(str(path), ignore_errors=True)


def run_adc_to_vital_signs(args):
    paths = infer_project_paths(
        project_root_arg=args.project_root,
        data_dir_arg=args.data_dir,
        config_dir_arg=args.config_dir,
        results_dir_arg=args.results_dir,
    )

    bin_path = Path(args.bin).expanduser().resolve() if args.bin else find_bin_file(paths.data_dir).resolve()
    results_dir = Path(args.vital_results_dir).expanduser().resolve() if args.vital_results_dir else paths.project_root / "Data_Processing" / "Vital_Signs_Results"
    _clean_dashboard_output_dir(results_dir)

    cfg, config_diagnostics = build_config_from_available_sources(
        bin_path=bin_path,
        paths=paths,
        explicit_config=None,
        explicit_log=None,
        explicit_profile_csv=None,
        explicit_raw_log=None,
        allow_root_xml_fallback=args.allow_root_xml_fallback,
    )

    parsed = read_dca1000_adc_bin(
        bin_path,
        cfg,
        allow_truncate=args.allow_truncate,
        force_iq_swap=args.force_iq_swap,
    )

    metadata = parsed["metadata"]
    metadata["config_diagnostics"] = config_diagnostics
    cube_frames = parsed["cube_frames"]

    if args.show_dashboard and args.dashboard_display_mode != "none":
        _configure_matplotlib_gui_backend(args.gui_backend)

    summary, out_paths = analyze_vital_signs_from_cube(
        cube_frames,
        metadata,
        args,
        input_bin_path=bin_path,
        results_dir=results_dir,
        stem=bin_path.stem,
    )

    subjects = summary.get("subjects", [])
    print("Vital-sign analysis complete")
    print("Detected subjects: {}".format(len(subjects)))
    for subject in subjects:
        print(
            "Subject {idx:02d}: heart {heart:.1f} bpm, breathing {breath:.1f}/min".format(
                idx=int(subject.get("subject_index", 0)),
                heart=float(subject.get("heart_rate_beats_per_min", float("nan"))),
                breath=float(subject.get("breathing_rate_breaths_per_min", float("nan"))),
            )
        )

    dashboard_path = out_paths.get("dashboard_plot") or (results_dir / "{}_validation_dashboard.png".format(bin_path.stem))
    print("Dashboard: {}".format(dashboard_path))

    if args.show_dashboard:
        _display_dashboard_window(
            dashboard_path,
            gui_backend=args.gui_backend,
            display_mode=args.dashboard_display_mode,
        )

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Minimal mmWave vital-sign dashboard runner.")
    parser.add_argument("--project-root", default=None, help="Project root. Default: auto-detected.")
    parser.add_argument("--data-dir", default=None, help="ADC data folder. Default: ADC_Recorded_Data.")
    parser.add_argument("--config-dir", default=None, help="Configuration folder. Default: mmWave_Configuration.")
    parser.add_argument("--bin", default=None, help="ADC .bin file. Default: first/best .bin in the data folder.")
    parser.add_argument("--output-dir", default=None, help="Dashboard output folder. Default: Data_Processing/Vital_Signs_Results.")
    parser.add_argument("--max-subjects", type=int, default=4, help="Maximum number of subjects to detect. Default: 4.")
    parser.add_argument("--display", default="file", choices=["file", "auto", "matplotlib", "none"], help="Dashboard display mode. Default: file.")
    return parser


def main(argv=None):
    cli = build_arg_parser().parse_args(argv)
    args = _default_args(cli)
    run_adc_to_vital_signs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
