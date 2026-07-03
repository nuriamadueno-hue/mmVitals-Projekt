#!/usr/bin/env python3
"""
ADC_To_Vital_Signs.py

One-command entry point for direct TI mmWave Studio / DCA1000 ADC processing.

The implementation is split into modules in the same folder:
    common.py              shared constants, data classes, helper functions
    radar_config.py        project discovery and mmWave Studio config parsing
    adc_parser.py          DCA1000 raw ADC parsing and radar axes
    angle_processing.py    range-angle AoA helpers and multi-target detection
    vital_processing.py    breathing/heart-rate extraction from beamformed phase
    plotting.py            result files and diagnostic plots

Typical use from the project root or Data_Processing folder:
    python Data_Processing/ADC_To_Vital_Signs.py --make-plots
"""

from common import *
from radar_config import *
from adc_parser import *
from vital_processing import analyze_vital_signs_from_cube
from plotting import _configure_matplotlib_gui_backend, _display_dashboard_window

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

    results_dir = Path(args.vital_results_dir).expanduser().resolve() if args.vital_results_dir else paths.project_root / "Data_Processing" / "Vital_Signs_Results"
    stem = args.output_stem if args.output_stem else bin_path.stem

    # Configure the GUI backend before any plotting function imports pyplot.
    # Otherwise Matplotlib may lock itself to the non-interactive Agg backend,
    # which can save PNG files but cannot open a window.
    dashboard_backend_status = None
    if args.make_plots and bool(getattr(args, "show_dashboard", True)) and str(getattr(args, "dashboard_display_mode", "auto")).lower() != "none":
        ok, backend_name, backend_error = _configure_matplotlib_gui_backend(getattr(args, "gui_backend", None))
        dashboard_backend_status = (ok, backend_name, backend_error)
        if ok:
            print("Dashboard backend prepared before plotting: {}".format(backend_name))
        else:
            print("Dashboard GUI backend could not be prepared before plotting. Current backend: {}. Last error: {}".format(backend_name, backend_error))

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
            validation = subject.get("validation", {})
            if validation:
                print("  Validation:")
                if validation.get("reference_breathing_rate_breaths_per_min") is not None:
                    print("    breath ref:     {:.2f}/min; error {:+.2f}/min".format(
                        validation["reference_breathing_rate_breaths_per_min"],
                        validation["breathing_error_breaths_per_min"],
                    ))
                if validation.get("reference_heart_rate_beats_per_min") is not None:
                    print("    heart ref:      {:.2f} bpm; error {:+.2f} bpm".format(
                        validation["reference_heart_rate_beats_per_min"],
                        validation["heart_error_beats_per_min"],
                    ))
            if subject.get("vital_score_db") is not None:
                print("  Vital score:      {:.1f} dB".format(subject["vital_score_db"]))
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

    if (not args.no_save) and args.make_plots and bool(getattr(args, "show_dashboard", True)):
        dashboard_path = out_paths.get("dashboard_plot")
        if dashboard_path is not None:
            print("")
            print("Displaying validation dashboard after processing...")
            _display_dashboard_window(
                dashboard_path,
                gui_backend=getattr(args, "gui_backend", None),
                display_mode=getattr(args, "dashboard_display_mode", "auto"),
            )

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
    p.add_argument("--vital-results-dir", default=None, help="Vital-sign result folder. Default: <project-root>/Data_Processing/Vital_Signs_Results.")
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
    p.add_argument("--heart-peak-method", default="harmonic_aware", choices=["harmonic_aware", "strongest"], help="Heart peak selection method. harmonic_aware ranks multiple peaks and penalizes breathing harmonics.")
    p.add_argument("--heart-harmonic-orders", default="2,3", help="Comma-separated breathing harmonic orders to penalize for heart selection. Default: 2,3.")
    p.add_argument("--heart-harmonic-reject-tolerance-bpm", type=float, default=4.0, help="Hard tolerance around breathing harmonics for heart candidate rejection/penalty.")
    p.add_argument("--heart-harmonic-soft-tolerance-bpm", type=float, default=6.0, help="Soft penalty tolerance around breathing harmonics.")
    p.add_argument("--heart-harmonic-penalty-db", type=float, default=8.0, help="Maximum score penalty for peaks very close to breathing harmonics.")
    p.add_argument("--heart-candidate-max-drop-db", type=float, default=10.0, help="Consider heart candidates within this dB range of the strongest heart-band peak.")
    p.add_argument("--heart-high-candidate-min-bpm", type=float, default=70.0, help="Minimum BPM for high-heart candidate bonus.")
    p.add_argument("--heart-high-candidate-bonus-db", type=float, default=0.0, help="Score bonus for plausible high-heart-rate candidates. Default 0; harmonic rejection and low-rate penalty usually provide enough separation.")
    p.add_argument("--heart-low-candidate-min-bpm", type=float, default=58.0, help="Heart candidates below this rate receive a small penalty to avoid low-frequency motion artifacts.")
    p.add_argument("--heart-low-candidate-penalty-db", type=float, default=6.0, help="Maximum penalty for very low heart-rate candidates.")
    p.add_argument("--filter-mode", default="auto", choices=["auto", "butter", "fft"], help="Vital bandpass filter method. auto uses scipy Butterworth when available and falls back to FFT masking. Default: auto.")
    p.add_argument("--filter-order", type=int, default=4, help="Butterworth bandpass order when scipy filtering is used. Default: 4.")
    p.add_argument("--vital-fft-zeropad-factor", type=float, default=8.0, help="Zero-padding factor for breathing/heart spectra before peak picking. Default: 8.")
    p.add_argument("--vital-fft-min-size", type=int, default=8192, help="Minimum FFT length for breathing/heart spectra. Default: 8192.")
    p.add_argument("--disable-vital-fft-interpolation", dest="vital_fft_interpolate", action="store_false", help="Disable quadratic interpolation around the vital-rate FFT peak.")
    p.set_defaults(vital_fft_interpolate=True)
    p.add_argument("--reference-breath-rates", default=None, help="Comma-separated reference breathing rates in breaths/min for subject 1..N, for example 18,22.")
    p.add_argument("--reference-heart-rates", default=None, help="Comma-separated reference heart rates in bpm for subject 1..N, for example 58,92.")
    p.add_argument("--reference-breath-subject-1", type=float, default=None, help="Reference breathing rate for subject 1 in breaths/min.")
    p.add_argument("--reference-heart-subject-1", type=float, default=None, help="Reference heart rate for subject 1 in bpm.")
    p.add_argument("--reference-breath-subject-2", type=float, default=None, help="Reference breathing rate for subject 2 in breaths/min.")
    p.add_argument("--reference-heart-subject-2", type=float, default=None, help="Reference heart rate for subject 2 in bpm.")
    p.add_argument("--rate-trend-window-s", type=float, default=30.0, help="Sliding-window length in seconds for the validation plot showing breathing/heart rate over time. Default: 30 s for stable display.")
    p.add_argument("--rate-trend-step-s", type=float, default=2.0, help="Step size in seconds for the validation rate-trend plot. Default: 2 s.")
    p.add_argument("--rate-trend-median-s", type=float, default=6.0, help="Median smoothing width in seconds for the displayed rate trend. Default: 6 s.")
    p.add_argument("--rate-trend-max-jump-bpm-per-s", type=float, default=2.0, help="Maximum displayed rate change in bpm per second before jump limiting. Default: 2 bpm/s.")
    p.add_argument("--rate-trend-ema-alpha", type=float, default=0.35, help="EMA smoothing alpha for displayed rate trend. Default: 0.35.")
    p.add_argument("--rate-trend-full-capture-blend", type=float, default=0.00, help="Blend factor pulling the displayed trend toward the robust full-capture estimate. Default: 0.00, so the dashboard shows the 30 s sliding-window trend.")
    p.add_argument("--rate-trend-candidate-max-drop-db", type=float, default=18.0, help="Consider short-window heart candidates within this dB drop from the strongest candidate for trend continuity. Default: 18 dB.")
    p.add_argument("--rate-trend-prior-penalty-db-per-bpm", type=float, default=0.20, help="Trend candidate penalty per bpm away from the full-capture heart-rate prior. Default: 0.20.")
    p.add_argument("--rate-trend-continuity-penalty-db-per-bpm", type=float, default=0.55, help="Trend candidate penalty per bpm away from previous displayed heart rate. Default: 0.55.")
    p.add_argument("--rate-trend-harmonic-extra-penalty-db", type=float, default=12.0, help="Extra trend-display penalty for candidates marked as breathing harmonics. Default: 12 dB.")
    p.add_argument("--no-show-dashboard", dest="show_dashboard", action="store_false", help="Do not display the final combined validation dashboard window after processing. By default the script attempts to show it when a GUI backend is available.")
    p.add_argument("--gui-backend", default=None, help="Optional Matplotlib GUI backend to force for the final dashboard window, for example TkAgg or QtAgg. Default: try TkAgg, QtAgg, Qt5Agg, then WXAgg if needed.")
    p.add_argument("--dashboard-display-mode", default="auto", choices=["auto", "matplotlib", "file", "none"], help="How to display the final dashboard after processing. auto tries Matplotlib GUI first and falls back to the OS image viewer; file opens the saved PNG with the OS viewer; none saves only. Default: auto.")
    p.set_defaults(show_dashboard=True)
    p.add_argument("--make-plots", action="store_true", help="Save PNG plots, including range-angle diagnostic maps in multi-target mode.")
    p.add_argument("--no-save", action="store_true", help="Do not save any result files; print summary only.")
    p.add_argument("--output-stem", default=None, help="Output filename prefix. Default: ADC bin stem.")
    return p

def main(argv=None):
    args = build_combined_arg_parser().parse_args(argv)
    run_adc_to_vital_signs(args)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
