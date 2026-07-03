#!/usr/bin/env python3
# Auto-split from ADC_To_Vital_Signs.py.
# Keep Python 3.8 compatibility.

from common import *

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

__all__ = [name for name in globals() if not name.startswith('__')]
