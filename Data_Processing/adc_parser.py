#!/usr/bin/env python3
"""DCA1000 raw ADC parser."""
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

