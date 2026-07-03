#!/usr/bin/env python3
# Auto-split from ADC_To_Vital_Signs.py.
# Keep Python 3.8 compatibility.

from common import *
from metadata_utils import *

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

def _range_bin_spacing_m(range_axis_pos, has_metric_axis):
    if has_metric_axis and len(range_axis_pos) > 1:
        diffs = np.diff(range_axis_pos)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size:
            return float(np.median(diffs))
    return 0.1

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

__all__ = [name for name in globals() if not name.startswith('__')]
