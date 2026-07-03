#!/usr/bin/env python3
# Auto-split from ADC_To_Vital_Signs.py.
# Keep Python 3.8 compatibility.

from common import *
from range_processing import *
from angle_processing import *
from plotting import *

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
    # Script normally lives in mmWave_Studio/Data_Processing/
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

def _fft_mask_bandpass(signal, fs, low_hz, high_hz):
    """Dependency-free FFT-domain bandpass fallback."""
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

def _butter_bandpass(signal, fs, low_hz, high_hz, order):
    """Zero-phase Butterworth bandpass. Requires scipy; caller handles fallback."""
    from scipy.signal import butter, sosfiltfilt
    x = np.asarray(signal, dtype=np.float64)
    if x.size < 8:
        return np.zeros_like(x)
    nyq = 0.5 * float(fs)
    low = max(float(low_hz), 1e-9)
    high = min(float(high_hz), nyq * 0.999)
    if not (0.0 < low < high < nyq):
        return _fft_mask_bandpass(x, fs, low_hz, high_hz)
    sos = butter(max(1, int(order)), [low, high], btype="bandpass", fs=float(fs), output="sos")
    return sosfiltfilt(sos, x - np.nanmean(x))

def _bandpass_filter(signal, fs, low_hz, high_hz, args=None):
    """Vital bandpass using scipy Butterworth when requested/available, else FFT fallback."""
    mode = str(getattr(args, "filter_mode", "auto") if args is not None else "auto").lower()
    order = int(getattr(args, "filter_order", 4) if args is not None else 4)
    if mode in {"auto", "butter"}:
        try:
            return _butter_bandpass(signal, fs, low_hz, high_hz, order)
        except Exception:
            if mode == "butter":
                raise
    return _fft_mask_bandpass(signal, fs, low_hz, high_hz)

# Backward-compatible name used by existing ROI code.
def _fft_bandpass(signal, fs, low_hz, high_hz, args=None):
    return _bandpass_filter(signal, fs, low_hz, high_hz, args=args)

def _parse_reference_list(text):
    if text is None or str(text).strip() == "":
        return []
    vals = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        try:
            vals.append(float(part))
        except Exception:
            vals.append(None)
    return vals

def _reference_for_subject(args, subject_index, kind):
    idx = int(subject_index)
    if kind == "breath":
        explicit = getattr(args, "reference_breath_subject_{}".format(idx), None)
        vals = _parse_reference_list(getattr(args, "reference_breath_rates", None))
    else:
        explicit = getattr(args, "reference_heart_subject_{}".format(idx), None)
        vals = _parse_reference_list(getattr(args, "reference_heart_rates", None))
    if explicit is not None:
        return float(explicit)
    if 0 <= idx - 1 < len(vals) and vals[idx - 1] is not None:
        return float(vals[idx - 1])
    return None

def _add_validation_and_vital_score(subject, args):
    """Add reference error fields and a separate vital quality score after vital extraction."""
    bq = float(subject.get("breathing_quality_db", float("nan")))
    hq = float(subject.get("heart_quality_db", float("nan")))
    finite = [x for x in [bq, hq] if math.isfinite(x)]
    subject["vital_score_db"] = float(sum(finite) / len(finite)) if finite else None
    subject["detection_score_db"] = subject.get("detection_quality_db", subject.get("target_quality_db"))

    ref_b = _reference_for_subject(args, subject.get("subject_index", 1), "breath")
    ref_h = _reference_for_subject(args, subject.get("subject_index", 1), "heart")
    validation = {}
    if ref_b is not None:
        est = float(subject.get("breathing_rate_breaths_per_min", float("nan")))
        validation["reference_breathing_rate_breaths_per_min"] = float(ref_b)
        validation["breathing_error_breaths_per_min"] = float(est - ref_b) if math.isfinite(est) else None
        validation["breathing_abs_error_breaths_per_min"] = abs(float(est - ref_b)) if math.isfinite(est) else None
    if ref_h is not None:
        est = float(subject.get("heart_rate_beats_per_min", float("nan")))
        validation["reference_heart_rate_beats_per_min"] = float(ref_h)
        validation["heart_error_beats_per_min"] = float(est - ref_h) if math.isfinite(est) else None
        validation["heart_abs_error_beats_per_min"] = abs(float(est - ref_h)) if math.isfinite(est) else None
    if validation:
        subject["validation"] = validation
    return subject

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



def _parse_int_list(value, default):
    if value is None:
        return list(default)
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    out = []
    for part in str(value).replace(';', ',').split(','):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out or list(default)

def _local_peak_indices(power, idxs):
    idxs = np.asarray(idxs, dtype=int)
    peaks = []
    if idxs.size == 0:
        return peaks
    idx_set = set(int(i) for i in idxs)
    for i in idxs:
        i = int(i)
        if i <= 0 or i >= len(power) - 1:
            continue
        if (i - 1) not in idx_set or (i + 1) not in idx_set:
            continue
        if power[i] >= power[i - 1] and power[i] >= power[i + 1]:
            peaks.append(i)
    if not peaks:
        peaks = [int(idxs[np.argmax(power[idxs])])]
    return peaks

def _nearest_breath_harmonic(rate_bpm, breathing_rate_bpm, orders):
    if breathing_rate_bpm is None or not math.isfinite(float(breathing_rate_bpm)) or float(breathing_rate_bpm) <= 0.0:
        return None, float('inf'), None
    best_order = None
    best_harm = None
    best_dist = float('inf')
    for order in orders:
        harm = float(order) * float(breathing_rate_bpm)
        dist = abs(float(rate_bpm) - harm)
        if dist < best_dist:
            best_dist = dist
            best_order = int(order)
            best_harm = float(harm)
    return best_order, best_dist, best_harm

def _estimate_heart_rate_fft(signal, fs, low_hz, high_hz, breathing_rate_bpm, args=None):
    """Estimate heart rate with breathing-harmonic-aware multi-peak selection.

    The previous method selected the strongest peak in the heart band. Validation
    showed that this often chooses the 3rd respiration harmonic. This method still
    computes the same FFT, but ranks several local peaks and penalizes candidates
    that are close to selected breathing harmonics.
    """
    baseline = _estimate_rate_fft(signal, fs, low_hz, high_hz, args=args)
    freqs = np.asarray(baseline.get('freqs_hz', []), dtype=np.float64)
    power = np.asarray(baseline.get('power', []), dtype=np.float64)
    if freqs.size == 0 or power.size == 0 or not math.isfinite(float(baseline.get('rate_per_min', float('nan')))):
        baseline.update({
            'peak_selection_method': 'strongest_peak_fallback',
            'heart_candidates': [],
            'harmonic_warning': False,
            'nearest_breath_harmonic_order': None,
            'nearest_breath_harmonic_bpm': None,
            'nearest_breath_harmonic_distance_bpm': None,
        })
        return baseline

    enabled = True
    if args is not None:
        enabled = str(getattr(args, 'heart_peak_method', 'harmonic_aware')).lower() != 'strongest'
    if not enabled:
        baseline.update({
            'peak_selection_method': 'strongest_peak',
            'heart_candidates': [],
            'harmonic_warning': False,
            'nearest_breath_harmonic_order': None,
            'nearest_breath_harmonic_bpm': None,
            'nearest_breath_harmonic_distance_bpm': None,
        })
        return baseline

    orders = _parse_int_list(getattr(args, 'heart_harmonic_orders', '2,3') if args is not None else '2,3', [2, 3])
    tol_bpm = float(getattr(args, 'heart_harmonic_reject_tolerance_bpm', 4.0) if args is not None else 4.0)
    max_drop_db = float(getattr(args, 'heart_candidate_max_drop_db', 10.0) if args is not None else 10.0)
    high_min_bpm = float(getattr(args, 'heart_high_candidate_min_bpm', 70.0) if args is not None else 70.0)
    high_bonus_db = float(getattr(args, 'heart_high_candidate_bonus_db', 0.0) if args is not None else 0.0)
    low_min_bpm = float(getattr(args, 'heart_low_candidate_min_bpm', 58.0) if args is not None else 58.0)
    low_penalty_db = float(getattr(args, 'heart_low_candidate_penalty_db', 6.0) if args is not None else 6.0)
    harmonic_penalty_db = float(getattr(args, 'heart_harmonic_penalty_db', 8.0) if args is not None else 8.0)
    near_harmonic_soft_bpm = max(tol_bpm, float(getattr(args, 'heart_harmonic_soft_tolerance_bpm', 6.0) if args is not None else 6.0))

    band = (freqs >= low_hz) & (freqs <= high_hz)
    idxs = np.where(band)[0]
    if idxs.size == 0:
        return baseline
    peak_idxs = _local_peak_indices(power, idxs)
    top_power = float(np.max(power[peak_idxs])) if peak_idxs else float(np.max(power[idxs]))
    med = float(np.median(power[idxs])) if idxs.size else 0.0

    candidates = []
    for idx in peak_idxs:
        raw_rate = 60.0 * float(freqs[idx])
        delta = _quadratic_peak_interpolation(power, idx) if bool(getattr(args, 'vital_fft_interpolate', True) if args is not None else True) else 0.0
        interp_hz = float((float(idx) + delta) * float(fs) / float(baseline.get('n_fft', len(power) * 2 - 2)))
        rate = 60.0 * max(float(low_hz), min(float(high_hz), interp_hz))
        rel_db = 10.0 * math.log10((float(power[idx]) + 1e-30) / (top_power + 1e-30))
        quality_db = 10.0 * math.log10((float(power[idx]) + 1e-30) / (med + 1e-30))
        order, dist, harm = _nearest_breath_harmonic(rate, breathing_rate_bpm, orders)
        is_harm = bool(dist <= tol_bpm)
        near_harm = bool(dist <= near_harmonic_soft_bpm)
        score = quality_db
        if near_harm:
            # Smooth penalty. Peaks almost exactly on breathing harmonics are heavily penalized;
            # candidates slightly outside the hard tolerance are penalized less.
            score -= harmonic_penalty_db * max(0.0, (near_harmonic_soft_bpm - dist) / max(near_harmonic_soft_bpm, 1e-9))
        if rate >= high_min_bpm:
            score += high_bonus_db
        if rate < low_min_bpm:
            score -= low_penalty_db
        candidates.append({
            'idx': int(idx),
            'rate_bpm': float(rate),
            'raw_rate_bpm': float(raw_rate),
            'quality_db': float(quality_db),
            'relative_to_top_db': float(rel_db),
            'score_db': float(score),
            'nearest_harmonic_order': int(order) if order is not None else None,
            'nearest_harmonic_bpm': float(harm) if harm is not None else None,
            'nearest_harmonic_distance_bpm': float(dist) if math.isfinite(dist) else None,
            'is_breathing_harmonic': bool(is_harm),
            'interpolation_delta_bins': float(delta),
        })

    # Limit consideration to peaks that are not too far below the strongest peak.
    considered = [c for c in candidates if c['relative_to_top_db'] >= -abs(max_drop_db)]
    if not considered:
        considered = candidates[:]

    # Prefer non-hard-harmonic candidates. If every peak is harmonic-like, fall back to score.
    non_harm = [c for c in considered if not c['is_breathing_harmonic']]
    pool = non_harm if non_harm else considered
    selected = max(pool, key=lambda c: (c['score_db'], c['quality_db'])) if pool else None
    if selected is None:
        return baseline

    # Low-rate refinement: in validation data, a rejected respiration harmonic can
    # create a split side-lobe cluster around the actual low heart rate. If the
    # selected low-HR peak has a nearby lower non-harmonic peak of almost the same
    # quality, prefer the lower peak. This avoids jumping to the upper side-lobe
    # while leaving high-HR cases unchanged.
    if float(selected.get('rate_bpm', 999.0)) < 70.0:
        nearby_lower = [
            c for c in pool
            if c is not selected
            and float(c.get('rate_bpm', 0.0)) < float(selected.get('rate_bpm', 0.0))
            and float(selected.get('rate_bpm', 0.0)) - float(c.get('rate_bpm', 0.0)) <= 5.0
            and float(selected.get('quality_db', -999.0)) - float(c.get('quality_db', -999.0)) <= 3.0
            and not bool(c.get('is_breathing_harmonic', False))
        ]
        if nearby_lower:
            selected = max(nearby_lower, key=lambda c: (c['quality_db'], c['score_db']))

    selected_idx = int(selected['idx'])
    rate_per_min = float(selected['rate_bpm'])
    raw_rate_per_min = float(selected['raw_rate_bpm'])
    quality_db = float(selected['quality_db'])

    candidates_sorted = sorted(candidates, key=lambda c: c['quality_db'], reverse=True)[:12]
    harmonic_warning = bool(selected['nearest_harmonic_distance_bpm'] is not None and selected['nearest_harmonic_distance_bpm'] <= tol_bpm)

    baseline.update({
        'rate_per_min': float(rate_per_min),
        'raw_bin_rate_per_min': float(raw_rate_per_min),
        'quality_db': float(quality_db),
        'interpolation_delta_bins': float(selected['interpolation_delta_bins']),
        'peak_bin_index': int(selected_idx),
        'peak_selection_method': 'harmonic_aware_multi_peak',
        'heart_candidates': candidates_sorted,
        'harmonic_warning': bool(harmonic_warning),
        'nearest_breath_harmonic_order': selected['nearest_harmonic_order'],
        'nearest_breath_harmonic_bpm': selected['nearest_harmonic_bpm'],
        'nearest_breath_harmonic_distance_bpm': selected['nearest_harmonic_distance_bpm'],
    })
    return baseline


def _median_filter_1d(values, kernel):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    k = int(kernel)
    if k <= 1 or arr.size < 3:
        return arr.copy()
    if k % 2 == 0:
        k += 1
    k = min(k, arr.size if arr.size % 2 == 1 else arr.size - 1)
    if k <= 1:
        return arr.copy()
    half = k // 2
    out = arr.copy()
    for i in range(arr.size):
        lo = max(0, i - half)
        hi = min(arr.size, i + half + 1)
        vals = arr[lo:hi]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = float(np.median(vals))
    return out

def _ema_filter_1d(values, alpha):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    a = float(alpha)
    if not math.isfinite(a):
        a = 0.35
    a = min(1.0, max(0.0, a))
    out = arr.copy()
    last = None
    for i, v in enumerate(arr):
        if not math.isfinite(float(v)):
            out[i] = float(last) if last is not None else float('nan')
            continue
        if last is None or not math.isfinite(float(last)):
            last = float(v)
        else:
            last = a * float(v) + (1.0 - a) * float(last)
        out[i] = float(last)
    return out

def _limit_rate_jumps(values, max_jump_bpm):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return arr.copy()
    max_jump = abs(float(max_jump_bpm))
    if not math.isfinite(max_jump) or max_jump <= 0.0:
        return arr.copy()
    out = arr.copy()
    last = None
    for i, v in enumerate(arr):
        if not math.isfinite(float(v)):
            out[i] = float(last) if last is not None else float('nan')
            continue
        if last is None or not math.isfinite(float(last)):
            last = float(v)
            out[i] = float(v)
            continue
        dv = float(v) - float(last)
        if abs(dv) > max_jump:
            v = float(last) + math.copysign(max_jump, dv)
        last = float(v)
        out[i] = float(v)
    return out

def _choose_trend_heart_from_candidates(est, prior_bpm, previous_bpm, args):
    """Choose a window heart-rate candidate for display trend stability.

    The final measurement uses full-capture estimation. This helper is only for the
    dashboard trend. It prevents the display from jumping between heartbeat,
    respiration harmonics, and side lobes in short FFT windows.
    """
    fallback = float(est.get('heart_rate_beats_per_min', est.get('rate_per_min', float('nan'))) if isinstance(est, dict) else float('nan'))
    if not isinstance(est, dict):
        return fallback
    candidates = est.get('heart_candidates', []) or []
    if not candidates:
        return fallback

    max_drop_db = abs(float(getattr(args, 'rate_trend_candidate_max_drop_db', 18.0)))
    prior_weight = float(getattr(args, 'rate_trend_prior_penalty_db_per_bpm', 0.20))
    continuity_weight = float(getattr(args, 'rate_trend_continuity_penalty_db_per_bpm', 0.55))
    harmonic_extra_penalty = float(getattr(args, 'rate_trend_harmonic_extra_penalty_db', 12.0))

    top_q = max(float(c.get('quality_db', -999.0)) for c in candidates)
    pool = []
    for c in candidates:
        q = float(c.get('quality_db', -999.0))
        if q < top_q - max_drop_db:
            continue
        rate = float(c.get('rate_bpm', float('nan')) )
        if not math.isfinite(rate):
            continue
        score = q
        if bool(c.get('is_breathing_harmonic', False)):
            score -= harmonic_extra_penalty
        if prior_bpm is not None and math.isfinite(float(prior_bpm)):
            score -= prior_weight * abs(rate - float(prior_bpm))
        if previous_bpm is not None and math.isfinite(float(previous_bpm)):
            score -= continuity_weight * abs(rate - float(previous_bpm))
        # Use the candidate score from the harmonic-aware estimator as a secondary signal.
        score += 0.15 * float(c.get('score_db', q))
        pool.append((score, q, rate, c))
    if not pool:
        return fallback
    pool.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return float(pool[0][2])

def _estimate_rate_trend(displacement_m, fs_vital, args):
    """Estimate breathing and heart rate over time for the dashboard.

    This is a visualization trace. The official value remains the full-capture
    estimate. The trace now uses a longer default window, the full-capture result
    as a candidate-selection prior, candidate continuity, median smoothing, and jump limiting so the
    plotted rate does not switch unrealistically between heartbeat, breathing
    harmonics, and side lobes.
    """
    x = np.asarray(displacement_m, dtype=np.float64)
    n = x.size
    default_win = 30.0
    default_step = 2.0
    if n < 8:
        return {
            'time_s': np.array([], dtype=np.float64),
            'breathing_rate_bpm': np.array([], dtype=np.float64),
            'heart_rate_bpm': np.array([], dtype=np.float64),
            'breathing_rate_raw_bpm': np.array([], dtype=np.float64),
            'heart_rate_raw_bpm': np.array([], dtype=np.float64),
            'window_s': float(getattr(args, 'rate_trend_window_s', default_win)),
            'step_s': float(getattr(args, 'rate_trend_step_s', default_step)),
        }

    full_est = _extract_vitals_from_displacement(x, fs_vital, args)
    full_breath = float(full_est.get('breathing_rate_breaths_per_min', float('nan')))
    full_heart = float(full_est.get('heart_rate_beats_per_min', float('nan')))

    win_s = max(10.0, float(getattr(args, 'rate_trend_window_s', default_win)))
    step_s = max(0.5, float(getattr(args, 'rate_trend_step_s', default_step)))
    win_n = max(8, int(round(win_s * float(fs_vital))))
    step_n = max(1, int(round(step_s * float(fs_vital))))

    # If the capture is not longer than the requested window, draw a flat validated
    # line across the whole recording instead of a single point or noisy short-window trace.
    if win_n >= n:
        t0 = 0.0
        t1 = max(0.0, (n - 1) / float(fs_vital))
        return {
            'time_s': np.asarray([t0, t1], dtype=np.float64),
            'breathing_rate_bpm': np.asarray([full_breath, full_breath], dtype=np.float64),
            'heart_rate_bpm': np.asarray([full_heart, full_heart], dtype=np.float64),
            'breathing_rate_raw_bpm': np.asarray([full_breath, full_breath], dtype=np.float64),
            'heart_rate_raw_bpm': np.asarray([full_heart, full_heart], dtype=np.float64),
            'window_s': float(n / float(fs_vital)),
            'step_s': float(step_n / float(fs_vital)),
        }

    centers = []
    breath_raw = []
    heart_raw = []
    heart_cont = []
    half = win_n // 2
    prev_heart = full_heart if math.isfinite(float(full_heart)) else None
    for center in range(half, n - (win_n - half) + 1, step_n):
        start = center - half
        stop = start + win_n
        seg = x[start:stop]
        if seg.size < 8:
            continue
        est = _extract_vitals_from_displacement(seg, fs_vital, args)
        br = float(est['breathing_rate_breaths_per_min'])
        hr_raw = float(est['heart_rate_beats_per_min'])
        hr = _choose_trend_heart_from_candidates(est, full_heart, prev_heart, args)
        centers.append((start + 0.5 * seg.size) / float(fs_vital))
        breath_raw.append(br)
        heart_raw.append(hr_raw)
        heart_cont.append(hr)
        if math.isfinite(float(hr)):
            prev_heart = float(hr)

    t = np.asarray(centers, dtype=np.float64)
    b_raw = np.asarray(breath_raw, dtype=np.float64)
    h_raw = np.asarray(heart_raw, dtype=np.float64)
    h = np.asarray(heart_cont, dtype=np.float64)

    # Smooth the display trend. The median filter removes isolated window errors;
    # the jump limiter and EMA remove non-physiological display jumps.
    kernel_s = max(0.0, float(getattr(args, 'rate_trend_median_s', 6.0)))
    kernel_n = int(round(kernel_s / max(step_s, 1e-9))) if kernel_s > 0.0 else 1
    if kernel_n % 2 == 0:
        kernel_n += 1
    h = _median_filter_1d(h, kernel_n)
    b = _median_filter_1d(b_raw, kernel_n)

    max_jump_per_s = max(0.0, float(getattr(args, 'rate_trend_max_jump_bpm_per_s', 2.0)))
    max_jump = max_jump_per_s * step_s
    h = _limit_rate_jumps(h, max_jump)
    b = _limit_rate_jumps(b, max(1.0, 0.75 * max_jump))

    ema_alpha = float(getattr(args, 'rate_trend_ema_alpha', 0.35))
    h = _ema_filter_1d(h, ema_alpha)
    b = _ema_filter_1d(b, ema_alpha)

    # Optional display blend toward the robust full-capture estimate.
    # Default is 0.0 so the dashboard shows the actual 30 s sliding-window trend.
    # Increase this only for a deliberately more stable display.
    blend = min(1.0, max(0.0, float(getattr(args, 'rate_trend_full_capture_blend', 0.00))))
    if math.isfinite(full_heart):
        h = (1.0 - blend) * h + blend * full_heart
    if math.isfinite(full_breath):
        b = (1.0 - min(0.50, blend)) * b + min(0.50, blend) * full_breath

    return {
        'time_s': t,
        'breathing_rate_bpm': np.asarray(b, dtype=np.float64),
        'heart_rate_bpm': np.asarray(h, dtype=np.float64),
        'breathing_rate_raw_bpm': b_raw,
        'heart_rate_raw_bpm': h_raw,
        'window_s': float(win_n / float(fs_vital)),
        'step_s': float(step_n / float(fs_vital)),
    }

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

    breathing = _fft_bandpass(displacement_m, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz), args=args)
    heart = _fft_bandpass(displacement_m, fs_vital, float(args.heart_low_hz), float(args.heart_high_hz), args=args)

    breath_est = _estimate_rate_fft(
        breathing, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz), args=args
    )
    heart_est = _estimate_heart_rate_fft(
        heart,
        fs_vital,
        float(args.heart_low_hz),
        float(args.heart_high_hz),
        float(breath_est["rate_per_min"]),
        args=args,
    )

    return {
        "displacement_m": displacement_m,
        "breathing": breathing,
        "heart": heart,
        "breathing_rate_breaths_per_min": float(breath_est["rate_per_min"]),
        "breathing_raw_bin_rate_breaths_per_min": float(breath_est["raw_bin_rate_per_min"]),
        "breathing_quality_db": float(breath_est["quality_db"]),
        "breathing_freqs_hz": breath_est["freqs_hz"],
        "breathing_power": breath_est["power"],
        "breathing_fft_n": int(breath_est["n_fft"]),
        "breathing_true_resolution_per_min": float(breath_est["true_resolution_per_min"]),
        "breathing_padded_spacing_per_min": float(breath_est["padded_spacing_per_min"]),
        "breathing_interpolation_delta_bins": float(breath_est["interpolation_delta_bins"]),
        "heart_rate_beats_per_min": float(heart_est["rate_per_min"]),
        "heart_raw_bin_rate_beats_per_min": float(heart_est["raw_bin_rate_per_min"]),
        "heart_quality_db": float(heart_est["quality_db"]),
        "heart_freqs_hz": heart_est["freqs_hz"],
        "heart_power": heart_est["power"],
        "heart_fft_n": int(heart_est["n_fft"]),
        "heart_true_resolution_per_min": float(heart_est["true_resolution_per_min"]),
        "heart_padded_spacing_per_min": float(heart_est["padded_spacing_per_min"]),
        "heart_interpolation_delta_bins": float(heart_est["interpolation_delta_bins"]),
        "heart_peak_selection_method": heart_est.get("peak_selection_method"),
        "heart_candidates": heart_est.get("heart_candidates", []),
        "heart_harmonic_warning": bool(heart_est.get("harmonic_warning", False)),
        "heart_nearest_breath_harmonic_order": heart_est.get("nearest_breath_harmonic_order"),
        "heart_nearest_breath_harmonic_bpm": heart_est.get("nearest_breath_harmonic_bpm"),
        "heart_nearest_breath_harmonic_distance_bpm": heart_est.get("nearest_breath_harmonic_distance_bpm"),
    }

def _extract_vitals_from_slow_complex(slow_complex, fs_vital, wavelength, args):
    displacement_m = _phase_to_displacement(slow_complex, fs_vital, wavelength, args)
    return _extract_vitals_from_displacement(displacement_m, fs_vital, args)

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
    seed_breath = _fft_bandpass(seed_disp, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz), args=args)

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

        breath = _fft_bandpass(disp, fs_vital, float(args.breath_low_hz), float(args.breath_high_hz), args=args)
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
        "roi_rejected_cells": rejected_cells,
    }
    return vitals, roi_info

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
            trend = _estimate_rate_trend(vitals["displacement_m"], fs_vital, args)
            subject.update({
                "selected_rx": -2,
                "rx_mode": rx_mode_text,
                "angle_method": angle_method_text,
                "angle_remove_static_mean": bool(args.angle_remove_static_mean),
                "rx_order": ra["rx_order"],
                "rx_spacing_lambda": float(args.rx_spacing_lambda),
                "rate_trend_time_s": trend["time_s"],
                "rate_trend_breath_bpm": trend["breathing_rate_bpm"],
                "rate_trend_heart_bpm": trend["heart_rate_bpm"],
                "rate_trend_window_s": trend["window_s"],
                "rate_trend_step_s": trend["step_s"],
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
                "heart_peak_selection_method": vitals.get("heart_peak_selection_method"),
                "heart_candidates": vitals.get("heart_candidates", []),
                "heart_harmonic_warning": bool(vitals.get("heart_harmonic_warning", False)),
                "heart_nearest_breath_harmonic_order": vitals.get("heart_nearest_breath_harmonic_order"),
                "heart_nearest_breath_harmonic_bpm": vitals.get("heart_nearest_breath_harmonic_bpm"),
                "heart_nearest_breath_harmonic_distance_bpm": vitals.get("heart_nearest_breath_harmonic_distance_bpm"),
            })
            subject.update(roi_info)
            _add_validation_and_vital_score(subject, args)
            subjects.append(subject)

            if not args.no_save:
                spaths = _make_subject_output_paths(results_dir, stem, subject["subject_index"])
                np.save(str(spaths["disp_npy"]), vitals["displacement_m"])
                np.save(str(spaths["breath_npy"]), vitals["breathing"])
                np.save(str(spaths["heart_npy"]), vitals["heart"])
                _write_timeseries_csv(spaths["csv"], time_s, vitals["displacement_m"], vitals["breathing"], vitals["heart"])
                _write_rate_trend_csv(spaths["rate_trend_csv"], trend["time_s"], trend["breathing_rate_bpm"], trend["heart_rate_bpm"])
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
                    _save_vital_spectrum_plot(
                        spaths["spectrum_plot"],
                        vitals,
                        float(args.breath_low_hz),
                        float(args.breath_high_hz),
                        float(args.heart_low_hz),
                        float(args.heart_high_hz),
                    )
                    _save_roi_diagnostic_plot(
                        spaths["roi_plot"],
                        subject,
                        ra,
                        range_axis_pos,
                        has_metric_axis,
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
            if args.make_plots:
                _save_range_angle_map_plot(paths["range_angle_plot"], ra, range_axis_pos, has_metric_axis, subjects)
                _save_subject_comparison_plot(paths["subject_comparison_plot"], subjects)
                _save_validation_dashboard_plot(
                    paths["dashboard_plot"],
                    subjects,
                    ra,
                    range_axis_pos,
                    has_metric_axis,
                    show_window=False,
                    gui_backend=getattr(args, "gui_backend", None),
                )
                out_paths_by_name["range_angle_plot"] = paths["range_angle_plot"]
                out_paths_by_name["subject_comparison_plot"] = paths["subject_comparison_plot"]
                out_paths_by_name["dashboard_plot"] = paths["dashboard_plot"]

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
        trend = _estimate_rate_trend(vitals["displacement_m"], fs_vital, args)
        subjects.append({
            "subject_index": 1,
            "selected_range_bin": int(selected_bin),
            "rate_trend_time_s": trend["time_s"],
            "rate_trend_breath_bpm": trend["breathing_rate_bpm"],
            "rate_trend_heart_bpm": trend["heart_rate_bpm"],
            "rate_trend_window_s": trend["window_s"],
            "rate_trend_step_s": trend["step_s"],
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
            "heart_peak_selection_method": vitals.get("heart_peak_selection_method"),
            "heart_candidates": vitals.get("heart_candidates", []),
            "heart_harmonic_warning": bool(vitals.get("heart_harmonic_warning", False)),
            "heart_nearest_breath_harmonic_order": vitals.get("heart_nearest_breath_harmonic_order"),
            "heart_nearest_breath_harmonic_bpm": vitals.get("heart_nearest_breath_harmonic_bpm"),
            "heart_nearest_breath_harmonic_distance_bpm": vitals.get("heart_nearest_breath_harmonic_distance_bpm"),
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
            _write_rate_trend_csv(paths["rate_trend_csv"], trend["time_s"], trend["breathing_rate_bpm"], trend["heart_rate_bpm"])
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

    for _subject in subjects:
        if "vital_score_db" not in _subject:
            _add_validation_and_vital_score(_subject, args)

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
        "vital_filtering": {
            "mode": str(getattr(args, "filter_mode", "auto")),
            "butterworth_order": int(getattr(args, "filter_order", 4)),
            "note": "auto uses scipy Butterworth zero-phase filtering when scipy is available; otherwise it falls back to FFT masking.",
        },
        "validation_mode": {
            "reference_breath_rates": _parse_reference_list(getattr(args, "reference_breath_rates", None)),
            "reference_heart_rates": _parse_reference_list(getattr(args, "reference_heart_rates", None)),
            "note": "Per-subject validation error is stored under each subject when reference values are supplied.",
        },
        "vital_frequency_estimation": {
            "method": "zero_padded_fft_with_quadratic_peak_interpolation" if bool(args.vital_fft_interpolate) else "zero_padded_fft_raw_peak",
            "heart_peak_method": str(getattr(args, "heart_peak_method", "harmonic_aware")),
            "heart_harmonic_orders": str(getattr(args, "heart_harmonic_orders", "2,3")),
            "heart_harmonic_reject_tolerance_bpm": float(getattr(args, "heart_harmonic_reject_tolerance_bpm", 4.0)),
            "heart_candidate_max_drop_db": float(getattr(args, "heart_candidate_max_drop_db", 10.0)),
            "zeropad_factor": float(args.vital_fft_zeropad_factor),
            "minimum_fft_size": int(args.vital_fft_min_size),
            "true_resolution_per_min": float(60.0 * fs_vital / float(n_frames)),
            "padded_spacing_per_min": float(60.0 * fs_vital / float(subjects[0].get("breathing_fft_n", max(1, n_frames))) if subjects else float("nan")),
            "note": "True resolution is limited by capture duration. Zero padding and interpolation improve peak readout, not physical separation of very close rates.",
        },
        "target_detection": {
            "detection_stage": "range_angle_motion_power_with_non_maximum_suppression",
            "vital_scoring_stage": "post_detection_breathing_and_heart_quality_score_per_subject",
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
        "rate_trend_estimation": {
            "window_s": float(getattr(args, "rate_trend_window_s", 30.0)),
            "step_s": float(getattr(args, "rate_trend_step_s", 2.0)),
            "note": "Sliding-window trend used for validation plots; final reported rates still come from the full analyzed capture.",
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

__all__ = [name for name in globals() if not name.startswith('__')]
