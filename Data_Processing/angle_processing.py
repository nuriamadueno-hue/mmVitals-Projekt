#!/usr/bin/env python3
"""Range/azimuth subject detection and ROI beamforming."""
from common import *
from range_processing import *

def _angle_grid_deg(min_deg, max_deg, step_deg):
    if step_deg <= 0:
        raise ValueError("angle step must be > 0")
    if max_deg < min_deg:
        raise ValueError("angle max must be >= angle min")
    count = int(math.floor((max_deg - min_deg) / step_deg)) + 1
    grid = min_deg + step_deg * np.arange(count, dtype=np.float64)
    if grid.size == 0 or grid[-1] < max_deg - 1e-9:
        grid = np.append(grid, float(max_deg))
    return grid

def _steering_vector_ula(num_rx, angle_deg, rx_spacing_lambda=0.5):
    """Return a simple azimuth steering vector for a uniform linear RX array.

    This assumes RX channels are ordered along the azimuth array and separated
    by rx_spacing_lambda wavelengths. For IWR1843/xWR16xx captures using only
    one TX, this gives a useful first-order azimuth AoA estimate from the 4 RX
    channels. If your hardware/channel order differs, use --invert-angle-sign
    or --rx-order to correct it.
    """
    idx = np.arange(int(num_rx), dtype=np.float64)
    theta = math.radians(float(angle_deg))
    phase = -2.0 * math.pi * float(rx_spacing_lambda) * np.sin(theta) * idx
    return np.exp(1j * phase).astype(np.complex128)

def _parse_rx_order(rx_order_text, num_rx):
    if rx_order_text is None or str(rx_order_text).strip() == "":
        return list(range(num_rx))
    parts = [x.strip() for x in str(rx_order_text).split(",") if x.strip() != ""]
    order = [int(x) for x in parts]
    if sorted(order) != list(range(num_rx)):
        raise ValueError("--rx-order must be a comma-separated permutation of 0..{}".format(num_rx - 1))
    return order

def _compute_range_angle_map(range_fft_pos, range_axis_pos, has_metric_axis, args):
    """Compute a moving-target range-angle power map for multi-person detection.

    range_fft_pos shape: (frames, rx, positive_range_bins)
    Returns a dictionary with map power shaped (candidate_ranges, angles).
    """
    n_frames, n_rx, n_bins = range_fft_pos.shape
    rx_order = _parse_rx_order(args.rx_order, n_rx)
    x_all = range_fft_pos[:, rx_order, :]

    if args.invert_angle_sign:
        min_angle = -float(args.angle_max_deg)
        max_angle = -float(args.angle_min_deg)
    else:
        min_angle = float(args.angle_min_deg)
        max_angle = float(args.angle_max_deg)

    angle_grid_internal = _angle_grid_deg(min_angle, max_angle, float(args.angle_step_deg))
    angle_grid_reported = -angle_grid_internal if args.invert_angle_sign else angle_grid_internal

    range_mask = _candidate_range_mask(
        range_axis_pos,
        has_metric_axis,
        n_bins,
        float(args.min_range_m),
        float(args.max_range_m),
        int(args.min_range_bin),
        args.max_range_bin,
    )
    candidate_bins = np.where(range_mask)[0]

    # Mean non-static range profile for plotting and fallback diagnostics.
    range_profile = np.mean(np.abs(x_all), axis=(0, 1))

    x_scan = x_all[:, :, candidate_bins]
    if args.angle_remove_static_mean:
        # Use moving energy for detection, not static reflectors from furniture/walls.
        x_scan = x_scan - np.mean(x_scan, axis=0, keepdims=True)

    power_map = np.zeros((candidate_bins.size, angle_grid_internal.size), dtype=np.float64)
    for a_idx, angle in enumerate(angle_grid_internal):
        steer = _steering_vector_ula(n_rx, angle, float(args.rx_spacing_lambda))
        weights = np.conj(steer) / float(n_rx)
        y = np.einsum('frb,r->fb', x_scan, weights)
        power_map[:, a_idx] = np.mean(np.abs(y) ** 2, axis=0)

    return {
        "power_map": power_map,
        "candidate_bins": candidate_bins,
        "angle_grid_internal": angle_grid_internal,
        "angle_grid_reported": angle_grid_reported,
        "range_profile": range_profile,
        "rx_order": rx_order,
        "rx_spacing_lambda": float(args.rx_spacing_lambda),
    }

def _detect_range_angle_targets(ra, range_axis_pos, has_metric_axis, args):
    """Find locally separated subject candidates in a range-angle map."""
    power_map = np.asarray(ra["power_map"], dtype=np.float64)
    candidate_bins = np.asarray(ra["candidate_bins"], dtype=int)
    angle_grid_reported = np.asarray(ra["angle_grid_reported"], dtype=np.float64)
    angle_grid_internal = np.asarray(ra["angle_grid_internal"], dtype=np.float64)

    if power_map.size == 0:
        return []

    median_power = float(np.median(power_map))
    max_power = float(np.max(power_map))
    if not np.isfinite(max_power) or max_power <= 0:
        return []

    min_rel_db = float(args.target_min_relative_db)
    min_abs_quality_db = float(args.target_min_quality_db)
    max_targets = max(1, int(args.max_subjects))

    # Convert suppression distances from engineering units to index distances.
    if has_metric_axis and len(range_axis_pos) > 1:
        diffs = np.diff(range_axis_pos)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        bin_spacing_m = float(np.median(diffs)) if diffs.size else 0.1
    else:
        bin_spacing_m = 0.1
    suppress_range_bins = max(1, int(round(float(args.target_min_separation_m) / max(bin_spacing_m, 1e-9))))
    suppress_angle_bins = max(1, int(round(float(args.target_min_separation_deg) / max(float(args.angle_step_deg), 1e-9))))

    work = power_map.copy()
    targets = []
    for subject_idx in range(max_targets):
        flat_idx = int(np.argmax(work))
        peak = float(work.flat[flat_idx])
        if not np.isfinite(peak) or peak <= 0:
            break

        rel_db = 10.0 * math.log10((peak + 1e-30) / (max_power + 1e-30))
        quality_db = 10.0 * math.log10((peak + 1e-30) / (median_power + 1e-30))
        if subject_idx > 0 and rel_db < min_rel_db:
            break
        if quality_db < min_abs_quality_db:
            break

        r_local, a_idx = np.unravel_index(flat_idx, work.shape)
        rbin = int(candidate_bins[r_local])
        angle_reported = float(angle_grid_reported[a_idx])
        angle_internal = float(angle_grid_internal[a_idx])
        selected_range = float(range_axis_pos[rbin]) if has_metric_axis else None

        targets.append({
            "subject_index": int(subject_idx + 1),
            "selected_range_bin": int(rbin),
            "selected_range_m": selected_range,
            "selected_angle_deg": angle_reported,
            "internal_angle_deg": angle_internal,
            "detection_power": peak,
            "detection_relative_power_db": float(rel_db),
            "detection_quality_db": float(quality_db),
            # Backward-compatible aliases for older summaries/prints.
            "target_power": peak,
            "target_relative_power_db": float(rel_db),
            "target_quality_db": float(quality_db),
        })

        # Non-maximum suppression around this target so adjacent bins/angles are not double-counted.
        r0 = max(0, r_local - suppress_range_bins)
        r1 = min(work.shape[0], r_local + suppress_range_bins + 1)
        a0 = max(0, a_idx - suppress_angle_bins)
        a1 = min(work.shape[1], a_idx + suppress_angle_bins + 1)
        if getattr(args, "target_suppress_same_range_all_angles", True):
            # With a 4-RX-only ULA, angular sidelobes can make one person appear as
            # several angle peaks at the same range. Suppress the full local range
            # interval by default. Use --allow-same-range-targets to disable this.
            work[r0:r1, :] = -np.inf
        else:
            work[r0:r1, a0:a1] = -np.inf

    return targets

def _beamform_slow_complex_for_target(range_fft_pos, target, rx_order, rx_spacing_lambda):
    n_rx = len(rx_order)
    rbin = int(target["selected_range_bin"])
    x_target = range_fft_pos[:, rx_order, rbin]
    steer = _steering_vector_ula(n_rx, float(target["internal_angle_deg"]), float(rx_spacing_lambda))
    return x_target @ (np.conj(steer) / float(n_rx))

def _make_subject_roi_cells(target, ra, range_axis_pos, has_metric_axis, args):
    """Return range/angle cells around the target that represent the chest ROI.

    The previous multi-subject version analyzed exactly one range-angle cell per
    subject. That can be too narrow: the chest reflection can span adjacent
    range bins and a few adjacent AoA bins. This ROI keeps cells around the
    detected peak and rejects weak local sidelobes using a local relative
    threshold.
    """
    power_map = np.asarray(ra["power_map"], dtype=np.float64)
    candidate_bins = np.asarray(ra["candidate_bins"], dtype=int)
    angles_internal = np.asarray(ra["angle_grid_internal"], dtype=np.float64)
    angles_reported = np.asarray(ra["angle_grid_reported"], dtype=np.float64)

    rbin = int(target["selected_range_bin"])
    a_internal = float(target["internal_angle_deg"])
    r_matches = np.where(candidate_bins == rbin)[0]
    if not r_matches.size:
        return []
    r_local_center = int(r_matches[0])
    a_center = int(np.argmin(np.abs(angles_internal - a_internal)))

    spacing_m = _range_bin_spacing_m(range_axis_pos, has_metric_axis)
    half_range_bins = max(0, int(round(float(args.chest_roi_range_m) / max(spacing_m, 1e-9))))
    half_range_bins = max(half_range_bins, int(args.chest_roi_min_range_bins))
    half_angle_bins = max(0, int(round(float(args.chest_roi_angle_deg) / max(float(args.angle_step_deg), 1e-9))))

    # Prefer the centre of the detected chest target while still allowing the
    # ROI to include adjacent bins. This prevents a stronger neighbouring edge
    # or multipath bin from dominating the averaged motion.
    sigma_range_bins = max(1.0, float(half_range_bins) * 0.65)
    sigma_angle_bins = max(1.0, float(half_angle_bins) * 0.65)

    r0 = max(0, r_local_center - half_range_bins)
    r1 = min(power_map.shape[0], r_local_center + half_range_bins + 1)
    a0 = max(0, a_center - half_angle_bins)
    a1 = min(power_map.shape[1], a_center + half_angle_bins + 1)

    local = power_map[r0:r1, a0:a1]
    if local.size == 0 or not np.any(np.isfinite(local)):
        return []

    local_peak = float(np.nanmax(local))
    min_power = local_peak * (10.0 ** (float(args.chest_roi_min_relative_db) / 10.0))

    cells = []
    for rr in range(r0, r1):
        for aa in range(a0, a1):
            pwr = float(power_map[rr, aa])
            if not np.isfinite(pwr) or pwr < min_power:
                continue
            dr = float(rr - r_local_center) / sigma_range_bins
            da = float(aa - a_center) / sigma_angle_bins
            distance_weight = math.exp(-0.5 * (dr * dr + da * da))
            cells.append({
                "range_local_index": int(rr),
                "angle_index": int(aa),
                "distance_weight": float(distance_weight),
                "range_bin": int(candidate_bins[rr]),
                "range_m": float(range_axis_pos[int(candidate_bins[rr])]) if has_metric_axis else None,
                "angle_internal_deg": float(angles_internal[aa]),
                "angle_reported_deg": float(angles_reported[aa]),
                "power": pwr,
            })

    # Keep strongest centre-weighted ROI cells first so the chest centre dominates over fringe cells.
    cells.sort(key=lambda c: c["power"] * c.get("distance_weight", 1.0), reverse=True)
    max_cells = max(1, int(args.chest_roi_max_cells))
    return cells[:max_cells]

def _beamform_slow_complex_for_cell(range_fft_pos, range_bin, angle_internal_deg, rx_order, rx_spacing_lambda):
    n_rx = len(rx_order)
    x_target = range_fft_pos[:, rx_order, int(range_bin)]
    steer = _steering_vector_ula(n_rx, float(angle_internal_deg), float(rx_spacing_lambda))
    return x_target @ (np.conj(steer) / float(n_rx))

def _detect_multi_subjects(range_fft_pos, range_axis_pos, has_metric_axis, args):
    ra = _compute_range_angle_map(range_fft_pos, range_axis_pos, has_metric_axis, args)
    targets = _detect_range_angle_targets(ra, range_axis_pos, has_metric_axis, args)
    return targets, ra

__all__ = [name for name in globals() if not name.startswith('__')]
