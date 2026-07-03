#!/usr/bin/env python3
# Metadata helpers split from ADC_To_Vital_Signs.py.

from common import *

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


__all__ = [name for name in globals() if not name.startswith("__")]
