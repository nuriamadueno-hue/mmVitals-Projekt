#!/usr/bin/env python3
"""Small helpers for reading numeric values from nested metadata."""
from common import *

def _safe_float(value, default=None):
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default

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
