from __future__ import annotations

import os
import json
import pickle
import hashlib
from dataclasses import asdict, is_dataclass
from typing import Any

import numpy as np



# ============================================================
# Cache utilities
# ============================================================

def _stable_for_hash(obj: Any) -> Any:
    if is_dataclass(obj):
        return _stable_for_hash(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _stable_for_hash(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_stable_for_hash(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return {
            "shape": obj.shape,
            "mean": float(np.nanmean(obj)) if obj.size else np.nan,
            "std": float(np.nanstd(obj)) if obj.size else np.nan,
            "min": float(np.nanmin(obj)) if obj.size else np.nan,
            "max": float(np.nanmax(obj)) if obj.size else np.nan,
        }
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    return obj


def _cache_key(payload: dict) -> str:
    s = json.dumps(_stable_for_hash(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:24]


def _cache_load(folder: str, key: str):
    path = os.path.join(folder, f"{key}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _cache_save(folder: str, key: str, obj):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path