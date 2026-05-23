from __future__ import annotations
import os, pickle, hashlib, json
from dataclasses import asdict, is_dataclass
from typing import Any

def _stable(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (list, tuple)):
        return [_stable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _stable(v) for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    return obj

def make_key(*, dataset_tag: str, theta_spec, boot_spec) -> str:
    payload = {"dataset_tag": dataset_tag, "theta_spec": _stable(theta_spec), "boot_spec": _stable(boot_spec)}
    s = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:24]

def load(cache_folder: str, key: str):
    path = os.path.join(cache_folder, f"{key}.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)

def save(cache_folder: str, key: str, obj):
    os.makedirs(cache_folder, exist_ok=True)
    path = os.path.join(cache_folder, f"{key}.pkl")
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
