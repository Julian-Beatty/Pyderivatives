# pyderivatives/density_evaluation/io.py

from pathlib import Path
import pickle


def load_forecast_part(path):
    path = Path(path)

    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, dict) and "forecasts" in obj:
        return obj

    return {
        "forecasts": obj,
        "path": str(path),
    }


def merge_forecast_parts(paths, sort=True):
    payloads = [load_forecast_part(p) for p in paths]

    forecasts = []
    metadata = []

    for payload in payloads:
        forecasts.extend(payload["forecasts"])
        metadata.append({
            k: v
            for k, v in payload.items()
            if k != "forecasts"
        })

    if sort:
        forecasts = sorted(forecasts, key=lambda x: x.date)

    return {
        "forecasts": forecasts,
        "metadata": metadata,
    }