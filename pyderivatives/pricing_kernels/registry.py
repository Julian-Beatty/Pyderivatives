from __future__ import annotations

from typing import Dict, List


TRANSFORM_REGISTRY: Dict[str, type] = {}


def register_transform(name: str):
    """Register a measure transformation class."""
    def decorator(cls):
        key = str(name).strip().lower()
        if key in TRANSFORM_REGISTRY:
            raise ValueError(f"Transform method '{key}' is already registered.")
        TRANSFORM_REGISTRY[key] = cls
        cls.method_name = key
        return cls
    return decorator


def get_transform(name: str, **kwargs):
    key = str(name).strip().lower()
    if key not in TRANSFORM_REGISTRY:
        raise ValueError(
            f"Unknown transform '{name}'. Available methods: {available_transforms()}"
        )
    return TRANSFORM_REGISTRY[key](**kwargs)


def available_transforms() -> List[str]:
    return sorted(TRANSFORM_REGISTRY.keys())