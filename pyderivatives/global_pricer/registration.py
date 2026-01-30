# pyderivatives/global_pricer/registration.py
from __future__ import annotations
from typing import Dict, Type, Protocol

class GlobalModelProto(Protocol):
    ...

_MODEL_REGISTRY: Dict[str, Type[GlobalModelProto]] = {}

def register_model(name: str):
    key = name.lower()

    def decorator(cls: Type[GlobalModelProto]):
        _MODEL_REGISTRY[key] = cls
        return cls

    return decorator

def get_model(name: str):
    key = name.lower()
    if key not in _MODEL_REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[key]

def available_models() -> list[str]:
    return sorted(_MODEL_REGISTRY.keys())
