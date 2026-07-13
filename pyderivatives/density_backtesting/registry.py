# registry.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence, Any


@dataclass
class ModelRegistry:
    models: list[Any] = field(default_factory=list)

    def __init__(self, *models):
        self.models = list(models)

    def add(self, model):
        self.models.append(model)
        return self

    def remove(self, name: str):
        self.models = [m for m in self.models if m.name != name]
        return self

    @property
    def names(self):
        return [m.name for m in self.models]

    def __iter__(self):
        return iter(self.models)

    def __len__(self):
        return len(self.models)

    def __getitem__(self, idx):
        return self.models[idx]


@dataclass
class TestRegistry:
    tests: list[Any] = field(default_factory=list)

    def __init__(self, *tests):
        self.tests = list(tests)

    def add(self, test):
        self.tests.append(test)
        return self

    def remove(self, test_id: str):
        self.tests = [t for t in self.tests if t.test_id != test_id]
        return self

    @property
    def ids(self):
        return [t.test_id for t in self.tests]

    def __iter__(self):
        return iter(self.tests)

    def __len__(self):
        return len(self.tests)

    def __getitem__(self, idx):
        return self.tests[idx]