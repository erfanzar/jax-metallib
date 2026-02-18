from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BenchCase:
    name: str
    build: Callable[[np.random.Generator], tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]]]
    description: str = ""


_REGISTRY: dict[str, BenchCase] = {}


def register_case(case: BenchCase) -> None:
    if case.name in _REGISTRY:
        raise ValueError(f"Duplicate benchmark case name: {case.name}")
    _REGISTRY[case.name] = case


def get_case(name: str) -> BenchCase:
    return _REGISTRY[name]


def all_cases() -> list[BenchCase]:
    return [c for _, c in sorted(_REGISTRY.items(), key=lambda kv: kv[0])]


def load_all() -> None:
    # Import side-effects: registers cases.
    from . import anchors as _anchors  # noqa: F401
    from . import linalg as _linalg  # noqa: F401
