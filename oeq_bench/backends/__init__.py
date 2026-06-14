from __future__ import annotations

from importlib.util import find_spec
from typing import Iterable


def build_instructions(irreps_in1: Iterable, irreps_in2: Iterable, irreps_out: Iterable) -> list[tuple]:
    instructions: list[tuple] = []
    for i, (_, ir1) in enumerate(irreps_in1):
        for j, (_, ir2) in enumerate(irreps_in2):
            for k, (_, iro) in enumerate(irreps_out):
                if iro in ir1 * ir2:
                    instructions.append((i, j, k, "uvu", True, 1.0))
    return instructions


def backend_available(module_name: str) -> bool:
    return find_spec(module_name) is not None
