from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def edge_throughput_medges_s(num_edges: int, ms: float) -> float:
    return round(num_edges / ms / 1e3, 3)


def bandwidth_gbs(bytes_per_edge: int, num_edges: int, ms: float) -> float:
    return round(bytes_per_edge * num_edges / ms / 1e6, 3)


def speedup(ms_ref: float, ms_backend: float) -> float:
    return round(ms_ref / ms_backend, 3)


def bytes_per_edge(
    irreps_in_dim: int,
    irreps_sh_dim: int,
    weight_numel: int,
    irreps_out_dim: int,
) -> int:
    return (irreps_in_dim + irreps_sh_dim + weight_numel + irreps_out_dim) * 4


def collect_versions(extra: dict[str, str | None]) -> dict[str, Any]:
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
    except Exception:
        torch_version = None
        cuda_version = None

    versions: dict[str, Any] = {
        "torch": torch_version,
        "cuda": cuda_version,
    }
    try:
        import e3nn

        versions["e3nn"] = e3nn.__version__
    except Exception:
        versions["e3nn"] = None
    versions.update(extra)
    return versions


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
