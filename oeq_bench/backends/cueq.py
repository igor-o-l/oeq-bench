from __future__ import annotations

import importlib.util


def _missing() -> str | None:
    for name in ("cuequivariance", "cuequivariance_torch"):
        if importlib.util.find_spec(name) is None:
            return name
    return None


def build_channelwise_tensor_product(cfg):
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    irreps_in1 = cue.Irreps("O3", cfg.irreps_in1)
    irreps_in2 = cue.Irreps("O3", cfg.irreps_in2)
    irreps_out = cue.Irreps("O3", cfg.irreps_out)
    descriptor = cue.descriptors.channelwise_tensor_product(irreps_in1, irreps_in2, irreps_out)
    return cuet.SegmentedPolynomial(descriptor.polynomial, method="uniform_1d").to(cfg.device)


def run_smoke(cfg) -> dict:
    missing = _missing()
    if missing:
        return {"available": False, "reason": f"{missing} is not installed"}

    try:
        import torch
        import cuequivariance as cue
        import cuequivariance_torch as cuet

        linear = cuet.Linear(
            cue.Irreps("O3", cfg.irreps_in1),
            cue.Irreps("O3", cfg.irreps_out),
            shared_weights=True,
            internal_weights=True,
            device=cfg.device,
            dtype=getattr(torch, cfg.dtype),
        )
        x = torch.randn(
            4,
            cue.Irreps("O3", cfg.irreps_in1).dim,
            device=cfg.device,
            dtype=getattr(torch, cfg.dtype),
        )
        y = linear(x)
        tp = build_channelwise_tensor_product(cfg)
        return {
            "available": True,
            "linear_shape": list(y.shape),
            "channelwise_tp": type(tp).__name__,
        }
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}


def version() -> str | None:
    try:
        import cuequivariance_torch as cuet

        return getattr(cuet, "__version__", "unknown")
    except Exception:
        return None
