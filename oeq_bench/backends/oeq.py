from __future__ import annotations

from . import build_instructions


def build_oeq_conv(cfg):
    import openequivariance as oeq

    irreps_in1 = oeq.Irreps(cfg.irreps_in1)
    irreps_in2 = oeq.Irreps(cfg.irreps_in2)
    irreps_out = oeq.Irreps(cfg.irreps_out)
    problem = oeq.TPProblem(
        irreps_in1,
        irreps_in2,
        irreps_out,
        build_instructions(irreps_in1, irreps_in2, irreps_out),
        shared_weights=False,
        internal_weights=False,
    )
    conv = oeq.TensorProductConv(problem, deterministic=False)
    return problem, conv


def run_oeq(conv, X, Y, W, rows, cols):
    return conv.forward(X, Y, W, rows, cols)


def version() -> str | None:
    try:
        import openequivariance as oeq

        return getattr(oeq, "__version__", "unknown")
    except Exception:
        return None
