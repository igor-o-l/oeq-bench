from __future__ import annotations

import inspect

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
    required_params = {"block_size", "l1_carveout"}
    missing_params = required_params - set(inspect.signature(oeq.TensorProductConv).parameters)
    if missing_params:
        version = getattr(oeq, "__version__", "unknown")
        raise RuntimeError(
            "oeq-bench tuning requires an OpenEquivariance build whose "
            "TensorProductConv constructor accepts "
            f"{', '.join(sorted(required_params))}; missing {', '.join(sorted(missing_params))}. "
            f"Found version {version!r}. "
            "Install the MLIPs external/OpenEquivariance fork or use a compatible OEQ release."
        )
    conv = oeq.TensorProductConv(
        problem,
        deterministic=False,
        block_size=cfg.block_size,
        l1_carveout=cfg.l1_carveout,
    )
    return problem, conv


def _launch_config_dict(schedule) -> dict:
    launch_config = schedule.launch_config
    num_threads = int(launch_config.num_threads)
    warp_size = int(launch_config.warp_size)
    return {
        "num_blocks": int(launch_config.num_blocks),
        "num_threads": num_threads,
        "warp_size": warp_size,
        "warps_per_block": num_threads // warp_size if warp_size else None,
        "smem": int(launch_config.smem),
    }


def describe_oeq_kernel(conv, cfg) -> dict:
    forward = _launch_config_dict(conv.forward_schedule)
    actual_l1_carveout = conv.kernel_prop.get("l1_carveout", -1)
    actual_l1_carveout = None if actual_l1_carveout == -1 else int(actual_l1_carveout)
    return {
        "requested_block_size": cfg.block_size,
        "requested_block_size_effective": forward["num_threads"] == cfg.block_size,
        "requested_l1_carveout": cfg.l1_carveout,
        "requested_l1_carveout_effective": None
        if cfg.l1_carveout is None
        else actual_l1_carveout == cfg.l1_carveout,
        "actual_l1_carveout": actual_l1_carveout,
        "forward": forward,
        "kernel_hash": conv.hash,
    }


def run_oeq(conv, X, Y, W, rows, cols):
    return conv.forward(X, Y, W, rows, cols)


def version() -> str | None:
    try:
        import openequivariance as oeq

        return getattr(oeq, "__version__", "unknown")
    except Exception:
        return None
