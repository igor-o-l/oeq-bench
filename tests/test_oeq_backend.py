from types import SimpleNamespace

import pytest

from oeq_bench.config import BenchConfig
from oeq_bench.backends.oeq import build_oeq_conv, describe_oeq_kernel


def test_build_oeq_conv_small_problem_has_weights():
    pytest.importorskip("openequivariance")
    cfg = BenchConfig(
        irreps_in1="4x0e+4x1o",
        irreps_in2="0e+1o",
        num_nodes=8,
        num_edges=16,
    )
    problem, conv = build_oeq_conv(cfg)
    assert problem.weight_numel > 0
    assert hasattr(conv, "forward")


def test_describe_oeq_kernel_reports_actual_forward_launch_config():
    cfg = BenchConfig(block_size=512)
    conv = SimpleNamespace(
        hash=123,
        forward_schedule=SimpleNamespace(
            launch_config=SimpleNamespace(num_blocks=84, num_threads=192, warp_size=32, smem=2048)
        ),
    )

    assert describe_oeq_kernel(conv, cfg) == {
        "requested_block_size": 512,
        "requested_block_size_effective": False,
        "forward": {
            "num_blocks": 84,
            "num_threads": 192,
            "warp_size": 32,
            "warps_per_block": 6,
            "smem": 2048,
        },
        "kernel_hash": 123,
    }
