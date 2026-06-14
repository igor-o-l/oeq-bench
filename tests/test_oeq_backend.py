import pytest

from oeq_bench.config import BenchConfig
from oeq_bench.backends.oeq import build_oeq_conv


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
