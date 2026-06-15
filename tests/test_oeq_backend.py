from types import SimpleNamespace
import sys

import pytest

from oeq_bench.config import BenchConfig
from oeq_bench.backends.oeq import build_oeq_conv, describe_oeq_kernel
import oeq_bench.backends.oeq as oeq_backend


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


def test_build_oeq_conv_passes_requested_block_size(monkeypatch):
    calls = {}

    fake_oeq = SimpleNamespace()
    fake_oeq.Irreps = lambda spec: spec

    class FakeTPProblem:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class FakeTensorProductConv:
        def __init__(self, problem, *, deterministic, block_size, l1_carveout):
            calls["problem"] = problem
            calls["deterministic"] = deterministic
            calls["block_size"] = block_size
            calls["l1_carveout"] = l1_carveout

    fake_oeq.TPProblem = FakeTPProblem
    fake_oeq.TensorProductConv = FakeTensorProductConv
    monkeypatch.setitem(sys.modules, "openequivariance", fake_oeq)
    monkeypatch.setattr(oeq_backend, "build_instructions", lambda *args: [])

    build_oeq_conv(BenchConfig(block_size=128, l1_carveout=0))

    assert isinstance(calls["problem"], FakeTPProblem)
    assert calls["deterministic"] is False
    assert calls["block_size"] == 128
    assert calls["l1_carveout"] == 0


def test_build_oeq_conv_requires_block_size_capable_oeq(monkeypatch):
    fake_oeq = SimpleNamespace()
    fake_oeq.Irreps = lambda spec: spec

    class FakeTPProblem:
        def __init__(self, *args, **kwargs):
            pass

    class FakeTensorProductConv:
        def __init__(self, problem, *, deterministic, block_size):
            pass

    fake_oeq.TPProblem = FakeTPProblem
    fake_oeq.TensorProductConv = FakeTensorProductConv
    fake_oeq.__version__ = "0.6.6"
    monkeypatch.setitem(sys.modules, "openequivariance", fake_oeq)
    monkeypatch.setattr(oeq_backend, "build_instructions", lambda *args: [])

    with pytest.raises(RuntimeError, match="requires .*l1_carveout"):
        build_oeq_conv(BenchConfig(block_size=128))


def test_describe_oeq_kernel_reports_actual_forward_launch_config():
    cfg = BenchConfig(block_size=512, l1_carveout=0)
    conv = SimpleNamespace(
        hash=123,
        kernel_prop={"l1_carveout": 0},
        forward_schedule=SimpleNamespace(
            launch_config=SimpleNamespace(
                num_blocks=84,
                num_threads=192,
                warp_size=32,
                smem=2048,
            )
        ),
    )

    assert describe_oeq_kernel(conv, cfg) == {
        "requested_block_size": 512,
        "requested_block_size_effective": False,
        "requested_l1_carveout": 0,
        "requested_l1_carveout_effective": True,
        "actual_l1_carveout": 0,
        "forward": {
            "num_blocks": 84,
            "num_threads": 192,
            "warp_size": 32,
            "warps_per_block": 6,
            "smem": 2048,
        },
        "kernel_hash": 123,
    }


def test_describe_oeq_kernel_marks_requested_block_size_effective():
    cfg = BenchConfig(block_size=128)
    conv = SimpleNamespace(
        hash=123,
        kernel_prop={"l1_carveout": -1},
        forward_schedule=SimpleNamespace(
            launch_config=SimpleNamespace(
                num_blocks=84,
                num_threads=128,
                warp_size=32,
                smem=2048,
            )
        ),
    )

    assert describe_oeq_kernel(conv, cfg)["requested_block_size_effective"] is True
