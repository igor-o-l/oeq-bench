import builtins
import types

import pytest

from oeq_bench.config import BenchConfig
from oeq_bench.backends.cueq import build_channelwise_tensor_product, run_smoke


def test_run_smoke_reports_missing_cueq_cleanly(monkeypatch):
    monkeypatch.setattr("importlib.util.find_spec", lambda name: None)
    result = run_smoke(BenchConfig(device="cpu"))
    assert result["available"] is False
    assert "cuequivariance" in result["reason"]


def test_run_smoke_reports_broken_cueq_import_cleanly(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            return types.SimpleNamespace(float32=object())
        if name == "cuequivariance":
            return types.SimpleNamespace()
        if name == "cuequivariance_torch":
            raise ImportError("broken cuet")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = run_smoke(BenchConfig(device="cpu"))

    assert result["available"] is False
    assert "ImportError" in result["reason"]
    assert "broken cuet" in result["reason"]


def test_build_channelwise_tensor_product_when_cueq_installed():
    pytest.importorskip("cuequivariance")
    pytest.importorskip("cuequivariance_torch")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("cuEquivariance tensor-product constructor requires a usable CUDA device")
    cfg = BenchConfig(
        irreps_in1="4x0e+4x1o",
        irreps_in2="0e+1o",
        num_nodes=8,
        num_edges=16,
    )
    module = build_channelwise_tensor_product(cfg)
    assert hasattr(module, "forward")
