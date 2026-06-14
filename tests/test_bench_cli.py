import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from oeq_bench.config import BenchConfig
from oeq_bench.bench import main


def test_dry_run_exits_zero(capsys):
    assert main(["--dry-run", "--profile", "ncu", "--backend", "oeq"]) == 0
    out = capsys.readouterr().out
    assert "BenchConfig" in out
    assert "ncu --set full --target-processes all" in out
    assert "oeq_bench.profiling" in out
    assert "--backend oeq" in out


def test_json_schema_helper_contains_spec_keys(tmp_path, monkeypatch):
    from oeq_bench import bench

    def fake_run(cfg):
        payload = bench.empty_result_payload(cfg)
        payload["device"] = "fake"
        payload["compute_capability"] = "0.0"
        bench.write_json(cfg.out, payload)
        return payload

    monkeypatch.setattr(bench, "run_benchmark", fake_run)
    out = tmp_path / "results.json"
    assert main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert {"device", "versions", "config", "results", "validation", "runtime_check"} <= set(payload)


def test_ncu_profile_without_dry_run_fails_before_importing_torch(monkeypatch):
    from oeq_bench import bench

    class TorchTrap:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch":
                raise AssertionError("torch import should not be attempted for ncu early failure")
            return None

    monkeypatch.setattr(sys, "meta_path", [TorchTrap(), *sys.meta_path])

    with pytest.raises(RuntimeError, match="ncu profiling execution is not wired yet"):
        bench.run_benchmark(BenchConfig(profile="ncu"))


def test_ncu_dry_run_both_backends_uses_distinct_output_stems(tmp_path, capsys):
    out = tmp_path / "profile.json"
    assert main(["--dry-run", "--profile", "ncu", "--backend", "both", "--out", str(out)]) == 0

    lines = capsys.readouterr().out.splitlines()
    commands = [line for line in lines if line.startswith("ncu ")]
    assert len(commands) == 2
    assert any("--backend oeq" in line and str(tmp_path / "profile_oeq") in line for line in commands)
    assert any("--backend cueq" in line and str(tmp_path / "profile_cueq") in line for line in commands)


def test_validation_failure_writes_diagnostics_and_skips_oeq_timing(tmp_path, monkeypatch):
    from oeq_bench import bench

    class FakeNoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (9, 0),
            get_device_name=lambda index: "fake gpu",
        ),
        no_grad=lambda: FakeNoGrad(),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_irreps = SimpleNamespace(dim=4)
    fake_tp = SimpleNamespace(
        weight_numel=3,
        irreps_in1=fake_irreps,
        irreps_in2=fake_irreps,
        irreps_out=fake_irreps,
    )
    monkeypatch.setattr(bench, "build_e3nn_tp", lambda cfg: fake_tp)

    class FakeTensor:
        def detach(self):
            return self

    monkeypatch.setattr(
        bench,
        "make_graph_data",
        lambda cfg, weight_numel: (FakeTensor(), FakeTensor(), FakeTensor(), "src", "dst"),
    )
    monkeypatch.setattr(bench, "e3nn_scatter", lambda *args: "e3nn-output")

    time_calls = []

    def fake_time_cuda_events(fn, warmup, repeats):
        time_calls.append(fn)
        return 2.0

    monkeypatch.setattr(bench, "time_cuda_events", fake_time_cuda_events)

    fake_oeq = ModuleType("oeq_bench.backends.oeq")
    fake_oeq.build_oeq_conv = lambda cfg: (
        SimpleNamespace(weight_numel=3),
        SimpleNamespace(forward=lambda *args: "oeq-output"),
    )
    fake_oeq.version = lambda: "fake-oeq"
    monkeypatch.setitem(sys.modules, "oeq_bench.backends.oeq", fake_oeq)

    validation = {"fwd_ok": False, "grad_ok": True, "rows_is_dst": None, "fwd_err": 1.0}
    fake_validation = ModuleType("oeq_bench.validation")
    fake_validation.validate_oeq = lambda *args: validation
    monkeypatch.setitem(sys.modules, "oeq_bench.validation", fake_validation)

    out = tmp_path / "invalid.json"
    cfg = BenchConfig(validate=True, skip_runtime_check=True, out=out)
    with pytest.raises(RuntimeError, match="validation failed"):
        bench.run_benchmark(cfg)

    assert len(time_calls) == 1
    payload = json.loads(out.read_text())
    assert payload["validation"] == validation
    assert "oeq" not in payload["results"]
