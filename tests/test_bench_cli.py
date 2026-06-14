import json
import subprocess
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
    assert {"device", "versions", "config", "results", "validation", "runtime_check", "kernel_config"} <= set(payload)


def test_ncu_profile_executes_ncu_and_merges_metrics_without_importing_torch(tmp_path, monkeypatch):
    from oeq_bench import bench

    class TorchTrap:
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torch":
                raise AssertionError("torch import should not be attempted by the ncu orchestrator")
            return None

    monkeypatch.setattr(sys, "meta_path", [TorchTrap(), *sys.meta_path])

    calls = []

    def fake_run_ncu_command(command, backend, output_stem):
        calls.append((command, backend, output_stem))
        output_stem.with_suffix(".ncu-rep").write_text("fake report")
        child_json = bench.ncu_child_json_path(output_stem)
        child_payload = bench.empty_result_payload(BenchConfig(backend=backend, out=child_json))
        child_payload["results"][backend] = {"ms": 1.25}
        child_payload["versions"] = {"torch": "fake"}
        child_json.write_text(json.dumps(child_payload))
        return {"command": command, "used_sudo": False}

    monkeypatch.setattr(bench, "run_ncu_command", fake_run_ncu_command)
    monkeypatch.setattr(
        bench,
        "parse_ncu_report",
        lambda path: {
            "achieved_occupancy": 0.82,
            "dram_throughput_pct": 67.4,
            "l2_hit_rate": 0.45,
            "stall_reason": "memory_dependency",
        },
    )

    out = tmp_path / "profile.json"
    payload = bench.run_benchmark(BenchConfig(profile="ncu", out=out))

    assert len(calls) == 1
    command, backend, output_stem = calls[0]
    assert backend == "oeq"
    assert output_stem == tmp_path / "profile"
    assert command[:4] == ["ncu", "--set", "full", "--target-processes"]
    assert payload["results"]["oeq"] == {"ms": 1.25}
    assert payload["versions"] == {"torch": "fake"}
    assert payload["ncu"]["oeq"]["dram_throughput_pct"] == 67.4
    assert payload["ncu"]["oeq"]["report_path"] == str(tmp_path / "profile.ncu-rep")
    assert payload["ncu"]["oeq"]["benchmark_json"] == str(tmp_path / "profile_bench.json")
    assert payload["ncu"]["oeq"]["used_sudo"] is False
    assert json.loads(out.read_text()) == payload


def test_ncu_profile_both_backends_uses_distinct_reports(tmp_path, monkeypatch):
    from oeq_bench import bench

    calls = []

    def fake_run_ncu_command(command, backend, output_stem):
        calls.append((backend, output_stem))
        output_stem.with_suffix(".ncu-rep").write_text("fake report")
        child_json = bench.ncu_child_json_path(output_stem)
        child_payload = bench.empty_result_payload(BenchConfig(backend=backend, out=child_json))
        child_payload["results"][backend] = {"ms": 1.0 if backend == "oeq" else 2.0}
        child_json.write_text(json.dumps(child_payload))
        return {"command": command, "used_sudo": backend == "cueq"}

    monkeypatch.setattr(bench, "run_ncu_command", fake_run_ncu_command)
    monkeypatch.setattr(bench, "parse_ncu_report", lambda path: {"dram_throughput_pct": 50.0})

    out = tmp_path / "profile.json"
    payload = bench.run_benchmark(BenchConfig(profile="ncu", backend="both", out=out))

    assert calls == [("oeq", tmp_path / "profile_oeq"), ("cueq", tmp_path / "profile_cueq")]
    assert payload["results"]["oeq"] == {"ms": 1.0}
    assert payload["results"]["cueq"] == {"ms": 2.0}
    assert payload["ncu"]["oeq"]["report_path"] == str(tmp_path / "profile_oeq.ncu-rep")
    assert payload["ncu"]["cueq"]["report_path"] == str(tmp_path / "profile_cueq.ncu-rep")
    assert payload["ncu"]["cueq"]["used_sudo"] is True


def test_ncu_profile_fails_when_report_is_missing(tmp_path, monkeypatch):
    from oeq_bench import bench

    def fake_run_ncu_command(command, backend, output_stem):
        bench.ncu_child_json_path(output_stem).write_text(json.dumps(bench.empty_result_payload(BenchConfig())))
        return {"command": command, "used_sudo": False}

    monkeypatch.setattr(bench, "run_ncu_command", fake_run_ncu_command)

    with pytest.raises(RuntimeError, match="ncu did not create report"):
        bench.run_benchmark(BenchConfig(profile="ncu", out=tmp_path / "profile.json"))


def test_ncu_profile_removes_stale_child_json_before_first_run(tmp_path, monkeypatch):
    from oeq_bench import bench

    out = tmp_path / "profile.json"
    output_stem = tmp_path / "profile"
    child_json = bench.ncu_child_json_path(output_stem)
    child_json.write_text(json.dumps({"results": {"oeq": {"ms": "stale"}}}))

    def fake_run(command, **kwargs):
        assert command[0] == "ncu"
        assert not child_json.exists()
        output_stem.with_suffix(".ncu-rep").write_text("fresh report")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="did not create JSON"):
        bench.run_benchmark(BenchConfig(profile="ncu", out=out))
    assert not child_json.exists()


def test_run_ncu_command_retries_permission_failure_with_sudo(tmp_path, monkeypatch):
    from oeq_bench import bench

    calls = []
    child_json = bench.ncu_child_json_path(tmp_path / "profile")
    child_json.write_text("{}")

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "ncu":
            assert not child_json.exists()
            raise subprocess.CalledProcessError(
                returncode=1,
                cmd=command,
                stderr="ERR_NVGPUCTRPERM: profiling is not permitted",
            )
        if command[0] == "sudo" and any(part.endswith("/ncu") or part == "ncu" for part in command):
            assert not child_json.exists()
            child_json.write_text("{}")
            (tmp_path / "profile.ncu-rep").write_text("report")
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        if command[0] == "sudo" and command[2] == "chown":
            assert child_json.exists()
            return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")
        assert not child_json.exists()
        return subprocess.CompletedProcess(command, 0, stdout="ok", stderr="")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)
    monkeypatch.setattr(bench.shutil, "which", lambda name: "/usr/local/cuda/bin/ncu" if name == "ncu" else None)

    result = bench.run_ncu_command(["ncu", "-o", str(tmp_path / "profile")], "oeq", tmp_path / "profile")

    assert calls[0][0] == "ncu"
    assert calls[1][:2] == ["sudo", "-n"]
    assert "/usr/local/cuda/bin/ncu" in calls[1]
    assert calls[2][:3] == ["sudo", "-n", "chown"]
    assert str(tmp_path / "profile.ncu-rep") in calls[2]
    assert str(child_json) in calls[2]
    assert result["used_sudo"] is True


def test_run_ncu_command_wraps_non_permission_failure(tmp_path, monkeypatch):
    from oeq_bench import bench

    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(returncode=2, cmd=command, stderr="bad option")

    monkeypatch.setattr(bench.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="ncu failed for backend oeq"):
        bench.run_ncu_command(["ncu", "--bad"], "oeq", tmp_path / "profile")


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
    fake_oeq.describe_oeq_kernel = lambda conv, cfg: {"requested_block_size": cfg.block_size}
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
