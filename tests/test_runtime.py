import importlib
import json

from oeq_bench import runtime
from oeq_bench.runtime import classify_runtime_paths, runtime_report


def test_classify_runtime_paths_groups_openmp_and_blas():
    paths = [
        "/opt/torch/lib/libgomp-a34b3233.so.1",
        "/opt/conda/lib/libgomp.so.1",
        "/opt/conda/lib/libopenblas.so.0",
    ]
    grouped = classify_runtime_paths(paths)
    assert grouped["openmp"] == sorted(paths[:2])
    assert grouped["blas"] == [paths[2]]


def test_classify_runtime_paths_ignores_omptarget():
    grouped = classify_runtime_paths(
        [
            "/opt/conda/lib/libomptarget.so",
            "/opt/conda/lib/libomp.so",
            "/opt/conda/lib/libiomp5.so",
        ]
    )
    assert grouped["openmp"] == [
        "/opt/conda/lib/libiomp5.so",
        "/opt/conda/lib/libomp.so",
    ]


def test_runtime_report_flags_duplicate_openmp():
    report = runtime_report(
        paths=[
            "/opt/torch/lib/libgomp-a34b3233.so.1",
            "/opt/conda/lib/libgomp.so.1",
            "/opt/conda/lib/libopenblas.so.0",
        ]
    )
    assert report["pass"] is False
    assert "openmp" in report["duplicates"]


def test_runtime_report_imports_probe_modules_before_scan(monkeypatch):
    events = []

    def fake_import_module(name):
        events.append(name)

    def fake_library_paths():
        assert events == ["torch"]
        return []

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(runtime, "current_process_library_paths", fake_library_paths)

    report = runtime_report(imports=["torch"])

    assert report["pass"] is True
    assert report["import_errors"] == {}


def test_runtime_report_import_errors_fail_report(monkeypatch):
    def fake_import_module(name):
        raise ModuleNotFoundError("no module")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(runtime, "current_process_library_paths", lambda: [])

    report = runtime_report(imports=["missing_runtime_probe"])

    assert report["pass"] is False
    assert report["import_errors"] == {
        "missing_runtime_probe": "ModuleNotFoundError: no module"
    }


def test_main_json_reports_pass_exit_code(monkeypatch, capsys):
    monkeypatch.setattr(runtime, "current_process_library_paths", lambda: [])

    exit_code = runtime.main(["--json"])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert json.loads(captured.out)["pass"] is True


def test_main_json_reports_import_failures_and_exit_code(monkeypatch, capsys):
    def fake_import_module(name):
        raise ImportError("probe failed")

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    monkeypatch.setattr(runtime, "current_process_library_paths", lambda: [])

    exit_code = runtime.main(["--json", "--import", "torch"])

    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert exit_code == 1
    assert report["pass"] is False
    assert report["import_errors"] == {"torch": "ImportError: probe failed"}
