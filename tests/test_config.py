from pathlib import Path

import pytest

from oeq_bench.bench import main
from oeq_bench.config import BenchConfig


def test_default_config_matches_spec3_shape():
    cfg = BenchConfig()
    assert cfg.irreps_in1 == "128x0e+128x1o+128x2e"
    assert cfg.irreps_in2 == "0e+1o+2e+3o"
    assert cfg.irreps_out == cfg.irreps_in1
    assert cfg.num_nodes == 12500
    assert cfg.num_edges == 200000
    assert cfg.chunk_edges == 16384
    assert cfg.backend == "oeq"
    assert cfg.validate is False
    assert cfg.profile == "cuda-events"
    assert cfg.out == Path("oeq_bench_results.json")
    assert cfg.block_size == 128
    assert cfg.l1_carveout is None


def test_config_from_namespace_translates_cli_aliases():
    ns = type(
        "Args",
        (),
        {
            "irreps": "32x0e+32x1o",
            "irreps_sh": "0e+1o+2e",
            "num_nodes": 1024,
            "num_edges": 8192,
            "chunk_edges": 512,
            "backend": "both",
            "validate": True,
            "profile": "ncu",
            "dry_run": True,
            "warmup": 2,
            "repeats": 10,
            "out": "/tmp/spec3.json",
            "skip_runtime_check": True,
            "rtol": 1e-4,
            "atol": 1e-4,
            "bw_peak_gbs": 896.0,
            "block_size": 256,
            "l1_carveout": 0,
        },
    )()
    cfg = BenchConfig.from_args(ns)
    assert cfg.irreps_in1 == "32x0e+32x1o"
    assert cfg.irreps_in2 == "0e+1o+2e"
    assert cfg.irreps_out == "32x0e+32x1o"
    assert cfg.backend == "both"
    assert cfg.validate is True
    assert cfg.profile == "ncu"
    assert cfg.dry_run is True
    assert cfg.skip_runtime_check is True
    assert cfg.out == Path("/tmp/spec3.json")
    assert cfg.l1_carveout == 0


def test_main_dry_run_prints_config_and_returns_zero(capsys):
    assert main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "BenchConfig(" in out
    assert "dry_run=True" in out


def test_main_ncu_dry_run_prints_config_and_returns_zero(capsys):
    assert main(["--dry-run", "--profile", "ncu", "--backend", "oeq"]) == 0
    out = capsys.readouterr().out
    assert "BenchConfig(" in out
    assert "profile='ncu'" in out
    assert "dry_run=True" in out


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"block_size": 64}, "block_size"),
        ({"l1_carveout": -1}, "l1_carveout"),
        ({"l1_carveout": 101}, "l1_carveout"),
        ({"rtol": -1e-4}, "rtol"),
        ({"atol": -1e-4}, "atol"),
        ({"bw_peak_gbs": 0.0}, "bw_peak_gbs"),
    ],
)
def test_config_rejects_invalid_public_numeric_fields(kwargs, match):
    with pytest.raises(ValueError, match=match):
        BenchConfig(**kwargs)


def test_both_backend_expands_to_oeq_then_cueq():
    assert BenchConfig(backend="both").backends_to_run() == ["oeq", "cueq"]
