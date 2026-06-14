from pathlib import Path

from oeq_bench.config import BenchConfig
from oeq_bench.profiling import build_ncu_command, main, parse_ncu_csv_text, run_single


def test_build_ncu_command_contains_profile_target():
    cfg = BenchConfig(num_edges=8192, repeats=1, validate=False)
    cmd = build_ncu_command(cfg, backend="oeq", output_stem=Path("/tmp/oeq_profile"))
    assert cmd[:4] == ["ncu", "--set", "full", "--target-processes"]
    assert "oeq_bench.profiling" in " ".join(cmd)
    assert "--backend oeq" in " ".join(cmd)


def test_parse_ncu_csv_text_extracts_metrics():
    csv_text = "\n".join(
        [
            '"Metric Name","Metric Unit","Metric Value"',
            '"dram__throughput.avg_pct_of_peak_sustained_elapsed","%","67.4"',
            '"sm__warps_active.avg.pct_of_peak_sustained_active","%","82.0"',
            '"lts__t_sectors_hit_rate.pct","%","45.0"',
            '"smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct","%","31.2"',
        ]
    )
    parsed = parse_ncu_csv_text(csv_text)
    assert parsed["dram_throughput_pct"] == 67.4
    assert parsed["achieved_occupancy"] == 0.82
    assert parsed["l2_hit_rate"] == 0.45
    assert parsed["stall_reason"] == "memory_dependency"


def test_run_single_and_main_propagate_bench_exit_code(monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(argv)
        return 17

    monkeypatch.setattr("oeq_bench.bench.main", fake_main)
    expected = [
        "--backend",
        "oeq",
        "--irreps",
        "16x0e",
        "--irreps-sh",
        "0e+1o",
        "--num-nodes",
        "128",
        "--num-edges",
        "512",
        "--chunk-edges",
        "64",
        "--repeats",
        "3",
        "--skip-runtime-check",
    ]

    assert run_single(
        backend="oeq",
        irreps="16x0e",
        irreps_sh="0e+1o",
        num_nodes=128,
        num_edges=512,
        chunk_edges=64,
        repeats=3,
    ) == 17
    assert calls[-1] == expected

    assert main(
        [
            "--backend",
            "oeq",
            "--irreps",
            "16x0e",
            "--irreps-sh",
            "0e+1o",
            "--num-nodes",
            "128",
            "--num-edges",
            "512",
            "--chunk-edges",
            "64",
            "--repeats",
            "3",
        ]
    ) == 17
    assert calls[-1] == expected


def test_parse_ncu_csv_text_uses_max_for_duplicate_metric_rows():
    csv_text = "\n".join(
        [
            '"Metric Name","Metric Unit","Metric Value"',
            '"dram__throughput.avg_pct_of_peak_sustained_elapsed","%","71.5"',
            '"dram__throughput.avg_pct_of_peak_sustained_elapsed","%","12.0"',
            '"sm__warps_active.avg.pct_of_peak_sustained_active","%","83.0"',
            '"sm__warps_active.avg.pct_of_peak_sustained_active","%","41.0"',
            '"lts__t_sectors_hit_rate.pct","%","58.0"',
            '"lts__t_sectors_hit_rate.pct","%","5.0"',
            '"smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct","%","21.0"',
            '"smsp__warp_issue_stalled_not_selected_per_warp_active.pct","%","52.0"',
            '"smsp__warp_issue_stalled_not_selected_per_warp_active.pct","%","2.0"',
        ]
    )
    parsed = parse_ncu_csv_text(csv_text)
    assert parsed["dram_throughput_pct"] == 71.5
    assert parsed["achieved_occupancy"] == 0.83
    assert parsed["l2_hit_rate"] == 0.58
    assert parsed["stall_reason"] == "not_selected"


def test_parse_ncu_csv_text_returns_unknown_stall_reason_without_stall_metrics():
    csv_text = "\n".join(
        [
            '"Metric Name","Metric Unit","Metric Value"',
            '"dram__throughput.avg_pct_of_peak_sustained_elapsed","%","67.4"',
            '"sm__warps_active.avg.pct_of_peak_sustained_active","%","82.0"',
            '"lts__t_sectors_hit_rate.pct","%","45.0"',
        ]
    )
    parsed = parse_ncu_csv_text(csv_text)
    assert parsed["stall_reason"] == "unknown"
