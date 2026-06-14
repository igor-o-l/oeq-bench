from pathlib import Path
import subprocess

import pytest

from oeq_bench.config import BenchConfig
from oeq_bench.profiling import (
    build_ncu_command,
    main,
    ncu_child_json_path,
    parse_ncu_csv_text,
    parse_ncu_report,
    run_single,
)


def test_build_ncu_command_contains_profile_target():
    cfg = BenchConfig(num_edges=8192, warmup=0, repeats=1, validate=False, block_size=128)
    cmd = build_ncu_command(cfg, backend="oeq", output_stem=Path("/tmp/oeq_profile"))
    assert cmd[:4] == ["ncu", "--set", "full", "--target-processes"]
    assert "oeq_bench.profiling" in " ".join(cmd)
    assert "--backend oeq" in " ".join(cmd)
    assert "--warmup 0" in " ".join(cmd)
    assert "--block-size 128" in " ".join(cmd)
    assert "--out /tmp/oeq_profile_bench.json" in " ".join(cmd)


def test_ncu_child_json_path_is_separate_from_parent_output():
    assert ncu_child_json_path(Path("/tmp/profile")) == Path("/tmp/profile_bench.json")


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


def test_parse_ncu_csv_text_extracts_wide_raw_metrics():
    csv_text = "\n".join(
        [
            (
                '"ID","Kernel Name","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",'
                '"sm__warps_active.avg.pct_of_peak_sustained_active","lts__t_sector_hit_rate.pct",'
                '"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",'
                '"smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio"'
            ),
            '"1","warmup","4.2","12.5","20.0","0.1","0.4"',
            '"2","target","61.5","78.0","57.0","0.9","0.2"',
        ]
    )
    parsed = parse_ncu_csv_text(csv_text)
    assert parsed["dram_throughput_pct"] == 61.5
    assert parsed["achieved_occupancy"] == 0.78
    assert parsed["l2_hit_rate"] == 0.57
    assert parsed["stall_reason"] == "long_scoreboard"


def test_parse_ncu_csv_text_filters_wide_rows_to_target_oeq_kernel():
    csv_text = "\n".join(
        [
            (
                '"ID","Kernel Name","Grid Size","Block Size",'
                '"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",'
                '"sm__warps_active.avg.pct_of_peak_sustained_active","lts__t_sector_hit_rate.pct",'
                '"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",'
                '"smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio"'
            ),
            '"12","void at::_scatter_gather_elementwise_kernel","(1024, 1, 1)","(128, 1, 1)","99.0","94.8","97.3","0.2","0.1"',
            '"945","forward(float *, float *, float *, float *, ConvData, void *)","(60, 1, 1)","(192, 1, 1)","84.294922","12.208787","24.497295","5.674761","0.022091"',
            '"946","fixup_forward(void *, float *)","(60, 1, 1)","(192, 1, 1)","0.089250","11.936230","91.587302","0.0","86.580556"',
        ]
    )

    parsed = parse_ncu_csv_text(
        csv_text,
        target_kernel={
            "forward": {
                "num_blocks": 60,
                "num_threads": 192,
            }
        },
    )

    assert parsed["dram_throughput_pct"] == 84.294922
    assert parsed["achieved_occupancy"] == 0.1221
    assert parsed["l2_hit_rate"] == 0.245
    assert parsed["stall_reason"] == "long_scoreboard"
    assert parsed["selected_kernel"]["id"] == "945"
    assert parsed["selected_kernel"]["grid_size"] == "(60, 1, 1)"
    assert parsed["selected_kernel"]["block_size"] == "(192, 1, 1)"
    assert parsed["selected_kernel"]["candidate_count"] == 2
    assert parsed["selected_kernel"]["name"].startswith("forward(")


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
        "--warmup",
        "0",
        "--block-size",
        "512",
        "--skip-runtime-check",
        "--out",
        "/tmp/profile.json",
    ]

    assert run_single(
        backend="oeq",
        irreps="16x0e",
        irreps_sh="0e+1o",
        num_nodes=128,
        num_edges=512,
        chunk_edges=64,
        repeats=3,
        warmup=0,
        block_size=512,
        out="/tmp/profile.json",
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
            "--warmup",
            "0",
            "--block-size",
            "512",
            "--out",
            "/tmp/profile.json",
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


def test_parse_ncu_report_imports_raw_csv(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="\n".join(
                [
                    '"Metric Name","Metric Unit","Metric Value"',
                    '"dram__throughput.avg_pct_of_peak_sustained_elapsed","%","67.4"',
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr("oeq_bench.profiling.subprocess.run", fake_run)

    parsed = parse_ncu_report("/tmp/profile.ncu-rep")

    assert calls == [
        (
            ["ncu", "--import", "/tmp/profile.ncu-rep", "--csv", "--page", "raw"],
            {"check": True, "capture_output": True, "text": True},
        )
    ]
    assert parsed["dram_throughput_pct"] == 67.4


def test_parse_ncu_report_wraps_import_failure(monkeypatch):
    def fake_run(command, **kwargs):
        raise subprocess.CalledProcessError(
            returncode=7,
            cmd=command,
            output="partial csv",
            stderr="import failed",
        )

    monkeypatch.setattr("oeq_bench.profiling.subprocess.run", fake_run)

    with pytest.raises(RuntimeError) as excinfo:
        parse_ncu_report("/tmp/bad.ncu-rep")
    message = str(excinfo.value)
    assert "ncu import failed for report /tmp/bad.ncu-rep" in message
    assert "exit 7" in message
    assert "import failed" in message
    assert "partial csv" in message
