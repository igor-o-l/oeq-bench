from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from io import StringIO
from pathlib import Path
from typing import Callable


def time_cuda_events(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end) / repeats)


def time_wall(fn: Callable[[], object], warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    return float((time.perf_counter() - t0) * 1e3 / repeats)


def build_tuning_report(architecture: str, device: str, configs: list[dict]) -> dict:
    winner = max(configs, key=lambda row: row["medges_s"])
    return {
        "architecture": architecture,
        "device": device,
        "configs_tested": configs,
        "recommended": {
            "block_size": winner["block_size"],
            "l1_carveout": winner["l1_carveout"],
        },
    }


def build_ncu_command(cfg, backend: str, output_stem: Path) -> list[str]:
    return [
        "ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "-o",
        str(output_stem),
        sys.executable,
        "-m",
        "oeq_bench.profiling",
        "--backend",
        backend,
        "--irreps",
        cfg.irreps_in1,
        "--irreps-sh",
        cfg.irreps_in2,
        "--num-nodes",
        str(cfg.num_nodes),
        "--num-edges",
        str(cfg.num_edges),
        "--chunk-edges",
        str(cfg.chunk_edges),
        "--repeats",
        "1",
    ]


def _max_numeric_metrics(csv_text: str) -> dict[str, float]:
    """Collapse duplicate ncu metric rows to best observed target-kernel values."""
    rows = csv.DictReader(StringIO(csv_text))
    values: dict[str, float] = {}
    for row in rows:
        try:
            value = float(row["Metric Value"])
        except (KeyError, TypeError, ValueError):
            continue
        name = row.get("Metric Name")
        if not name:
            continue
        values[name] = max(value, values.get(name, value))
    return values


def parse_ncu_csv_text(csv_text: str) -> dict:
    values = _max_numeric_metrics(csv_text)
    dram = values.get("dram__throughput.avg_pct_of_peak_sustained_elapsed", 0.0)
    occ_pct = values.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    l2_pct = values.get("lts__t_sectors_hit_rate.pct", 0.0)
    stall_metrics = {
        "memory_dependency": "smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",
        "not_selected": "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
        "no_instruction": "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
    }
    stall_reason = "unknown"
    if any(metric in values for metric in stall_metrics.values()):
        stall_reason = max(stall_metrics, key=lambda name: values.get(stall_metrics[name], 0.0))
    return {
        "achieved_occupancy": round(occ_pct / 100.0, 4),
        "dram_throughput_pct": dram,
        "l2_hit_rate": round(l2_pct / 100.0, 4),
        "stall_reason": stall_reason,
    }


def parse_ncu_report(ncu_rep_path: str) -> dict:
    proc = subprocess.run(
        ["ncu", "--import", ncu_rep_path, "--csv", "--page", "raw"],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_ncu_csv_text(proc.stdout)


def run_single(**kwargs) -> int:
    from .bench import main

    args = [
        "--backend",
        kwargs["backend"],
        "--irreps",
        kwargs["irreps"],
        "--irreps-sh",
        kwargs["irreps_sh"],
        "--num-nodes",
        str(kwargs["num_nodes"]),
        "--num-edges",
        str(kwargs["num_edges"]),
        "--chunk-edges",
        str(kwargs["chunk_edges"]),
        "--repeats",
        str(kwargs["repeats"]),
        "--skip-runtime-check",
    ]
    return main(args)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one oeq-bench target under a profiler")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--irreps", required=True)
    parser.add_argument("--irreps-sh", required=True)
    parser.add_argument("--num-nodes", type=int, required=True)
    parser.add_argument("--num-edges", type=int, required=True)
    parser.add_argument("--chunk-edges", type=int, required=True)
    parser.add_argument("--repeats", type=int, required=True)
    args = parser.parse_args(argv)
    return run_single(
        backend=args.backend,
        irreps=args.irreps,
        irreps_sh=args.irreps_sh,
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        chunk_edges=args.chunk_edges,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    raise SystemExit(main())
