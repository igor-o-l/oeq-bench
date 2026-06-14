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


def ncu_child_json_path(output_stem: Path) -> Path:
    output_stem = Path(output_stem)
    return output_stem.with_name(f"{output_stem.name}_bench").with_suffix(".json")


def build_ncu_command(cfg, backend: str, output_stem: Path) -> list[str]:
    output_stem = Path(output_stem)
    return [
        "ncu",
        "--set",
        "full",
        "--target-processes",
        "all",
        "--force-overwrite",
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
        "--warmup",
        str(cfg.warmup),
        "--out",
        str(ncu_child_json_path(output_stem)),
    ]


def _max_numeric_metrics(csv_text: str) -> dict[str, float]:
    """Collapse duplicate ncu metric rows to best observed target-kernel values."""
    rows = csv.DictReader(StringIO(csv_text))
    if rows.fieldnames and "Metric Name" not in rows.fieldnames:
        return _max_wide_numeric_metrics(rows)
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


def _max_wide_numeric_metrics(rows: csv.DictReader) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        for name, raw_value in row.items():
            if not name:
                continue
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            values[name] = max(value, values.get(name, value))
    return values


def _first_metric(values: dict[str, float], names: tuple[str, ...], default: float | None = 0.0) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return default


def parse_ncu_csv_text(csv_text: str) -> dict:
    values = _max_numeric_metrics(csv_text)
    dram = _first_metric(
        values,
        (
            "dram__throughput.avg_pct_of_peak_sustained_elapsed",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__bytes.sum.pct_of_peak_sustained_elapsed",
        ),
    )
    occ_pct = _first_metric(values, ("sm__warps_active.avg.pct_of_peak_sustained_active",))
    l2_pct = _first_metric(values, ("lts__t_sectors_hit_rate.pct", "lts__t_sector_hit_rate.pct"))
    stall_metrics = {
        "memory_dependency": ("smsp__warp_issue_stalled_memory_dependency_per_warp_active.pct",),
        "not_selected": (
            "smsp__warp_issue_stalled_not_selected_per_warp_active.pct",
            "smsp__average_warps_issue_stalled_not_selected_per_issue_active.ratio",
        ),
        "no_instruction": (
            "smsp__warp_issue_stalled_no_instruction_per_warp_active.pct",
            "smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio",
        ),
        "long_scoreboard": ("smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",),
        "short_scoreboard": ("smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",),
        "lg_throttle": ("smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio",),
        "mio_throttle": ("smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio",),
    }
    stall_reason = "unknown"
    stall_values = {
        name: _first_metric(values, metrics, None) for name, metrics in stall_metrics.items()
    }
    stall_values = {name: value for name, value in stall_values.items() if value is not None}
    if stall_values:
        stall_reason = max(stall_values, key=stall_values.get)
    return {
        "achieved_occupancy": round(occ_pct / 100.0, 4),
        "dram_throughput_pct": dram,
        "l2_hit_rate": round(l2_pct / 100.0, 4),
        "stall_reason": stall_reason,
    }


def parse_ncu_report(ncu_rep_path: str) -> dict:
    command = ["ncu", "--import", ncu_rep_path, "--csv", "--page", "raw"]
    try:
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        output = "\n".join(str(part) for part in (exc.stdout, exc.stderr) if part)
        raise RuntimeError(
            f"ncu import failed for report {ncu_rep_path} with exit {exc.returncode}: {output}"
        ) from exc
    try:
        return parse_ncu_csv_text(proc.stdout)
    except Exception as exc:
        raise RuntimeError(f"ncu CSV parse failed for report {ncu_rep_path}: {exc}") from exc


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
        "--warmup",
        str(kwargs.get("warmup", 0)),
        "--skip-runtime-check",
        "--out",
        str(kwargs.get("out", "oeq_bench_results.json")),
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
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    return run_single(
        backend=args.backend,
        irreps=args.irreps,
        irreps_sh=args.irreps_sh,
        num_nodes=args.num_nodes,
        num_edges=args.num_edges,
        chunk_edges=args.chunk_edges,
        repeats=args.repeats,
        warmup=args.warmup,
        out=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
