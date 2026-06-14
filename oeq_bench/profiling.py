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
    report = {
        "architecture": architecture,
        "device": device,
        "configs_tested": configs,
        "recommended": {
            "block_size": winner["block_size"],
            "l1_carveout": winner["l1_carveout"],
        },
    }
    if winner.get("requested_block_size_effective") is False:
        report["fastest_request"] = dict(report["recommended"])
        report["recommended"] = {
            "block_size": None,
            "l1_carveout": None,
            "effective": False,
        }
    return report


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
        "--block-size",
        str(cfg.block_size),
        "--out",
        str(ncu_child_json_path(output_stem)),
    ]


def _max_numeric_metrics(rows: list[dict], fieldnames: list[str] | None) -> dict[str, float]:
    """Collapse duplicate ncu metric rows when no target launch is available."""
    if fieldnames and "Metric Name" not in fieldnames:
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


def _max_wide_numeric_metrics(rows: list[dict]) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in rows:
        for name, value in _wide_numeric_metrics(row).items():
            values[name] = max(value, values.get(name, value))
    return values


def _wide_numeric_metrics(row: dict) -> dict[str, float]:
    values: dict[str, float] = {}
    for name, raw_value in row.items():
        if not name:
            continue
        try:
            values[name] = float(raw_value)
        except (TypeError, ValueError):
            continue
    return values


def _parse_dim3(value: str) -> tuple[int, int, int] | None:
    try:
        parts = [int(part.strip()) for part in value.strip().strip("()").split(",")]
    except (AttributeError, ValueError):
        return None
    if len(parts) != 3:
        return None
    return parts[0], parts[1], parts[2]


def _target_forward_config(target_kernel: dict | None) -> dict:
    if not target_kernel:
        return {}
    forward = target_kernel.get("forward")
    return forward if isinstance(forward, dict) else target_kernel


def _row_matches_target_launch(row: dict, target_kernel: dict | None) -> bool:
    target = _target_forward_config(target_kernel)
    if not target:
        return False
    grid = _parse_dim3(row.get("Grid Size", ""))
    block = _parse_dim3(row.get("Block Size", ""))
    if grid is None or block is None:
        return False
    if target.get("num_blocks") is not None and grid[0] != int(target["num_blocks"]):
        return False
    if target.get("num_threads") is not None and block[0] != int(target["num_threads"]):
        return False
    return True


def _is_main_oeq_forward(row: dict) -> bool:
    name = row.get("Kernel Name", "")
    return name.startswith("forward(") and "ConvData" in name


def _select_target_kernel_row(rows: list[dict], target_kernel: dict | None) -> tuple[dict, int, str]:
    candidates = [row for row in rows if _row_matches_target_launch(row, target_kernel)]
    if not candidates:
        raise ValueError(f"ncu target kernel not found for launch config: {target_kernel}")
    main_forward = [row for row in candidates if _is_main_oeq_forward(row)]
    if main_forward:
        return main_forward[-1], len(candidates), "last_matching_oeq_forward"
    return candidates[-1], len(candidates), "last_matching_launch"


def _first_metric(values: dict[str, float], names: tuple[str, ...], default: float | None = 0.0) -> float | None:
    for name in names:
        if name in values:
            return values[name]
    return default


def _summarize_ncu_metrics(values: dict[str, float]) -> dict:
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


def parse_ncu_csv_text(csv_text: str, target_kernel: dict | None = None) -> dict:
    reader = csv.DictReader(StringIO(csv_text))
    rows = list(reader)
    selected_kernel = None
    if target_kernel and reader.fieldnames and "Metric Name" not in reader.fieldnames:
        selected_row, candidate_count, strategy = _select_target_kernel_row(rows, target_kernel)
        values = _wide_numeric_metrics(selected_row)
        selected_kernel = {
            "id": selected_row.get("ID"),
            "name": selected_row.get("Kernel Name"),
            "grid_size": selected_row.get("Grid Size"),
            "block_size": selected_row.get("Block Size"),
            "candidate_count": candidate_count,
            "selection_strategy": strategy,
        }
    else:
        values = _max_numeric_metrics(rows, reader.fieldnames)
    metrics = _summarize_ncu_metrics(values)
    if selected_kernel is not None:
        metrics["selected_kernel"] = selected_kernel
    return metrics


def parse_ncu_report(ncu_rep_path: str, target_kernel: dict | None = None) -> dict:
    command = ["ncu", "--import", ncu_rep_path, "--csv", "--page", "raw"]
    try:
        proc = subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        output = "\n".join(str(part) for part in (exc.stdout, exc.stderr) if part)
        raise RuntimeError(
            f"ncu import failed for report {ncu_rep_path} with exit {exc.returncode}: {output}"
        ) from exc
    try:
        return parse_ncu_csv_text(proc.stdout, target_kernel=target_kernel)
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
        "--block-size",
        str(kwargs["block_size"]),
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
    parser.add_argument("--block-size", type=int, default=128)
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
        block_size=args.block_size,
        out=args.out,
    )


if __name__ == "__main__":
    raise SystemExit(main())
