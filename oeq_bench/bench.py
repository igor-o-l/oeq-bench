"""bench — time OEQ (and optionally cuEquivariance) vs the chunked e3nn oracle; validate; report.

This is the CLI entry (`oeq-bench`). The full implementation is ported from L13's
benchmark_oeq_vs_e3nn.py; here it is a stub with the argument surface + the intended flow so the package is
installable and the work is well-scoped.

Intended flow:
  1. build uvu instructions from (irreps_in1, irreps_in2, irreps_out)  [shared by both backends]
  2. e3nn oracle (oracle.e3nn_scatter / e3nn_grad_chunked, chunked)
  3. OEQ TensorProductConv  (and, if available, cuEquivariance)
  4. validate forward + dL/dX to FP32 tol; auto-detect rows/cols orientation
  5. time both; report ms/call, Medges/s, achieved GB/s, % of measured peak BW
  6. write results JSON (device, cc, versions, irreps, edges, speedup, val)
"""
from __future__ import annotations

import argparse

from .config import BenchConfig


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="OEQ/cuEq vs e3nn fused TP-conv benchmark")
    ap.add_argument("--irreps", default=BenchConfig.irreps_in1, help="node input and output irreps")
    ap.add_argument("--irreps-sh", default=BenchConfig.irreps_in2, help="edge spherical-harmonics irreps")
    ap.add_argument("--num-nodes", type=int, default=BenchConfig.num_nodes)
    ap.add_argument("--num-edges", type=int, default=BenchConfig.num_edges)
    ap.add_argument("--chunk-edges", type=int, default=BenchConfig.chunk_edges)
    ap.add_argument("--backend", choices=["oeq", "cueq", "both"], default=BenchConfig.backend)
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--profile", choices=["cuda-events", "ncu"], default=BenchConfig.profile)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--warmup", type=int, default=BenchConfig.warmup)
    ap.add_argument("--repeats", type=int, default=BenchConfig.repeats)
    ap.add_argument("--out", default=str(BenchConfig.out))
    ap.add_argument("--skip-runtime-check", action="store_true")
    ap.add_argument("--rtol", type=float, default=BenchConfig.rtol)
    ap.add_argument("--atol", type=float, default=BenchConfig.atol)
    ap.add_argument("--bw-peak-gbs", type=float, default=BenchConfig.bw_peak_gbs)
    ap.add_argument("--block-size", type=int, choices=[128, 256, 512], default=BenchConfig.block_size)
    return ap


def run_benchmark(cfg: BenchConfig) -> dict:
    raise NotImplementedError("Task 6 wires the benchmark orchestration")


def main(argv: list[str] | None = None) -> int:
    cfg = BenchConfig.from_args(build_parser().parse_args(argv))
    if cfg.dry_run:
        print(cfg)
        return 0
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
