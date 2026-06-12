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


def main() -> int:
    ap = argparse.ArgumentParser(description="OEQ/cuEq vs e3nn fused TP-conv benchmark")
    ap.add_argument("--irreps", default="128x0e+128x1o+128x2e", help="in1 == out irreps")
    ap.add_argument("--irreps-sh", default="0e+1o+2e+3o", help="edge spherical-harmonics irreps")
    ap.add_argument("--num-nodes", type=int, default=12500)
    ap.add_argument("--num-edges", type=int, default=200000)
    ap.add_argument("--chunk-edges", type=int, default=16384, help="tile the e3nn oracle (0=off)")
    ap.add_argument("--backend", choices=["oeq", "cueq", "both"], default="oeq")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--out", default="oeq_bench_results.json")
    a = ap.parse_args()
    raise NotImplementedError(
        "Port the body from MLIPs/experiments/labs/L13-oeq-fused-tp-conv-benchmark/"
        "benchmark_oeq_vs_e3nn.py (uses oeq_bench.oracle for the chunked reference). "
        f"Parsed config: {vars(a)}")


if __name__ == "__main__":
    raise SystemExit(main())
