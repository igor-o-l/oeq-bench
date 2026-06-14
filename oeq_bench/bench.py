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
from pathlib import Path

from .config import BenchConfig
from .oracle import build_e3nn_tp, e3nn_scatter, make_graph_data
from .profiling import build_ncu_command, parse_ncu_report, time_cuda_events
from .report import (
    bandwidth_gbs,
    bytes_per_edge,
    collect_versions,
    edge_throughput_medges_s,
    speedup,
    write_json,
)
from .runtime import runtime_report


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


def empty_result_payload(cfg: BenchConfig) -> dict:
    return {
        "device": None,
        "compute_capability": None,
        "versions": {},
        "config": {
            "irreps_in1": cfg.irreps_in1,
            "irreps_in2": cfg.irreps_in2,
            "irreps_out": cfg.irreps_out,
            "num_nodes": cfg.num_nodes,
            "num_edges": cfg.num_edges,
            "chunk_edges": cfg.chunk_edges,
            "block_size": cfg.block_size,
        },
        "results": {},
        "validation": {},
        "runtime_check": {},
        "ncu": {},
    }


def _ncu_output_stem(cfg: BenchConfig, backend: str) -> Path:
    stem = cfg.out.with_suffix("")
    if len(cfg.backends_to_run()) == 1:
        return stem
    return stem.with_name(f"{stem.name}_{backend}")


def _print_ncu_commands(cfg: BenchConfig) -> None:
    for backend in cfg.backends_to_run():
        print(" ".join(build_ncu_command(cfg, backend=backend, output_stem=_ncu_output_stem(cfg, backend))))


def run_benchmark(cfg: BenchConfig) -> dict:
    if cfg.profile == "ncu" and cfg.dry_run:
        _print_ncu_commands(cfg)
        return empty_result_payload(cfg)
    if cfg.profile == "ncu":
        raise RuntimeError("ncu profiling execution is not wired yet; use --dry-run to print the ncu command")

    import torch

    if cfg.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; run this benchmark on rog in the mlip env")

    payload = empty_result_payload(cfg)
    runtime = {"pass": True, "duplicates": {}, "runtimes": {}} if cfg.skip_runtime_check else runtime_report()
    payload["runtime_check"] = runtime
    if not runtime["pass"]:
        raise RuntimeError(f"duplicate runtimes detected: {runtime['duplicates']}")

    cc = torch.cuda.get_device_capability()
    payload["device"] = torch.cuda.get_device_name(0)
    payload["compute_capability"] = f"{cc[0]}.{cc[1]}"

    tp_e3nn = build_e3nn_tp(cfg)
    weight_numel = tp_e3nn.weight_numel
    X, Y, W, src, dst = make_graph_data(cfg, weight_numel)

    def e3nn_fn():
        with torch.no_grad():
            return e3nn_scatter(tp_e3nn, X, Y, W, src, dst, cfg.num_nodes, cfg.chunk_edges)

    ms_e3nn = time_cuda_events(e3nn_fn, cfg.warmup, cfg.repeats)
    payload["results"]["e3nn"] = {
        "ms": round(ms_e3nn, 4),
        "medges_s": edge_throughput_medges_s(cfg.num_edges, ms_e3nn),
    }

    extra_versions: dict[str, str | None] = {}
    validation: dict = {}
    for backend in cfg.backends_to_run():
        if backend == "oeq":
            from .backends.oeq import build_oeq_conv, version
            from .validation import validate_oeq

            problem, conv = build_oeq_conv(cfg)
            if problem.weight_numel != weight_numel:
                raise RuntimeError(f"weight layout mismatch: e3nn={weight_numel}, oeq={problem.weight_numel}")
            validation = validate_oeq(tp_e3nn, conv, X, Y, W, src, dst, cfg) if cfg.validate else {}
            payload["validation"] = validation
            if cfg.validate and (validation.get("fwd_ok") is not True or validation.get("grad_ok") is not True):
                write_json(cfg.out, payload)
                raise RuntimeError(f"validation failed: {validation}")
            rows, cols = (dst, src) if validation.get("rows_is_dst", True) else (src, dst)

            def fn():
                with torch.no_grad():
                    return conv.forward(X.detach(), Y, W, rows, cols)

            ms = time_cuda_events(fn, cfg.warmup, cfg.repeats)
            bpe = bytes_per_edge(
                tp_e3nn.irreps_in1.dim,
                tp_e3nn.irreps_in2.dim,
                weight_numel,
                tp_e3nn.irreps_out.dim,
            )
            payload["results"]["oeq"] = {
                "ms": round(ms, 4),
                "medges_s": edge_throughput_medges_s(cfg.num_edges, ms),
                "bw_gbs": bandwidth_gbs(bpe, cfg.num_edges, ms),
                "bw_pct_peak": round(100.0 * bandwidth_gbs(bpe, cfg.num_edges, ms) / cfg.bw_peak_gbs, 3),
            }
            payload["results"]["speedup_oeq_vs_e3nn"] = speedup(ms_e3nn, ms)
            extra_versions["oeq"] = version()
        if backend == "cueq":
            from .backends.cueq import run_smoke, version

            payload["results"]["cueq"] = run_smoke(cfg)
            extra_versions["cueq"] = version()

    payload["validation"] = validation
    payload["versions"] = collect_versions(extra_versions)
    write_json(cfg.out, payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    cfg = BenchConfig.from_args(build_parser().parse_args(argv))
    if cfg.dry_run:
        print(cfg)
        if cfg.profile == "ncu":
            run_benchmark(cfg)
        return 0
    run_benchmark(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
