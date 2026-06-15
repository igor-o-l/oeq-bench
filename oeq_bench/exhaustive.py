from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable

from .bench import run_benchmark
from .config import BenchConfig


@dataclass(frozen=True)
class ExhaustiveCase:
    problem_id: str
    irreps: str
    irreps_sh: str
    num_nodes: int
    num_edges: int
    chunk_edges: int
    seed: int
    block_size: int
    l1_carveout: int | None
    oeq_load_strategy: str
    oeq_schedule_strategy: str
    edge_ordering: str


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(part) for part in _parse_csv(value)]


def _parse_l1_csv(value: str) -> list[int | None]:
    result: list[int | None] = []
    for part in _parse_csv(value):
        result.append(None if part in {"default", "none", "null"} else int(part))
    return result


def build_matrix(
    *,
    block_sizes: Iterable[int],
    l1_carveouts: Iterable[int | None],
    load_strategies: Iterable[str],
    schedule_strategies: Iterable[str],
    edge_orderings: Iterable[str],
    seeds: Iterable[int],
    problems: Iterable[tuple[str, str, str, int, int, int]],
) -> list[ExhaustiveCase]:
    cases: list[ExhaustiveCase] = []
    for problem_id, irreps, irreps_sh, num_nodes, num_edges, chunk_edges in problems:
        for seed in seeds:
            for block_size in block_sizes:
                for l1_carveout in l1_carveouts:
                    for load_strategy in load_strategies:
                        for schedule_strategy in schedule_strategies:
                            for edge_ordering in edge_orderings:
                                cases.append(
                                    ExhaustiveCase(
                                        problem_id=problem_id,
                                        irreps=irreps,
                                        irreps_sh=irreps_sh,
                                        num_nodes=num_nodes,
                                        num_edges=num_edges,
                                        chunk_edges=chunk_edges,
                                        seed=seed,
                                        block_size=block_size,
                                        l1_carveout=l1_carveout,
                                        oeq_load_strategy=load_strategy,
                                        oeq_schedule_strategy=schedule_strategy,
                                        edge_ordering=edge_ordering,
                                    )
                                )
    return cases


def variant_id(case: ExhaustiveCase) -> str:
    l1 = "default" if case.l1_carveout is None else str(case.l1_carveout)
    return (
        f"{case.problem_id}_seed{case.seed}_bs{case.block_size}_l1{l1}_"
        f"load{case.oeq_load_strategy}_sched{case.oeq_schedule_strategy}_"
        f"edges{case.edge_ordering}"
    )


def _case_config(base_cfg: BenchConfig, case: ExhaustiveCase, out: Path) -> BenchConfig:
    return replace(
        base_cfg,
        irreps_in1=case.irreps,
        irreps_in2=case.irreps_sh,
        irreps_out=case.irreps,
        num_nodes=case.num_nodes,
        num_edges=case.num_edges,
        chunk_edges=case.chunk_edges,
        seed=case.seed,
        backend="oeq",
        validate=True,
        profile="cuda-events",
        out=out,
        block_size=case.block_size,
        l1_carveout=case.l1_carveout,
        oeq_load_strategy=case.oeq_load_strategy,
        oeq_schedule_strategy=case.oeq_schedule_strategy,
        edge_ordering=case.edge_ordering,
    )


def _passed_correctness(payload: dict) -> bool:
    validation = payload.get("validation", {})
    required = ("fwd_ok", "grad_ok", "grad_x_ok", "grad_y_ok", "grad_w_ok")
    return all(validation.get(key) is True for key in required)


def _run_record(case: ExhaustiveCase, payload: dict, run_path: Path) -> dict:
    return {
        "variant_id": variant_id(case),
        "problem_id": case.problem_id,
        "seed": case.seed,
        "block_size": case.block_size,
        "l1_carveout": case.l1_carveout,
        "oeq_load_strategy": case.oeq_load_strategy,
        "oeq_schedule_strategy": case.oeq_schedule_strategy,
        "edge_ordering": case.edge_ordering,
        "passed_correctness": _passed_correctness(payload),
        "run_json": str(run_path),
        "run_json_retained": run_path.exists(),
        "device": payload.get("device"),
        "compute_capability": payload.get("compute_capability"),
        "versions": payload.get("versions", {}),
        "results": payload.get("results", {}),
        "validation": payload.get("validation", {}),
        "kernel_config": payload.get("kernel_config", {}),
        "ncu": payload.get("ncu", {}),
    }


def _failure_payload(run_path: Path) -> dict:
    if run_path.exists():
        return json.loads(run_path.read_text())
    return {
        "device": None,
        "compute_capability": None,
        "versions": {},
        "results": {},
        "validation": {},
        "kernel_config": {},
        "ncu": {},
    }


def _summary(runs: list[dict]) -> dict:
    passed = [run for run in runs if run["passed_correctness"]]
    best = None
    if passed:
        best = max(
            passed,
            key=lambda run: run.get("results", {}).get("oeq", {}).get("medges_s", float("-inf")),
        )
    return {
        "total": len(runs),
        "passed": len(passed),
        "failed": len(runs) - len(passed),
        "best_variant_id": None if best is None else best["variant_id"],
    }


def run_exhaustive_eval(
    base_cfg: BenchConfig,
    cases: list[ExhaustiveCase],
    out: Path,
    *,
    keep_run_jsons: bool = False,
    run_one: Callable[[BenchConfig], dict] = run_benchmark,
) -> dict:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []
    for idx, case in enumerate(cases, start=1):
        run_path = out.with_name(f"{out.stem}_{variant_id(case)}.json")
        cfg = _case_config(base_cfg, case, run_path)
        print(f"[{idx}/{len(cases)}] {variant_id(case)}", flush=True)
        error = None
        try:
            payload = run_one(cfg)
        except Exception as exc:
            payload = _failure_payload(run_path)
            error = f"{type(exc).__name__}: {exc}"
        record = _run_record(case, payload, run_path)
        if error is not None:
            record["error"] = error
        if not keep_run_jsons and run_path.exists():
            run_path.unlink()
            record["run_json_retained"] = False
        runs.append(record)
        status = "PASS" if record["passed_correctness"] else "FAIL"
        medges = record.get("results", {}).get("oeq", {}).get("medges_s")
        print(f"[{idx}/{len(cases)}] {status} medges_s={medges}", flush=True)
    report = {
        "schema": "oeq-bench-exhaustive-v1",
        "runs": runs,
        "summary": _summary(runs),
    }
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run exhaustive OEQ correctness/performance matrix")
    parser.add_argument("--out", required=True)
    parser.add_argument("--irreps", default=BenchConfig.irreps_in1)
    parser.add_argument("--irreps-sh", default=BenchConfig.irreps_in2)
    parser.add_argument("--num-nodes", type=int, default=BenchConfig.num_nodes)
    parser.add_argument("--num-edges", type=int, default=BenchConfig.num_edges)
    parser.add_argument("--chunk-edges", type=int, default=BenchConfig.chunk_edges)
    parser.add_argument("--block-sizes", default="128,256,512")
    parser.add_argument("--l1-carveouts", default="default,0,50")
    parser.add_argument("--load-strategies", default="scalar,vectorized")
    parser.add_argument("--schedule-strategies", default="default,persistent")
    parser.add_argument("--edge-orderings", default="random,dst-src,src-dst")
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--warmup", type=int, default=BenchConfig.warmup)
    parser.add_argument("--repeats", type=int, default=BenchConfig.repeats)
    parser.add_argument("--skip-runtime-check", action="store_true")
    parser.add_argument(
        "--keep-run-jsons",
        action="store_true",
        help="Keep the transient per-case benchmark JSONs next to the aggregate report.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    base_cfg = BenchConfig(
        warmup=args.warmup,
        repeats=args.repeats,
        skip_runtime_check=args.skip_runtime_check,
    )
    problems = [
        (
            "custom",
            args.irreps,
            args.irreps_sh,
            args.num_nodes,
            args.num_edges,
            args.chunk_edges,
        )
    ]
    cases = build_matrix(
        block_sizes=_parse_int_csv(args.block_sizes),
        l1_carveouts=_parse_l1_csv(args.l1_carveouts),
        load_strategies=_parse_csv(args.load_strategies),
        schedule_strategies=_parse_csv(args.schedule_strategies),
        edge_orderings=_parse_csv(args.edge_orderings),
        seeds=_parse_int_csv(args.seeds),
        problems=problems,
    )
    run_exhaustive_eval(base_cfg, cases, Path(args.out), keep_run_jsons=args.keep_run_jsons)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
