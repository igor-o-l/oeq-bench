import json
from pathlib import Path

from oeq_bench.config import BenchConfig
from oeq_bench.exhaustive import (
    ExhaustiveCase,
    build_matrix,
    build_parser,
    run_exhaustive_eval,
    variant_id,
)


def test_build_matrix_crosses_experimental_variants():
    cases = build_matrix(
        block_sizes=[128],
        l1_carveouts=[None, 0],
        load_strategies=["scalar", "vectorized"],
        schedule_strategies=["default", "persistent"],
        edge_orderings=["random"],
        seeds=[42, 43],
        problems=[("smoke", "4x0e", "0e", 8, 16, 0)],
    )

    assert len(cases) == 16
    assert {case.oeq_load_strategy for case in cases} == {"scalar", "vectorized"}
    assert {case.oeq_schedule_strategy for case in cases} == {"default", "persistent"}
    assert {case.l1_carveout for case in cases} == {None, 0}
    assert {case.seed for case in cases} == {42, 43}


def test_cli_defaults_match_documented_108_case_matrix():
    args = build_parser().parse_args(["--out", "/tmp/report.json"])
    cases = build_matrix(
        block_sizes=[int(value) for value in args.block_sizes.split(",")],
        l1_carveouts=[None if value == "default" else int(value) for value in args.l1_carveouts.split(",")],
        load_strategies=args.load_strategies.split(","),
        schedule_strategies=args.schedule_strategies.split(","),
        edge_orderings=args.edge_orderings.split(","),
        seeds=[int(value) for value in args.seeds.split(",")],
        problems=[("smoke", "4x0e", "0e", 8, 16, 0)],
    )

    assert len(cases) == 108


def test_variant_id_is_stable_and_filename_safe():
    case = ExhaustiveCase(
        problem_id="mace",
        irreps="128x0e+128x1o",
        irreps_sh="0e+1o",
        num_nodes=12500,
        num_edges=200000,
        chunk_edges=16384,
        seed=42,
        block_size=128,
        l1_carveout=None,
        oeq_load_strategy="vectorized",
        oeq_schedule_strategy="persistent",
        edge_ordering="dst-src",
    )

    assert variant_id(case) == "mace_seed42_bs128_l1default_loadvectorized_schedpersistent_edgesdst-src"


def test_run_exhaustive_eval_writes_summary_and_forces_correctness(tmp_path):
    calls = []

    def fake_run(cfg):
        calls.append(cfg)
        payload = {
            "device": "fake gpu",
            "compute_capability": "12.0",
            "versions": {"e3nn": "0.6.0"},
            "config": {"seed": cfg.seed, "edge_ordering": cfg.edge_ordering},
            "results": {"oeq": {"ms": 2.0, "medges_s": 8.0}},
            "validation": {
                "fwd_ok": True,
                "grad_ok": True,
                "grad_x_ok": True,
                "grad_y_ok": True,
                "grad_w_ok": True,
            },
            "kernel_config": {"oeq": {"actual_load_strategy": cfg.oeq_load_strategy}},
            "ncu": {},
        }
        Path(cfg.out).write_text(json.dumps(payload))
        return payload

    cases = build_matrix(
        block_sizes=[128],
        l1_carveouts=[None],
        load_strategies=["scalar", "vectorized"],
        schedule_strategies=["default"],
        edge_orderings=["random"],
        seeds=[42],
        problems=[("smoke", "4x0e", "0e", 8, 16, 0)],
    )
    out = tmp_path / "summary.json"
    report = run_exhaustive_eval(BenchConfig(device="cpu"), cases, out, run_one=fake_run)

    assert len(calls) == 2
    assert all(call.validate is True for call in calls)
    assert all(call.profile == "cuda-events" for call in calls)
    assert all(call.seed == 42 for call in calls)
    assert report["summary"] == {
        "total": 2,
        "passed": 2,
        "failed": 0,
        "best_variant_id": "smoke_seed42_bs128_l1default_loadscalar_scheddefault_edgesrandom",
    }
    assert len(report["runs"]) == 2
    assert report["runs"][0]["run_json_retained"] is False
    assert not Path(report["runs"][0]["run_json"]).exists()
    assert json.loads(out.read_text()) == report


def test_run_exhaustive_eval_can_keep_per_run_jsons(tmp_path):
    def fake_run(cfg):
        payload = {
            "device": "fake gpu",
            "compute_capability": "12.0",
            "versions": {},
            "config": {},
            "results": {"oeq": {"ms": 2.0, "medges_s": 8.0}},
            "validation": {
                "fwd_ok": True,
                "grad_ok": True,
                "grad_x_ok": True,
                "grad_y_ok": True,
                "grad_w_ok": True,
            },
            "kernel_config": {},
            "ncu": {},
        }
        Path(cfg.out).write_text(json.dumps(payload))
        return payload

    case = ExhaustiveCase(
        problem_id="smoke",
        irreps="4x0e",
        irreps_sh="0e",
        num_nodes=8,
        num_edges=16,
        chunk_edges=0,
        seed=42,
        block_size=128,
        l1_carveout=None,
        oeq_load_strategy="scalar",
        oeq_schedule_strategy="default",
        edge_ordering="random",
    )

    report = run_exhaustive_eval(
        BenchConfig(device="cpu"),
        [case],
        tmp_path / "summary.json",
        keep_run_jsons=True,
        run_one=fake_run,
    )

    assert report["runs"][0]["run_json_retained"] is True
    assert Path(report["runs"][0]["run_json"]).exists()


def test_run_exhaustive_eval_marks_failed_gradient_variant(tmp_path):
    def fake_run(cfg):
        payload = {
            "device": "fake gpu",
            "compute_capability": "12.0",
            "versions": {},
            "config": {},
            "results": {"oeq": {"ms": 2.0, "medges_s": 8.0}},
            "validation": {
                "fwd_ok": True,
                "grad_ok": False,
                "grad_x_ok": True,
                "grad_y_ok": True,
                "grad_w_ok": False,
            },
            "kernel_config": {},
            "ncu": {},
        }
        Path(cfg.out).write_text(json.dumps(payload))
        return payload

    cases = build_matrix(
        block_sizes=[128],
        l1_carveouts=[None],
        load_strategies=["scalar"],
        schedule_strategies=["default"],
        edge_orderings=["random"],
        seeds=[42],
        problems=[("smoke", "4x0e", "0e", 8, 16, 0)],
    )
    report = run_exhaustive_eval(BenchConfig(device="cpu"), cases, tmp_path / "summary.json", run_one=fake_run)

    assert report["summary"]["passed"] == 0
    assert report["summary"]["failed"] == 1
    assert report["summary"]["best_variant_id"] is None
    assert report["runs"][0]["passed_correctness"] is False


def test_run_exhaustive_eval_records_run_benchmark_failure_and_continues(tmp_path):
    calls = []

    def fake_run(cfg):
        calls.append(cfg.out)
        payload = {
            "device": "fake gpu",
            "compute_capability": "12.0",
            "versions": {},
            "config": {},
            "results": {},
            "validation": {"fwd_ok": False, "grad_ok": False},
            "kernel_config": {},
            "ncu": {},
        }
        Path(cfg.out).write_text(json.dumps(payload))
        if len(calls) == 1:
            raise RuntimeError("validation failed")
        return {
            **payload,
            "results": {"oeq": {"medges_s": 1.0}},
            "validation": {
                "fwd_ok": True,
                "grad_ok": True,
                "grad_x_ok": True,
                "grad_y_ok": True,
                "grad_w_ok": True,
            },
        }

    cases = build_matrix(
        block_sizes=[128, 256],
        l1_carveouts=[None],
        load_strategies=["scalar"],
        schedule_strategies=["default"],
        edge_orderings=["random"],
        seeds=[42],
        problems=[("smoke", "4x0e", "0e", 8, 16, 0)],
    )
    report = run_exhaustive_eval(BenchConfig(device="cpu"), cases, tmp_path / "summary.json", run_one=fake_run)

    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 1
    assert report["runs"][0]["passed_correctness"] is False
    assert report["runs"][0]["error"] == "RuntimeError: validation failed"
    assert report["runs"][1]["passed_correctness"] is True
