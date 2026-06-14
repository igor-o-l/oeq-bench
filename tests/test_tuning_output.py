from oeq_bench.profiling import build_tuning_report


def test_build_tuning_report_selects_fastest_config():
    report = build_tuning_report(
        architecture="sm_120",
        device="RTX 5080 Laptop GPU",
        configs=[
            {"block_size": 128, "l1_carveout": 0, "medges_s": 28.5, "bw_pct": 52.1},
            {"block_size": 256, "l1_carveout": 0, "medges_s": 32.2, "bw_pct": 57.2},
            {"block_size": 512, "l1_carveout": 0, "medges_s": 30.1, "bw_pct": 54.8},
        ],
    )
    assert report["recommended"] == {"block_size": 256, "l1_carveout": 0}
