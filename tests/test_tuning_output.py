from oeq_bench.profiling import build_tuning_report


def test_build_tuning_report_selects_fastest_config():
    report = build_tuning_report(
        architecture="sm_120",
        device="RTX 5080 Laptop GPU",
        configs=[
            {"block_size": 128, "l1_carveout": None, "medges_s": 45.887, "bw_pct": 81.613},
            {"block_size": 256, "l1_carveout": None, "medges_s": 26.043, "bw_pct": 46.320},
            {"block_size": 512, "l1_carveout": None, "medges_s": 23.552, "bw_pct": 41.890},
        ],
    )
    assert report["recommended"] == {"block_size": 128, "l1_carveout": None}


def test_build_tuning_report_marks_ineffective_requested_knobs():
    report = build_tuning_report(
        architecture="sm_120",
        device="RTX 5080 Laptop GPU",
        configs=[
            {
                "block_size": 128,
                "l1_carveout": 0,
                "medges_s": 32.2,
                "bw_pct": 57.2,
                "requested_block_size_effective": False,
            }
        ],
    )

    assert report["fastest_request"] == {"block_size": 128, "l1_carveout": 0}
    assert report["recommended"] == {
        "block_size": None,
        "l1_carveout": None,
        "effective": False,
    }
