import json

from oeq_bench.report import (
    bandwidth_gbs,
    bytes_per_edge,
    edge_throughput_medges_s,
    speedup,
    write_json,
)


def test_throughput_and_bandwidth_math_matches_l13_units():
    assert edge_throughput_medges_s(num_edges=200000, ms=6.25) == 32.0
    assert bandwidth_gbs(bytes_per_edge=16000, num_edges=200000, ms=6.25) == 512.0
    assert speedup(51.2, 6.4) == 8.0


def test_bytes_per_edge_counts_four_byte_float_fields():
    assert bytes_per_edge(
        irreps_in_dim=96,
        irreps_sh_dim=16,
        weight_numel=128,
        irreps_out_dim=64,
    ) == 1216


def test_write_json_creates_parent_and_trailing_newline(tmp_path):
    path = tmp_path / "nested" / "results.json"
    write_json(path, {"backend": "oeq", "ms": 6.25})

    text = path.read_text()
    assert text.endswith("\n")
    assert json.loads(text) == {"backend": "oeq", "ms": 6.25}
