import pytest

from oeq_bench.config import BenchConfig
from oeq_bench.oracle import build_e3nn_tp, make_graph_data


def test_make_graph_data_shapes_cpu():
    torch = pytest.importorskip("torch")
    pytest.importorskip("e3nn.o3")
    cfg = BenchConfig(
        irreps_in1="4x0e+4x1o",
        irreps_in2="0e+1o",
        num_nodes=8,
        num_edges=16,
        device="cpu",
    )
    tp = build_e3nn_tp(cfg)
    assert tp.shared_weights is False
    assert tp.internal_weights is False
    X, Y, W, src, dst = make_graph_data(cfg, tp.weight_numel)
    assert X.shape == (8, tp.irreps_in1.dim)
    assert Y.shape == (16, tp.irreps_in2.dim)
    assert W.shape == (16, tp.weight_numel)
    assert src.dtype == torch.int64
    assert dst.dtype == torch.int64
