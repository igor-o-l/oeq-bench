import pytest

from oeq_bench.backends import build_instructions


def test_build_instructions_matches_l13_small_selection():
    o3 = pytest.importorskip("e3nn.o3")
    i1 = o3.Irreps("32x0e+32x1o")
    i2 = o3.Irreps("0e+1o+2e")
    io = o3.Irreps("32x0e+32x1o")
    instructions = build_instructions(i1, i2, io)
    assert instructions == [
        (0, 0, 0, "uvu", True, 1.0),
        (0, 1, 1, "uvu", True, 1.0),
        (1, 0, 1, "uvu", True, 1.0),
        (1, 1, 0, "uvu", True, 1.0),
        (1, 2, 1, "uvu", True, 1.0),
    ]
    assert instructions
    assert all(ins[3] == "uvu" for ins in instructions)
    assert all(ins[4] is True for ins in instructions)
    assert all(ins[5] == 1.0 for ins in instructions)
