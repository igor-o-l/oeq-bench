import importlib

import pytest


def test_validation_module_imports_without_torch():
    validation = importlib.import_module("oeq_bench.validation")
    assert hasattr(validation, "choose_orientation")
    assert hasattr(validation, "validate_oeq")
    assert hasattr(validation, "e3nn_grads_chunked")


def test_choose_orientation_prefers_rows_dst_cols_src():
    torch = pytest.importorskip("torch")
    from oeq_bench.validation import choose_orientation

    ref = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    out_expected = torch.tensor([[1.0, 0.0], [0.0, 2.0]])
    out_swapped = torch.tensor([[0.0, 2.0], [1.0, 0.0]])
    orient = choose_orientation(ref, out_expected, out_swapped)
    assert orient.orientation == "rows=dst, cols=src"
    assert orient.rows_is_dst is True
    assert orient.forward_error == 0.0


class _NoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeTorch:
    @staticmethod
    def no_grad():
        return _NoGrad()

    @staticmethod
    def allclose(*args, **kwargs):
        return True


class _FakeTensor:
    def detach(self):
        return self


class _Cfg:
    num_nodes = 2
    chunk_edges = 0
    rtol = 1e-4
    atol = 1e-4


def test_validate_oeq_returns_payload_when_orientation_forward_raises(monkeypatch):
    validation = importlib.import_module("oeq_bench.validation")
    monkeypatch.setattr(validation, "torch", _FakeTorch)
    monkeypatch.setattr(validation, "e3nn_scatter", lambda *args: object())

    class RaisingConv:
        def forward(self, *args):
            raise RuntimeError("kernel launch failed")

    result = validation.validate_oeq(
        object(),
        RaisingConv(),
        _FakeTensor(),
        object(),
        object(),
        object(),
        object(),
        _Cfg(),
    )

    assert result["orientation"] is None
    assert result["rows_is_dst"] is None
    assert result["fwd_ok"] is False
    assert result["grad_ok"] is False
    assert "RuntimeError: kernel launch failed" in result["fwd_error"]


class _FakeErrorValue:
    def abs(self):
        return self

    def max(self):
        return self

    def item(self):
        return 0.0


class _FakeValue:
    def __sub__(self, other):
        return _FakeErrorValue()


class _TrackTensor:
    def __init__(self, name, *, detached=False):
        self.name = name
        self.detached = detached
        self.grad = None

    def detach(self):
        return _TrackTensor(self.name, detached=True)

    def clone(self):
        return _TrackTensor(self.name, detached=self.detached)

    def requires_grad_(self, enabled):
        return self


class _BackwardValue:
    def __init__(self, tensors):
        self.tensors = tensors if isinstance(tensors, list) else [tensors]

    def sum(self):
        return self

    def backward(self):
        for tensor in self.tensors:
            tensor.grad = _FakeValue()


def test_validate_oeq_detaches_y_and_w_for_gradient_validation(monkeypatch):
    validation = importlib.import_module("oeq_bench.validation")
    monkeypatch.setattr(validation, "torch", _FakeTorch)
    monkeypatch.setattr(validation, "e3nn_scatter", lambda *args: _FakeValue())

    calls = {}

    def fake_grads(*args):
        calls["grad_ref_y"] = args[2]
        calls["grad_ref_w"] = args[3]
        return {
            "x": _FakeValue(),
            "y": _FakeValue(),
            "w": _FakeValue(),
        }

    monkeypatch.setattr(validation, "e3nn_grads_chunked", fake_grads)

    class PassingConv:
        def __init__(self):
            self.calls = []

        def forward(self, X, Y, W, rows, cols):
            self.calls.append((X, Y, W, rows, cols))
            if len(self.calls) == 4:
                calls["oeq_grad_x"] = X
                calls["oeq_grad_y"] = Y
                calls["oeq_grad_w"] = W
                return _BackwardValue([X, Y, W])
            return _FakeValue()

    y = _TrackTensor("Y")
    w = _TrackTensor("W")
    result = validation.validate_oeq(
        object(),
        PassingConv(),
        _TrackTensor("X"),
        y,
        w,
        object(),
        object(),
        _Cfg(),
    )

    assert result["grad_ok"] is True
    assert result["grad_x_ok"] is True
    assert result["grad_y_ok"] is True
    assert result["grad_w_ok"] is True
    assert calls["grad_ref_y"].detached is True
    assert calls["grad_ref_w"].detached is True
    assert calls["oeq_grad_x"].detached is True
    assert calls["oeq_grad_y"].detached is True
    assert calls["oeq_grad_w"].detached is True
    assert y.detached is False
    assert w.detached is False
