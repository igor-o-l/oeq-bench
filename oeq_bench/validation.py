from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in optional-dependency environments
    torch = None

from .oracle import e3nn_grads_chunked, e3nn_scatter


@dataclass(frozen=True)
class Orientation:
    orientation: str
    rows_is_dst: bool
    forward_error: float
    swapped_error: float


def max_abs_err(a, b) -> float:
    return float((a - b).abs().max().item())


def _error_message(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _failure_result(cfg, *, fwd_error: str | None, grad_error: str | None) -> dict:
    return {
        "orientation": None,
        "rows_is_dst": None,
        "fwd_ok": False,
        "fwd_err": None,
        "grad_ok": False,
        "grad_err": None,
        "grad_x_ok": False,
        "grad_x_err": None,
        "grad_y_ok": False,
        "grad_y_err": None,
        "grad_w_ok": False,
        "grad_w_err": None,
        "chunk_edges": cfg.chunk_edges,
        "fwd_error": fwd_error,
        "grad_error": grad_error,
        "orientation_forward_error": None,
        "orientation_swapped_error": None,
    }


def choose_orientation(ref, out_rows_dst, out_rows_src) -> Orientation:
    err_dst = max_abs_err(ref, out_rows_dst)
    err_src = max_abs_err(ref, out_rows_src)
    if err_dst <= err_src:
        return Orientation("rows=dst, cols=src", True, err_dst, err_src)
    return Orientation("rows=src, cols=dst", False, err_src, err_dst)


def validate_oeq(tp_e3nn, oeq_conv, X, Y, W, src, dst, cfg) -> dict:
    if torch is None:
        error = _error_message(ModuleNotFoundError("torch"))
        return _failure_result(cfg, fwd_error=error, grad_error=error)

    try:
        with torch.no_grad():
            ref = e3nn_scatter(tp_e3nn, X, Y, W, src, dst, cfg.num_nodes, cfg.chunk_edges)
            out_dst = oeq_conv.forward(X.detach(), Y, W, dst, src)
            out_src = oeq_conv.forward(X.detach(), Y, W, src, dst)
        orient = choose_orientation(ref, out_dst, out_src)
    except Exception as exc:
        error = _error_message(exc)
        return _failure_result(
            cfg,
            fwd_error=error,
            grad_error=f"skipped because orientation failed: {error}",
        )
    rows, cols = (dst, src) if orient.rows_is_dst else (src, dst)

    fwd_error = None
    try:
        with torch.no_grad():
            out = oeq_conv.forward(X.detach(), Y, W, rows, cols)
        fwd_err = max_abs_err(ref, out)
        fwd_ok = bool(torch.allclose(ref, out, rtol=cfg.rtol, atol=cfg.atol))
    except Exception as exc:
        fwd_err = None
        fwd_ok = False
        fwd_error = _error_message(exc)

    grad_err = None
    grad_ok = False
    grad_error = None
    if fwd_error is None:
        try:
            Yd = Y.detach()
            Wd = W.detach()
            grad_ref = e3nn_grads_chunked(tp_e3nn, X, Yd, Wd, src, dst, cfg.chunk_edges)
            missing_ref = [name for name, grad in grad_ref.items() if grad is None]
            if missing_ref:
                raise RuntimeError(
                    "e3nn gradient reference returned None for "
                    + ", ".join(sorted(missing_ref))
                )
            Xo = X.detach().clone().requires_grad_(True)
            Yo = Yd.clone().requires_grad_(True)
            Wo = Wd.clone().requires_grad_(True)
            oeq_conv.forward(Xo, Yo, Wo, rows, cols).sum().backward()
            grad_out = {"x": Xo.grad, "y": Yo.grad, "w": Wo.grad}
            missing_out = [name for name, grad in grad_out.items() if grad is None]
            if missing_out:
                raise RuntimeError(
                    "OEQ backward produced no gradient for "
                    + ", ".join(sorted(missing_out))
                )
            grad_errors = {
                name: max_abs_err(grad_ref[name], grad_out[name])
                for name in ("x", "y", "w")
            }
            grad_oks = {
                name: bool(
                    torch.allclose(
                        grad_ref[name],
                        grad_out[name],
                        rtol=cfg.rtol,
                        atol=cfg.atol,
                    )
                )
                for name in ("x", "y", "w")
            }
            grad_err = max(grad_errors.values())
            grad_ok = all(grad_oks.values())
        except Exception as exc:
            grad_error = _error_message(exc)
            grad_errors = {"x": None, "y": None, "w": None}
            grad_oks = {"x": False, "y": False, "w": False}
    else:
        grad_error = "skipped because forward validation failed"
        grad_errors = {"x": None, "y": None, "w": None}
        grad_oks = {"x": False, "y": False, "w": False}

    return {
        "orientation": orient.orientation,
        "rows_is_dst": orient.rows_is_dst,
        "fwd_ok": fwd_ok,
        "fwd_err": fwd_err,
        "grad_ok": grad_ok,
        "grad_err": grad_err,
        "grad_x_ok": grad_oks["x"],
        "grad_x_err": grad_errors["x"],
        "grad_y_ok": grad_oks["y"],
        "grad_y_err": grad_errors["y"],
        "grad_w_ok": grad_oks["w"],
        "grad_w_err": grad_errors["w"],
        "chunk_edges": cfg.chunk_edges,
        "fwd_error": fwd_error,
        "grad_error": grad_error,
        "orientation_forward_error": orient.forward_error,
        "orientation_swapped_error": orient.swapped_error,
    }
