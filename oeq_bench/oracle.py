"""Scatter-chunked e3nn TP-conv reference.

The naive `tp(X[src], Y, W)` + scatter materialises the full [E, out_dim] message tensor (and, for the
backward, all of e3nn's per-path intermediates), which OOMs at paper-scale edge counts. Tiling the edge
loop bounds peak memory to O(chunk), letting the *reference* reach 200K-800K edges where the one-shot form
dies — so the OEQ/cuEq comparison can be made at scale (the fused kernel needs no chunking; that's its win).

Ported from MLIPs/experiments/labs/L13-oeq-fused-tp-conv-benchmark/benchmark_oeq_vs_e3nn.py.
Requires torch + e3nn in the environment.
"""
from __future__ import annotations

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in optional-dependency environments
    torch = None

from .backends import build_instructions


def _torch_dtype(name: str):
    if torch is None:
        raise ModuleNotFoundError("torch")
    return getattr(torch, name)


def build_e3nn_tp(cfg):
    from e3nn.o3 import Irreps, TensorProduct

    irreps_in1 = Irreps(cfg.irreps_in1)
    irreps_in2 = Irreps(cfg.irreps_in2)
    irreps_out = Irreps(cfg.irreps_out)
    tp = TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        build_instructions(irreps_in1, irreps_in2, irreps_out),
        shared_weights=False,
        internal_weights=False,
    ).to(_torch_dtype(cfg.dtype))
    return tp.to(cfg.device)


def make_graph_data(cfg, weight_numel: int, seed: int | None = None):
    from e3nn.o3 import Irreps

    if torch is None:
        raise ModuleNotFoundError("torch")
    torch.manual_seed(cfg.seed if seed is None else seed)
    irreps_in1 = Irreps(cfg.irreps_in1)
    irreps_in2 = Irreps(cfg.irreps_in2)
    dtype = _torch_dtype(cfg.dtype)
    X = torch.randn(cfg.num_nodes, irreps_in1.dim, dtype=dtype, device=cfg.device)
    Y = torch.randn(cfg.num_edges, irreps_in2.dim, dtype=dtype, device=cfg.device)
    W = torch.randn(cfg.num_edges, weight_numel, dtype=dtype, device=cfg.device)
    src = torch.randint(0, cfg.num_nodes, (cfg.num_edges,), dtype=torch.int64, device=cfg.device)
    dst = torch.randint(0, cfg.num_nodes, (cfg.num_edges,), dtype=torch.int64, device=cfg.device)
    if cfg.edge_ordering != "random":
        if cfg.edge_ordering == "dst-src":
            key = dst * cfg.num_nodes + src
        elif cfg.edge_ordering == "src-dst":
            key = src * cfg.num_nodes + dst
        else:
            raise ValueError(f"unsupported edge_ordering: {cfg.edge_ordering}")
        perm = torch.argsort(key)
        Y = Y[perm].contiguous()
        W = W[perm].contiguous()
        src = src[perm].contiguous()
        dst = dst[perm].contiguous()
    return X, Y, W, src, dst


def e3nn_scatter(tp, X, Y, W, src, dst, num_nodes, chunk=0):
    """Gather X[src] → per-edge TP → scatter-add into dst, tiled over edges if chunk>0.
    Pure-value path (wrap caller in torch.no_grad() for forward/timing)."""
    out_dim = tp.irreps_out.dim
    Z = torch.zeros(num_nodes, out_dim, dtype=X.dtype, device=X.device)
    E = src.shape[0]
    step = chunk if chunk and chunk > 0 else E
    for s in range(0, E, step):
        e = min(s + step, E)
        msg = tp(X[src[s:e]], Y[s:e], W[s:e])
        Z.scatter_add_(0, dst[s:e].unsqueeze(-1).expand_as(msg), msg)
    return Z


def e3nn_grad_chunked(tp, X, Y, W, src, dst, chunk=0):
    """dL/dX for L = sum(Z), tiled so each chunk's autograd graph is freed before the next
    (bounded memory at paper scale; autograd accumulates per-chunk grads into X.grad)."""
    return e3nn_grads_chunked(tp, X, Y, W, src, dst, chunk)["x"]


def e3nn_grads_chunked(tp, X, Y, W, src, dst, chunk=0):
    """dL/d(X,Y,W) for L = sum(Z), tiled over edges for bounded reference memory."""
    Xg = X.detach().clone().requires_grad_(True)
    Yg = Y.detach().clone().requires_grad_(True)
    Wg = W.detach().clone().requires_grad_(True)
    E = src.shape[0]
    step = chunk if chunk and chunk > 0 else E
    for s in range(0, E, step):
        e = min(s + step, E)
        tp(Xg[src[s:e]], Yg[s:e], Wg[s:e]).sum().backward()
    return {"x": Xg.grad, "y": Yg.grad, "w": Wg.grad}
