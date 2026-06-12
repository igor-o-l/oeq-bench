"""Scatter-chunked e3nn TP-conv reference.

The naive `tp(X[src], Y, W)` + scatter materialises the full [E, out_dim] message tensor (and, for the
backward, all of e3nn's per-path intermediates), which OOMs at paper-scale edge counts. Tiling the edge
loop bounds peak memory to O(chunk), letting the *reference* reach 200K-800K edges where the one-shot form
dies — so the OEQ/cuEq comparison can be made at scale (the fused kernel needs no chunking; that's its win).

Ported from MLIPs/experiments/labs/L13-oeq-fused-tp-conv-benchmark/benchmark_oeq_vs_e3nn.py.
Requires torch + e3nn in the environment.
"""
from __future__ import annotations
import torch


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
    Xg = X.detach().clone().requires_grad_(True)
    E = src.shape[0]
    step = chunk if chunk and chunk > 0 else E
    for s in range(0, E, step):
        e = min(s + step, E)
        tp(Xg[src[s:e]], Y[s:e], W[s:e]).sum().backward()
    return Xg.grad
