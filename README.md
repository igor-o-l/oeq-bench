# oeq-bench

A reproducible, arch-pinned benchmark + correctness suite for fused equivariant tensor-product kernels
(**OpenEquivariance**, **NVIDIA cuEquivariance**) vs the e3nn reference, on current GPUs (Blackwell sm_120 /
CUDA 13). Generalizes `MLIPs/experiments/labs/L13-oeq-fused-tp-conv-benchmark/`. Implements the
validation/benchmark half of [Spec 3](../../docs/community-contributions/03-blackwell-fused-equivariant-kernels.md).

## Why
Fused TP-conv + symmetric-contraction kernels dominate MACE/NequIP/Allegro inference; coverage/tuning lags
new arches. This gives per-arch, version-pinned numbers (speedup, achieved BW, fwd+grad correctness) and —
crucially — a **scatter-chunked e3nn oracle** so the reference can *reach* paper-scale edge counts where the
naive e3nn OOMs (the kernel's memory advantage is as important as its speed).

## What's here
- `oeq_bench/oracle.py` — chunked e3nn TP-conv reference (FFT-free; tiles edges so neither the [E, out]
  message tensor nor e3nn's intermediates blow up at 200K–800K edges).
- `oeq_bench/bench.py` — times OEQ (and optionally cuEquivariance) vs the oracle; validates forward +
  `dL/dX`; reports edge throughput + % of measured peak DRAM BW; writes JSON.
- `pyproject.toml` — installable `oeq-bench` CLI.

## Quickstart (GPU box, mlip env)
```bash
pip install -e .
oeq-bench --irreps "128x0e+128x1o+128x2e" --num-edges 200000 --chunk-edges 16384
```

## cuEquivariance

Install the PyTorch frontend plus CUDA ops into the existing `mlip` env; keep the pinned torch wheel:

```bash
pixi run -e mlip pip install cuequivariance-torch cuequivariance-ops-torch-cu13
# For CUDA-12 torch environments instead:
pixi run -e mlip pip install cuequivariance-torch cuequivariance-ops-torch-cu12
```

`oeq-bench --backend cueq` first runs a `cuequivariance_torch.Linear` smoke check and constructs the documented
channelwise tensor-product `SegmentedPolynomial`. The OEQ backend remains the decision-grade TP-conv path until
cuEq's graph scatter path is validated against the same e3nn oracle.

## Status
On rog (sm_120): OEQ 0.6.6 beat chunked-e3nn 8.4× @ 200K edges, fwd+grad validated. cuEquivariance has a smoke
wrapper and channelwise tensor-product constructor, but not yet the decision-grade graph scatter benchmark. TODO:
OEQ GCC14/CUDA13 build-fix PR (`-include cstdint`), Nsight roofline (needs `ncu` unblock), publish per-arch
numbers. Submit to OpenEquivariance (build/CI) + ACEsuit/mace (fused path docs).
