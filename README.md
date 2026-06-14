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
  `dL/dX`; reports edge throughput + % of measured peak DRAM BW; records OEQ launch config; writes JSON.
- `pyproject.toml` — installable `oeq-bench` CLI.

## Quickstart (GPU box, mlip env)
```bash
pip install -e .
oeq-bench --irreps "128x0e+128x1o+128x2e" --num-edges 200000 --chunk-edges 16384
```

## OpenEquivariance on GCC 14 / CUDA 13

OEQ 0.6.x vendors `json11`, which can fail under GCC 14 because `uint8_t` is used without an explicit `<cstdint>` include. Use the installer wrapper in the target env:

```bash
cd external/oeq-bench
pixi run -e mlip bash scripts/install_oeq.sh
```

The wrapper only sets `CXXFLAGS="-include cstdint"` for the build and then imports `openequivariance` to verify the install. The root `pixi.toml` also sets this activation variable for the Linux `mlip` env.

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
On rog (sm_120): OEQ 0.6.6 beat chunked-e3nn 10.2× @ 200K edges under e3nn 0.6.0, fwd+grad validated.
cuEquivariance has a smoke wrapper and channelwise tensor-product constructor, but not yet the decision-grade
graph scatter benchmark. TODO: OEQ GCC14/CUDA13 build-fix PR (`-include cstdint`), targeted Nsight kernel filters,
publish per-arch numbers. Submit to OpenEquivariance (build/CI) + ACEsuit/mace (fused path docs).

## Spec 3 Verification (2026-06-14)

- command: `pixi run -e mlip python -m pytest src/tests -q` from MLIPs root -> `11 passed in 0.75s`
- command: `pixi run -e mlip python -m pytest tests -q` from `external/oeq-bench` -> `51 passed, 1 warning in 1.65s`
- command: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 bash scripts/test_spec3_e2e.sh` from MLIPs root -> `Spec 3 smoke: NVIDIA GeForce RTX 5080 Laptop GPU 176.648`
- command: `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 pixi run -e mlip oeq-bench --irreps "128x0e+128x1o+128x2e" --irreps-sh "0e+1o+2e+3o" --num-nodes 12500 --num-edges 200000 --chunk-edges 16384 --backend oeq --validate --profile cuda-events --repeats 20 --out experiments/labs/L13-oeq-fused-tp-conv-benchmark/l13_results.json`
- git SHA: pre-evidence/run SHAs were MLIPs `21ff556a4271548dc71cf3c2aae35eb54b3d1468` and oeq-bench `c42904542908c50a8ed4cef3504f19230b9129a5`
- device: NVIDIA GeForce RTX 5080 Laptop GPU, compute capability 12.0, torch 2.12.0+cu132, CUDA 13.2, e3nn 0.6.0, OEQ 0.6.6
- validation: PASS (`fwd_ok=true`, `grad_ok=true`, fwd err `9.5367431640625e-06`, grad err `9.5367431640625e-06`, rows=dst/cols=src)
- runtime check: PASS (`libgomp.so.1` from torch and `libopenblasp-r0.3.33.so` from the `mlip` env; no duplicates or import errors)
- speedup: e3nn `61.8381 ms`, OEQ `6.0519 ms`, OEQ/e3nn speedup `10.218x` for the fastest 200K-edge cuda-events sweep point
- bandwidth: OEQ `526.64 GB/s`, `58.777%` of 896 GB/s measured peak
- launch config: requested `--block-size 128`, actual OEQ forward config `60` blocks, `192` threads, `6` warps/block, `74112` bytes shared memory; the requested block size is not effective in OEQ 0.6.6's public constructor.
- ncu status: `--profile ncu` ran the generated command, retried successfully with `sudo -n ncu`, wrote `sm120_e3nn060_bs128_ncu.ncu-rep`, imported counters with `ncu --import ... --csv --page raw`, and wrote parsed metrics under `payload["ncu"]["oeq"]`: occupancy `0.9482`, DRAM throughput `84.666647%`, L2 hit rate `0.973`, stall reason `long_scoreboard`.
