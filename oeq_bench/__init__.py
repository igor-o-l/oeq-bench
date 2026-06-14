"""oeq-bench — fused equivariant TP-conv kernel benchmark + correctness suite.

Modules:
  config  — frozen benchmark configuration dataclass and CLI namespace adapter.
  oracle  — scatter-chunked e3nn TP-conv reference (reaches paper-scale edge counts without OOM).
  bench   — time OEQ / cuEquivariance vs the oracle; validate fwd + dL/dX; report throughput + BW.

Provenance: generalized from MLIPs/experiments/labs/L13-oeq-fused-tp-conv-benchmark/.
"""
from .config import BenchConfig

__version__ = "0.1.0.dev0"

__all__ = ["BenchConfig", "__version__"]
