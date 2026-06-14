from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchConfig:
    irreps_in1: str = "128x0e+128x1o+128x2e"
    irreps_in2: str = "0e+1o+2e+3o"
    irreps_out: str | None = None
    num_nodes: int = 12500
    num_edges: int = 200000
    chunk_edges: int = 16384
    backend: str = "oeq"
    validate: bool = False
    profile: str = "cuda-events"
    dry_run: bool = False
    warmup: int = 5
    repeats: int = 20
    out: Path = Path("oeq_bench_results.json")
    skip_runtime_check: bool = False
    rtol: float = 1e-4
    atol: float = 1e-4
    bw_peak_gbs: float = 896.0
    block_size: int = 128
    dtype: str = "float32"
    device: str = "cuda"

    def __post_init__(self) -> None:
        if self.irreps_out is None:
            object.__setattr__(self, "irreps_out", self.irreps_in1)
        if self.backend not in {"oeq", "cueq", "both"}:
            raise ValueError(f"unsupported backend: {self.backend}")
        if self.profile not in {"cuda-events", "ncu"}:
            raise ValueError(f"unsupported profile mode: {self.profile}")
        if self.num_nodes <= 0 or self.num_edges <= 0:
            raise ValueError("num_nodes and num_edges must be positive")
        if self.chunk_edges < 0:
            raise ValueError("chunk_edges must be non-negative")
        if self.warmup < 0 or self.repeats <= 0:
            raise ValueError("warmup must be non-negative and repeats must be positive")
        if self.block_size not in {128, 256, 512}:
            raise ValueError("block_size must be one of 128, 256, or 512")
        if self.rtol < 0:
            raise ValueError("rtol must be non-negative")
        if self.atol < 0:
            raise ValueError("atol must be non-negative")
        if self.bw_peak_gbs <= 0:
            raise ValueError("bw_peak_gbs must be positive")
        object.__setattr__(self, "out", Path(self.out))

    @classmethod
    def from_args(cls, args: object) -> "BenchConfig":
        return cls(
            irreps_in1=args.irreps,
            irreps_in2=args.irreps_sh,
            irreps_out=args.irreps,
            num_nodes=args.num_nodes,
            num_edges=args.num_edges,
            chunk_edges=args.chunk_edges,
            backend=args.backend,
            validate=args.validate,
            profile=args.profile,
            dry_run=args.dry_run,
            warmup=args.warmup,
            repeats=args.repeats,
            out=Path(args.out),
            skip_runtime_check=args.skip_runtime_check,
            rtol=args.rtol,
            atol=args.atol,
            bw_peak_gbs=args.bw_peak_gbs,
            block_size=args.block_size,
        )

    def backends_to_run(self) -> list[str]:
        return ["oeq", "cueq"] if self.backend == "both" else [self.backend]
