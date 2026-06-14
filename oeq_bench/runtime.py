from __future__ import annotations

import argparse
import importlib
import json
import os
import re
from pathlib import Path

OPENMP_RE = re.compile(r"^(?:libgomp|libomp|libiomp5?)(?:[-.]|$)", re.IGNORECASE)
BLAS_RE = re.compile(r"lib(openblas|mkl_rt|blas)", re.IGNORECASE)


def current_process_library_paths() -> list[str]:
    maps = Path("/proc/self/maps")
    if not maps.exists():
        return []
    paths: set[str] = set()
    for line in maps.read_text(errors="ignore").splitlines():
        if "/" not in line:
            continue
        path = line.rsplit(maxsplit=1)[-1]
        if path.startswith("/") and os.path.exists(path):
            paths.add(path)
    return sorted(paths)


def classify_runtime_paths(paths: list[str]) -> dict[str, list[str]]:
    grouped = {"openmp": [], "blas": []}
    for path in paths:
        name = Path(path).name
        if OPENMP_RE.search(name):
            grouped["openmp"].append(path)
        if BLAS_RE.search(name):
            grouped["blas"].append(path)
    return {key: sorted(value) for key, value in grouped.items()}


def import_probe_modules(module_names: list[str]) -> dict[str, str]:
    errors = {}
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as exc:
            errors[module_name] = f"{type(exc).__name__}: {exc}"
    return errors


def runtime_report(paths: list[str] | None = None, imports: list[str] | None = None) -> dict:
    import_errors = import_probe_modules(imports or [])
    grouped = classify_runtime_paths(paths if paths is not None else current_process_library_paths())
    duplicates = {key: value for key, value in grouped.items() if len(value) > 1}
    return {
        "pass": not duplicates and not import_errors,
        "duplicates": duplicates,
        "import_errors": import_errors,
        "runtimes": grouped,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check loaded OpenMP/BLAS runtime libraries")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--import", dest="imports", action="append", default=[])
    args = parser.parse_args(argv)
    report = runtime_report(imports=args.imports)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("PASS" if report["pass"] else "FAIL")
        for group, paths in report["duplicates"].items():
            print(f"{group}:")
            for path in paths:
                print(f"  {path}")
        for module_name, error in report["import_errors"].items():
            print(f"{module_name}: {error}")
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
