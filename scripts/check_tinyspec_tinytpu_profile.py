#!/usr/bin/env python3
"""Check that TinyTPU backend docs mention current lowering classes/opcodes."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tinygrad"))

from tinygrad.runtime.ops_tinytpu import _SXU_OPS, _VPU_OPS  # noqa: E402
from tinygrad.renderer.tinytpu import KernelClass  # noqa: E402


DOCS = [
    ROOT / "doc" / "tinyspec_coverage.md",
    ROOT / "doc" / "plan-tinyspec-tinytpu-sync.md",
]


def _doc_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in DOCS)


def main() -> int:
    text = _doc_text()
    missing: list[str] = []

    for klass in KernelClass:
        if klass is KernelClass.UNSUPPORTED:
            continue
        if klass.name not in text:
            missing.append(f"KernelClass.{klass.name}")

    for name in _VPU_OPS:
        if name not in text:
            missing.append(f"VPU.{name}")

    for name in _SXU_OPS:
        if name not in text:
            missing.append(f"SXU.{name}")

    if missing:
        print("TinyTPU docs are missing current backend names:")
        for item in missing:
            print(f"  - {item}")
        print("\nUpdate doc/tinyspec_coverage.md or doc/plan-tinyspec-tinytpu-sync.md.")
        return 1

    print("TinyTPU Tinyspec profile docs mention all current lowering classes and opcodes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
