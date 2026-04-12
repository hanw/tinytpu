#!/usr/bin/env python3
"""Latency/throughput benchmarks for TinyTPU.

Drives a set of representative kernels through the BSV simulator,
captures the `cycles N` line emitted by the testbench, and reports
per-kernel latency (cycles) and throughput (int32 elements/cycle).

Usage:
    python3 scripts/benchmark_tinytpu.py              # all kernels
    python3 scripts/benchmark_tinytpu.py --filter col # substring match
    python3 scripts/benchmark_tinytpu.py --json       # JSON output
    python3 scripts/benchmark_tinytpu.py --csv        # CSV output

Requires the runtime sim to be built:
    make build/mkTbTinyTPURuntime.bexe
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
TINYGRAD = REPO_ROOT / "tinygrad"
if str(TINYGRAD) not in sys.path: sys.path.insert(0, str(TINYGRAD))

# Ensure tinygrad picks up our backend and sim.
os.environ["TINYTPU_SIM"] = str(SIM)


@dataclass
class Bench:
    name: str
    run: Callable[[], None]
    work_elems: int  # int32 elements "done" by this kernel


@dataclass
class Result:
    name: str
    cycles: int
    wall_ms: float
    work_elems: int

    @property
    def elems_per_cycle(self) -> float:
        return self.work_elems / self.cycles if self.cycles else 0.0

    @property
    def bytes_per_cycle(self) -> float:
        return self.work_elems * 4 / self.cycles if self.cycles else 0.0


def _parse_cycles(stdout: str) -> int:
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("cycles "):
            return int(line.split()[1])
    raise RuntimeError("no `cycles` line in sim stdout")


def _run_and_get_cycles(bundle_text: str) -> int:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(bundle_text)
        path = f.name
    try:
        env = {**os.environ, "TINYTPU_BUNDLE": path}
        proc = subprocess.run([str(SIM)], env=env, capture_output=True, text=True, timeout=30)
    finally:
        os.unlink(path)
    if proc.returncode != 0:
        raise RuntimeError(f"sim exited {proc.returncode}\n{proc.stdout}\n{proc.stderr}")
    return _parse_cycles(proc.stdout)


def _run_tinygrad(run_fn) -> tuple[int, float]:
    """Run a tinygrad callable and capture cycles from the last sim invocation.

    Monkey-patches _run_bundle in ops_tinytpu to record cycles from every
    bundle executed. Returns (total_cycles, wall_ms).
    """
    import tinygrad.runtime.ops_tinytpu as o
    original = o._run_bundle
    captured: list[int] = []

    def wrapped(sim, bundle_text):
        stdout = original(sim, bundle_text)
        try:
            captured.append(_parse_cycles(stdout))
        except Exception:
            pass
        return stdout

    o._run_bundle = wrapped
    t0 = time.perf_counter()
    try:
        run_fn()
    finally:
        o._run_bundle = original
    wall = (time.perf_counter() - t0) * 1000
    return sum(captured), wall


# --- Benchmark kernels ----------------------------------------------------

def _bench_add_16():
    import numpy as np
    from tinygrad import Tensor
    a = Tensor(np.arange(16, dtype=np.int32), dtype="int32", device="TINYTPU")
    b = Tensor(np.arange(16, dtype=np.int32) + 1, dtype="int32", device="TINYTPU")
    (a + b).numpy()


def _bench_add_256():
    import numpy as np
    from tinygrad import Tensor
    a = Tensor(np.arange(256, dtype=np.int32), dtype="int32", device="TINYTPU")
    b = Tensor(np.arange(256, dtype=np.int32) + 1, dtype="int32", device="TINYTPU")
    (a + b).numpy()


def _bench_sum_16():
    import numpy as np
    from tinygrad import Tensor
    Tensor(np.arange(16, dtype=np.int32), dtype="int32", device="TINYTPU").sum().numpy()


def _bench_sum_256():
    import numpy as np
    from tinygrad import Tensor
    Tensor(np.arange(256, dtype=np.int32), dtype="int32", device="TINYTPU").sum().numpy()


def _bench_rowsum_8x8():
    import numpy as np
    from tinygrad import Tensor
    Tensor(np.arange(64, dtype=np.int32).reshape(8, 8), dtype="int32",
           device="TINYTPU").sum(axis=1).numpy()


def _bench_colsum_8x8():
    import numpy as np
    from tinygrad import Tensor
    Tensor(np.arange(64, dtype=np.int32).reshape(8, 8), dtype="int32",
           device="TINYTPU").sum(axis=0).numpy()


def _bench_gemm_1x4x4():
    """1x4 @ 4x4 GEMM — smallest supported shape."""
    import numpy as np
    from tinygrad import Tensor
    a = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int32), dtype="int32", device="TINYTPU")
    w = Tensor(np.eye(4, dtype=np.int32), dtype="int32", device="TINYTPU")
    (a @ w).numpy()


def _bench_gemm_4x4x4():
    """4x4 @ 4x4 GEMM."""
    import numpy as np
    from tinygrad import Tensor
    a = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
    w = Tensor(np.eye(4, dtype=np.int32), dtype="int32", device="TINYTPU")
    (a @ w).numpy()


def _bench_gemm_4x8x4():
    """4x8 @ 8x4 GEMM — deep-K two tiles."""
    import numpy as np
    from tinygrad import Tensor
    a = Tensor(np.arange(32, dtype=np.int32).reshape(4, 8), dtype="int32", device="TINYTPU")
    w = Tensor(np.ones((8, 4), dtype=np.int32), dtype="int32", device="TINYTPU")
    (a @ w).numpy()


def _bench_relu_64():
    import numpy as np
    from tinygrad import Tensor
    a = Tensor(np.arange(64, dtype=np.int32) - 32, dtype="int32", device="TINYTPU")
    a.relu().numpy()


def _bench_reshape_64():
    import numpy as np
    from tinygrad import Tensor
    Tensor(np.arange(64, dtype=np.int32), dtype="int32",
           device="TINYTPU").reshape(8, 8).numpy()


BENCHES: list[Bench] = [
    Bench("add_16",          _bench_add_16,      16),
    Bench("add_256",         _bench_add_256,     256),
    Bench("relu_64",         _bench_relu_64,     64),
    Bench("reshape_64",      _bench_reshape_64,  64),
    Bench("sum_16_scalar",   _bench_sum_16,      16),
    Bench("sum_256_scalar",  _bench_sum_256,     256),
    Bench("rowsum_8x8",      _bench_rowsum_8x8,  64),
    Bench("colsum_8x8",      _bench_colsum_8x8,  64),
    Bench("gemm_1x4x4",      _bench_gemm_1x4x4,  16),  # 1*4*4 MACs
    Bench("gemm_4x4x4",      _bench_gemm_4x4x4,  64),  # 4*4*4 MACs
    Bench("gemm_4x8x4",      _bench_gemm_4x8x4,  128), # 4*8*4 MACs
]


def run_benches(filter_: str | None = None) -> list[Result]:
    if not SIM.exists():
        raise FileNotFoundError(f"sim not built: {SIM}. Run `make build/mkTbTinyTPURuntime.bexe`.")
    out: list[Result] = []
    for b in BENCHES:
        if filter_ and filter_ not in b.name:
            continue
        cycles, wall_ms = _run_tinygrad(b.run)
        out.append(Result(name=b.name, cycles=cycles, wall_ms=wall_ms, work_elems=b.work_elems))
    return out


def print_table(results: list[Result]) -> None:
    hdr = f"{'kernel':<22}{'cycles':>10}{'wall (ms)':>14}{'elems':>10}{'elems/cyc':>14}{'bytes/cyc':>14}"
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        print(f"{r.name:<22}{r.cycles:>10}{r.wall_ms:>14.2f}{r.work_elems:>10}"
              f"{r.elems_per_cycle:>14.3f}{r.bytes_per_cycle:>14.3f}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--filter", help="only run benchmarks whose name contains this substring")
    ap.add_argument("--json", action="store_true", help="emit JSON results")
    ap.add_argument("--csv", action="store_true", help="emit CSV results")
    args = ap.parse_args(argv[1:])
    results = run_benches(args.filter)
    if args.json:
        print(json.dumps([r.__dict__ | {"elems_per_cycle": r.elems_per_cycle,
                                         "bytes_per_cycle": r.bytes_per_cycle}
                          for r in results], indent=2))
    elif args.csv:
        print("kernel,cycles,wall_ms,elems,elems_per_cycle,bytes_per_cycle")
        for r in results:
            print(f"{r.name},{r.cycles},{r.wall_ms:.2f},{r.work_elems},"
                  f"{r.elems_per_cycle:.3f},{r.bytes_per_cycle:.3f}")
    else:
        print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
