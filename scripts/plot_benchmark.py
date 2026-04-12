#!/usr/bin/env python3
"""Plot TinyTPU benchmark progress: experiment # vs geomean elems/cycle.

Reads doc/benchmark_history.tsv and renders:
  - grey dots for discarded experiments (kept=0)
  - green dots for kept experiments (kept=1)
  - green step line tracing the running best (higher is better)
  - labels on each kept experiment

Higher elems/cycle is better — the chip may scale (wider MXU/VPU) so we
track a scale-invariant aggregate (geomean of per-kernel elems/cycle)
instead of total cycles, which would shift with the kernel mix.

Usage:
    python3 scripts/plot_benchmark.py                  # writes doc/benchmark_progress.png
    python3 scripts/plot_benchmark.py --out other.png
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY = REPO_ROOT / "doc" / "benchmark_history.tsv"
DEFAULT_OUT = REPO_ROOT / "doc" / "benchmark_progress.png"


def load_history(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            rows.append({
                "experiment": int(r["experiment"]),
                "date":       r["date"],
                "commit":     r["commit"],
                "label":      r["label"],
                "kept":       int(r["kept"]),
                "metric":     float(r["geomean_elems_per_cycle"]),
                "notes":      r.get("notes", ""),
            })
    rows.sort(key=lambda r: r["experiment"])
    return rows


def running_best(rows: list[dict]) -> list[tuple[int, float]]:
    """Step function: (experiment #, best metric so far), only over kept rows."""
    best = float("-inf")
    trace: list[tuple[int, float]] = []
    for r in rows:
        if not r["kept"]:
            continue
        if r["metric"] > best:
            best = r["metric"]
        trace.append((r["experiment"], best))
    return trace


def plot(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

    kept   = [r for r in rows if r["kept"]]
    disc   = [r for r in rows if not r["kept"]]
    trace  = running_best(rows)

    fig, ax = plt.subplots(figsize=(10, 5))

    if disc:
        ax.scatter([r["experiment"] for r in disc],
                   [r["metric"] for r in disc],
                   color="lightgray", s=18, label="Discarded", zorder=2)

    if kept:
        ax.scatter([r["experiment"] for r in kept],
                   [r["metric"] for r in kept],
                   color="#2ca67a", s=60, label="Kept", zorder=4)

    if trace:
        xs = [trace[0][0]] + [t[0] for t in trace]
        ys = [trace[0][1]] + [t[1] for t in trace]
        ax.step(xs, ys, color="#2ca67a", where="post", linewidth=1.3,
                label="Running best", zorder=3)

    for r in kept:
        ax.annotate(r["label"], xy=(r["experiment"], r["metric"]),
                    xytext=(6, 4), textcoords="offset points",
                    fontsize=9, color="#2a2a2a", rotation=20)

    ax.set_xlabel("Experiment #")
    ax.set_ylabel("Geomean elements / cycle (higher is better)")
    total_kept = len(kept)
    total_all  = len(rows)
    ax.set_title(f"TinyTPU Perf Progress: {total_all} Experiments, {total_kept} Kept Improvements")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    ax.set_xlim(left=0)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", default=str(HISTORY), help="TSV history file")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output PNG path")
    args = ap.parse_args(argv[1:])
    rows = load_history(Path(args.history))
    if not rows:
        print("no rows in history", file=sys.stderr)
        return 1
    plot(rows, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
