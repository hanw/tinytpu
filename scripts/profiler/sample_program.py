#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path

if __package__ in {None, ""}:
  sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
  from profiler.bundle import Bundle, BundleInstr, write_bundle_file
else:
  from .bundle import Bundle, BundleInstr, write_bundle_file


def make_sample_bundle() -> Bundle:
  return Bundle(
    weight_tiles=[(0, [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ])],
    act_tiles=[(1, [1, -2, 3, -4])],
    instructions=[
      BundleInstr(0, 2, 0, 0, 0, 0, 0, 0, 0),
      BundleInstr(2, 0, 1, 0, 2, 0, 0, 0, 0),
      BundleInstr(1, 3, 0, 1, 0, 0, 0, 0, 0),
      BundleInstr(3, 0, 0, 0, 0, 0, 0, 1, 1),
      BundleInstr(4, 0, 0, 0, 0, 0, 0, 0, 0),
      BundleInstr(3, 0, 0, 0, 0, 0, 0, 1, 1),
      BundleInstr(4, 0, 0, 0, 0, 0, 0, 0, 0),
      BundleInstr(5, 0, 0, 0, 0, 0, 0, 0, 0),
    ],
    output_mxu=True,
  )


def main(argv:list[str]) -> int:
  if len(argv) != 2:
    print(f"usage: {argv[0]} <bundle-path>", file=sys.stderr)
    return 1
  out = Path(argv[1])
  out.parent.mkdir(parents=True, exist_ok=True)
  write_bundle_file(out, make_sample_bundle())
  print(out)
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
