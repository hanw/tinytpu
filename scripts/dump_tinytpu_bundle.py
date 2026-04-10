#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, sys

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from profiler.bundle import parse_bundle_file
from profiler.sample_program import make_sample_bundle
from profiler.tinygrad_bridge import bundle_from_tinygrad_script


def main(argv:list[str]) -> int:
  parser = argparse.ArgumentParser(description="Dump a normalized TinyTPU numeric bundle.")
  parser.add_argument("bundle", nargs="?", help="Existing numeric TinyTPU bundle file")
  parser.add_argument("--from-tinygrad", help="Python script that runs a tinygrad TINYTPU program")
  parser.add_argument("--sample", action="store_true", help="Dump the built-in sample bundle")
  parser.add_argument("--out", help="Write normalized bundle text to this path instead of stdout")
  args = parser.parse_args(argv[1:])

  sources = sum(bool(x) for x in (args.bundle, args.from_tinygrad, args.sample))
  if sources != 1:
    parser.error("choose exactly one input source: <bundle>, --from-tinygrad, or --sample")

  if args.sample:
    bundle = make_sample_bundle()
  elif args.from_tinygrad is not None:
    bundle = bundle_from_tinygrad_script(args.from_tinygrad)
  else:
    bundle = parse_bundle_file(args.bundle)

  text = bundle.to_text()
  if args.out:
    Path(args.out).write_text(text, encoding="utf-8")
  else:
    print(text, end="")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
