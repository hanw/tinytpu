#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, os, subprocess, sys, tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))

from profiler.bundle import Bundle, parse_bundle_file, write_bundle_file
from profiler.perfetto_emitter import write_perfetto
from profiler.reports import (
  print_bubbles, print_hotspots, print_instruction_mix, print_mxu_breakdown,
  print_summary, print_utilization, print_vpu_breakdown,
)
from profiler.sample_program import make_sample_bundle
from profiler.tinygrad_bridge import bundle_from_tinygrad_script
from profiler.trace_parser import parse_trace_output


def _trace_sim_path() -> Path:
  if (env := os.environ.get("TINYTPU_TRACE_SIM")):
    return Path(env)
  return Path(__file__).resolve().parents[1] / "build" / "mkTbTinyTPURuntimeTrace.bexe"


def _run_trace(bundle:Bundle) -> tuple[str, list, list[str]]:
  sim = _trace_sim_path()
  if not sim.exists():
    raise FileNotFoundError(f"traced simulator not found: {sim}. Run `make runtime-tb-trace`.")
  with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    write_bundle_file(f.name, bundle)
    bundle_path = f.name
  try:
    proc = subprocess.run([str(sim)], env={**os.environ, "TINYTPU_BUNDLE": bundle_path}, capture_output=True, text=True, check=False)
  finally:
    os.unlink(bundle_path)
  if proc.returncode != 0:
    raise RuntimeError(f"trace simulator exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
  events, lines = parse_trace_output(proc.stdout)
  return proc.stdout, events, lines


def _load_bundle(args:argparse.Namespace) -> Bundle:
  if args.sample: return make_sample_bundle()
  if args.from_tinygrad is not None: return bundle_from_tinygrad_script(args.from_tinygrad)
  if args.bundle is not None: return parse_bundle_file(args.bundle)
  raise ValueError("bundle source missing")


def main(argv:list[str]) -> int:
  parser = argparse.ArgumentParser(description="Profile TinyTPU runtime bundles and emit Perfetto traces.")
  parser.add_argument("bundle", nargs="?", help="Numeric TinyTPU bundle file")
  parser.add_argument("--trace-out", default="trace.json", help="Perfetto JSON output path")
  parser.add_argument("--from-tinygrad", help="Python script that runs a tinygrad TINYTPU program")
  parser.add_argument("--sample", action="store_true", help="Run the built-in sample bundle")
  parser.add_argument("--dump-raw-trace", action="store_true", help="Print raw TRACE stdout before the report")
  args = parser.parse_args(argv[1:])

  bundle_sources = sum(bool(x) for x in (args.bundle, args.from_tinygrad, args.sample))
  if bundle_sources != 1:
    parser.error("choose exactly one input source: <bundle>, --from-tinygrad, or --sample")

  bundle = _load_bundle(args)
  raw_stdout, events, lines = _run_trace(bundle)
  write_perfetto(args.trace_out, events)

  if args.dump_raw_trace:
    print(raw_stdout.rstrip())
    print()
  print_summary(bundle, events, lines)
  print_instruction_mix(bundle)
  print_utilization(events, lines)
  print_hotspots(bundle, events)
  print_mxu_breakdown(events)
  print_vpu_breakdown(events)
  print_bubbles(events, lines)
  print(f"\ntrace_json: {args.trace_out}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
