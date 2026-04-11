#!/usr/bin/env python3
"""
Profile TinyTPU runtime bundles and emit Perfetto/HTML-compatible traces.

Usage:
    # Profile a tinygrad model script (captures ALL bundles, merges trace)
    python scripts/profile_tpu.py --from-tinygrad scripts/my_model.py

    # Profile a single bundle file
    python scripts/profile_tpu.py my_bundle.txt

    # Run the built-in sample
    python scripts/profile_tpu.py --sample

    # Open the HTML visualizer with the trace
    open scripts/viz_pipeline.html   # then Load JSON → trace.json
"""
from __future__ import annotations
from pathlib import Path
import argparse, json, os, subprocess, sys, tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))

from profiler.bundle import Bundle, parse_bundle_file, write_bundle_file
from profiler.perfetto_emitter import write_perfetto, emit_perfetto, PID, THREAD_IDS
from profiler.reports import (
  print_bubbles, print_hotspots, print_instruction_mix, print_mxu_breakdown,
  print_summary, print_utilization, print_vpu_breakdown,
)
from profiler.sample_program import make_sample_bundle
from profiler.tinygrad_bridge import bundle_from_tinygrad_script, bundles_from_tinygrad_script
from profiler.trace_parser import parse_trace_output, Event


def _trace_sim_path() -> Path:
  if (env := os.environ.get("TINYTPU_TRACE_SIM")):
    return Path(env)
  return Path(__file__).resolve().parents[1] / "build" / "mkTbTinyTPURuntimeTrace.bexe"


def _run_trace(bundle: Bundle) -> tuple[str, list[Event], list[str]]:
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


def _run_multi_trace(bundles: list[Bundle], trace_out: str, dump_raw: bool) -> None:
  """Trace multiple bundles and merge into one timeline JSON."""
  sim = _trace_sim_path()
  if not sim.exists():
    raise FileNotFoundError(f"traced simulator not found: {sim}. Run `make runtime-tb-trace`.")

  trace_events = [
    {"ph": "M", "pid": PID, "name": "process_name", "args": {"name": "TinyTPU Multi-Bundle Trace"}},
    {"ph": "M", "pid": PID, "tid": 0, "name": "thread_name", "args": {"name": "Layer"}},
  ]
  for unit, tid in THREAD_IDS.items():
    trace_events.append({"ph": "M", "pid": PID, "tid": tid, "name": "thread_name", "args": {"name": unit}})

  cycle_offset = 0
  total_instrs = 0
  all_events: list[Event] = []

  for i, bundle in enumerate(bundles):
    raw_stdout, events, lines = _run_trace(bundle)
    n_instrs = len(bundle.instructions) if hasattr(bundle, 'instructions') else 0
    total_instrs += n_instrs

    if dump_raw:
      print(f"--- Bundle {i} ({n_instrs} instrs) ---")
      print(raw_stdout.rstrip())
      print()

    layer_start = cycle_offset
    for ev in events:
      tid = THREAD_IDS.get(ev.unit)
      if tid is None: continue
      trace_events.append({
        "ph": "X", "name": ev.ev, "cat": ev.unit,
        "ts": ev.cycle + cycle_offset, "dur": 1,
        "pid": PID, "tid": tid,
      })
      all_events.append(Event(cycle=ev.cycle + cycle_offset, unit=ev.unit, ev=ev.ev, fields=ev.fields))

    if events:
      max_c = max(e.cycle for e in events)
      trace_events.append({
        "ph": "X", "name": f"── Bundle {i} ({n_instrs} instrs) ──", "cat": "MAIN",
        "ts": layer_start, "dur": max_c, "pid": PID, "tid": 0,
      })
      print(f"  Bundle {i}: {n_instrs} instrs, {max_c} cycles")
      cycle_offset += max_c + 5
    else:
      print(f"  Bundle {i}: {n_instrs} instrs, no trace events")

  Path(trace_out).write_text(json.dumps({"traceEvents": trace_events}), encoding="utf-8")
  print(f"\nTotal: {len(bundles)} bundles, {total_instrs} instructions, {cycle_offset} cycles")
  print(f"Trace: {trace_out}")


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description="Profile TinyTPU runtime bundles and emit traces.")
  parser.add_argument("bundle", nargs="?", help="Numeric TinyTPU bundle file")
  parser.add_argument("--trace-out", default="trace.json", help="Perfetto JSON output path")
  parser.add_argument("--from-tinygrad", help="Python script that runs a tinygrad TINYTPU program")
  parser.add_argument("--sample", action="store_true", help="Run the built-in sample bundle")
  parser.add_argument("--dump-raw-trace", action="store_true", help="Print raw TRACE stdout before the report")
  args = parser.parse_args(argv[1:])

  bundle_sources = sum(bool(x) for x in (args.bundle, args.from_tinygrad, args.sample))
  if bundle_sources != 1:
    parser.error("choose exactly one input source: <bundle>, --from-tinygrad, or --sample")

  # Multi-bundle path for tinygrad scripts
  if args.from_tinygrad is not None:
    bundles = bundles_from_tinygrad_script(args.from_tinygrad)
    print(f"Captured {len(bundles)} bundle(s) from {args.from_tinygrad}")
    if len(bundles) > 1:
      _run_multi_trace(bundles, args.trace_out, args.dump_raw_trace)
      return 0
    # Single bundle — fall through to detailed report
    bundle = bundles[0]
  elif args.sample:
    bundle = make_sample_bundle()
  else:
    bundle = parse_bundle_file(args.bundle)

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
