#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, json, os, subprocess, sys, tempfile

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
  sys.path.insert(0, str(SCRIPT_DIR))

from profiler.bundle import Bundle, BundleInstr, parse_bundle_file, write_bundle_file


def _sim_path() -> Path:
  if (env := os.environ.get("TINYTPU_SIM")):
    return Path(env)
  return SCRIPT_DIR.parent / "build" / "mkTbTinyTPURuntime.bexe"


def _parse_result(stdout: str) -> dict:
  result: dict[str, object] = {}
  for raw in stdout.splitlines():
    line = raw.strip()
    if not line:
      continue
    if line.startswith("FAIL:") or line.startswith("ERROR:"):
      result["failure"] = line
    if line.startswith("mxu_result "):
      result["mxu_result"] = [int(x) for x in line.split()[1:]]
    elif line.startswith("vmem_result "):
      result["vmem_result"] = [int(x) for x in line.split()[1:]]
    elif line.startswith("cycles "):
      result["cycles"] = int(line.split()[1])
    elif line.startswith("status "):
      result["status"] = line.split(maxsplit=1)[1]
  return result


def _make_sample_bundle() -> Bundle:
  return Bundle(
    weight_tiles=[(0, [
      1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 1, 0,
      0, 0, 0, 1,
    ])],
    act_tiles=[(1, [1, -2, 3, -4])],
    instructions=[
      BundleInstr(3, 0, 0, 0, 0, 0, 0, 1, 1),
      BundleInstr(4, 0, 0, 0, 0, 0, 0, 0, 0),
      BundleInstr(5, 0, 0, 0, 0, 0, 0, 0, 0),
    ],
    output_mxu=True,
  )


def _load_bundle(args: argparse.Namespace) -> tuple[Bundle, str]:
  if args.sample:
    return _make_sample_bundle(), "<sample>"
  if args.bundle is not None:
    return parse_bundle_file(args.bundle), args.bundle
  raise ValueError("bundle source missing")


def _run(bundle: Bundle) -> tuple[dict, str]:
  sim = _sim_path()
  if not sim.exists():
    raise FileNotFoundError(f"simulator not found: {sim}. Run `make runtime-tb` or set TINYTPU_SIM.")
  with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
    write_bundle_file(f.name, bundle)
    bundle_path = f.name
  try:
    proc = subprocess.run(
      [str(sim)],
      env={**os.environ, "TINYTPU_BUNDLE": bundle_path},
      capture_output=True,
      text=True,
      check=False,
    )
  finally:
    os.unlink(bundle_path)
  if proc.returncode != 0:
    raise RuntimeError(f"simulator exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
  result = _parse_result(proc.stdout)
  if "failure" in result:
    raise RuntimeError(f"simulator reported failure: {result['failure']}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
  if result.get("status") != "ok":
    raise RuntimeError(f"simulator did not report `status ok`\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
  return result, proc.stdout


def main(argv: list[str]) -> int:
  parser = argparse.ArgumentParser(description="Run a TinyTPU numeric bundle through the runtime simulator.")
  parser.add_argument("bundle", nargs="?", help="Numeric TinyTPU bundle file")
  parser.add_argument("--sample", action="store_true", help="Run the built-in sample bundle")
  parser.add_argument("--dump-raw", action="store_true", help="Print raw simulator stdout before the parsed JSON result")
  args = parser.parse_args(argv[1:])

  bundle_sources = int(bool(args.bundle)) + int(args.sample)
  if bundle_sources != 1:
    parser.error("choose exactly one input source: <bundle> or --sample")

  bundle, source = _load_bundle(args)
  result, raw_stdout = _run(bundle)
  payload = {"bundle_source": source, **result}

  if args.dump_raw:
    print(raw_stdout.rstrip())
  print(json.dumps(payload, indent=2, sort_keys=True))
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
