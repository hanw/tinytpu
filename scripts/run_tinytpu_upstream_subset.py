#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import argparse, os, subprocess, sys

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "tests" / "tinytpu_upstream_subset.txt"


def _load_manifest(path:Path) -> list[str]:
  tests: list[str] = []
  for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if line and not line.startswith("#"):
      tests.append(line)
  return tests


def main(argv:list[str]) -> int:
  parser = argparse.ArgumentParser(description="Run the selected upstream tinygrad tests on TINYTPU.")
  parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="Manifest of tinygrad pytest node ids")
  parser.add_argument("--dry-run", action="store_true", help="Print the pytest command without running it")
  args = parser.parse_args(argv[1:])

  manifest = Path(args.manifest)
  tests = _load_manifest(manifest)
  if not tests:
    raise RuntimeError(f"no tests listed in {manifest}")
  cmd = [sys.executable, "-m", "pytest", "-q", *tests]
  env = {
    **os.environ,
    "PYTHONPATH": str(REPO_ROOT / "tinygrad"),
    "DEVICE": "TINYTPU",
    "TINYTPU_SIM": os.environ.get("TINYTPU_SIM", str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")),
  }
  if args.dry_run:
    print("cwd:", REPO_ROOT / "tinygrad")
    print("env: DEVICE=TINYTPU")
    print("cmd:", " ".join(cmd))
    return 0
  return subprocess.run(cmd, cwd=REPO_ROOT / "tinygrad", env=env, check=False).returncode


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
