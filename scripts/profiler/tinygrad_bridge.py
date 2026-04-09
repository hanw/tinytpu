from __future__ import annotations
from pathlib import Path
import os, runpy, sys
from .bundle import Bundle, parse_bundle_text


class _BundleCaptured(Exception): pass


def bundle_from_tinygrad_script(script_path:str|Path) -> Bundle:
  repo_root = Path(__file__).resolve().parents[2]
  tinygrad_dir = repo_root / "tinygrad"
  if not tinygrad_dir.exists():
    raise RuntimeError("tinygrad submodule not found; run `git submodule update --init`")

  sys.path.insert(0, str(tinygrad_dir))
  import tinygrad.runtime.ops_tinytpu as ops_tinytpu

  captured: dict[str, str] = {}
  orig_run = ops_tinytpu.subprocess.run

  def fake_run(*args, **kwargs):
    env = kwargs.get("env") or os.environ
    bundle_path = env.get("TINYTPU_BUNDLE")
    if bundle_path and Path(bundle_path).exists():
      captured["bundle_text"] = Path(bundle_path).read_text(encoding="utf-8")
    raise _BundleCaptured()

  ops_tinytpu.subprocess.run = fake_run
  try:
    try:
      runpy.run_path(str(script_path), run_name="__main__")
    except _BundleCaptured:
      pass
  finally:
    ops_tinytpu.subprocess.run = orig_run

  if "bundle_text" not in captured:
    raise RuntimeError("tinygrad script did not execute a TINYTPU program")
  return parse_bundle_text(captured["bundle_text"])
