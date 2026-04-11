from __future__ import annotations
from pathlib import Path
import os, runpy, subprocess, sys
from .bundle import Bundle, parse_bundle_text


def bundles_from_tinygrad_script(script_path: str | Path) -> list[Bundle]:
    """Run a tinygrad script, capture all TinyTPU bundles, and return them.

    Intercepts subprocess.run calls to capture bundle texts while letting
    the real simulator run so the model executes to completion.
    """
    repo_root = Path(__file__).resolve().parents[2]
    tinygrad_dir = repo_root / "tinygrad"
    if not tinygrad_dir.exists():
        raise RuntimeError("tinygrad submodule not found; run `git submodule update --init`")

    sys.path.insert(0, str(tinygrad_dir))
    import tinygrad.runtime.ops_tinytpu as ops_tinytpu

    captured: list[str] = []
    orig_run = ops_tinytpu.subprocess.run

    def capture_run(*args, **kwargs):
        env = kwargs.get("env") or os.environ
        bundle_path = env.get("TINYTPU_BUNDLE")
        if bundle_path and Path(bundle_path).exists():
            captured.append(Path(bundle_path).read_text(encoding="utf-8"))
        return orig_run(*args, **kwargs)

    ops_tinytpu.subprocess.run = capture_run
    try:
        runpy.run_path(str(script_path), run_name="__main__")
    finally:
        ops_tinytpu.subprocess.run = orig_run

    if not captured:
        raise RuntimeError("tinygrad script did not execute any TINYTPU programs")
    return [parse_bundle_text(t) for t in captured]


# Backwards compat — returns first bundle only
def bundle_from_tinygrad_script(script_path: str | Path) -> Bundle:
    return bundles_from_tinygrad_script(script_path)[0]
