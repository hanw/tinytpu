"""pytest session guard: refuse to run against a stale or missing simulator.

The BSV simulator at ``build/mkTbTinyTPURuntime.bexe.so`` is a build artifact
(``build/`` is gitignored). When the BSV source under ``src/`` advances but the
simulator is not rebuilt, the sim-backed tests run against outdated hardware and
fail with timeouts or garbage output — failures that look like broken hardware
but are really a stale build. This guard stops the session up front with one
actionable message instead.

Set ``TINYTPU_SKIP_SIM_STALENESS_CHECK=1`` to bypass the guard (e.g. to run the
non-sim tests on a machine without the Bluespec toolchain).
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SIM_SO = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe.so"
_REBUILD_HINT = "run `make runtime-tb` to rebuild it"


def _sim_sources() -> list[Path]:
  """Files the runtime simulator is compiled from (see the Makefile rule)."""
  srcs = sorted((REPO_ROOT / "src").glob("*.bsv"))
  for extra in ("test/TbTinyTPURuntime.bsv", "bdpi/tinytpu_io.c"):
    path = REPO_ROOT / extra
    if path.exists():
      srcs.append(path)
  return srcs


def pytest_sessionstart(session: pytest.Session) -> None:
  if os.environ.get("TINYTPU_SKIP_SIM_STALENESS_CHECK"):
    return

  if not _SIM_SO.exists():
    pytest.exit(f"TinyTPU simulator not built ({_SIM_SO} missing) — {_REBUILD_HINT}.",
                returncode=1)

  sim_mtime = _SIM_SO.stat().st_mtime
  stale = [p for p in _sim_sources() if p.stat().st_mtime > sim_mtime]
  if stale:
    shown = ", ".join(p.relative_to(REPO_ROOT).as_posix() for p in stale[:3])
    more = f" (+{len(stale) - 3} more)" if len(stale) > 3 else ""
    pytest.exit(
      f"TinyTPU simulator is stale — {len(stale)} source file(s) newer than "
      f"{_SIM_SO.relative_to(REPO_ROOT).as_posix()}: {shown}{more}. {_REBUILD_HINT}.",
      returncode=1,
    )
