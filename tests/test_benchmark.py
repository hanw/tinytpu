from __future__ import annotations
import subprocess, sys, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestBenchmark(unittest.TestCase):
  def test_benchmark_runs_add_kernel(self):
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    proc = subprocess.run(
      [sys.executable, str(REPO_ROOT / "scripts" / "benchmark_tinytpu.py"),
       "--filter", "add_16", "--json"],
      cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
    import json
    results = json.loads(proc.stdout)
    self.assertEqual(len(results), 1)
    r = results[0]
    self.assertEqual(r["name"], "add_16")
    self.assertGreater(r["cycles"], 0)
    self.assertEqual(r["work_elems"], 16)
    self.assertGreater(r["elems_per_cycle"], 0.0)
    self.assertGreater(r["bytes_per_cycle"], 0.0)

  def test_benchmark_csv_output(self):
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    proc = subprocess.run(
      [sys.executable, str(REPO_ROOT / "scripts" / "benchmark_tinytpu.py"),
       "--filter", "relu", "--csv"],
      cwd=REPO_ROOT, capture_output=True, text=True, check=False,
    )
    self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    self.assertEqual(lines[0], "kernel,cycles,wall_ms,elems,elems_per_cycle,bytes_per_cycle")
    self.assertTrue(any(ln.startswith("relu_64,") for ln in lines[1:]))


if __name__ == "__main__":
  unittest.main()
