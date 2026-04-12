from __future__ import annotations
import json, os, stat, subprocess, sys, tempfile, textwrap, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


class TestRunTinyTPU(unittest.TestCase):
  def test_sample_bundle_via_fake_sim(self):
    with tempfile.TemporaryDirectory() as td:
      td_path = Path(td)
      fake_sim = td_path / "fake_sim.py"
      fake_sim.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import os, sys
        bundle = os.environ.get("TINYTPU_BUNDLE")
        if not bundle or not os.path.exists(bundle):
          print("missing bundle", file=sys.stderr)
          raise SystemExit(2)
        print("mxu_result 1 2 3 4")
        print("cycles 12")
        print("status ok")
      """), encoding="utf-8")
      fake_sim.chmod(fake_sim.stat().st_mode | stat.S_IEXEC)

      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu.py"), "--sample"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "TINYTPU_SIM": str(fake_sim)},
        check=False,
      )

      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      payload = json.loads(proc.stdout)
      self.assertEqual(payload["bundle_source"], "<sample>")
      self.assertEqual(payload["mxu_result"], [1, 2, 3, 4])
      self.assertEqual(payload["cycles"], 12)
      self.assertEqual(payload["status"], "ok")

  def test_vmem_bundle_via_runtime_sim(self):
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    with tempfile.TemporaryDirectory() as td:
      bundle = Path(td) / "vpu_add.txt"
      bundle.write_text(textwrap.dedent("""\
        5 0 1 2 3 0 0 0 0 0 0 0 0 0 0 0 0 0
        5 1 4 5 6 0 0 0 0 0 0 0 0 0 0 0 0 0
        2 0 0 0 0 0 0 0 0 0
        2 0 1 1 0 0 0 0 0 0
        2 2 0 2 0 0 1 0 0 0
        2 1 2 0 2 0 0 0 0 0
        2 7 0 0 0 0 0 0 0 0
        6 2
        4
      """), encoding="utf-8")
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu.py"), str(bundle)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env={**os.environ, "TINYTPU_SIM": str(sim)},
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      payload = json.loads(proc.stdout)
      self.assertEqual(payload["vmem_result"][:3], [5, 7, 9])
      self.assertEqual(payload["status"], "ok")

  def test_vpu_sum_reduce_tile_via_runtime_sim(self):
    """End-to-end: VPU_SUM_REDUCE_TILE on 1..16 broadcasts scalar 136 to full tile."""
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    with tempfile.TemporaryDirectory() as td:
      bundle = Path(td) / "vpu_sum_reduce_tile.txt"
      # Preload VMEM[0] with 1..16, run SUM_REDUCE_TILE (opcode 32), store to VMEM[1]
      bundle.write_text(textwrap.dedent("""\
        5 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        2 0 0 0 0 0 0 0 0 0
        2 2 0 1 0 32 0 0 0 0
        2 1 1 0 1 0 0 0 0 0
        2 7 0 0 0 0 0 0 0 0
        6 1
        4
      """), encoding="utf-8")
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu.py"), str(bundle)],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={**os.environ, "TINYTPU_SIM": str(sim)}, check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      payload = json.loads(proc.stdout)
      self.assertEqual(payload["status"], "ok")
      # Full-tile sum of 1..16 = 136, broadcast to every slot
      self.assertEqual(payload["vmem_result"], [136] * 16)

  def test_vpu_mul_reduce_tile_via_runtime_sim(self):
    """End-to-end: VPU_MUL_REDUCE_TILE on a small tile broadcasts product to full tile."""
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    with tempfile.TemporaryDirectory() as td:
      bundle = Path(td) / "vpu_mul_reduce_tile.txt"
      # Tile with one 2, one 3, one 4, rest 1s → product 24.
      bundle.write_text(textwrap.dedent("""\
        5 0 2 1 1 1 1 1 1 1 1 1 3 1 1 1 4 1
        2 0 0 0 0 0 0 0 0 0
        2 2 0 1 0 37 0 0 0 0
        2 1 1 0 1 0 0 0 0 0
        2 7 0 0 0 0 0 0 0 0
        6 1
        4
      """), encoding="utf-8")
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu.py"), str(bundle)],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={**os.environ, "TINYTPU_SIM": str(sim)}, check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      payload = json.loads(proc.stdout)
      self.assertEqual(payload["status"], "ok")
      self.assertEqual(payload["vmem_result"], [24] * 16)

  def test_vpu_sum_reduce_col_via_runtime_sim(self):
    """End-to-end: VPU_SUM_REDUCE_COL on 1..16 gives per-col sums broadcast down each column."""
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    with tempfile.TemporaryDirectory() as td:
      bundle = Path(td) / "vpu_sum_reduce_col.txt"
      # VMEM[0] = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
      # col sums = [28, 32, 36, 40], broadcast down each column
      bundle.write_text(textwrap.dedent("""\
        5 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        2 0 0 0 0 0 0 0 0 0
        2 2 0 1 0 29 0 0 0 0
        2 1 1 0 1 0 0 0 0 0
        2 7 0 0 0 0 0 0 0 0
        6 1
        4
      """), encoding="utf-8")
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu.py"), str(bundle)],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={**os.environ, "TINYTPU_SIM": str(sim)}, check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      payload = json.loads(proc.stdout)
      self.assertEqual(payload["status"], "ok")
      # Col sums [28, 32, 36, 40] broadcast to all 4 rows
      expected = [28, 32, 36, 40] * 4
      self.assertEqual(payload["vmem_result"], expected)

  def test_sxu_xlu_transpose_via_runtime_sim(self):
    """End-to-end: SXU_DISPATCH_XLU_TRANSPOSE transposes a 4x4 tile."""
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    with tempfile.TemporaryDirectory() as td:
      bundle = Path(td) / "xlu_transpose.txt"
      # VMEM[0] = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
      # Transpose → [[1,5,9,13],[2,6,10,14],[3,7,11,15],[4,8,12,16]]
      # SXU opcodes: 0=LOAD, 1=STORE, 12=DISPATCH_XLU_TRANSPOSE, 7=HALT
      bundle.write_text(textwrap.dedent("""\
        5 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16
        2 0 0 0 0 0 0 0 0 0
        2 12 0 1 0 0 0 0 0 0
        2 1 1 0 1 0 0 0 0 0
        2 7 0 0 0 0 0 0 0 0
        6 1
        4
      """), encoding="utf-8")
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu.py"), str(bundle)],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={**os.environ, "TINYTPU_SIM": str(sim)}, check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      payload = json.loads(proc.stdout)
      self.assertEqual(payload["status"], "ok")
      self.assertEqual(payload["vmem_result"],
                       [1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16])

  def test_vpu_mul_reduce_col_via_runtime_sim(self):
    """End-to-end: VPU_MUL_REDUCE_COL gives per-column products broadcast down each column."""
    sim = REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe"
    if not sim.exists():
      self.skipTest("runtime binary not built")
    with tempfile.TemporaryDirectory() as td:
      bundle = Path(td) / "vpu_mul_reduce_col.txt"
      # 4x4 tile with col-product targets: col0=2, col1=6, col2=12, col3=24
      bundle.write_text(textwrap.dedent("""\
        5 0 1 1 1 1 2 1 2 2 1 2 3 3 1 3 2 4
        2 0 0 0 0 0 0 0 0 0
        2 2 0 1 0 36 0 0 0 0
        2 1 1 0 1 0 0 0 0 0
        2 7 0 0 0 0 0 0 0 0
        6 1
        4
      """), encoding="utf-8")
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu.py"), str(bundle)],
        cwd=REPO_ROOT, text=True, capture_output=True,
        env={**os.environ, "TINYTPU_SIM": str(sim)}, check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      payload = json.loads(proc.stdout)
      self.assertEqual(payload["status"], "ok")
      # Tile: [1,1,1,1 ; 2,1,2,2 ; 1,2,3,3 ; 1,3,2,4]
      # col products: 1*2*1*1=2, 1*1*2*3=6, 1*2*3*2=12, 1*2*3*4=24
      expected = [2, 6, 12, 24] * 4
      self.assertEqual(payload["vmem_result"], expected)

  def test_upstream_subset_wrapper_dry_run(self):
    proc = subprocess.run(
      [sys.executable, str(REPO_ROOT / "scripts" / "run_tinytpu_upstream_subset.py"), "--dry-run"],
      cwd=REPO_ROOT,
      text=True,
      capture_output=True,
      check=False,
    )
    self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
    self.assertIn("DEVICE=TINYTPU", proc.stdout)
    self.assertIn("test/test_tiny.py::TestTiny::test_plus_int", proc.stdout)


if __name__ == "__main__":
  unittest.main()
