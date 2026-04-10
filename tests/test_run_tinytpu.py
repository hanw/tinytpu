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
