from __future__ import annotations
import io, json, os, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from profiler.bundle import parse_bundle_text
from profiler.reports import print_summary, print_utilization
from profiler.sample_program import make_sample_bundle
from profiler.trace_parser import parse_trace_output


class TestProfilerHelpers(unittest.TestCase):
  def test_bundle_roundtrip(self):
    bundle = make_sample_bundle()
    parsed = parse_bundle_text(bundle.to_text())
    self.assertEqual(parsed.instructions[0].opcode_name, "SXU_LOAD_VREG")
    self.assertEqual(len(parsed.instructions), len(bundle.instructions))
    self.assertTrue(parsed.output_mxu)

  def test_bundle_parse_reports_bad_integer_line(self):
    with self.assertRaisesRegex(ValueError, "line 2: invalid integer 'nope'"):
      parse_bundle_text("3 1\n2 5 nope\n4\n")

  def test_trace_parser(self):
    events, lines = parse_trace_output(
      "TRACE cycle=7 unit=SXU ev=WAIT_MXU pc=3\n"
      "TRACE cycle=7 unit=MXU ev=STREAM_A cyc=1\n"
      "mxu_result 1 2 3 4\n"
      "cycles 12\n"
      "status ok\n"
    )
    self.assertEqual(len(events), 2)
    self.assertEqual(events[0].fields["pc"], "3")
    self.assertEqual(lines[-1], "status ok")

  def test_reports_render_expected_sections(self):
    bundle = make_sample_bundle()
    events, lines = parse_trace_output(
      "TRACE cycle=0 unit=SXU ev=FETCH pc=0\n"
      "TRACE cycle=1 unit=VMEM ev=READ_REQ addr=2\n"
      "TRACE cycle=2 unit=VPU ev=EXEC op=2\n"
      "TRACE cycle=3 unit=MXU ev=STREAM_A cyc=0\n"
      "cycles 6\n"
      "status ok\n"
    )
    out = io.StringIO()
    with redirect_stdout(out):
      print_summary(bundle, events, lines)
      print_utilization(events, lines)
    rendered = out.getvalue()
    self.assertIn("SXU: busy=", rendered)
    self.assertIn("MXU: busy=", rendered)
    self.assertIn("VPU: busy=", rendered)
    self.assertIn("VMEM: busy=", rendered)


@unittest.skipUnless((REPO_ROOT / "build" / "mkTbTinyTPURuntimeTrace.bexe").exists(), "traced runtime binary not built")
class TestProfilerIntegration(unittest.TestCase):
  def test_profile_cli_sample(self):
    with tempfile.TemporaryDirectory() as td:
      trace_out = Path(td) / "trace.json"
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "profile_tpu.py"), "--sample", "--trace-out", str(trace_out)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=os.environ.copy(),
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      self.assertTrue(trace_out.exists())
      data = json.loads(trace_out.read_text(encoding="utf-8"))
      self.assertTrue(data["traceEvents"])
      self.assertIn("SXU: busy=", proc.stdout)
      self.assertIn("MXU: busy=", proc.stdout)
      self.assertIn("VPU: busy=", proc.stdout)
      self.assertIn("VMEM: busy=", proc.stdout)


if __name__ == "__main__":
  unittest.main()
