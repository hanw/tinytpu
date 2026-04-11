from __future__ import annotations
import io, json, os, subprocess, sys, tempfile, unittest
from contextlib import redirect_stdout
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from profiler.bundle import Bundle, make_vpu_binary_bundle, parse_bundle_text
from profiler.reports import print_summary, print_utilization, print_vpu_breakdown
from profiler.sample_program import make_sample_bundle
from profiler.trace_parser import parse_trace_output


class TestProfilerHelpers(unittest.TestCase):
  def test_bundle_roundtrip(self):
    bundle = make_sample_bundle()
    parsed = parse_bundle_text(bundle.to_text())
    self.assertEqual(parsed.instructions[0].opcode_name, "SXU_LOAD_VREG")
    self.assertEqual(len(parsed.instructions), len(bundle.instructions))
    self.assertTrue(parsed.output_mxu)

  def test_bundle_roundtrip_vmem_records(self):
    bundle = Bundle(vmem_tiles=[(2, list(range(16)))], output_vmem_addr=2)
    parsed = parse_bundle_text(bundle.to_text())
    self.assertEqual(parsed.vmem_tiles, [(2, list(range(16)))])
    self.assertEqual(parsed.output_vmem_addr, 2)
    self.assertFalse(parsed.output_mxu)

  def test_make_vpu_binary_bundle(self):
    bundle = make_vpu_binary_bundle([1, 2, 3], [4, 5, 6], vpu_op=0)
    parsed = parse_bundle_text(bundle.to_text())
    self.assertEqual(parsed.vmem_tiles[0], (0, [1, 2, 3] + [0] * 13))
    self.assertEqual(parsed.vmem_tiles[1], (1, [4, 5, 6] + [0] * 13))
    self.assertEqual([instr.opcode for instr in parsed.instructions], [0, 0, 2, 1, 6])
    self.assertEqual(parsed.instructions[2].vpu_op, 0)
    self.assertEqual(parsed.output_vmem_addr, 2)

  def test_make_vpu_binary_bundle_rejects_bad_width(self):
    with self.assertRaisesRegex(ValueError, "expects 1..16 elements"):
      make_vpu_binary_bundle(list(range(17)), list(range(17)), vpu_op=0)

  def test_bundle_parse_reports_bad_integer_line(self):
    with self.assertRaisesRegex(ValueError, "line 2: invalid integer 'nope'"):
      parse_bundle_text("3 1\n2 5 nope\n4\n")

  def test_bundle_parse_rejects_non_boolean_output_flag(self):
    with self.assertRaisesRegex(ValueError, "line 1: output flag must be 0 or 1"):
      parse_bundle_text("3 7\n4\n")

  def test_bundle_parse_rejects_bad_vmem_tile_width(self):
    with self.assertRaisesRegex(ValueError, "line 1: vmem tile record expects 17 integers"):
      parse_bundle_text("5 2 1 2 3\n4\n")

  def test_bundle_parse_rejects_bad_output_vmem_width(self):
    with self.assertRaisesRegex(ValueError, "line 1: output vmem record expects 1 integer"):
      parse_bundle_text("6 2 3\n4\n")

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

  def test_trace_parser_rejects_malformed_trace_line(self):
    with self.assertRaisesRegex(ValueError, "line 1: malformed TRACE line"):
      parse_trace_output("TRACE unit=SXU ev=FETCH\n")

  def test_trace_parser_rejects_malformed_field_token(self):
    with self.assertRaisesRegex(ValueError, "line 1: malformed TRACE field 'pc3'"):
      parse_trace_output("TRACE cycle=0 unit=SXU ev=FETCH pc3\n")

  def test_trace_parser_rejects_empty_field_value(self):
    with self.assertRaisesRegex(ValueError, "line 1: empty TRACE field value for pc"):
      parse_trace_output("TRACE cycle=0 unit=SXU ev=FETCH pc=\n")

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

  def test_vpu_breakdown_reports_exec_cycles(self):
    events, _ = parse_trace_output(
      "TRACE cycle=1 unit=VPU ev=EXEC op=0\n"
      "TRACE cycle=2 unit=VPU ev=EXEC op=0\n"
      "TRACE cycle=3 unit=VPU ev=EXEC op=5\n"
    )
    out = io.StringIO()
    with redirect_stdout(out):
      print_vpu_breakdown(events)
    rendered = out.getvalue()
    self.assertIn("op=0: exec_cycles=2", rendered)
    self.assertIn("op=5: exec_cycles=1", rendered)

  def test_dump_bundle_cli_writes_sample(self):
    with tempfile.TemporaryDirectory() as td:
      out = Path(td) / "bundle.txt"
      proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "dump_tinytpu_bundle.py"), "--sample", "--out", str(out)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      parsed = parse_bundle_text(out.read_text(encoding="utf-8"))
      self.assertTrue(parsed.output_mxu)
      self.assertGreater(len(parsed.instructions), 0)

  def test_dump_bundle_cli_rejects_multiple_sources(self):
    proc = subprocess.run(
      [sys.executable, str(REPO_ROOT / "scripts" / "dump_tinytpu_bundle.py"), "--sample", "bundle.txt"],
      cwd=REPO_ROOT,
      text=True,
      capture_output=True,
      check=False,
    )
    self.assertNotEqual(proc.returncode, 0)
    self.assertIn("choose exactly one input source", proc.stderr)


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
