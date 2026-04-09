#!/usr/bin/env python3.12
from __future__ import annotations
import argparse, io, json, os, subprocess, sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
TINYGRAD_ROOT = REPO_ROOT / "tinygrad"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for p in (TINYGRAD_ROOT, SCRIPTS_ROOT):
  if str(p) not in sys.path: sys.path.insert(0, str(p))

from tinygrad import Tensor
from tinygrad.device import Device
from tinygrad.codegen import get_program
from tinygrad.nn.onnx import OnnxRunner
from tinygrad.runtime.ops_tinytpu import analyze_tinytpu_uops
import tinygrad.runtime.ops_tinytpu as ops_tinytpu

from profiler.bundle import parse_bundle_text
from profiler.perfetto_emitter import write_perfetto
from profiler.reports import (
  print_bubbles, print_hotspots, print_instruction_mix, print_mxu_breakdown,
  print_summary, print_utilization,
)
from profiler.trace_parser import parse_trace_output


def build_sample_onnx_model(model_path:Path, with_relu:bool=False) -> Path:
  try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper
  except ModuleNotFoundError as e:
    raise RuntimeError("sample ONNX generation requires the `onnx` package; install it or pass --onnx <model.onnx>") from e

  weight = np.array([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ], dtype=np.int32)
  nodes = [helper.make_node("MatMul", ["input", "weight"], ["matmul_out"])]
  output_name = "matmul_out"
  if with_relu:
    nodes.append(helper.make_node("Relu", ["matmul_out"], ["output"]))
    output_name = "output"

  graph = helper.make_graph(
    nodes=nodes,
    name="tinytpu_matmul_relu" if with_relu else "tinytpu_matmul",
    inputs=[helper.make_tensor_value_info("input", TensorProto.INT32, [1, 4])],
    outputs=[helper.make_tensor_value_info(output_name, TensorProto.INT32, [1, 4])],
    initializer=[numpy_helper.from_array(weight, name="weight")],
  )
  model = helper.make_model(graph)
  model_path.parent.mkdir(parents=True, exist_ok=True)
  onnx.save(model, model_path)
  return model_path


class _BundleCaptured(Exception): pass


def make_tinytpu_outputs(onnx_path:Path, input_values:np.ndarray) -> tuple[dict[str, Tensor], dict[str, list]]:
  runner = OnnxRunner(onnx_path).to("TINYTPU")
  inputs = {"input": Tensor(input_values, dtype="int32", device="TINYTPU")}
  outputs = runner(inputs)
  return outputs, {"input": input_values.tolist()}


def compile_onnx_to_tinytpu(outputs:dict[str, Tensor]) -> list[dict]:
  compiled: list[dict] = []
  for kernel_idx, si in enumerate(Tensor.schedule(*outputs.values())):
    if si.ast.op.name != "SINK": continue
    prg = get_program(si.ast, Device["TINYTPU"].renderer)
    src = json.loads(prg.src)
    diag = analyze_tinytpu_uops(prg.uops)
    compiled.append({
      "kernel_index": kernel_idx,
      "program_name": prg.name,
      "src": src,
      "supported": diag["supported"],
      "reason": diag["reason"],
      "missing_instructions": diag["missing_instructions"],
      "notes": diag["notes"],
      "op_counts": diag["op_counts"],
    })
  return compiled


def lower_supported_matmul_to_bundle(onnx_path:Path, input_values:np.ndarray) -> tuple[str, dict]:
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
    outputs, input_meta = make_tinytpu_outputs(onnx_path, input_values)
    try:
      Tensor.realize(*outputs.values())
    except _BundleCaptured:
      pass
  finally:
    ops_tinytpu.subprocess.run = orig_run

  if "bundle_text" not in captured:
    raise RuntimeError("failed to capture TinyTPU bundle from ONNX execution")
  return captured["bundle_text"], input_meta


def run_trace(bundle_path:Path, trace_sim:Path) -> tuple[str, list, list[str]]:
  proc = subprocess.run(
    [str(trace_sim)],
    env={**os.environ, "TINYTPU_BUNDLE": str(bundle_path)},
    text=True,
    capture_output=True,
    check=False,
  )
  if proc.returncode != 0:
    raise RuntimeError(f"trace simulator exited {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
  events, lines = parse_trace_output(proc.stdout)
  return proc.stdout, events, lines


def write_compile_report(out_dir:Path, model_path:Path, input_meta:dict, compile_info:list[dict]) -> Path:
  report_path = out_dir / "compile_report.txt"
  buf = io.StringIO()
  print(f"onnx_model: {model_path}", file=buf)
  print(f"inputs: {input_meta}", file=buf)
  print(f"kernels: {len(compile_info)}", file=buf)
  for item in compile_info:
    print(f"\n== Kernel {item['kernel_index']} ==", file=buf)
    print(f"program_name: {item['program_name']}", file=buf)
    print(f"supported: {item['supported']}", file=buf)
    print(f"reason: {item['reason']}", file=buf)
    if item["missing_instructions"]:
      print("missing_instructions:", ", ".join(item["missing_instructions"]), file=buf)
    if item["notes"]:
      for note in item["notes"]:
        print(f"note: {note}", file=buf)
    print(f"src: {json.dumps(item['src'])}", file=buf)
    print(f"op_counts: {json.dumps(item['op_counts'], sort_keys=True)}", file=buf)
  report_path.write_text(buf.getvalue(), encoding="utf-8")
  (out_dir / "compile_report.json").write_text(json.dumps(compile_info, indent=2), encoding="utf-8")
  return report_path


def main(argv:list[str]) -> int:
  parser = argparse.ArgumentParser(description="Compile ONNX through tinygrad on TINYTPU, report missing instructions, and trace supported runs.")
  parser.add_argument("--onnx", help="Path to an ONNX model. If omitted, generate a sample MatMul model in this directory.")
  parser.add_argument("--sample", choices=["matmul", "matmul_relu"], default="matmul", help="Generated sample model when --onnx is omitted.")
  parser.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "artifacts"), help="Artifact directory for model, compile report, bundle, trace, and report.")
  parser.add_argument("--input", default="1,2,3,4", help="Comma-separated int32 input vector for the sample or single-input model.")
  parser.add_argument("--trace-sim", default=str(REPO_ROOT / "build" / "mkTbTinyTPURuntimeTrace.bexe"), help="Path to traced TinyTPU runtime binary.")
  args = parser.parse_args(argv[1:])

  out_dir = Path(args.out_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  trace_sim = Path(args.trace_sim)
  input_values = np.array([int(x) for x in args.input.split(",")], dtype=np.int32).reshape(1, 4)

  if args.onnx:
    model_path = Path(args.onnx)
  else:
    model_path = build_sample_onnx_model(out_dir / f"sample_{args.sample}.onnx", with_relu=(args.sample == "matmul_relu"))

  outputs, input_meta = make_tinytpu_outputs(model_path, input_values)
  compile_info = compile_onnx_to_tinytpu(outputs)
  compile_report = write_compile_report(out_dir, model_path, input_meta, compile_info)

  unsupported = [item for item in compile_info if not item["supported"]]
  if unsupported:
    print(f"compile_report: {compile_report}")
    print("unsupported kernels found:")
    for item in unsupported:
      print(f"  kernel {item['kernel_index']}: {item['reason']}")
      if item["missing_instructions"]:
        print(f"    missing: {', '.join(item['missing_instructions'])}")
    return 0

  if not trace_sim.exists():
    raise FileNotFoundError(f"traced simulator not found: {trace_sim}. Run `make runtime-tb-trace`.")

  bundle_text, _ = lower_supported_matmul_to_bundle(model_path, input_values)
  bundle_path = out_dir / "captured_bundle.txt"
  bundle_path.write_text(bundle_text, encoding="utf-8")
  bundle = parse_bundle_text(bundle_text)

  raw_stdout, events, lines = run_trace(bundle_path, trace_sim)
  trace_path = out_dir / "trace.json"
  write_perfetto(trace_path, events)

  report_buf = io.StringIO()
  with redirect_stdout(report_buf):
    print_summary(bundle, events, lines)
    print_instruction_mix(bundle)
    print_utilization(events, lines)
    print_hotspots(bundle, events)
    print_mxu_breakdown(events)
    print_bubbles(events, lines)
  report_path = out_dir / "report.txt"
  report_path.write_text(report_buf.getvalue(), encoding="utf-8")
  trace_stdout_path = out_dir / "trace_stdout.txt"
  trace_stdout_path.write_text(raw_stdout, encoding="utf-8")

  metadata = {
    "onnx_model": str(model_path),
    "input": input_meta["input"],
    "compile_report": str(compile_report),
    "bundle": str(bundle_path),
    "trace_json": str(trace_path),
    "report": str(report_path),
    "trace_stdout": str(trace_stdout_path),
  }
  (out_dir / "run.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

  print(f"compile_report: {compile_report}")
  print(f"bundle: {bundle_path}")
  print(f"trace_json: {trace_path}")
  print(f"report: {report_path}")
  print(f"trace_stdout: {trace_stdout_path}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main(sys.argv))
