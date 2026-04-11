from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Sequence

SXU_OP_NAMES = {
  0: "SXU_LOAD_VREG",
  1: "SXU_STORE_VREG",
  2: "SXU_DISPATCH_VPU",
  3: "SXU_DISPATCH_XLU_BROADCAST",
  4: "SXU_DISPATCH_MXU",
  5: "SXU_WAIT_MXU",
  6: "SXU_LOAD_MXU_RESULT",
  7: "SXU_HALT",
  8: "SXU_DISPATCH_SELECT",
  9: "SXU_BROADCAST_SCALAR",
  10: "SXU_BROADCAST_ROW",
  11: "SXU_BROADCAST_COL",
}

VPU_OP_NAMES = {
  0: "VPU_ADD",
  1: "VPU_MUL",
  2: "VPU_RELU",
  3: "VPU_MAX",
  4: "VPU_SUM_REDUCE",
  5: "VPU_CMPLT",
  6: "VPU_CMPNE",
  7: "VPU_SUB",
  8: "VPU_CMPEQ",
  9: "VPU_MAX_REDUCE",
  10: "VPU_SHL",
  11: "VPU_SHR",
  12: "VPU_MIN",
  13: "VPU_MIN_REDUCE",
  14: "VPU_DIV",
  15: "VPU_AND",
  16: "VPU_OR",
  17: "VPU_XOR",
  18: "VPU_FADD",
  19: "VPU_FMUL",
  20: "VPU_FSUB",
  21: "VPU_FMAX",
  22: "VPU_FCMPLT",
  23: "VPU_FRECIP",
  24: "VPU_I2F",
  25: "VPU_F2I",
  26: "VPU_NOT",
  27: "VPU_SELECT",
  28: "VPU_COPY",
}


@dataclass(frozen=True)
class BundleInstr:
  opcode: int
  vmem_addr: int
  vreg_dst: int
  vreg_src: int
  vpu_op: int
  vreg_src2: int
  mxu_w_base: int
  mxu_a_base: int
  mxu_t_len: int

  @property
  def opcode_name(self) -> str:
    return SXU_OP_NAMES.get(self.opcode, f"OP_{self.opcode}")

  @property
  def vpu_op_name(self) -> str:
    return VPU_OP_NAMES.get(self.vpu_op, f"VPU_{self.vpu_op}")

  def to_record(self) -> str:
    return "2 " + " ".join(str(x) for x in (
      self.opcode, self.vmem_addr, self.vreg_dst, self.vreg_src, self.vpu_op,
      self.vreg_src2, self.mxu_w_base, self.mxu_a_base, self.mxu_t_len,
    ))


@dataclass
class Bundle:
  weight_tiles: list[tuple[int, list[int]]] = field(default_factory=list)
  act_tiles: list[tuple[int, list[int]]] = field(default_factory=list)
  vmem_tiles: list[tuple[int, list[int]]] = field(default_factory=list)
  instructions: list[BundleInstr] = field(default_factory=list)
  output_mxu: bool = False
  output_vmem_addr: int | None = None

  def to_text(self) -> str:
    lines: list[str] = []
    for addr, vals in self.weight_tiles:
      if len(vals) != 16: raise ValueError(f"weight tile at addr {addr} must have 16 values")
      lines.append("0 " + " ".join([str(addr)] + [str(v) for v in vals]))
    for addr, vals in self.act_tiles:
      if len(vals) != 4: raise ValueError(f"activation tile at addr {addr} must have 4 values")
      lines.append("1 " + " ".join([str(addr)] + [str(v) for v in vals]))
    for addr, vals in self.vmem_tiles:
      if len(vals) != 16: raise ValueError(f"vmem tile at addr {addr} must have 16 values")
      lines.append("5 " + " ".join([str(addr)] + [str(v) for v in vals]))
    lines.extend(instr.to_record() for instr in self.instructions)
    lines.append(f"3 {1 if self.output_mxu else 0}")
    if self.output_vmem_addr is not None:
      lines.append(f"6 {self.output_vmem_addr}")
    lines.append("4")
    return "\n".join(lines) + "\n"


def parse_bundle_text(text:str) -> Bundle:
  bundle = Bundle()
  for lineno, raw in enumerate(text.splitlines(), start=1):
    line = raw.strip()
    if not line or line.startswith("#"): continue
    parts = line.split()
    rec_type = parts[0]
    try:
      vals = [int(x) for x in parts[1:]]
    except ValueError as exc:
      bad = next((x for x in parts[1:] if not x.lstrip("-").isdigit()), parts[1])
      raise ValueError(f"line {lineno}: invalid integer {bad!r}") from exc
    if rec_type == "0":
      if len(vals) != 17: raise ValueError(f"line {lineno}: weight tile record expects 17 integers")
      bundle.weight_tiles.append((vals[0], vals[1:]))
    elif rec_type == "1":
      if len(vals) != 5: raise ValueError(f"line {lineno}: activation tile record expects 5 integers")
      bundle.act_tiles.append((vals[0], vals[1:]))
    elif rec_type == "2":
      if len(vals) != 9: raise ValueError(f"line {lineno}: instruction record expects 9 integers")
      bundle.instructions.append(BundleInstr(*vals))
    elif rec_type == "3":
      if len(vals) != 1: raise ValueError(f"line {lineno}: output flag record expects 1 integer")
      if vals[0] not in (0, 1): raise ValueError(f"line {lineno}: output flag must be 0 or 1")
      bundle.output_mxu = vals[0] != 0
    elif rec_type == "4":
      break
    elif rec_type == "5":
      if len(vals) != 17: raise ValueError(f"line {lineno}: vmem tile record expects 17 integers")
      bundle.vmem_tiles.append((vals[0], vals[1:]))
    elif rec_type == "6":
      if len(vals) != 1: raise ValueError(f"line {lineno}: output vmem record expects 1 integer")
      bundle.output_vmem_addr = vals[0]
    else:
      raise ValueError(f"line {lineno}: unknown record type {rec_type!r}")
  return bundle


def parse_bundle_file(path:str|Path) -> Bundle:
  return parse_bundle_text(Path(path).read_text(encoding="utf-8"))


def write_bundle_file(path:str|Path, bundle:Bundle) -> None:
  Path(path).write_text(bundle.to_text(), encoding="utf-8")


def make_vpu_binary_bundle(lhs:Sequence[int], rhs:Sequence[int], vpu_op:int, num_elems:int|None=None) -> Bundle:
  elem_count = len(lhs) if num_elems is None else num_elems
  if not 0 < elem_count <= 16:
    raise ValueError(f"VPU binary bundle expects 1..16 elements, got {elem_count}")
  if len(lhs) < elem_count or len(rhs) < elem_count:
    raise ValueError(f"VPU binary operands shorter than num_elems={elem_count}")

  def tile(vals:Sequence[int]) -> list[int]:
    padded = [0] * 16
    padded[:elem_count] = [int(v) for v in vals[:elem_count]]
    return padded

  return Bundle(
    vmem_tiles=[(0, tile(lhs)), (1, tile(rhs))],
    instructions=[
      BundleInstr(0, 0, 0, 0, 0, 0, 0, 0, 0),
      BundleInstr(0, 1, 1, 0, 0, 0, 0, 0, 0),
      BundleInstr(2, 0, 2, 0, int(vpu_op), 1, 0, 0, 0),
      BundleInstr(1, 2, 0, 2, 0, 0, 0, 0, 0),
      BundleInstr(7, 0, 0, 0, 0, 0, 0, 0, 0),
    ],
    output_vmem_addr=2,
  )
