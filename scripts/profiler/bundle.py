from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

SXU_OP_NAMES = {
  0: "SXU_LOAD_VREG",
  1: "SXU_STORE_VREG",
  2: "SXU_DISPATCH_VPU",
  3: "SXU_DISPATCH_MXU",
  4: "SXU_WAIT_MXU",
  5: "SXU_HALT",
}

VPU_OP_NAMES = {
  0: "VPU_ADD",
  1: "VPU_MUL",
  2: "VPU_RELU",
  3: "VPU_MAX",
  4: "VPU_SUM_REDUCE",
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
  instructions: list[BundleInstr] = field(default_factory=list)
  output_mxu: bool = False

  def to_text(self) -> str:
    lines: list[str] = []
    for addr, vals in self.weight_tiles:
      if len(vals) != 16: raise ValueError(f"weight tile at addr {addr} must have 16 values")
      lines.append("0 " + " ".join([str(addr)] + [str(v) for v in vals]))
    for addr, vals in self.act_tiles:
      if len(vals) != 4: raise ValueError(f"activation tile at addr {addr} must have 4 values")
      lines.append("1 " + " ".join([str(addr)] + [str(v) for v in vals]))
    lines.extend(instr.to_record() for instr in self.instructions)
    lines.append(f"3 {1 if self.output_mxu else 0}")
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
    else:
      raise ValueError(f"line {lineno}: unknown record type {rec_type!r}")
  return bundle


def parse_bundle_file(path:str|Path) -> Bundle:
  return parse_bundle_text(Path(path).read_text(encoding="utf-8"))


def write_bundle_file(path:str|Path, bundle:Bundle) -> None:
  Path(path).write_text(bundle.to_text(), encoding="utf-8")
