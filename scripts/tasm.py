#!/usr/bin/env python3
"""
tasm.py — TinyTPU bundle assembler / disassembler

Converts human-readable TinyTPU assembly (TASM) to the numeric wire format
consumed by the BSV testbench (TbTinyTPURuntime.bsv via tinytpu_io.c),
and vice versa.

Spec: doc/tinytpu_asm.md
Wire format: doc/tinytpu_bundle_format.md

CLI usage:
    python3 scripts/tasm.py assemble   program.tasm   # wire → stdout
    python3 scripts/tasm.py disassemble bundle.txt    # TASM → stdout
    python3 scripts/tasm.py assemble   -              # read stdin

API usage:
    from scripts.tasm import assemble, disassemble
    wire = assemble(tasm_text)
    tasm = disassemble(wire_text)
"""

import re
import sys

# ---------------------------------------------------------------------------
# Opcode tables (must match BSV enum Bits encoding — 0-based, in order)
# ---------------------------------------------------------------------------

# SXU opcodes: BSV SxuOpCode enum (ScalarUnit.bsv)
_SXU = {
    "LOAD_VREG":              0,
    "STORE_VREG":             1,
    "DISPATCH_VPU":           2,
    "DISPATCH_XLU_BROADCAST": 3,
    "DISPATCH_MXU":           4,
    "WAIT_MXU":               5,
    "LOAD_MXU_RESULT":        6,
    "HALT":                   7,
    "DISPATCH_SELECT":        8,
    "BROADCAST_SCALAR":       9,
    "BROADCAST_ROW":          10,
    "BROADCAST_COL":          11,
    "DISPATCH_XLU_TRANSPOSE": 12,
    "LOAD_VPU_RESULT":        13,
    "LOAD_XLU_RESULT":        14,
    "PSUM_WRITE":             15,
    "PSUM_ACCUMULATE":        16,
    "PSUM_READ":              17,
    "PSUM_READ_ROW":          18,
    "PSUM_CLEAR":             19,
    "SET_PRED_IF_ZERO":       20,
    "SKIP_IF_PRED":           21,
    "PSUM_ACCUMULATE_ROW":    22,
    "DISPATCH_MXU_OS":        23,
}
_SXU_INV = {v: k for k, v in _SXU.items()}

# VPU opcodes: BSV VpuOp enum (VPU.bsv)
_VPU = {
    "ADD":        0,
    "MUL":        1,
    "RELU":       2,
    "MAX":        3,
    "SUM_REDUCE": 4,
    "CMPLT":      5,
    "CMPNE":      6,
    "SUB":        7,
    "CMPEQ":      8,
    "MAX_REDUCE": 9,
    "SHL":        10,
    "SHR":        11,
    "MIN":        12,
    "MIN_REDUCE": 13,
    "DIV":        14,
    "AND":        15,
    "OR":         16,
    "XOR":        17,
    "FADD":       18,
    "FMUL":       19,
    "FSUB":       20,
    "FMAX":       21,
    "FCMPLT":     22,
    "FRECIP":     23,
    "I2F":        24,
    "F2I":        25,
    "NOT":        26,
    "SELECT":     27,
    "COPY":       28,
    "SUM_REDUCE_COL":  29,
    "MAX_REDUCE_COL":  30,
    "MIN_REDUCE_COL":  31,
    "SUM_REDUCE_TILE": 32,
    "MAX_REDUCE_TILE": 33,
    "MIN_REDUCE_TILE": 34,
    "MUL_REDUCE":      35,
    "MUL_REDUCE_COL":  36,
    "MUL_REDUCE_TILE": 37,
    "FSUM_REDUCE_TILE": 38,
    "FMAX_REDUCE_TILE": 39,
    "FMIN_REDUCE_TILE": 40,
    "FMIN":            41,
    "FSUM_REDUCE":     42,
    "FMAX_REDUCE":     43,
    "FMIN_REDUCE":     44,
    "FSUM_REDUCE_COL": 45,
    "FMAX_REDUCE_COL": 46,
    "FMIN_REDUCE_COL": 47,
    "FPROD_REDUCE_TILE": 48,
    "FPROD_REDUCE":      49,
    "FPROD_REDUCE_COL":  50,
    "EXP2":              51,
    "LOG2":              52,
    "SIN":               53,
    "COS":               54,
}
_VPU_INV = {v: k for k, v in _VPU.items()}

# Ops that take a single source register (src2 slot is unused by hardware)
_VPU_UNARY = {"RELU", "SUM_REDUCE", "MAX_REDUCE", "MIN_REDUCE", "NOT", "COPY",
              "SUM_REDUCE_COL", "MAX_REDUCE_COL", "MIN_REDUCE_COL",
              "SUM_REDUCE_TILE", "MAX_REDUCE_TILE", "MIN_REDUCE_TILE",
              "MUL_REDUCE", "MUL_REDUCE_COL", "MUL_REDUCE_TILE",
              "FSUM_REDUCE_TILE", "FMAX_REDUCE_TILE", "FMIN_REDUCE_TILE",
              "FSUM_REDUCE", "FMAX_REDUCE", "FMIN_REDUCE",
              "FSUM_REDUCE_COL", "FMAX_REDUCE_COL", "FMIN_REDUCE_COL",
              "FPROD_REDUCE_TILE", "FPROD_REDUCE", "FPROD_REDUCE_COL",
              "EXP2", "LOG2", "SIN", "COS"}

# VMEM/WMEM/AMEM tile geometry
_VMEM_ELEMS = 16   # 4×4 Int32
_WMEM_ELEMS = 16   # 4×4 Int8
_AMEM_ELEMS = 4    # 4×1 Int8


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _strip_comment(line: str) -> str:
    """Remove trailing # comment and strip whitespace."""
    idx = line.find("#")
    return (line[:idx] if idx >= 0 else line).strip()


def _parse_vreg(s: str) -> int:
    """Parse 'vN' (with optional trailing comma) → register index N."""
    s = s.strip().rstrip(",").strip()
    if re.fullmatch(r"v\d+", s, re.IGNORECASE):
        return int(s[1:])
    raise SyntaxError(f"expected vector register like v0, got {s!r}")


def _parse_mem(tag: str, s: str) -> int:
    """Parse 'TAG[N]' (case-insensitive, optional trailing comma) → slot N."""
    s = s.strip().rstrip(",").strip()
    m = re.fullmatch(rf"{tag}\[(\d+)\]", s, re.IGNORECASE)
    if not m:
        raise SyntaxError(f"expected {tag}[N], got {s!r}")
    return int(m.group(1))


def _instr(opc: int, vmemAddr: int = 0, vregDst: int = 0, vregSrc: int = 0,
           vpuOp: int = 0, vregSrc2: int = 0,
           mxuWBase: int = 0, mxuABase: int = 0, mxuTLen: int = 0) -> str:
    """Emit one SXU instruction wire line (record type 2)."""
    return (f"2 {opc} {vmemAddr} {vregDst} {vregSrc} "
            f"{vpuOp} {vregSrc2} {mxuWBase} {mxuABase} {mxuTLen}")


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------

def assemble(text: str) -> str:
    """
    Convert TASM source text to the numeric wire format expected by the
    TinyTPU BSV testbench.

    Raises SyntaxError with line number on any parse error.
    """
    out: list[str] = []

    for lineno, raw in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw)
        if not line:
            continue

        tokens = line.split()
        kw = tokens[0].upper()

        try:
            # ── Data declarations ──────────────────────────────────────────

            if "=" in line and kw.startswith("VMEM["):
                # VMEM[N] = v0 v1 ... v15   (16 int32 values)
                lhs, rhs = line.split("=", 1)
                addr = _parse_mem("VMEM", lhs)
                vals = [int(x) for x in rhs.split()]
                if len(vals) != _VMEM_ELEMS:
                    raise SyntaxError(
                        f"VMEM tile requires {_VMEM_ELEMS} values, got {len(vals)}")
                out.append("5 " + str(addr) + " " + " ".join(str(v) for v in vals))

            elif "=" in line and kw.startswith("WMEM["):
                # WMEM[N] = w00 ... w15   (16 int8 values, row-major)
                lhs, rhs = line.split("=", 1)
                addr = _parse_mem("WMEM", lhs)
                vals = [int(x) for x in rhs.split()]
                if len(vals) != _WMEM_ELEMS:
                    raise SyntaxError(
                        f"WMEM tile requires {_WMEM_ELEMS} values, got {len(vals)}")
                out.append("0 " + str(addr) + " " + " ".join(str(v) for v in vals))

            elif "=" in line and kw.startswith("AMEM["):
                # AMEM[N] = a0 a1 a2 a3   (4 int8 values)
                lhs, rhs = line.split("=", 1)
                addr = _parse_mem("AMEM", lhs)
                vals = [int(x) for x in rhs.split()]
                if len(vals) != _AMEM_ELEMS:
                    raise SyntaxError(
                        f"AMEM tile requires {_AMEM_ELEMS} values, got {len(vals)}")
                out.append("1 " + str(addr) + " " + " ".join(str(v) for v in vals))

            # ── Instructions ───────────────────────────────────────────────

            elif kw == "LOAD":
                # LOAD vD, VMEM[S]
                rest = line[len("LOAD"):].strip()
                parts = [p.strip() for p in rest.split(",", 1)]
                if len(parts) != 2:
                    raise SyntaxError("LOAD syntax: LOAD vD, VMEM[S]")
                vd = _parse_vreg(parts[0])
                vs = _parse_mem("VMEM", parts[1])
                out.append(_instr(_SXU["LOAD_VREG"], vmemAddr=vs, vregDst=vd))

            elif kw == "STORE":
                # STORE VMEM[D], vS
                rest = line[len("STORE"):].strip()
                parts = [p.strip() for p in rest.split(",", 1)]
                if len(parts) != 2:
                    raise SyntaxError("STORE syntax: STORE VMEM[D], vS")
                vd = _parse_mem("VMEM", parts[0])
                vs = _parse_vreg(parts[1])
                out.append(_instr(_SXU["STORE_VREG"], vmemAddr=vd, vregSrc=vs))

            elif kw == "VPU":
                # VPU vD = OP(vA) or VPU vD = OP(vA, vB)
                rest = line[len("VPU"):].strip()
                m = re.fullmatch(r"(v\d+)\s*=\s*(\w+)\(([^)]+)\)", rest,
                                 re.IGNORECASE)
                if not m:
                    raise SyntaxError(
                        "VPU syntax: VPU vD = OP(vA) or VPU vD = OP(vA, vB)")
                vd = _parse_vreg(m.group(1))
                op_name = m.group(2).upper()
                if op_name not in _VPU:
                    raise SyntaxError(
                        f"unknown VPU op {op_name!r}. "
                        f"Valid ops: {', '.join(sorted(_VPU))}")
                op_int = _VPU[op_name]
                args = [a.strip() for a in m.group(3).split(",")]
                va = _parse_vreg(args[0])
                vb = _parse_vreg(args[1]) if len(args) > 1 else 0
                out.append(_instr(_SXU["DISPATCH_VPU"],
                                  vregDst=vd, vregSrc=va,
                                  vpuOp=op_int, vregSrc2=vb))

            elif kw == "BROADCAST":
                # BROADCAST vN [, lane=L]
                rest = line[len("BROADCAST"):].strip()
                lane = 0
                if "," in rest:
                    vreg_part, lane_part = rest.split(",", 1)
                    vn = _parse_vreg(vreg_part.strip())
                    lm = re.fullmatch(r"lane=(\d+)", lane_part.strip(),
                                      re.IGNORECASE)
                    if not lm:
                        raise SyntaxError(
                            f"BROADCAST optional arg must be lane=N, "
                            f"got {lane_part.strip()!r}")
                    lane = int(lm.group(1))
                else:
                    vn = _parse_vreg(rest)
                out.append(_instr(_SXU["DISPATCH_XLU_BROADCAST"],
                                  vregDst=vn, vregSrc=vn, vregSrc2=lane))

            elif kw == "SELECT":
                # SELECT vD = SELECT(vCond, vTrue, vFalse)
                rest = line[len("SELECT"):].strip()
                m = re.fullmatch(r"(v\d+)\s*=\s*SELECT\(([^)]+)\)", rest,
                                 re.IGNORECASE)
                if not m:
                    raise SyntaxError(
                        "SELECT syntax: SELECT vD = SELECT(vCond, vTrue, vFalse)")
                vd = _parse_vreg(m.group(1))
                args = [a.strip() for a in m.group(2).split(",")]
                if len(args) != 3:
                    raise SyntaxError(
                        "SELECT syntax: SELECT vD = SELECT(vCond, vTrue, vFalse)")
                vcond = _parse_vreg(args[0])
                vtrue = _parse_vreg(args[1])
                vfalse = _parse_vreg(args[2])
                out.append(_instr(_SXU["DISPATCH_SELECT"],
                                  vregDst=vd, vregSrc=vcond,
                                  vregSrc2=vtrue, mxuWBase=vfalse))

            elif kw == "BROADCAST_SCALAR":
                rest = line[len("BROADCAST_SCALAR"):].strip()
                m = re.fullmatch(r"(v\d+)\s*=\s*(v\d+)\[(\d+),\s*(\d+)\]", rest, re.IGNORECASE)
                if not m:
                    raise SyntaxError(
                        "BROADCAST_SCALAR syntax: BROADCAST_SCALAR vD = vS[row,col]")
                vd = _parse_vreg(m.group(1))
                vs = _parse_vreg(m.group(2))
                row = int(m.group(3))
                col = int(m.group(4))
                sel = (row << 2) | col
                out.append(_instr(_SXU["BROADCAST_SCALAR"],
                                  vregDst=vd, vregSrc=vs, vregSrc2=sel))

            elif kw == "BROADCAST_ROW":
                rest = line[len("BROADCAST_ROW"):].strip()
                m = re.fullmatch(r"(v\d+)\s*=\s*ROW\((v\d+),\s*row=(\d+)\)", rest, re.IGNORECASE)
                if not m:
                    raise SyntaxError(
                        "BROADCAST_ROW syntax: BROADCAST_ROW vD = ROW(vS, row=N)")
                vd = _parse_vreg(m.group(1))
                vs = _parse_vreg(m.group(2))
                row = int(m.group(3))
                out.append(_instr(_SXU["BROADCAST_ROW"],
                                  vregDst=vd, vregSrc=vs, vregSrc2=row))

            elif kw == "BROADCAST_COL":
                rest = line[len("BROADCAST_COL"):].strip()
                m = re.fullmatch(r"(v\d+)\s*=\s*COL\((v\d+),\s*col=(\d+)\)", rest, re.IGNORECASE)
                if not m:
                    raise SyntaxError(
                        "BROADCAST_COL syntax: BROADCAST_COL vD = COL(vS, col=N)")
                vd = _parse_vreg(m.group(1))
                vs = _parse_vreg(m.group(2))
                col = int(m.group(3))
                out.append(_instr(_SXU["BROADCAST_COL"],
                                  vregDst=vd, vregSrc=vs, vregSrc2=col))

            elif kw == "MXU":
                # MXU WMEM[W], AMEM[A], tiles=N
                rest = line[len("MXU"):].strip()
                parts = [p.strip() for p in rest.split(",")]
                if len(parts) not in (3, 5):
                    raise SyntaxError(
                        "MXU syntax: MXU WMEM[W], AMEM[A], tiles=N "
                        "[, psum_write=PSUM[A], psum_row=R | "
                        "psum_acc=PSUM[A], psum_row=R]")
                wbase = _parse_mem("WMEM", parts[0])
                abase = _parse_mem("AMEM", parts[1])
                tm = re.fullmatch(r"tiles=(\d+)", parts[2], re.IGNORECASE)
                if not tm:
                    raise SyntaxError(
                        f"expected tiles=N, got {parts[2]!r}")
                tlen = int(tm.group(1))
                psum_addr = 0
                psum_row = 0
                psum_mode = 0   # PSUM_OFF
                if len(parts) == 5:
                    pm = re.fullmatch(r"psum_(write|acc)=PSUM\[(\d+)\]",
                                      parts[3], re.IGNORECASE)
                    if not pm:
                        raise SyntaxError(
                            "expected psum_write=PSUM[A] or "
                            f"psum_acc=PSUM[A], got {parts[3]!r}")
                    psum_addr = int(pm.group(2))
                    psum_mode = 1 if pm.group(1).lower() == "write" else 2
                    rm = re.fullmatch(r"psum_row=(\d+)", parts[4], re.IGNORECASE)
                    if not rm:
                        raise SyntaxError(
                            f"expected psum_row=R, got {parts[4]!r}")
                    psum_row = int(rm.group(1))
                # psumAddr packs into vregDst, psumRow into vregSrc, psumMode
                # into vregSrc2 — matching ScalarUnit.do_mxu decoding.
                out.append(_instr(_SXU["DISPATCH_MXU"],
                                  vregDst=psum_addr,
                                  vregSrc=psum_row,
                                  vregSrc2=psum_mode,
                                  mxuWBase=wbase, mxuABase=abase,
                                  mxuTLen=tlen))

            elif kw == "MXU_OS":
                # MXU_OS WMEM[W], AMEM[A], tiles=N
                # Routes through Controller.startOS (dfMode=OS); operand
                # fields identical to MXU. PSUM routing not supported here
                # yet (OS + PSUM combination lands in a later iter).
                rest = line[len("MXU_OS"):].strip()
                parts = [p.strip() for p in rest.split(",")]
                if len(parts) != 3:
                    raise SyntaxError(
                        "MXU_OS syntax: MXU_OS WMEM[W], AMEM[A], tiles=N")
                wbase = _parse_mem("WMEM", parts[0])
                abase = _parse_mem("AMEM", parts[1])
                tm = re.fullmatch(r"tiles=(\d+)", parts[2], re.IGNORECASE)
                if not tm:
                    raise SyntaxError(
                        f"expected tiles=N, got {parts[2]!r}")
                tlen = int(tm.group(1))
                out.append(_instr(_SXU["DISPATCH_MXU_OS"],
                                  mxuWBase=wbase, mxuABase=abase,
                                  mxuTLen=tlen))

            elif kw == "WAIT_MXU":
                out.append(_instr(_SXU["WAIT_MXU"]))

            elif kw == "LOAD_MXU_RESULT":
                dst = _parse_vreg(tokens[1])
                out.append(_instr(_SXU["LOAD_MXU_RESULT"], vregDst=dst))

            elif kw == "LOAD_VPU_RESULT":
                dst = _parse_vreg(tokens[1])
                out.append(_instr(_SXU["LOAD_VPU_RESULT"], vregDst=dst))

            elif kw == "LOAD_XLU_RESULT":
                dst = _parse_vreg(tokens[1])
                out.append(_instr(_SXU["LOAD_XLU_RESULT"], vregDst=dst))

            elif kw == "HALT":
                out.append(_instr(_SXU["HALT"]))

            # ── Output directives ──────────────────────────────────────────

            elif kw == "OUTPUT_MXU":
                out.append("3 1")

            elif kw == "OUTPUT_VMEM":
                rest = line[len("OUTPUT_VMEM"):].strip()
                addr = _parse_mem("VMEM", rest)
                out.append(f"6 {addr}")

            elif kw == "END":
                out.append("4")

            else:
                raise SyntaxError(f"unknown mnemonic {tokens[0]!r}")

        except SyntaxError as exc:
            raise SyntaxError(f"line {lineno}: {exc}") from None

    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# Disassembler
# ---------------------------------------------------------------------------

def disassemble(wire: str) -> str:
    """
    Convert numeric wire format to readable TASM source.

    Produces valid TASM that round-trips back through assemble() to the
    same wire bytes (modulo whitespace).
    """
    out: list[str] = []
    in_program = False   # True once we've emitted the first instruction

    for lineno, raw in enumerate(wire.splitlines(), 1):
        line = _strip_comment(raw)
        if not line:
            continue

        tokens = line.split()
        try:
            rec = int(tokens[0])
        except (ValueError, IndexError):
            out.append(f"# UNPARSEABLE: {raw.rstrip()}")
            continue

        rest = tokens[1:]

        try:
            if rec == 0:
                # WMEM tile: addr + 16 int8 values
                addr = int(rest[0])
                vals = rest[1:]
                out.append(f"WMEM[{addr}] = " + " ".join(vals))

            elif rec == 1:
                # AMEM tile: addr + 4 int8 values
                addr = int(rest[0])
                vals = rest[1:]
                out.append(f"AMEM[{addr}] = " + " ".join(vals))

            elif rec == 2:
                # SXU instruction: 9 fields after record type
                if len(rest) < 9:
                    raise ValueError(
                        f"INSTR record needs 9 fields, got {len(rest)}")
                opc      = int(rest[0])
                vmemAddr = int(rest[1])
                vregDst  = int(rest[2])
                vregSrc  = int(rest[3])
                vpuOp    = int(rest[4])
                vregSrc2 = int(rest[5])
                mxuWBase = int(rest[6])
                mxuABase = int(rest[7])
                mxuTLen  = int(rest[8])

                if not in_program:
                    out.append("")   # blank line between data and program
                    in_program = True

                if opc == _SXU["LOAD_VREG"]:
                    out.append(f"LOAD  v{vregDst}, VMEM[{vmemAddr}]")

                elif opc == _SXU["STORE_VREG"]:
                    out.append(f"STORE VMEM[{vmemAddr}], v{vregSrc}")

                elif opc == _SXU["DISPATCH_VPU"]:
                    op_name = _VPU_INV.get(vpuOp, f"VPU_OP_{vpuOp}")
                    if op_name in _VPU_UNARY:
                        out.append(
                            f"VPU   v{vregDst} = {op_name}(v{vregSrc})")
                    else:
                        out.append(
                            f"VPU   v{vregDst} = {op_name}(v{vregSrc}, v{vregSrc2})")

                elif opc == _SXU["DISPATCH_XLU_BROADCAST"]:
                    lane_sfx = f", lane={vregSrc2}" if vregSrc2 != 0 else ""
                    out.append(f"BROADCAST v{vregSrc}{lane_sfx}")

                elif opc == _SXU["DISPATCH_SELECT"]:
                    out.append(
                        f"SELECT v{vregDst} = SELECT(v{vregSrc}, v{vregSrc2}, v{mxuWBase})")

                elif opc == _SXU["BROADCAST_SCALAR"]:
                    row = (vregSrc2 >> 2) & 0x3
                    col = vregSrc2 & 0x3
                    out.append(f"BROADCAST_SCALAR v{vregDst} = v{vregSrc}[{row},{col}]")

                elif opc == _SXU["BROADCAST_ROW"]:
                    out.append(f"BROADCAST_ROW v{vregDst} = ROW(v{vregSrc}, row={vregSrc2})")

                elif opc == _SXU["BROADCAST_COL"]:
                    out.append(f"BROADCAST_COL v{vregDst} = COL(v{vregSrc}, col={vregSrc2})")

                elif opc == _SXU["DISPATCH_MXU"]:
                    psum_mode = vregSrc2 & 0x3
                    if psum_mode == 0:
                        out.append(
                            f"MXU   WMEM[{mxuWBase}], AMEM[{mxuABase}], "
                            f"tiles={mxuTLen}")
                    else:
                        mode_kw = "psum_write" if psum_mode == 1 else "psum_acc"
                        out.append(
                            f"MXU   WMEM[{mxuWBase}], AMEM[{mxuABase}], "
                            f"tiles={mxuTLen}, {mode_kw}=PSUM[{vregDst}], "
                            f"psum_row={vregSrc & 0x3}")

                elif opc == _SXU["DISPATCH_MXU_OS"]:
                    out.append(
                        f"MXU_OS WMEM[{mxuWBase}], AMEM[{mxuABase}], "
                        f"tiles={mxuTLen}")

                elif opc == _SXU["WAIT_MXU"]:
                    out.append("WAIT_MXU")

                elif opc == _SXU["LOAD_MXU_RESULT"]:
                    out.append(f"LOAD_MXU_RESULT v{vregDst}")

                elif opc == _SXU["LOAD_VPU_RESULT"]:
                    out.append(f"LOAD_VPU_RESULT v{vregDst}")

                elif opc == _SXU["LOAD_XLU_RESULT"]:
                    out.append(f"LOAD_XLU_RESULT v{vregDst}")

                elif opc == _SXU["HALT"]:
                    out.append("HALT")

                else:
                    out.append(f"# UNKNOWN SXU OPCODE {opc}")

            elif rec == 3:
                # OUTPUT_MXU flag
                out.append("")
                out.append("OUTPUT_MXU")

            elif rec == 4:
                # END
                out.append("END")

            elif rec == 5:
                # VMEM tile: addr + 16 int32 values
                addr = int(rest[0])
                vals = rest[1:]
                out.append(f"VMEM[{addr}] = " + " ".join(vals))

            elif rec == 6:
                # OUTPUT_VMEM addr
                addr = int(rest[0])
                out.append("")
                out.append(f"OUTPUT_VMEM VMEM[{addr}]")

            else:
                out.append(f"# UNKNOWN RECORD TYPE {rec}: {raw.rstrip()}")

        except (ValueError, IndexError) as exc:
            out.append(f"# ERROR line {lineno}: {exc}: {raw.rstrip()}")

    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _usage() -> None:
    print(__doc__)
    sys.exit(1)


def main() -> None:
    if len(sys.argv) < 3:
        _usage()

    cmd = sys.argv[1].lower()
    path = sys.argv[2]

    if path == "-":
        text = sys.stdin.read()
    else:
        with open(path) as f:
            text = f.read()

    if cmd == "assemble":
        sys.stdout.write(assemble(text))
    elif cmd == "disassemble":
        sys.stdout.write(disassemble(text))
    else:
        print(f"Unknown command {cmd!r}. Use 'assemble' or 'disassemble'.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
