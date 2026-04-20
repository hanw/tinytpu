"""Unit tests for scripts/tasm.py — assembler and disassembler."""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.tasm import assemble, disassemble


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wire_lines(text: str) -> list[str]:
    """Return non-empty lines from wire output."""
    return [l for l in text.splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# Data declarations
# ---------------------------------------------------------------------------

def test_vmem_tile():
    vals = list(range(16))
    src = "VMEM[0] = " + " ".join(str(v) for v in vals) + "\nEND\n"
    wire = wire_lines(assemble(src))
    assert wire[0] == "5 0 " + " ".join(str(v) for v in vals)
    assert wire[1] == "4"


def test_wmem_tile():
    vals = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    src = "WMEM[2] = " + " ".join(str(v) for v in vals) + "\nEND\n"
    wire = wire_lines(assemble(src))
    assert wire[0] == "0 2 " + " ".join(str(v) for v in vals)


def test_amem_tile():
    src = "AMEM[1] = 3 7 -2 5\nEND\n"
    wire = wire_lines(assemble(src))
    assert wire[0] == "1 1 3 7 -2 5"


def test_wrong_vmem_count():
    with pytest.raises(SyntaxError, match="16 values"):
        assemble("VMEM[0] = 1 2 3\nEND\n")


def test_wrong_amem_count():
    with pytest.raises(SyntaxError, match="4 values"):
        assemble("AMEM[0] = 1 2\nEND\n")


# ---------------------------------------------------------------------------
# Instructions
# ---------------------------------------------------------------------------

def test_load():
    wire = wire_lines(assemble("LOAD v3, VMEM[5]\nHALT\nEND\n"))
    # LOAD: rec=2, opc=0(LOAD_VREG), vmemAddr=5, vregDst=3, rest 0
    assert wire[0] == "2 0 5 3 0 0 0 0 0 0"


def test_store():
    wire = wire_lines(assemble("STORE VMEM[4], v7\nHALT\nEND\n"))
    # STORE: rec=2, opc=1(STORE_VREG), vmemAddr=4, vregDst=0, vregSrc=7
    assert wire[0] == "2 1 4 0 7 0 0 0 0 0"


def test_vpu_binary():
    wire = wire_lines(assemble("VPU v2 = ADD(v0, v1)\nHALT\nEND\n"))
    # DISPATCH_VPU opc=2, vmemAddr=0, vregDst=2, vregSrc=0, vpuOp=0(ADD), vregSrc2=1
    assert wire[0] == "2 2 0 2 0 0 1 0 0 0"


def test_vpu_sub():
    wire = wire_lines(assemble("VPU v3 = SUB(v1, v0)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 3 1 7 0 0 0 0"


def test_vpu_unary_relu():
    wire = wire_lines(assemble("VPU v1 = RELU(v0)\nHALT\nEND\n"))
    # unary: vregSrc2=0
    assert wire[0] == "2 2 0 1 0 2 0 0 0 0"


def test_vpu_max_reduce():
    wire = wire_lines(assemble("VPU v2 = MAX_REDUCE(v0)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 2 0 9 0 0 0 0"


def test_vpu_sum_reduce_col():
    wire = wire_lines(assemble("VPU v2 = SUM_REDUCE_COL(v0)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 2 0 29 0 0 0 0"


def test_vpu_max_reduce_col():
    wire = wire_lines(assemble("VPU v3 = MAX_REDUCE_COL(v1)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 3 1 30 0 0 0 0"


def test_vpu_min_reduce_col():
    wire = wire_lines(assemble("VPU v3 = MIN_REDUCE_COL(v1)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 3 1 31 0 0 0 0"


def test_vpu_sum_reduce_tile():
    wire = wire_lines(assemble("VPU v1 = SUM_REDUCE_TILE(v0)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 1 0 32 0 0 0 0"


def test_vpu_max_reduce_tile():
    wire = wire_lines(assemble("VPU v1 = MAX_REDUCE_TILE(v0)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 1 0 33 0 0 0 0"


def test_vpu_min_reduce_tile():
    wire = wire_lines(assemble("VPU v1 = MIN_REDUCE_TILE(v0)\nHALT\nEND\n"))
    assert wire[0] == "2 2 0 1 0 34 0 0 0 0"


def test_vpu_unknown_op():
    with pytest.raises(SyntaxError, match="unknown VPU op"):
        assemble("VPU v0 = BADOP(v1)\nHALT\nEND\n")


def test_broadcast_default_lane():
    wire = wire_lines(assemble("BROADCAST v2\nHALT\nEND\n"))
    # DISPATCH_XLU_BROADCAST opc=3, vregDst=2, vregSrc=2, vregSrc2=0 (lane)
    assert wire[0] == "2 3 0 2 2 0 0 0 0 0"


def test_broadcast_explicit_lane():
    wire = wire_lines(assemble("BROADCAST v1, lane=3\nHALT\nEND\n"))
    assert wire[0] == "2 3 0 1 1 0 3 0 0 0"


def test_select():
    wire = wire_lines(assemble("SELECT v4 = SELECT(v0, v1, v2)\nHALT\nEND\n"))
    # DISPATCH_SELECT opc=8, vregDst=4, vregSrc=0(cond), vregSrc2=1(lhs), mxuWBase=2(rhs)
    assert wire[0] == "2 8 0 4 0 0 1 2 0 0"


def test_broadcast_scalar():
    wire = wire_lines(assemble("BROADCAST_SCALAR v4 = v1[2,3]\nHALT\nEND\n"))
    assert wire[0] == "2 9 0 4 1 0 11 0 0 0"


def test_broadcast_row():
    wire = wire_lines(assemble("BROADCAST_ROW v3 = ROW(v1, row=2)\nHALT\nEND\n"))
    assert wire[0] == "2 10 0 3 1 0 2 0 0 0"


def test_broadcast_col():
    wire = wire_lines(assemble("BROADCAST_COL v5 = COL(v2, col=1)\nHALT\nEND\n"))
    assert wire[0] == "2 11 0 5 2 0 1 0 0 0"


def test_mxu():
    wire = wire_lines(assemble("MXU WMEM[0], AMEM[1], tiles=2\nHALT\nEND\n"))
    # DISPATCH_MXU opc=4, mxuWBase=0, mxuABase=1, mxuTLen=2
    assert wire[0] == "2 4 0 0 0 0 0 0 1 2"


def test_mxu_psum_write():
    wire = wire_lines(assemble(
        "MXU WMEM[0], AMEM[1], tiles=2, psum_write=PSUM[3], psum_row=2\n"
        "HALT\nEND\n"))
    # vregDst=3 (psum addr), vregSrc=2 (psum row), vregSrc2=1 (WRITE mode)
    assert wire[0] == "2 4 0 3 2 0 1 0 1 2"


def test_mxu_psum_accumulate():
    wire = wire_lines(assemble(
        "MXU WMEM[5], AMEM[7], tiles=1, psum_acc=PSUM[1], psum_row=0\n"
        "HALT\nEND\n"))
    # vregDst=1 (addr), vregSrc=0 (row), vregSrc2=2 (ACCUMULATE mode)
    assert wire[0] == "2 4 0 1 0 0 2 5 7 1"


def test_mxu_psum_write_roundtrip():
    src = ("MXU WMEM[0], AMEM[1], tiles=2, psum_write=PSUM[3], psum_row=2\n"
           "HALT\nEND\n")
    text = disassemble(assemble(src))
    assert "psum_write=PSUM[3]" in text
    assert "psum_row=2" in text


def test_mxu_psum_acc_roundtrip():
    src = ("MXU WMEM[5], AMEM[7], tiles=1, psum_acc=PSUM[1], psum_row=0\n"
           "HALT\nEND\n")
    text = disassemble(assemble(src))
    assert "psum_acc=PSUM[1]" in text
    assert "psum_row=0" in text


def test_mxu_default_round_trip():
    # A plain MXU dispatch without psum fields should not emit psum syntax.
    src = "MXU WMEM[0], AMEM[1], tiles=2\nHALT\nEND\n"
    text = disassemble(assemble(src))
    assert "psum" not in text


def test_wait_mxu():
    wire = wire_lines(assemble("WAIT_MXU\nHALT\nEND\n"))
    assert wire[0] == "2 5 0 0 0 0 0 0 0 0"


def test_mxu_os():
    wire = wire_lines(assemble("MXU_OS WMEM[2], AMEM[4], tiles=3\nHALT\nEND\n"))
    # DISPATCH_MXU_OS opc=23, mxuWBase=2, mxuABase=4, mxuTLen=3.
    # Other operand fields zero — OS dispatch has no psum plumbing.
    assert wire[0] == "2 23 0 0 0 0 0 2 4 3"


def test_mxu_os_roundtrip():
    src = "MXU_OS WMEM[2], AMEM[4], tiles=3\nHALT\nEND\n"
    text = disassemble(assemble(src))
    assert "MXU_OS WMEM[2], AMEM[4], tiles=3" in text


def test_mxu_clear():
    wire = wire_lines(assemble("MXU_CLEAR\nHALT\nEND\n"))
    # MXU_CLEAR opc=24, no operand fields.
    assert wire[0] == "2 24 0 0 0 0 0 0 0 0"


def test_mxu_clear_roundtrip():
    src = "MXU_CLEAR\nHALT\nEND\n"
    text = disassemble(assemble(src))
    assert "MXU_CLEAR" in text


def test_halt():
    wire = wire_lines(assemble("HALT\nEND\n"))
    assert wire[0] == "2 7 0 0 0 0 0 0 0 0"


def test_load_vpu_result_assembles():
    wire = wire_lines(assemble("LOAD_VPU_RESULT v3\nHALT\nEND\n"))
    assert wire[0] == "2 13 0 3 0 0 0 0 0 0"


def test_load_xlu_result_assembles():
    wire = wire_lines(assemble("LOAD_XLU_RESULT v5\nHALT\nEND\n"))
    assert wire[0] == "2 14 0 5 0 0 0 0 0 0"


def test_load_vpu_result_roundtrip():
    from scripts.tasm import disassemble
    wire = assemble("LOAD_VPU_RESULT v7\nHALT\nEND\n")
    text = disassemble(wire)
    assert "LOAD_VPU_RESULT v7" in text


def test_load_xlu_result_roundtrip():
    from scripts.tasm import disassemble
    wire = assemble("LOAD_XLU_RESULT v1\nHALT\nEND\n")
    text = disassemble(wire)
    assert "LOAD_XLU_RESULT v1" in text


# ---------------------------------------------------------------------------
# Output directives
# ---------------------------------------------------------------------------

def test_output_mxu():
    wire = wire_lines(assemble("HALT\nOUTPUT_MXU\nEND\n"))
    assert "3 1" in wire


def test_output_vmem():
    wire = wire_lines(assemble("HALT\nOUTPUT_VMEM VMEM[3]\nEND\n"))
    assert "6 3" in wire


def test_end():
    wire = wire_lines(assemble("HALT\nEND\n"))
    assert wire[-1] == "4"


# ---------------------------------------------------------------------------
# Case insensitivity and comments
# ---------------------------------------------------------------------------

def test_case_insensitive():
    a = assemble("load v0, vmem[0]\nhalt\nend\n")
    b = assemble("LOAD v0, VMEM[0]\nHALT\nEND\n")
    assert a == b


def test_comments_ignored():
    src = """
# full-line comment
VMEM[0] = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16  # inline comment
HALT   # another comment
END
"""
    wire = wire_lines(assemble(src))
    assert wire[0].startswith("5 0 ")
    assert wire[-1] == "4"


def test_blank_lines_ignored():
    a = assemble("HALT\nEND\n")
    b = assemble("\n\nHALT\n\n\nEND\n\n")
    assert a == b


# ---------------------------------------------------------------------------
# Disassembler
# ---------------------------------------------------------------------------

def test_disassemble_load():
    tasm = disassemble("2 0 5 3 0 0 0 0 0 0\n4\n")
    assert "LOAD  v3, VMEM[5]" in tasm


def test_disassemble_store():
    tasm = disassemble("2 1 4 0 7 0 0 0 0 0\n4\n")
    assert "STORE VMEM[4], v7" in tasm


def test_disassemble_vpu_binary():
    tasm = disassemble("2 2 0 2 0 0 1 0 0 0\n4\n")
    assert "VPU   v2 = ADD(v0, v1)" in tasm


def test_disassemble_vpu_unary():
    tasm = disassemble("2 2 0 1 0 2 0 0 0 0\n4\n")
    assert "VPU   v1 = RELU(v0)" in tasm


def test_disassemble_broadcast_no_lane():
    tasm = disassemble("2 3 0 2 2 0 0 0 0 0\n4\n")
    assert "BROADCAST v2" in tasm
    assert "lane" not in tasm


def test_disassemble_broadcast_with_lane():
    tasm = disassemble("2 3 0 1 1 0 3 0 0 0\n4\n")
    assert "BROADCAST v1, lane=3" in tasm


def test_disassemble_select():
    tasm = disassemble("2 8 0 4 0 0 1 2 0 0\n4\n")
    assert "SELECT v4 = SELECT(v0, v1, v2)" in tasm


def test_disassemble_broadcast_scalar():
    tasm = disassemble("2 9 0 4 1 0 11 0 0 0\n4\n")
    assert "BROADCAST_SCALAR v4 = v1[2,3]" in tasm


def test_disassemble_broadcast_row():
    tasm = disassemble("2 10 0 3 1 0 2 0 0 0\n4\n")
    assert "BROADCAST_ROW v3 = ROW(v1, row=2)" in tasm


def test_disassemble_broadcast_col():
    tasm = disassemble("2 11 0 5 2 0 1 0 0 0\n4\n")
    assert "BROADCAST_COL v5 = COL(v2, col=1)" in tasm


def test_disassemble_mxu():
    tasm = disassemble("2 4 0 0 0 0 0 0 1 2\n4\n")
    assert "MXU   WMEM[0], AMEM[1], tiles=2" in tasm


def test_disassemble_vmem_tile():
    tasm = disassemble("5 0 -1 2 -3 4 0 0 0 0 0 0 0 0 0 0 0 0\n4\n")
    assert "VMEM[0] = -1 2 -3 4" in tasm


def test_disassemble_output_vmem():
    tasm = disassemble("2 7 0 0 0 0 0 0 0 0\n6 3\n4\n")
    assert "OUTPUT_VMEM VMEM[3]" in tasm


def test_disassemble_output_mxu():
    tasm = disassemble("2 7 0 0 0 0 0 0 0 0\n3 1\n4\n")
    assert "OUTPUT_MXU" in tasm


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------

def test_roundtrip_add():
    src = (
        "VMEM[0] = 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16\n"
        "VMEM[1] = 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n"
        "LOAD  v0, VMEM[0]\n"
        "LOAD  v1, VMEM[1]\n"
        "VPU   v2 = ADD(v0, v1)\n"
        "STORE VMEM[2], v2\n"
        "HALT\n"
        "OUTPUT_VMEM VMEM[2]\n"
        "END\n"
    )
    wire = assemble(src)
    assert assemble(disassemble(wire)) == wire


def test_roundtrip_gemm():
    src = (
        "WMEM[0] = 1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1\n"
        "AMEM[1] = 3 7 -2 5\n"
        "MXU   WMEM[0], AMEM[1], tiles=1\n"
        "WAIT_MXU\n"
        "HALT\n"
        "OUTPUT_MXU\n"
        "END\n"
    )
    wire = assemble(src)
    assert assemble(disassemble(wire)) == wire


def test_roundtrip_abs():
    src = (
        "VMEM[0] = -1 2 -3 4 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "VMEM[1] = 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n"
        "LOAD  v0, VMEM[0]\n"
        "LOAD  v1, VMEM[1]\n"
        "VPU   v2 = SUB(v1, v0)\n"
        "VPU   v3 = MAX(v0, v2)\n"
        "STORE VMEM[2], v3\n"
        "HALT\n"
        "OUTPUT_VMEM VMEM[2]\n"
        "END\n"
    )
    wire = assemble(src)
    assert assemble(disassemble(wire)) == wire


# ---------------------------------------------------------------------------
# _tasm helpers integration tests (ops_tinytpu internal helpers)
# ---------------------------------------------------------------------------

def test_tasm_helpers_import():
    """Verify _tasm helpers exist and are importable."""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import (
        _load, _store, _vpu, _halt, _output_vmem, _end, _bundle,
    )
    assert _load(3, 5)   == "2 0 5 3 0 0 0 0 0 0"
    assert _store(4, 7)  == "2 1 4 0 7 0 0 0 0 0"
    assert _halt()       == "2 7 0 0 0 0 0 0 0 0"
    assert _output_vmem(2) == "6 2"
    assert _end()        == "4"


def test_tasm_helpers_vpu_binary():
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _vpu, _VPU_OPS
    # ADD: vd=2, va=0, op=ADD(0), vb=1
    assert _vpu(2, 0, _VPU_OPS["ADD"], 1) == "2 2 0 2 0 0 1 0 0 0"
    # RELU (unary, vb=0): vd=1, va=0, op=RELU(2)
    assert _vpu(1, 0, _VPU_OPS.get("RELU", 2)) == "2 2 0 1 0 2 0 0 0 0"
    # SUB: vd=3, va=1, op=SUB(7), vb=0
    assert _vpu(3, 1, _VPU_OPS["SUB"], 0) == "2 2 0 3 1 7 0 0 0 0"


def test_binary_bundle_roundtrips_through_tasm():
    """_build_vpu_binary_bundle output must survive assemble(disassemble(wire))."""
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _build_vpu_binary_bundle, _VPU_OPS
    lhs = np.array([1, 2, 3, 4] + [0] * 12, dtype=np.int32)
    rhs = np.array([10, 20, 30, 40] + [0] * 12, dtype=np.int32)
    wire = _build_vpu_binary_bundle(lhs, rhs, 4, _VPU_OPS["ADD"])
    assert assemble(disassemble(wire)) == wire


def test_unary_bundle_roundtrips_through_tasm():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _build_vpu_unary_bundle
    src = np.array([-1, 2, -3, 4] + [0] * 12, dtype=np.int32)
    wire = _build_vpu_unary_bundle(src, 4, 2)  # RELU
    assert assemble(disassemble(wire)) == wire


def test_gemm_bundle_roundtrips_through_tasm():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _build_full_gemm_bundle
    w = np.eye(4, dtype=np.int8)
    a = np.array([[1, 2, 3, 4]], dtype=np.int8)
    wire = _build_full_gemm_bundle(a, w, num_vecs=1, num_k_tiles=1, num_weight_tiles=1)
    assert assemble(disassemble(wire)) == wire


def test_program_bundle_roundtrips_through_tasm():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _build_vpu_program_bundle, _VPU_OPS
    # abs = SUB(zeros, input) then MAX(input, neg_input)
    inp = np.array([-1, 2, -3, 4] + [0] * 12, dtype=np.int32)
    zeros = np.zeros(16, dtype=np.int32)
    steps = [
        {"op": _VPU_OPS["SUB"], "lhs": 1, "rhs": 0, "dst": 2},
        {"op": _VPU_OPS["MAX"], "lhs": 0, "rhs": 2, "dst": 3},
    ]
    wire = _build_vpu_program_bundle([inp, zeros], 4, steps, 3)
    assert assemble(disassemble(wire)) == wire


def test_disassemble_program_bundle_readable():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _build_vpu_program_bundle, _VPU_OPS
    inp = np.array([-1, 2, -3, 4] + [0] * 12, dtype=np.int32)
    zeros = np.zeros(16, dtype=np.int32)
    steps = [
        {"op": _VPU_OPS["SUB"], "lhs": 1, "rhs": 0, "dst": 2},
        {"op": _VPU_OPS["MAX"], "lhs": 0, "rhs": 2, "dst": 3},
    ]
    wire = _build_vpu_program_bundle([inp, zeros], 4, steps, 3)
    tasm = disassemble(wire)
    assert "VPU   v2 = SUB(v1, v0)" in tasm
    assert "VPU   v3 = MAX(v0, v2)" in tasm
    assert "STORE VMEM[2], v3" in tasm


def test_vpu_opcode_table_matches_hardware():
    """VPU opcodes in tasm.py must match BSV VpuOp enum order."""
    from scripts.tasm import _VPU as TASM_VPU
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _VPU_OPS as BACKEND_VPU
    # Check the ops that both tables share
    shared = {"ADD": 0, "MUL": 1, "MAX": 3, "SUB": 7, "CMPEQ": 8, "MAX_REDUCE": 9,
              "MIN": 12, "MIN_REDUCE": 13, "DIV": 14}
    for op, code in shared.items():
        assert TASM_VPU[op] == code, f"TASM {op}={TASM_VPU[op]} != {code}"
        assert BACKEND_VPU[op] == code, f"BACKEND {op}={BACKEND_VPU[op]} != {code}"


def test_broadcast_instruction_in_bundle():
    """BROADCAST instruction in a VPU binary bundle (lhs_broadcast=True)."""
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _build_vpu_binary_bundle, _VPU_OPS
    lhs = np.array([7] + [0] * 15, dtype=np.int32)
    rhs = np.arange(16, dtype=np.int32)
    wire = _build_vpu_binary_bundle(lhs, rhs, 4, _VPU_OPS["ADD"], lhs_broadcast=True)
    tasm = disassemble(wire)
    assert "BROADCAST v0" in tasm
    # Round-trip
    assert assemble(tasm) == wire


def test_mxu_instruction():
    wire = wire_lines(assemble("MXU WMEM[0], AMEM[1], tiles=3\nHALT\nEND\n"))
    assert wire[0] == "2 4 0 0 0 0 0 0 1 3"


def test_wait_mxu_instruction():
    wire = wire_lines(assemble("WAIT_MXU\nHALT\nEND\n"))
    assert wire[0] == "2 5 0 0 0 0 0 0 0 0"


def test_output_mxu_directive():
    wire = wire_lines(assemble("HALT\nOUTPUT_MXU\nEND\n"))
    assert "3 1" in wire
    assert "4" in wire


def test_wmem_and_amem_declarations():
    src = "WMEM[2] = 1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1\nAMEM[3] = 5 -2 7 0\nEND\n"
    wire = wire_lines(assemble(src))
    assert wire[0].startswith("0 2 ")
    assert wire[1].startswith("1 3 5 -2 7 0")


def test_disassemble_vmem_negative():
    tasm = disassemble("5 0 -1 -2 -3 -4 0 0 0 0 0 0 0 0 0 0 0 0\n4\n")
    assert "VMEM[0] = -1 -2 -3 -4" in tasm


def test_vpu_ops_cover_full_range():
    from scripts.tasm import _VPU
    # Full VPU opcode range including EXP2/LOG2/SIN/COS transcendentals.
    assert len(_VPU) == 55
    codes = sorted(_VPU.values())
    assert codes == list(range(55))


def test_vpu_exp2_roundtrip():
    prog = "LOAD  v0, VMEM[0]\nVPU   v1 = EXP2(v0)\nSTORE VMEM[1], v1\nHALT\nEND\n"
    wire = assemble(prog)
    # Check wire matches expected: opcode 2 (DISPATCH_VPU) with vpuOp=51.
    lines = wire.strip().splitlines()
    vpu_line = next(ln for ln in lines if ln.startswith("2 2 "))
    # fields: record op vmemAddr vregDst vregSrc vpuOp vregSrc2 ...
    fields = vpu_line.split()
    assert fields[5] == "51", f"expected EXP2 opcode 51, got {fields[5]}"
    back = disassemble(wire)
    assert "EXP2(v0)" in back


def test_assemble_error_bad_vreg():
    with pytest.raises(SyntaxError, match="expected vector register"):
        assemble("LOAD x0, VMEM[0]\nHALT\nEND\n")


def test_assemble_error_bad_vmem():
    with pytest.raises(SyntaxError, match="expected VMEM"):
        assemble("LOAD v0, WMEM[0]\nHALT\nEND\n")


def test_assemble_error_unknown_mnemonic():
    with pytest.raises(SyntaxError, match="unknown mnemonic"):
        assemble("JUMP v0\nHALT\nEND\n")
