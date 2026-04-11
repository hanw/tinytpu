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


def test_mxu():
    wire = wire_lines(assemble("MXU WMEM[0], AMEM[1], tiles=2\nHALT\nEND\n"))
    # DISPATCH_MXU opc=4, mxuWBase=0, mxuABase=1, mxuTLen=2
    assert wire[0] == "2 4 0 0 0 0 0 0 1 2"


def test_wait_mxu():
    wire = wire_lines(assemble("WAIT_MXU\nHALT\nEND\n"))
    assert wire[0] == "2 5 0 0 0 0 0 0 0 0"


def test_halt():
    wire = wire_lines(assemble("HALT\nEND\n"))
    assert wire[0] == "2 6 0 0 0 0 0 0 0 0"


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


def test_disassemble_mxu():
    tasm = disassemble("2 4 0 0 0 0 0 0 1 2\n4\n")
    assert "MXU   WMEM[0], AMEM[1], tiles=2" in tasm


def test_disassemble_vmem_tile():
    tasm = disassemble("5 0 -1 2 -3 4 0 0 0 0 0 0 0 0 0 0 0 0\n4\n")
    assert "VMEM[0] = -1 2 -3 4" in tasm


def test_disassemble_output_vmem():
    tasm = disassemble("2 6 0 0 0 0 0 0 0 0\n6 3\n4\n")
    assert "OUTPUT_VMEM VMEM[3]" in tasm


def test_disassemble_output_mxu():
    tasm = disassemble("2 6 0 0 0 0 0 0 0 0\n3 1\n4\n")
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
    assert _halt()       == "2 6 0 0 0 0 0 0 0 0"
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
    from tinygrad.runtime.ops_tinytpu import _build_gemm_bundle
    w = np.eye(4, dtype=np.int8)
    a = np.array([1, 2, 3, 4], dtype=np.int8)
    wire = _build_gemm_bundle(w, a)
    assert assemble(disassemble(wire)) == wire


def test_where_bundle_roundtrips_through_tasm():
    import numpy as np
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
    from tinygrad.runtime.ops_tinytpu import _build_vpu_where_bundle
    cond = np.array([1, 0, 1, 0] + [0] * 12, dtype=np.int32)
    lhs  = np.arange(16, dtype=np.int32)
    rhs  = -np.arange(16, dtype=np.int32)
    wire = _build_vpu_where_bundle(cond, lhs, rhs, 4)
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
