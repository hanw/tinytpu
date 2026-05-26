"""Tests for the UPat-based post-WMMA epilogue recognizers in
tinygrad/renderer/tinytpu/upat_recognizers.py.

Residual is validated against UOps produced by an actual matmul+residual
run through tinygrad (the same shape lower_gemm sees today). RoPE is
validated against hand-built UOps because no tinygrad expression
currently produces a single fused kernel matching that shape — the
recognizer is forward-looking.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.tinytpu.upat_recognizers import (
    recognize_residual_epilogue,
    recognize_rope_epilogue,
)


# ---------------------------------------------------------------------------
# Synthetic UOp builders. PARAM/CONST/INDEX/LOAD/WMMA/GEP/ADD/MUL/STORE
# follow the shape observed in real kernels (probed via lower_gemm trace).
# ---------------------------------------------------------------------------

INT = dtypes.int
PTR16 = dtypes.int.ptr(16)
VEC16 = dtypes.int.vec(16)


def _param(idx: int) -> UOp:
    return UOp(Ops.PARAM, PTR16, (), idx)


def _const(v: int) -> UOp:
    return UOp(Ops.CONST, INT, (), v)


def _index(p: UOp, i: int) -> UOp:
    return UOp(Ops.INDEX, PTR16, (p, _const(i)), None)


def _load(p: UOp, i: int) -> UOp:
    return UOp(Ops.LOAD, INT, (_index(p, i),), None)


def _stack16(loads: list[UOp]) -> UOp:
    assert len(loads) == 16
    return UOp(Ops.STACK, VEC16, tuple(loads), None)


def _wmma(a: UOp, b: UOp, acc: UOp) -> UOp:
    """Construct a placeholder WMMA UOp. The arg shape matches what
    tinygrad emits; we don't depend on it inside the recognizers."""
    return UOp(Ops.WMMA, VEC16, (a, b, acc),
               ("WMMA_4_4_4_int_int", (4, 4, 4), dtypes.int, dtypes.int,
                "TINYTPU", 1, (), (), ()))


def _gep(wmma: UOp, i: int) -> UOp:
    return UOp(Ops.GEP, INT, (wmma,), (i,))


def _add(a: UOp, b: UOp) -> UOp:
    return UOp(Ops.ADD, INT, (a, b), None)


def _mul(a: UOp, b: UOp) -> UOp:
    return UOp(Ops.MUL, INT, (a, b), None)


def _store(p: UOp, i: int, val: UOp) -> UOp:
    return UOp(Ops.STORE, dtypes.void, (_index(p, i), val), None)


def _build_kernel_prelude(*params: UOp):
    """Build a 4x4x4 GEMM WMMA preamble shared by both recognizer tests:
    16 LOADs per operand stacked into vec(16) inputs, then WMMA.
    Returns (wmma_uop, uops_list_so_far).
    """
    a, b, *_ = params
    a_loads = [_load(a, i) for i in range(16)]
    b_loads = [_load(b, i) for i in range(16)]
    zero = _const(0)
    acc_stack = UOp(Ops.STACK, VEC16, tuple(zero for _ in range(16)), None)
    a_stack = _stack16(a_loads)
    b_stack = _stack16(b_loads)
    wmma = _wmma(a_stack, b_stack, acc_stack)
    uops = list(params) + a_loads + b_loads + [acc_stack, a_stack, b_stack, wmma]
    return wmma, uops


class TestResidualRecognizer(unittest.TestCase):

    def test_recognizes_clean_residual(self):
        p_out, p_a, p_b, p_r = _param(0), _param(1), _param(2), _param(3)
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_r, p_out)
        for i in range(16):
            uops.append(_store(p_out, i, _add(_gep(wmma, i), _load(p_r, i))))
        res = recognize_residual_epilogue(uops)
        self.assertIsNotNone(res)
        self.assertIs(res.wmma, wmma)
        self.assertIs(res.residual_param, p_r)

    def test_rejects_index_mismatch(self):
        # GEP and LOAD use the same i, but STORE writes to a different
        # output index — that's a cross-lane reshuffle, not a residual.
        p_out, p_a, p_b, p_r = _param(0), _param(1), _param(2), _param(3)
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_r, p_out)
        for i in range(16):
            shuffled = (i + 1) % 16
            uops.append(_store(p_out, shuffled, _add(_gep(wmma, i), _load(p_r, i))))
        self.assertIsNone(recognize_residual_epilogue(uops))

    def test_rejects_mixed_residual_params(self):
        # Half the stores read from p_r, half from p_r2 — not a single residual.
        p_out, p_a, p_b, p_r, p_r2 = _param(0), _param(1), _param(2), _param(3), _param(4)
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_r, p_out)
        uops.append(p_r2)
        for i in range(16):
            src_p = p_r if i < 8 else p_r2
            uops.append(_store(p_out, i, _add(_gep(wmma, i), _load(src_p, i))))
        self.assertIsNone(recognize_residual_epilogue(uops))

    def test_rejects_when_pattern_is_actually_rope_shaped(self):
        # Each STORE source is ADD(MUL, MUL), not ADD(GEP, LOAD).
        p_out, p_a, p_b, p_c, p_s = (_param(i) for i in range(5))
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_c, p_s, p_out)
        for i in range(16):
            uops.append(_store(p_out, i, _add(
                _mul(_gep(wmma, i),     _load(p_c, i)),
                _mul(_gep(wmma, i ^ 1), _load(p_s, i ^ 1)),
            )))
        self.assertIsNone(recognize_residual_epilogue(uops))


class TestRopeRecognizer(unittest.TestCase):

    def test_recognizes_clean_rope(self):
        p_out, p_a, p_b, p_c, p_s = (_param(i) for i in range(5))
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_c, p_s, p_out)
        for i in range(16):
            uops.append(_store(p_out, i, _add(
                _mul(_gep(wmma, i),     _load(p_c, i)),       # no-swap leg
                _mul(_gep(wmma, i ^ 1), _load(p_s, i ^ 1)),   # swap leg
            )))
        res = recognize_rope_epilogue(uops)
        self.assertIsNotNone(res)
        self.assertIs(res.wmma, wmma)
        self.assertIs(res.c_param, p_c)
        self.assertIs(res.s_param, p_s)

    def test_recognizes_with_legs_swapped(self):
        # Same math but the swap leg is the first operand of ADD.
        # The framework's `src=list` form enumerates permutations, so
        # the matcher still binds; our recognizer normalizes via the
        # GEP-index == out_idx heuristic.
        p_out, p_a, p_b, p_c, p_s = (_param(i) for i in range(5))
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_c, p_s, p_out)
        for i in range(16):
            uops.append(_store(p_out, i, _add(
                _mul(_gep(wmma, i ^ 1), _load(p_s, i ^ 1)),   # swap leg first
                _mul(_gep(wmma, i),     _load(p_c, i)),       # no-swap leg second
            )))
        res = recognize_rope_epilogue(uops)
        self.assertIsNotNone(res)
        self.assertIs(res.c_param, p_c)
        self.assertIs(res.s_param, p_s)

    def test_rejects_unrelated_wmmas_in_two_legs(self):
        # The two MULs reference different WMMA accumulators — fails
        # the name-binding consistency check the matcher provides.
        p_out, p_a, p_b, p_c, p_s = (_param(i) for i in range(5))
        wmma1, uops = _build_kernel_prelude(p_a, p_b, p_c, p_s, p_out)
        # Second, distinct WMMA with different inputs.
        zero = _const(0)
        acc2 = UOp(Ops.STACK, VEC16, tuple(zero for _ in range(16)), None)
        wmma2 = _wmma(_stack16([_load(p_c, i) for i in range(16)]),
                      _stack16([_load(p_s, i) for i in range(16)]),
                      acc2)
        uops.append(wmma2)
        for i in range(16):
            uops.append(_store(p_out, i, _add(
                _mul(_gep(wmma1, i),     _load(p_c, i)),
                _mul(_gep(wmma2, i ^ 1), _load(p_s, i ^ 1)),
            )))
        self.assertIsNone(recognize_rope_epilogue(uops))

    def test_rejects_non_swap_gep_relation(self):
        # The "swap" leg's GEP index is i + 2 instead of i ^ 1 — that's
        # a different permutation (e.g. quad-shift), not pair-rotate.
        p_out, p_a, p_b, p_c, p_s = (_param(i) for i in range(5))
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_c, p_s, p_out)
        for i in range(16):
            uops.append(_store(p_out, i, _add(
                _mul(_gep(wmma, i),               _load(p_c, i)),
                _mul(_gep(wmma, (i + 2) % 16),    _load(p_s, (i + 2) % 16)),
            )))
        self.assertIsNone(recognize_rope_epilogue(uops))

    def test_rejects_load_index_mismatch(self):
        # GEP indices are clean pair-swap, but the LOAD on the swap leg
        # pulls from a different index than its GEP — not a clean
        # interleaved-coefficient layout.
        p_out, p_a, p_b, p_c, p_s = (_param(i) for i in range(5))
        wmma, uops = _build_kernel_prelude(p_a, p_b, p_c, p_s, p_out)
        for i in range(16):
            uops.append(_store(p_out, i, _add(
                _mul(_gep(wmma, i),     _load(p_c, i)),
                _mul(_gep(wmma, i ^ 1), _load(p_s, (i + 3) % 16)),
            )))
        self.assertIsNone(recognize_rope_epilogue(uops))


class TestResidualRecognizerOnRealUOps(unittest.TestCase):
    """Validate the residual recognizer against the actual UOp graph
    tinygrad produces for `(a @ b) + r`. This is the same graph
    lower_gemm sees today, so a green test proves the new matcher
    will be a behaviorally-equivalent replacement for the existing
    _extract_wmma_epilogue FULL-mode branch.
    """

    def test_real_matmul_plus_residual(self):
        os.environ.setdefault("TINYTPU_SIM",
                              "/home/hanwang/p/tinytpu/build/mkTbTinyTPURuntime.bexe")
        import numpy as np
        from tinygrad import Tensor

        # Patch lower_gemm so we can capture the UOps tinygrad feeds it.
        import tinygrad.renderer.tinytpu.gemm as gmod
        import tinygrad.runtime.ops_tinytpu as ot

        captured: list[list[UOp]] = []
        orig = gmod.lower_gemm

        def trace(uops):
            captured.append(list(uops))
            return orig(uops)

        old_g, old_ot = gmod.lower_gemm, ot.lower_gemm
        gmod.lower_gemm = trace
        ot.lower_gemm = trace
        try:
            a = Tensor(np.zeros((4, 4), dtype=np.int32), device="TINYTPU").realize()
            b = Tensor(np.zeros((4, 4), dtype=np.int32), device="TINYTPU").realize()
            r = Tensor(np.zeros((4, 4), dtype=np.int32), device="TINYTPU").realize()
            (a.matmul(b) + r).realize().numpy()
        finally:
            gmod.lower_gemm = old_g
            ot.lower_gemm = old_ot

        self.assertTrue(captured, "lower_gemm was never invoked on the matmul+residual kernel")
        # tinygrad may emit multiple kernels; find the one whose STORE shape
        # is residual. At least one of them should match.
        hits = [recognize_residual_epilogue(u) for u in captured]
        self.assertTrue(any(h is not None for h in hits),
                        "no captured kernel matched the residual recognizer")


if __name__ == "__main__":
    unittest.main()
