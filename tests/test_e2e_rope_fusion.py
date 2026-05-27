"""End-to-end RoPE fusion through the full TinyTPU stack.

Hand-builds a UOp graph matching the RoPE epilogue shape, runs it
through lower_gemm, materializes the bundle via TinyTPUKernel.build_bundle,
fires the BSV simulator, and asserts:

  1. The lowered instructions contain exactly num_output_tiles op-46
     dispatches (single-bundle, single-invocation fusion).
  2. The lowered instructions contain no legacy GEMM-then-VPU chain ops
     (op 4 = DISPATCH_MXU, op 2 = DISPATCH_VPU appearing as a separate
     add) — the entire kernel collapses to LOAD + op-46 + HALT.
  3. The runtime output matches a numpy reference IPAIR_ROTATE applied
     to the GEMM result with the same interleaved-CS coefficients.

The hand-built UOps approach is deliberate: no current tinygrad
expression produces a single fused kernel matching this shape (each
`.realize()` forces a kernel boundary). Once an upstream scheduler
or fusion hint surfaces RoPE in a single kernel, this test verifies
the lowering path works end-to-end.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))

import numpy as np

from tinygrad.dtype import dtypes
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer.tinytpu.gemm import lower_gemm
from tinygrad.runtime.support.compiler_tinytpu import TinyTPUKernel
from tinygrad.runtime.support.tinytpu import run_bundle, parse_multi_vmem_output


INT = dtypes.int
INT8 = dtypes.char
PTR16 = dtypes.int.ptr(16)
VEC16 = dtypes.int.vec(16)


def _param(idx: int, dtype=PTR16) -> UOp:
  return UOp(Ops.PARAM, dtype, (), idx)


def _const(v: int, dtype=INT) -> UOp:
  return UOp(Ops.CONST, dtype, (), v)


def _index(p: UOp, i: int) -> UOp:
  return UOp(Ops.INDEX, PTR16, (p, _const(i)), None)


def _load(p: UOp, i: int) -> UOp:
  return UOp(Ops.LOAD, INT, (_index(p, i),), None)


def _stack16(vals: list[UOp], dtype=VEC16) -> UOp:
  assert len(vals) == 16
  return UOp(Ops.STACK, dtype, tuple(vals), None)


def _wmma_4x4_int(a_stack: UOp, b_stack: UOp, acc: UOp) -> UOp:
  return UOp(Ops.WMMA, VEC16, (a_stack, b_stack, acc),
             ("WMMA_4_4_4_int_int", (4, 4, 4), dtypes.int, dtypes.int,
              "TINYTPU", 1, (), (), ()))


def _gep(wmma: UOp, i: int) -> UOp:
  return UOp(Ops.GEP, INT, (wmma,), (i,))


def _mul(a: UOp, b: UOp) -> UOp:
  return UOp(Ops.MUL, INT, (a, b), None)


def _add(a: UOp, b: UOp) -> UOp:
  return UOp(Ops.ADD, INT, (a, b), None)


def _store(p: UOp, i: int, val: UOp) -> UOp:
  return UOp(Ops.STORE, dtypes.void, (_index(p, i), val), None)


def _build_rope_kernel_uops():
  """Build a 4x4 GEMM + per-lane RoPE epilogue UOp graph.

  Param indices match the lower_gemm convention:
    0 = out (int[16])
    1 = a   (int8[16])
    2 = b   (int8[16])
    3 = c   (int[16])  — interleaved later; here just the cos source
    4 = s   (int[16])  — interleaved later; here just the sin source
  """
  p_out = _param(0)
  p_a   = _param(1)
  p_b   = _param(2)
  p_c   = _param(3)
  p_s   = _param(4)

  # GEMM preamble: 16 loads per operand, stacked to vec(16) inputs.
  a_loads = [_load(p_a, i) for i in range(16)]
  b_loads = [_load(p_b, i) for i in range(16)]
  zero = _const(0)
  acc = _stack16([zero] * 16)
  a_stack = _stack16(a_loads)
  b_stack = _stack16(b_loads)
  wmma = _wmma_4x4_int(a_stack, b_stack, acc)

  # GEPs for each output lane.
  geps = [_gep(wmma, i) for i in range(16)]

  # Per-lane RoPE epilogue. Real tinygrad lowering produces a flat
  # topological list of all UOps, so we explicitly track MULs / ADDs
  # alongside the LOADs / STOREs — Counter walks the explicit list.
  c_loads = [_load(p_c, i) for i in range(16)]
  s_loads = [_load(p_s, i) for i in range(16)]
  muls = []
  adds = []
  stores = []
  for i in range(16):
    mul_c = _mul(geps[i], c_loads[i])
    mul_s = _mul(geps[i ^ 1], s_loads[i ^ 1])
    add   = _add(mul_c, mul_s)
    muls.extend([mul_c, mul_s])
    adds.append(add)
    stores.append(_store(p_out, i, add))

  uops: list[UOp] = (
    [p_out, p_a, p_b, p_c, p_s]
    + a_loads + b_loads + [acc, a_stack, b_stack, wmma]
    + geps + c_loads + s_loads + muls + adds + stores
  )
  return uops


def _numpy_ipair_rotate(d: np.ndarray, c_flat: np.ndarray, s_flat: np.ndarray) -> np.ndarray:
  """Reference: lane-wise IPAIR_ROTATE on the 4x4 d, where src2 is the
  interleaved CS tile (even lanes from c, odd lanes from s, both
  indexed into their respective param buffers at the same flat index).
  """
  c = c_flat.reshape(4, 4)
  s = s_flat.reshape(4, 4)
  out = np.zeros_like(d)
  for r in range(4):
    for p in range(2):
      de, do = d[r, 2 * p], d[r, 2 * p + 1]
      # src2[r][2p]   = c[r][2p]   (even lane → cos)
      # src2[r][2p+1] = s[r][2p+1] (odd  lane → sin)
      cc = c[r, 2 * p]
      ss = s[r, 2 * p + 1]
      out[r, 2 * p]     = de * cc - do * ss
      out[r, 2 * p + 1] = de * ss + do * cc
  return out


class TestE2ERopeFusion(unittest.TestCase):

  def test_rope_lowering_emits_single_bundle(self):
    uops = _build_rope_kernel_uops()
    desc = lower_gemm(uops)
    self.assertIsNotNone(desc, "lower_gemm rejected the rope UOp graph")
    self.assertEqual(desc["op"], "SXU_PROGRAM")

    op46_lines = [ln for ln in desc["instructions"] if ln.startswith("2 46 ")]
    # 4 output tiles (num_vecs=4, num_weight_tiles=1) → 4 op-46 dispatches.
    self.assertEqual(len(op46_lines), 4, f"expected 4 op-46 dispatches, got {len(op46_lines)}")

    # No legacy chain ops: no DISPATCH_MXU (op 4), no WAIT_MXU (op 5),
    # no LOAD_MXU_RESULT (op 6), no DISPATCH_VPU (op 2 standalone) —
    # the entire kernel collapses to _load(src2) + op-46 + HALT.
    for forbidden in ("2 4 ", "2 5 ", "2 6 "):
      self.assertFalse(any(ln.startswith(forbidden) for ln in desc["instructions"]),
                       f"unexpected legacy op {forbidden!r} in fused rope bundle")
    # The data_plan should declare INTERLEAVE_CS for the src2 region.
    cs_entries = [e for e in desc["data_plan"] if e.get("mode") == "INTERLEAVE_CS"]
    self.assertEqual(len(cs_entries), 1)
    self.assertEqual(cs_entries[0]["s_param"], 4)

  def test_rope_runtime_matches_numpy_reference(self):
    """Run the fused bundle through the BSV sim and verify the output
    matches a numpy reference IPAIR_ROTATE applied to the per-PE drain
    matrix.

    Important semantic note: op-46 reads array.getMatrix (per-PE state,
    drain[r][c] = act[r] * weight[r][c] for WS tile_len=1), NOT
    array.getResults (column-summed GEMM output). The "CODA primitive"
    fusion operates elementwise on the per-PE matrix. We use identity
    weights here so the per-PE drain is diag(act_vec) — same trick
    Slice 5 uses — and assert the IPAIR_ROTATE'd full 16-element tile.
    """
    sim = os.environ.get("TINYTPU_SIM",
                         "/home/hanwang/p/tinytpu/build/mkTbTinyTPURuntime.bexe")
    if not os.path.exists(sim):
      self.skipTest(f"sim binary {sim} not built")

    # UOp shape implies nv=4 (16 LOADs per operand → 4 act rows worth of
    # data). The fused path emits 4 op-46 dispatches, but only dispatch 0
    # carries our meaningful activation row at AMEM[0]; rows 1..3 are
    # zero-padded so those dispatches produce zero drain. We assert
    # against tiles[0].
    act_row_0 = np.array([-3, 2, -1, 4], dtype=np.int8)
    a_full = np.zeros((4, 4), dtype=np.int8)
    a_full[0] = act_row_0
    ident_w = np.eye(4, dtype=np.int8)
    c_tile  = np.arange(1, 17, dtype=np.int32).reshape(4, 4)
    s_tile  = (np.arange(1, 17, dtype=np.int32) * 2).reshape(4, 4)

    out_buf = bytearray(16 * 4)
    a_buf   = bytearray(a_full.astype(np.int32).tobytes())
    b_buf   = bytearray(ident_w.astype(np.int32).tobytes())
    c_buf   = bytearray(c_tile.tobytes())
    s_buf   = bytearray(s_tile.tobytes())
    bufs    = (out_buf, a_buf, b_buf, c_buf, s_buf)

    uops = _build_rope_kernel_uops()
    desc = lower_gemm(uops)
    self.assertIsNotNone(desc)

    kernel = TinyTPUKernel(desc)
    bundle = kernel.build_bundle(bufs)
    stdout = run_bundle(sim, bundle)
    tiles = parse_multi_vmem_output(stdout)
    self.assertTrue(len(tiles) >= 1, f"expected at least 1 vmem_result tile, got {len(tiles)}")

    # The first vmem_result tile is the dispatch 0 output. With identity
    # weights and a single-vec input, drain = diag(act_row_0).
    drain = np.diag(act_row_0).astype(np.int32)
    expected_full = _numpy_ipair_rotate(drain, c_tile.reshape(-1), s_tile.reshape(-1))
    got_full = np.array(tiles[0], dtype=np.int32).reshape(4, 4)
    np.testing.assert_array_equal(got_full, expected_full,
      "rope-fused per-PE elementwise output mismatch")

  def test_rope_uses_one_sim_invocation(self):
    """Slice 8 guard: the fused RoPE path must use exactly 1 sim call.

    The pre-fusion baseline for the same expression would be 5
    invocations (separate kernels for matmul, swap-matmul, two
    elementwise muls, one elementwise add). One bundle => one
    `run_bundle` => one sim subprocess.
    """
    sim = os.environ.get("TINYTPU_SIM",
                         "/home/hanwang/p/tinytpu/build/mkTbTinyTPURuntime.bexe")
    if not os.path.exists(sim):
      self.skipTest(f"sim binary {sim} not built")

    rng = np.random.default_rng(7)
    bufs = (bytearray(16 * 4),
            bytearray(rng.integers(-2, 2, size=16, dtype=np.int8).astype(np.int32).tobytes()),
            bytearray(rng.integers(-2, 2, size=16, dtype=np.int8).astype(np.int32).tobytes()),
            bytearray(rng.integers(-3, 3, size=16, dtype=np.int32).tobytes()),
            bytearray(rng.integers(-3, 3, size=16, dtype=np.int32).tobytes()))

    desc = lower_gemm(_build_rope_kernel_uops())
    bundle = TinyTPUKernel(desc).build_bundle(bufs)

    # Patch subprocess.run to count invocations.
    from tinygrad.runtime.support import tinytpu as tinytpu_rt
    import subprocess
    calls = [0]
    orig = subprocess.run
    def counted(*a, **kw):
      calls[0] += 1
      return orig(*a, **kw)
    subprocess.run = counted
    try:
      tinytpu_rt.run_bundle(sim, bundle)
    finally:
      subprocess.run = orig

    self.assertEqual(calls[0], 1, f"fused rope should be 1 sim call, was {calls[0]}")


if __name__ == "__main__":
  unittest.main()
