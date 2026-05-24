"""
test_e2e_epilogues.py — End-to-end TinyTPU tests for the five CODA-style
GEMM epilogue primitive classes (Snell et al., "CODA: Coordinated Diverse
Accelerators", §3.1):

  1. Elementwise & pairwise maps    (residual, activation, RoPE, SwiGLU-style)
  2. Vector (rank-1) loads & stores (row/col bias broadcast, vector output)
  3. Tile   (rank-2) loads & stores (residual tile, saved-activation tile)
  4. Tile   (rank-2) reductions     (row/col/full sum, max, min)
  5. Stateful transforms             (stable softmax preamble, log-sum-exp,
                                     cross-entropy skeleton)

Each test wraps the epilogue around a real GEMM (x @ W) so the lowering
path exercises the same MXU + VPU/SXU surface a production epilogue would
hit. Inputs are kept in int8 range so the matmul fires on the MXU; the
epilogue itself runs on int32.

Known gap: column-vector (M,1) broadcast over a (M,N) tile currently
mis-lowers as elementwise multiply. The vector-load test that exercises it
is marked expectedFailure with a pointer to TODO.md.
"""

from __future__ import annotations
import sys, os, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tinygrad"))
sys.path.insert(0, str(REPO_ROOT))
os.environ["TINYTPU_SIM"]           = str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")
os.environ["DISABLE_COMPILER_CACHE"] = "1"

import numpy as np
from tinygrad import Tensor


DEV = "TINYTPU"


def _t(a: np.ndarray) -> Tensor:
  return Tensor(a, device=DEV)


# ── 1. Elementwise & pairwise maps ───────────────────────────────────────────

class TestElementwiseEpilogues(unittest.TestCase):
  """Local per-element transforms of accumulator values."""

  def test_residual_add(self):
    """y = (x @ W) + r — additive residual update."""
    rng = np.random.default_rng(100)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    r = rng.integers(-5, 5, (4, 4), dtype=np.int32)
    out = ((_t(x) @ _t(W)).realize() + _t(r)).realize().numpy()
    np.testing.assert_array_equal(out, (x @ W) + r)

  def test_relu_activation(self):
    """y = relu(x @ W) — pointwise activation."""
    rng = np.random.default_rng(101)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    out = (_t(x) @ _t(W)).realize().relu().realize().numpy()
    np.testing.assert_array_equal(out, np.maximum(x @ W, 0))

  def test_swiglu_style_gated(self):
    """y = (x @ Wg) * relu(x @ Wu) — ReGLU stand-in for SwiGLU (integer)."""
    rng = np.random.default_rng(102)
    x  = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    Wg = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    Wu = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    g = (_t(x) @ _t(Wg)).realize()
    u = (_t(x) @ _t(Wu)).realize().relu().realize()
    out = (g * u).realize().numpy()
    np.testing.assert_array_equal(out, (x @ Wg) * np.maximum(x @ Wu, 0))

  def test_rope_style_rotation(self):
    """Pairwise rotation: (a, b) -> (a*c - b*s, a*s + b*c) with integer (c,s).

    RoPE rotates pairs of activations by a precomputed (cos, sin). With
    integer (c, s) the rotation reduces to four pairwise mul/sub/add ops
    on two GEMM-produced halves a and b.
    """
    rng = np.random.default_rng(103)
    x  = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    Wa = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    Wb = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    c, s = 3, -2
    a = (_t(x) @ _t(Wa)).realize()
    b = (_t(x) @ _t(Wb)).realize()
    out_a = ((a * c).realize() - (b * s).realize()).realize().numpy()
    out_b = ((a * s).realize() + (b * c).realize()).realize().numpy()
    np.testing.assert_array_equal(out_a, (x @ Wa) * c - (x @ Wb) * s)
    np.testing.assert_array_equal(out_b, (x @ Wa) * s + (x @ Wb) * c)


# ── 2. Vector (rank-1) loads and stores ──────────────────────────────────────

class TestVectorEpilogues(unittest.TestCase):
  """Load a row/col vector into the epilogue, or emit a vector aux output."""

  def test_row_vector_load(self):
    """Load a (1, N) bias and broadcast it over the tile rows."""
    rng = np.random.default_rng(200)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    b = rng.integers(-5, 5, (1, 4), dtype=np.int32)
    out = ((_t(x) @ _t(W)).realize() + _t(b)).realize().numpy()
    np.testing.assert_array_equal(out, (x @ W) + b)

  def test_col_vector_load(self):
    """Load a (M, 1) bias and broadcast it over the tile columns."""
    rng = np.random.default_rng(201)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    b = rng.integers(-5, 5, (4, 1), dtype=np.int32)
    out = ((_t(x) @ _t(W)).realize() + _t(b)).realize().numpy()
    np.testing.assert_array_equal(out, (x @ W) + b)

  def test_row_vector_store(self):
    """Emit a (M, 1) aux vector: per-row sum of the GEMM output."""
    rng = np.random.default_rng(202)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    rs = (_t(x) @ _t(W)).realize().sum(axis=1, keepdim=True).realize().numpy()
    np.testing.assert_array_equal(rs.flatten(), (x @ W).sum(axis=1))

  def test_col_vector_store(self):
    """Emit a (1, N) aux vector: per-column sum of the GEMM output."""
    rng = np.random.default_rng(203)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    cs = (_t(x) @ _t(W)).realize().sum(axis=0, keepdim=True).realize().numpy()
    np.testing.assert_array_equal(cs.flatten(), (x @ W).sum(axis=0))


# ── 3. Tile (rank-2) loads and stores ────────────────────────────────────────

class TestTileEpilogues(unittest.TestCase):
  """Load an external (M, N) tile (residual / saved activation) or store one."""

  def test_load_residual_tile(self):
    """y = (x @ W) + R, where R is a full (M, N) tile loaded into the epilogue."""
    rng = np.random.default_rng(300)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    R = rng.integers(-5, 5, (4, 4), dtype=np.int32)
    out = ((_t(x) @ _t(W)).realize() + _t(R)).realize().numpy()
    np.testing.assert_array_equal(out, (x @ W) + R)

  def test_store_saved_activation(self):
    """Emit two tile results: pre-activation z (saved for backward) and relu(z)."""
    rng = np.random.default_rng(301)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    z = (_t(x) @ _t(W)).realize()
    z_host = z.numpy()                       # tile store #1 — saved activation
    y = z.relu().realize().numpy()           # tile store #2 — forward output
    np.testing.assert_array_equal(z_host, x @ W)
    np.testing.assert_array_equal(y, np.maximum(x @ W, 0))

  def test_chained_residual_tiles(self):
    """Two-layer block: h = relu(x @ W1) + x; y = relu(h @ W2) + h."""
    rng = np.random.default_rng(302)
    x  = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W1 = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W2 = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    xt = _t(x).realize()
    h = ((_t(x) @ _t(W1)).realize().relu().realize() + xt).realize()
    y = ((h @ _t(W2)).realize().relu().realize() + h).realize().numpy()
    h_ref = np.maximum(x @ W1, 0) + x
    y_ref = np.maximum(h_ref @ W2, 0) + h_ref
    np.testing.assert_array_equal(y, y_ref)


# ── 4. Tile (rank-2) reductions ──────────────────────────────────────────────

class TestReductionEpilogues(unittest.TestCase):
  """Partial reductions over rows or columns of the GEMM output tile."""

  def test_row_sum(self):
    rng = np.random.default_rng(400)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    out = (_t(x) @ _t(W)).realize().sum(axis=1, keepdim=True).realize().numpy()
    np.testing.assert_array_equal(out.flatten(), (x @ W).sum(axis=1))

  def test_col_sum(self):
    rng = np.random.default_rng(401)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    out = (_t(x) @ _t(W)).realize().sum(axis=0, keepdim=True).realize().numpy()
    np.testing.assert_array_equal(out.flatten(), (x @ W).sum(axis=0))

  def test_full_tile_sum(self):
    rng = np.random.default_rng(402)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    out = (_t(x) @ _t(W)).realize().sum().realize().numpy()
    self.assertEqual(int(out), int((x @ W).sum()))

  def test_row_max(self):
    rng = np.random.default_rng(403)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    out = (_t(x) @ _t(W)).realize().max(axis=1, keepdim=True).realize().numpy()
    np.testing.assert_array_equal(out.flatten(), (x @ W).max(axis=1))

  def test_row_min(self):
    rng = np.random.default_rng(404)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    out = (_t(x) @ _t(W)).realize().min(axis=1, keepdim=True).realize().numpy()
    np.testing.assert_array_equal(out.flatten(), (x @ W).min(axis=1))


# ── 5. Stateful transforms ───────────────────────────────────────────────────

class TestStatefulEpilogues(unittest.TestCase):
  """Maintain running tile state (running max, running sum) for online softmax
  / log-sum-exp / cross-entropy. Tested as the per-tile structural pattern;
  multi-tile streaming combines via the row-max and row-sum primitives below.
  """

  def test_stable_softmax_preamble(self):
    """z - rowmax(z) — the numerical-stability step for online softmax.

    The two outputs (rowmax m, centered logits z - m) are exactly the state a
    streaming softmax kernel would persist between K-block tiles.
    """
    rng = np.random.default_rng(500)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    z = (_t(x) @ _t(W)).realize()
    m = z.max(axis=1, keepdim=True).realize()
    centered = (z - m).realize().numpy()
    z_ref = x @ W
    np.testing.assert_array_equal(m.numpy().flatten(), z_ref.max(axis=1))
    np.testing.assert_array_equal(centered, z_ref - z_ref.max(axis=1, keepdims=True))

  def test_log_sum_exp_integer_skeleton(self):
    """Integer skeleton of log-sum-exp: m = rowmax(z); s = rowsum(relu(z - m)).

    Replaces exp with relu so the test stays on-chip and exact; exercises the
    full (rowmax -> centered -> rowsum) sequence that an int-only or fixed-
    point LSE would use. With true exp this becomes: m + log(rowsum(exp(z-m))).
    """
    rng = np.random.default_rng(501)
    x = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    z = (_t(x) @ _t(W)).realize()
    m = z.max(axis=1, keepdim=True).realize()
    centered = (z - m).realize()
    s = centered.relu().realize().sum(axis=1, keepdim=True).realize().numpy()
    z_ref = x @ W
    m_ref = z_ref.max(axis=1, keepdims=True)
    s_ref = np.maximum(z_ref - m_ref, 0).sum(axis=1, keepdims=True)
    np.testing.assert_array_equal(m.numpy(), m_ref)
    np.testing.assert_array_equal(s, s_ref)

  def test_cross_entropy_skeleton(self):
    """Cross-entropy skeleton: pick target logit via one-hot mask, subtract
    log-sum-exp-stand-in. Per-row: ce_row = lse_stand_in_row - z[row, tgt].

    Gather isn't lowered, so the target index is materialized as a one-hot
    mask on the host, then z * mask is summed on TPU to extract z[row, tgt].

    Each intermediate is named and .realize()'d before being consumed.
    Inlining the reduce inside the surrounding add (e.g. `(m + tile.sum(...))`)
    produces a fused kernel that the renderer currently does not lower; after
    the GEMM-fallback hardening that pattern now raises NotImplementedError
    rather than silently producing wrong output. See `test_inline_reduce_in_add_raises`.
    """
    rng = np.random.default_rng(502)
    x  = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    W  = rng.integers(-2, 3, (4, 4), dtype=np.int32)
    tgt = np.array([0, 2, 1, 3], dtype=np.int32)
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[np.arange(4), tgt] = 1

    z = (_t(x) @ _t(W)).realize()
    m = z.max(axis=1, keepdim=True).realize()
    centered = (z - m).realize()
    relud = centered.relu().realize()
    ssum = relud.sum(axis=1, keepdim=True).realize()
    lse_stand_in = (m + ssum).realize()
    masked = (z * _t(mask)).realize()
    z_tgt = masked.sum(axis=1, keepdim=True).realize()
    ce = (lse_stand_in - z_tgt).realize().numpy()

    z_ref = x @ W
    m_ref = z_ref.max(axis=1, keepdims=True)
    lse_ref = m_ref + np.maximum(z_ref - m_ref, 0).sum(axis=1, keepdims=True)
    z_tgt_ref = z_ref[np.arange(4), tgt].reshape(-1, 1)
    np.testing.assert_array_equal(ce, lse_ref - z_tgt_ref)


  def test_inline_reduce_in_add_raises(self):
    """`(m + tile.sum(axis=1, keepdim=True)).realize()` fuses a reduce inside
    an add. No fused lowering exists for that pattern. Before the GEMM-fallback
    hardening (`tinygrad/renderer/tinytpu/gemm.py: lower_gemm_fallback`), this
    kernel silently fell into the phantom-GEMM path and produced wrong output.
    It must now raise NotImplementedError so callers either lift the reduce to
    a named intermediate or add a real fused lowerer.
    """
    rng = np.random.default_rng(503)
    m = rng.integers(-5, 5, (4, 1), dtype=np.int32)
    t = rng.integers(-5, 5, (4, 4), dtype=np.int32)
    mt = _t(m).realize()
    tt = _t(t).realize()
    with self.assertRaises(NotImplementedError):
      (mt + tt.sum(axis=1, keepdim=True)).realize().numpy()


if __name__ == "__main__":
  unittest.main()
