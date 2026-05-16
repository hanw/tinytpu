"""
test_e2e_attention.py — End-to-end TinyTPU attention block tests

Exercises a single-head and multi-head integer attention block on TinyTPU.
The block performs:

    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    S = Q @ K^T                    # scores (matmul + XLU transpose)
    A = attention_fn(S)            # softmax (host), relu, or argmax mask (TPU)
    O = A @ V                      # context
    Y = O @ Wo                     # output projection

Inputs and projection weights stay in [-1, 1] so every MXU operand fits
int8 range, which lets all four matmuls execute on the MXU (the K^T
operand also stays bounded). Softmax has no hardware path, so the soft
variants run softmax on the host (matches the embedding-lookup pattern
in test_e2e_pipeclean.py). The relu-attention variant runs the entire
forward pass on TinyTPU.
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
from scripts.tinytpu_model import linear


# ── reference numpy attention ─────────────────────────────────────────────────

def _np_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x.astype(np.float64))
    return e / e.sum(axis=-1, keepdims=True)


def _np_attention(x, Wq, Wk, Wv, Wo, kind: str):
    Q = x @ Wq
    K = x @ Wk
    V = x @ Wv
    S = Q @ K.T
    if kind == "softmax":
        A = _np_softmax(S)
        ctx = A @ V.astype(np.float64)
        ctx = np.rint(ctx).astype(np.int32)
    elif kind == "relu":
        A = np.maximum(S, 0)
        ctx = A @ V
    else:
        raise ValueError(kind)
    return ctx @ Wo


# ── TinyTPU attention block ───────────────────────────────────────────────────

def _attn_scores(x_t: Tensor, Wq_t: Tensor, Wk_t: Tensor):
    """Q, K projections + S = Q @ K^T, all realized on TINYTPU.

    The .contiguous() after permute is load-bearing: without it the
    TinyTPU matmul scheduler folds the permute view away and reads
    the original (un-transposed) K buffer.
    """
    Q = linear(x_t, Wq_t)
    K = linear(x_t, Wk_t)
    K_T = K.permute(1, 0).contiguous().realize()
    S = (Q @ K_T).realize()
    return Q, K, S


def _tpu_relu_attention(x, Wq, Wk, Wv, Wo):
    """Full attention block on TINYTPU with relu(scores) in place of softmax."""
    dev = "TINYTPU"
    x_t  = Tensor(x,  device=dev)
    Wq_t = Tensor(Wq, device=dev)
    Wk_t = Tensor(Wk, device=dev)
    Wv_t = Tensor(Wv, device=dev)
    Wo_t = Tensor(Wo, device=dev)
    _, _, S = _attn_scores(x_t, Wq_t, Wk_t)
    A = S.relu().realize()
    V = linear(x_t, Wv_t)
    ctx = (A @ V).realize()
    return linear(ctx, Wo_t).numpy()


def _tpu_softmax_attention(x, Wq, Wk, Wv, Wo):
    """Q,K,V proj + scores on TPU; softmax on host; A@V and out-proj on TPU.

    Mirrors the embedding-lookup hybrid in test_e2e_pipeclean.py: do the
    unsupported step (softmax) on the host, keep every matmul on the MXU.
    """
    dev = "TINYTPU"
    x_t  = Tensor(x,  device=dev)
    Wq_t = Tensor(Wq, device=dev)
    Wk_t = Tensor(Wk, device=dev)
    Wv_t = Tensor(Wv, device=dev)
    Wo_t = Tensor(Wo, device=dev)

    _, _, S = _attn_scores(x_t, Wq_t, Wk_t)
    V = linear(x_t, Wv_t)
    V_host = V.numpy()

    A_host = _np_softmax(S.numpy())              # host softmax (no float MXU)
    ctx_host = np.rint(A_host @ V_host.astype(np.float64)).astype(np.int32)

    return linear(Tensor(ctx_host, device=dev), Wo_t).numpy()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestE2EAttention(unittest.TestCase):

    # ── components ────────────────────────────────────────────────────────

    def test_qkv_projection_4x4(self):
        """Each of Q, K, V is a separate linear projection from x."""
        rng = np.random.default_rng(0)
        x  = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wq = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wk = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wv = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        dev = "TINYTPU"
        x_t = Tensor(x, device=dev)
        Q = linear(x_t, Tensor(Wq, device=dev)).numpy()
        K = linear(x_t, Tensor(Wk, device=dev)).numpy()
        V = linear(x_t, Tensor(Wv, device=dev)).numpy()
        np.testing.assert_array_equal(Q, x @ Wq)
        np.testing.assert_array_equal(K, x @ Wk)
        np.testing.assert_array_equal(V, x @ Wv)

    def test_scores_qkT_4x4(self):
        """S = Q @ K^T on TINYTPU using permute + matmul."""
        rng = np.random.default_rng(1)
        x  = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wq = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wk = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        dev = "TINYTPU"
        _, _, S_t = _attn_scores(
            Tensor(x, device=dev), Tensor(Wq, device=dev), Tensor(Wk, device=dev),
        )
        Q_ref = x @ Wq
        K_ref = x @ Wk
        np.testing.assert_array_equal(S_t.numpy(), Q_ref @ K_ref.T)

    def test_value_aggregation_4x4(self):
        """A @ V with A produced on host (e.g. softmax weights as integer mask)."""
        rng = np.random.default_rng(2)
        V = rng.integers(-2, 3, (4, 4), dtype=np.int32)
        A = rng.integers( 0, 3, (4, 4), dtype=np.int32)
        dev = "TINYTPU"
        out = (Tensor(A, device=dev) @ Tensor(V, device=dev)).realize().numpy()
        np.testing.assert_array_equal(out, A @ V)

    # ── relu-attention (entirely on TINYTPU) ──────────────────────────────

    def test_relu_attention_block_4x4(self):
        """Full attention block with relu(S) replacing softmax — TINYTPU only."""
        rng = np.random.default_rng(10)
        x  = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wq = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wk = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wv = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wo = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        out = _tpu_relu_attention(x, Wq, Wk, Wv, Wo)
        exp = _np_attention(x, Wq, Wk, Wv, Wo, "relu")
        np.testing.assert_array_equal(out, exp)

    def test_relu_attention_block_random(self):
        rng = np.random.default_rng(11)
        x  = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wq = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wk = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wv = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wo = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        out = _tpu_relu_attention(x, Wq, Wk, Wv, Wo)
        exp = _np_attention(x, Wq, Wk, Wv, Wo, "relu")
        np.testing.assert_array_equal(out, exp)

    # ── softmax attention (host fallback for softmax only) ────────────────

    def test_softmax_attention_block_4x4(self):
        """Full block with softmax on host; every matmul on TINYTPU MXU."""
        rng = np.random.default_rng(30)
        x  = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wq = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wk = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wv = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        Wo = rng.integers(-1, 2, (4, 4), dtype=np.int32)
        out = _tpu_softmax_attention(x, Wq, Wk, Wv, Wo)
        exp = _np_attention(x, Wq, Wk, Wv, Wo, "softmax")
        np.testing.assert_array_equal(out, exp)

    # ── multi-head attention (heads split on host, per-head block on TPU) ─

    def test_multihead_relu_attention_2heads(self):
        """Split d_model=8 into 2 heads of 4; run each head end-to-end on TPU."""
        rng = np.random.default_rng(40)
        seq, d_model, n_heads = 4, 8, 2
        head_dim = d_model // n_heads
        x  = rng.integers(-1, 2, (seq, d_model), dtype=np.int32)
        Wq = rng.integers(-1, 2, (d_model, d_model), dtype=np.int32)
        Wk = rng.integers(-1, 2, (d_model, d_model), dtype=np.int32)
        Wv = rng.integers(-1, 2, (d_model, d_model), dtype=np.int32)
        Wo = rng.integers(-1, 2, (d_model, d_model), dtype=np.int32)

        # Split projection weights into per-head blocks (along output dim).
        heads = []
        for h in range(n_heads):
            s = slice(h * head_dim, (h + 1) * head_dim)
            Wq_h = np.ascontiguousarray(Wq[:, s])
            Wk_h = np.ascontiguousarray(Wk[:, s])
            Wv_h = np.ascontiguousarray(Wv[:, s])
            # Per-head context (head_dim wide); output projection waits until concat.
            dev = "TINYTPU"
            x_t = Tensor(x, device=dev)
            _, _, S = _attn_scores(x_t, Tensor(Wq_h, device=dev), Tensor(Wk_h, device=dev))
            A = S.relu().realize()
            V = linear(x_t, Tensor(Wv_h, device=dev))
            heads.append((A @ V).realize().numpy())
        concat = np.concatenate(heads, axis=-1)  # (seq, d_model)
        out = linear(Tensor(concat, device="TINYTPU"), Tensor(Wo, device="TINYTPU")).numpy()

        # Reference: same per-head split.
        ref_heads = []
        for h in range(n_heads):
            s = slice(h * head_dim, (h + 1) * head_dim)
            Q = x @ Wq[:, s]
            K = x @ Wk[:, s]
            V = x @ Wv[:, s]
            A = np.maximum(Q @ K.T, 0)
            ref_heads.append(A @ V)
        exp = np.concatenate(ref_heads, axis=-1) @ Wo
        np.testing.assert_array_equal(out, exp)


if __name__ == "__main__":
    unittest.main()
