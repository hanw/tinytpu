"""
test_e2e_pipeclean.py — End-to-end pipeclean for complex model patterns

Documents which patterns work natively on TinyTPU and which require
explicit .realize() calls to prevent tinygrad kernel fusion.

Working:
  ✓ Linear layer (any batch size 1-4, with/without bias)
  ✓ ReLU after linear
  ✓ Multi-layer MLP with bias, 1-4 layers
  ✓ Residual (skip) connections
  ✓ Embedding table lookup + linear projection (host gather)
  ✓ 4-layer integer classifier (forward pass)
  ✓ Row-broadcast bias-add (VPU_ROWBC_BINARY)

Known limitations:
  ✗ Attention with large intermediate activations (MXU requires int8 range)
  ✗ Fused (x@W+b).relu() without .realize() (4-param kernel not yet detected)
  ✗ Softmax (requires float or large-range division)
  ✗ Gather/scatter indexing (not yet lowered)
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
from scripts.tinytpu_model import linear, relu, linear_relu, mlp_forward


class TestE2EPipeclean(unittest.TestCase):

    # ── Residual block ────────────────────────────────────────────────────

    def test_residual_block_4x4(self):
        """relu(x @ W) + x — skip connection."""
        rng = np.random.default_rng(10)
        x = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        W = rng.integers(-1, 1, (4, 4), dtype=np.int32)
        xt = Tensor(x, device="TINYTPU")
        out = (linear_relu(xt, Tensor(W, device="TINYTPU")) + xt.realize()).realize().numpy()
        exp = np.maximum(x @ W, 0) + x
        np.testing.assert_array_equal(out, exp)

    def test_residual_block_with_bias(self):
        rng = np.random.default_rng(11)
        x = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        W = rng.integers(-1, 1, (4, 4), dtype=np.int32)
        b = rng.integers(-2, 2, (1, 4), dtype=np.int32)
        xt = Tensor(x, device="TINYTPU")
        h  = linear_relu(xt, Tensor(W, device="TINYTPU"), Tensor(b, device="TINYTPU"))
        out = (h + xt.realize()).realize().numpy()
        exp = np.maximum(x @ W + b, 0) + x
        np.testing.assert_array_equal(out, exp)

    def test_two_residual_blocks(self):
        rng = np.random.default_rng(12)
        x  = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        W1 = rng.integers(-1, 1, (4, 4), dtype=np.int32)
        W2 = rng.integers(-1, 1, (4, 4), dtype=np.int32)
        xt = Tensor(x, device="TINYTPU")
        h1 = (linear_relu(xt, Tensor(W1, device="TINYTPU")) + xt.realize()).realize()
        h2 = (linear_relu(h1, Tensor(W2, device="TINYTPU")) + h1).realize()
        exp1 = np.maximum(x @ W1, 0) + x
        exp2 = np.maximum(exp1 @ W2, 0) + exp1
        np.testing.assert_array_equal(h2.numpy(), exp2)

    # ── Embedding + linear ────────────────────────────────────────────────

    def test_embedding_lookup_then_linear(self):
        """Gather on host, project on TinyTPU."""
        rng = np.random.default_rng(20)
        vocab, dim = 4, 4
        E = rng.integers(-3, 3, (vocab, dim), dtype=np.int32)
        W = rng.integers(-2, 2, (dim, dim), dtype=np.int32)
        ids = np.array([0, 2, 1, 3], dtype=np.int32)
        embeds = E[ids]  # host gather
        out = linear(Tensor(embeds, device="TINYTPU"),
                     Tensor(W, device="TINYTPU")).numpy()
        np.testing.assert_array_equal(out, embeds @ W)

    def test_embedding_mlp_pipeline(self):
        """Embedding → 2-layer MLP."""
        rng = np.random.default_rng(21)
        E  = rng.integers(-3, 3, (4, 4), dtype=np.int32)
        W1 = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        W2 = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        ids = np.array([1, 0, 3, 2], dtype=np.int32)
        embeds = E[ids]
        out = mlp_forward(Tensor(embeds, device="TINYTPU"), [W1, W2]).numpy()
        exp = _np_mlp(embeds, [W1, W2])
        np.testing.assert_array_equal(out, exp)

    # ── Integer classifier ────────────────────────────────────────────────

    def test_4layer_classifier_forward(self):
        """4-layer MLP used as a classifier: logits and argmax prediction."""
        rng = np.random.default_rng(30)
        x  = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        Ws = [rng.integers(-2, 2, (4, 4), dtype=np.int32) for _ in range(4)]
        bs = [rng.integers(-2, 2, (1, 4), dtype=np.int32) for _ in range(4)]
        logits = mlp_forward(Tensor(x, device="TINYTPU"), Ws, biases=bs).numpy()
        exp    = _np_mlp(x, Ws, bs)
        np.testing.assert_array_equal(logits, exp)
        # Predictions (argmax on host)
        preds     = logits.argmax(axis=1)
        exp_preds = exp.argmax(axis=1)
        np.testing.assert_array_equal(preds, exp_preds)

    def test_classifier_batch1(self):
        rng = np.random.default_rng(31)
        x  = rng.integers(-2, 2, (1, 4), dtype=np.int32)
        Ws = [rng.integers(-2, 2, (4, 4), dtype=np.int32) for _ in range(3)]
        bs = [rng.integers(-2, 2, (1, 4), dtype=np.int32) for _ in range(3)]
        logits = mlp_forward(Tensor(x, device="TINYTPU"), Ws, biases=bs).numpy()
        np.testing.assert_array_equal(logits, _np_mlp(x, Ws, bs))

    # ── Bias-add broadcast ────────────────────────────────────────────────

    def test_rowbc_bias_add_1x4_output(self):
        """GEMM output (1x4) + bias (1x4) — plain VPU_BINARY."""
        rng = np.random.default_rng(40)
        x = rng.integers(-2, 2, (1, 4), dtype=np.int32)
        W = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        b = rng.integers(-5, 5, (1, 4), dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU"),
                     Tensor(b, device="TINYTPU")).numpy()
        np.testing.assert_array_equal(out, x @ W + b)

    def test_rowbc_bias_add_4x4_output(self):
        """GEMM output (4x4) + bias (1x4) — VPU_ROWBC_BINARY row-broadcast."""
        rng = np.random.default_rng(41)
        x = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        W = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        b = rng.integers(-5, 5, (1, 4), dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU"),
                     Tensor(b, device="TINYTPU")).numpy()
        np.testing.assert_array_equal(out, x @ W + b)

    def test_rowbc_bias_add_random_large(self):
        rng = np.random.default_rng(42)
        x = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        W = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        b = rng.integers(-10, 10, (1, 4), dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU"),
                     Tensor(b, device="TINYTPU")).numpy()
        np.testing.assert_array_equal(out, x @ W + b)

    # ── Step-by-step fused operations ────────────────────────────────────

    def test_gemm_bias_relu_via_realize(self):
        """(x@W+b).relu() split into 3 kernels via .realize()."""
        rng = np.random.default_rng(50)
        x = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        W = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        b = rng.integers(-3, 3, (1, 4), dtype=np.int32)
        h1 = (Tensor(x, device="TINYTPU") @ Tensor(W, device="TINYTPU")).realize()
        h2 = (h1 + Tensor(b, device="TINYTPU")).realize()
        out = h2.relu().numpy()
        np.testing.assert_array_equal(out, np.maximum(x @ W + b, 0))

    def test_mlp_with_postproc_on_host(self):
        """MLP forward + host-side argmax classification."""
        rng = np.random.default_rng(60)
        x   = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        Ws  = [rng.integers(-2, 2, (4, 4), dtype=np.int32) for _ in range(3)]
        bs  = [rng.integers(-2, 2, (1, 4), dtype=np.int32) for _ in range(3)]
        logits = mlp_forward(Tensor(x, device="TINYTPU"), Ws, biases=bs).numpy()
        preds  = logits.argmax(axis=1)       # host-side post-processing
        exp    = _np_mlp(x, Ws, bs).argmax(axis=1)
        np.testing.assert_array_equal(preds, exp)


# ── reference implementation ──────────────────────────────────────────────────

def _np_mlp(x, weights, biases=None, use_relu=True):
    if biases is None:
        biases = [None] * len(weights)
    h = x.copy()
    for i, (W, b) in enumerate(zip(weights, biases)):
        h = h @ W
        if b is not None:
            h = h + b
        if use_relu and i < len(weights) - 1:
            h = np.maximum(h, 0)
    return h


if __name__ == "__main__":
    unittest.main()
