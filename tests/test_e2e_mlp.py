"""
test_e2e_mlp.py — End-to-end TinyTPU MLP forward pass tests

Tests a 2-layer and 3-layer integer MLP on TinyTPU.  Each layer is:
    y = relu(x @ W + b)   (or no bias for the last layer)

All weights stay in int8 range so the MXU path fires.  Biases and
activations are int32.  Results are compared against numpy reference.
"""

from __future__ import annotations
import sys, os, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tinygrad"))
sys.path.insert(0, str(REPO_ROOT))
os.environ["TINYTPU_SIM"]          = str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")
os.environ["DISABLE_COMPILER_CACHE"] = "1"

import numpy as np
from tinygrad import Tensor
from scripts.tinytpu_model import linear, relu, mlp_forward


def _np_mlp(x, weights, biases=None, use_relu=True):
    """Numpy reference for mlp_forward."""
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


class TestE2EMLP(unittest.TestCase):

    # ── helpers ──────────────────────────────────────────────────────────

    def _check(self, result: np.ndarray, expected: np.ndarray, name: str):
        np.testing.assert_array_equal(
            result, expected,
            err_msg=f"{name}: result={result.flatten()[:8]} expected={expected.flatten()[:8]}"
        )

    # ── single linear layer ──────────────────────────────────────────────

    def test_linear_1x4_identity(self):
        x = np.array([[1, 2, 3, 4]], dtype=np.int32)
        W = np.eye(4, dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU")).numpy()
        self._check(out, x, "linear identity 1x4")

    def test_linear_1x4_with_bias(self):
        x = np.array([[1, 2, 3, 4]], dtype=np.int32)
        W = np.eye(4, dtype=np.int32)
        b = np.array([[10, 20, 30, 40]], dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU"),
                     Tensor(b, device="TINYTPU")).numpy()
        self._check(out, x + b, "linear+bias 1x4")

    def test_linear_2x4_batch(self):
        np.random.seed(0)
        x = np.random.randint(-3, 3, (2, 4), dtype=np.int32)
        W = np.random.randint(-3, 3, (4, 4), dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU")).numpy()
        self._check(out, x @ W, "linear 2x4 batch")

    def test_linear_4x4_batch(self):
        np.random.seed(1)
        x = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU")).numpy()
        self._check(out, x @ W, "linear 4x4 batch")

    def test_linear_4x4_with_bias(self):
        np.random.seed(2)
        x = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        b = np.random.randint(-5, 5, (1, 4), dtype=np.int32)
        out = linear(Tensor(x, device="TINYTPU"), Tensor(W, device="TINYTPU"),
                     Tensor(b, device="TINYTPU")).numpy()
        self._check(out, x @ W + b, "linear+bias 4x4 batch")

    # ── relu ─────────────────────────────────────────────────────────────

    def test_relu_1x4(self):
        x = np.array([[-1, 2, -3, 4]], dtype=np.int32)
        out = relu(Tensor(x, device="TINYTPU")).numpy()
        self._check(out, np.maximum(x, 0), "relu 1x4")

    def test_relu_4x4(self):
        np.random.seed(3)
        x = np.random.randint(-5, 5, (4, 4), dtype=np.int32)
        out = relu(Tensor(x, device="TINYTPU")).numpy()
        self._check(out, np.maximum(x, 0), "relu 4x4")

    # ── 2-layer MLP (no bias) ────────────────────────────────────────────

    def test_mlp_2layer_1x4_no_bias(self):
        np.random.seed(10)
        x  = np.random.randint(-2, 2, (1, 4), dtype=np.int32)
        W1 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W2 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        out = mlp_forward(Tensor(x, device="TINYTPU"), [W1, W2]).numpy()
        self._check(out, _np_mlp(x, [W1, W2]), "2-layer MLP 1x4 no bias")

    def test_mlp_2layer_4x4_no_bias(self):
        np.random.seed(11)
        x  = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W1 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W2 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        out = mlp_forward(Tensor(x, device="TINYTPU"), [W1, W2]).numpy()
        self._check(out, _np_mlp(x, [W1, W2]), "2-layer MLP 4x4 no bias")

    # ── 2-layer MLP with bias ────────────────────────────────────────────

    def test_mlp_2layer_1x4_with_bias(self):
        np.random.seed(20)
        x  = np.random.randint(-2, 2, (1, 4), dtype=np.int32)
        W1 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W2 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        b1 = np.random.randint(-5, 5, (1, 4), dtype=np.int32)
        b2 = np.random.randint(-5, 5, (1, 4), dtype=np.int32)
        out = mlp_forward(Tensor(x, device="TINYTPU"), [W1, W2],
                          biases=[b1, b2]).numpy()
        self._check(out, _np_mlp(x, [W1, W2], [b1, b2]), "2-layer+bias MLP 1x4")

    def test_mlp_2layer_4x4_with_bias(self):
        np.random.seed(21)
        x  = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W1 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W2 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        b1 = np.random.randint(-3, 3, (1, 4), dtype=np.int32)
        b2 = np.random.randint(-3, 3, (1, 4), dtype=np.int32)
        out = mlp_forward(Tensor(x, device="TINYTPU"), [W1, W2],
                          biases=[b1, b2]).numpy()
        self._check(out, _np_mlp(x, [W1, W2], [b1, b2]), "2-layer+bias MLP 4x4")

    # ── 3-layer MLP ──────────────────────────────────────────────────────

    def test_mlp_3layer_1x4_no_bias(self):
        np.random.seed(30)
        x  = np.random.randint(-2, 2, (1, 4), dtype=np.int32)
        W1 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W2 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W3 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        out = mlp_forward(Tensor(x, device="TINYTPU"), [W1, W2, W3]).numpy()
        self._check(out, _np_mlp(x, [W1, W2, W3]), "3-layer MLP 1x4 no bias")

    def test_mlp_3layer_4x4_no_bias(self):
        np.random.seed(31)
        x  = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W1 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W2 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        W3 = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        out = mlp_forward(Tensor(x, device="TINYTPU"), [W1, W2, W3]).numpy()
        self._check(out, _np_mlp(x, [W1, W2, W3]), "3-layer MLP 4x4 no bias")

    def test_mlp_3layer_4x4_with_bias(self):
        np.random.seed(32)
        x  = np.random.randint(-2, 2, (4, 4), dtype=np.int32)
        Ws = [np.random.randint(-2, 2, (4, 4), dtype=np.int32) for _ in range(3)]
        bs = [np.random.randint(-3, 3, (1, 4), dtype=np.int32) for _ in range(3)]
        out = mlp_forward(Tensor(x, device="TINYTPU"), Ws, biases=bs).numpy()
        self._check(out, _np_mlp(x, Ws, bs), "3-layer+bias MLP 4x4")

    # ── random stress tests ───────────────────────────────────────────────

    def test_mlp_random_weights_batch1(self):
        rng = np.random.default_rng(42)
        x  = rng.integers(-3, 3, (1, 4), dtype=np.int32)
        Ws = [rng.integers(-3, 3, (4, 4), dtype=np.int32) for _ in range(3)]
        bs = [rng.integers(-5, 5, (1, 4), dtype=np.int32) for _ in range(3)]
        out = mlp_forward(Tensor(x, device="TINYTPU"), Ws, biases=bs).numpy()
        self._check(out, _np_mlp(x, Ws, bs), "random 3-layer+bias batch=1")

    def test_mlp_random_weights_batch4(self):
        rng = np.random.default_rng(99)
        x  = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        Ws = [rng.integers(-2, 2, (4, 4), dtype=np.int32) for _ in range(3)]
        bs = [rng.integers(-4, 4, (1, 4), dtype=np.int32) for _ in range(3)]
        out = mlp_forward(Tensor(x, device="TINYTPU"), Ws, biases=bs).numpy()
        self._check(out, _np_mlp(x, Ws, bs), "random 3-layer+bias batch=4")

    # ── wide MLP (4-hidden, 4-wide) ──────────────────────────────────────

    def test_mlp_4layer_4x4_with_bias(self):
        rng = np.random.default_rng(77)
        x  = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        Ws = [rng.integers(-2, 2, (4, 4), dtype=np.int32) for _ in range(4)]
        bs = [rng.integers(-3, 3, (1, 4), dtype=np.int32) for _ in range(4)]
        out = mlp_forward(Tensor(x, device="TINYTPU"), Ws, biases=bs).numpy()
        self._check(out, _np_mlp(x, Ws, bs), "4-layer+bias MLP 4x4")

    # ── no-relu MLP (linear stack) ───────────────────────────────────────

    def test_mlp_3layer_no_relu_4x4(self):
        rng = np.random.default_rng(55)
        x  = rng.integers(-2, 2, (4, 4), dtype=np.int32)
        Ws = [rng.integers(-2, 2, (4, 4), dtype=np.int32) for _ in range(3)]
        out = mlp_forward(Tensor(x, device="TINYTPU"), Ws, use_relu=False).numpy()
        self._check(out, _np_mlp(x, Ws, use_relu=False), "3-layer no-relu 4x4")


if __name__ == "__main__":
    unittest.main()
