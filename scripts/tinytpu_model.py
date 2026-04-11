"""
tinytpu_model.py — TinyTPU-compatible model primitives

TinyTPU's tinygrad backend does not yet lower fused multi-op kernels
(e.g. GEMM+bias or GEMM+ReLU as a single kernel).  These primitives
force eager evaluation between every operation with .realize() so that
each step maps to exactly one supported TinyTPU kernel.

Usage:
    from scripts.tinytpu_model import linear, relu, mlp_forward, conv1x1

These are int32 only.  Weights must stay within int8 range [-128,127]
for the MXU (GEMM) path.
"""

from tinygrad import Tensor
import numpy as np


def linear(x: Tensor, W: Tensor, bias: Tensor | None = None) -> Tensor:
    """Linear layer: y = x @ W [+ bias].  Forces realize between GEMM and bias-add."""
    out = (x @ W).realize()
    if bias is not None:
        out = (out + bias).realize()
    return out


def relu(x: Tensor) -> Tensor:
    """ReLU: max(0, x).  Forces realize so it runs as a separate VPU kernel."""
    return x.relu().realize()


def softmax_int(x: Tensor) -> Tensor:
    """Integer softmax (argmax variant): returns index of max element."""
    return x.max(axis=-1, keepdim=True).realize()


def layer_norm_host(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Layer norm computed on the host (float).  Accepts numpy, returns numpy."""
    mean = x.mean(axis=-1, keepdims=True)
    std  = x.std(axis=-1, keepdims=True) + eps
    return ((x - mean) / std).astype(np.float32)


def mlp_forward(x: Tensor, weights: list[np.ndarray],
                biases: list[np.ndarray | None] | None = None,
                use_relu: bool = True) -> Tensor:
    """
    Multi-layer perceptron forward pass on TinyTPU.

    Args:
        x       : input tensor (already on TINYTPU device)
        weights : list of weight matrices (numpy int32, must be int8-range for MXU)
        biases  : optional list of bias vectors (numpy int32) or None per layer
        use_relu: if True, apply ReLU after each hidden layer (not the last)

    Returns:
        Output tensor (on TINYTPU device, realized).
    """
    if biases is None:
        biases = [None] * len(weights)

    h = x
    for i, (W, b) in enumerate(zip(weights, biases)):
        W_t = Tensor(W, device=x.device)
        b_t = Tensor(b, device=x.device) if b is not None else None
        h = linear(h, W_t, b_t)
        if use_relu and i < len(weights) - 1:
            h = relu(h)
    return h


def linear_relu(x: Tensor, W: Tensor, bias: Tensor | None = None) -> Tensor:
    """Linear layer followed by ReLU: relu(x @ W [+ bias]).

    Forces realize after each step so tinygrad does not fuse the three
    operations (GEMM + bias-add + ReLU) into a single 4-param kernel that
    the TinyTPU backend cannot yet lower.
    """
    return relu(linear(x, W, bias))


def conv1x1(x: Tensor, W: Tensor) -> Tensor:
    """
    1x1 convolution as a matrix multiply (batch, C_in) @ (C_in, C_out).
    Equivalent to a linear layer without bias.
    """
    return (x @ W).realize()


# ---------------------------------------------------------------------------
# Known fusion limitation
# ---------------------------------------------------------------------------
# Tinygrad's lazy evaluation fuses adjacent operations into single kernels.
# The TinyTPU backend recognizes a growing set of kernel shapes, but some
# multi-op fusions are not yet detected:
#
#   NOT YET SUPPORTED (use .realize() to split):
#     (x @ W + b).relu()       -- GEMM + bias + ReLU fused into 4-param kernel
#     (x @ W).relu()           -- GEMM + ReLU fused (unfused batch>1 case)
#
#   SUPPORTED via explicit realize chain:
#     h = (x @ W).realize()
#     h = (h + b).realize()    -- VPU_ROWBC_BINARY
#     h = h.relu().realize()   -- VPU_UNARY RELU
#
# Use linear(), linear_relu(), and mlp_forward() from this module to get
# correct results without worrying about fusion.
