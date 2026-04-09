#!/usr/bin/env python3.12
"""
TinyTPU co-simulation end-to-end test.

Runs a 4x4 identity GEMM on the TinyTPU BSV simulator via the tinygrad
TINYTPU device and checks the result against numpy.

Usage:
    cd <repo_root>
    python3.12 scripts/test_cosim.py

Requires:
    - make runtime-tb   (builds build/mkTbTinyTPURuntime.bexe)
    - tinygrad submodule checked out
"""

import sys, os

# Make the local tinygrad submodule importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TINYGRAD_DIR = os.path.join(REPO_ROOT, "tinygrad")
sys.path.insert(0, TINYGRAD_DIR)

# Point TINYTPU_SIM at the built simulator
os.environ["TINYTPU_SIM"] = os.path.join(REPO_ROOT, "build", "mkTbTinyTPURuntime.bexe")

import numpy as np
from tinygrad import Tensor

def test_identity_gemm():
    """identity weight: output == input"""
    a = Tensor([[1, 2, 3, 4]], dtype='int32', device='TINYTPU')
    w = Tensor([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype='int32', device='TINYTPU')
    result = a.matmul(w).numpy()
    expected = np.array([[1, 2, 3, 4]], dtype=np.int32)
    assert (result == expected).all(), f"FAIL identity: got {result}"
    print("PASS  identity GEMM:", result.flatten().tolist())

def test_scale_gemm():
    """2x scale on diagonal: output == 2 * input"""
    a = Tensor([[3, 1, 4, 1]], dtype='int32', device='TINYTPU')
    w = Tensor([[2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 2]], dtype='int32', device='TINYTPU')
    result = a.matmul(w).numpy()
    expected = np.array([[6, 2, 8, 2]], dtype=np.int32)
    assert (result == expected).all(), f"FAIL scale: got {result}"
    print("PASS  scale GEMM:   ", result.flatten().tolist())

def test_permute_gemm():
    """permutation matrix: reverses the vector"""
    a = Tensor([[1, 2, 3, 4]], dtype='int32', device='TINYTPU')
    w = Tensor([[0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0]], dtype='int32', device='TINYTPU')
    result = a.matmul(w).numpy()
    expected = np.array([[4, 3, 2, 1]], dtype=np.int32)
    assert (result == expected).all(), f"FAIL permute: got {result}"
    print("PASS  permute GEMM: ", result.flatten().tolist())

def test_numpy_reference():
    """Cross-check: verify each test against numpy on CPU."""
    tests = [
        ([[1,2,3,4]], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]),
        ([[3,1,4,1]], [[2,0,0,0],[0,2,0,0],[0,0,2,0],[0,0,0,2]]),
        ([[1,2,3,4]], [[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]]),
    ]
    for a_vals, w_vals in tests:
        a_np = np.array(a_vals, dtype=np.int32)
        w_np = np.array(w_vals, dtype=np.int32)
        a_t = Tensor(a_vals, dtype='int32', device='TINYTPU')
        w_t = Tensor(w_vals, dtype='int32', device='TINYTPU')
        tinytpu_result = a_t.matmul(w_t).numpy()
        numpy_result   = a_np @ w_np
        assert (tinytpu_result == numpy_result).all(), \
            f"FAIL mismatch: TinyTPU={tinytpu_result} numpy={numpy_result}"
    print("PASS  numpy cross-check (3 cases)")

if __name__ == "__main__":
    print(f"Simulator: {os.environ['TINYTPU_SIM']}")
    if not os.path.exists(os.environ["TINYTPU_SIM"]):
        print("ERROR: simulator not found — run  make runtime-tb  first")
        sys.exit(1)

    test_identity_gemm()
    test_scale_gemm()
    test_permute_gemm()
    test_numpy_reference()
    print("\nAll tests passed.")
