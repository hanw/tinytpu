from __future__ import annotations
import os, sys, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tinygrad"))
os.environ["TINYTPU_SIM"] = str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")

import numpy as np
from tinygrad import Tensor


@unittest.skipUnless((REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe").exists(), "runtime binary not built")
class TestTinyTPUBackendGemm(unittest.TestCase):
  def test_multi_row_gemm_matches_numpy(self):
    a_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    w_np = np.eye(4, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_batched_gemm_matches_numpy(self):
    a_np = np.array([[[1, 2, 3, 4]], [[5, 6, 7, 8]]], dtype=np.int32)
    w_np = np.array([[2, 0, 0, 0], [0, 2, 0, 0], [0, 0, 2, 0], [0, 0, 0, 2]], dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_wide_output_gemm_matches_numpy(self):
    a_np = np.array([[1, 2, 3, 4]], dtype=np.int32)
    w_np = np.arange(32, dtype=np.int32).reshape(4, 8)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_multi_row_wide_output_gemm_matches_numpy(self):
    a_np = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    w_np = np.arange(32, dtype=np.int32).reshape(4, 8)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_deep_k_gemm_matches_numpy(self):
    a_np = np.arange(8, dtype=np.int32).reshape(1, 8)
    w_np = np.arange(32, dtype=np.int32).reshape(8, 4)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_deep_and_wide_gemm_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(2, 8)
    w_np = np.arange(64, dtype=np.int32).reshape(8, 8)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_batched_deep_and_wide_gemm_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(2, 1, 8)
    w_np = np.arange(64, dtype=np.int32).reshape(8, 8)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_activation_out_of_int8_range_raises(self):
    a_np = np.array([[128, 0, 0, 0]], dtype=np.int32)
    w_np = np.eye(4, dtype=np.int32)
    with self.assertRaisesRegex(ValueError, "activation values must fit in signed int8"):
      (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()

  def test_weight_out_of_int8_range_raises(self):
    a_np = np.array([[1, 2, 3, 4]], dtype=np.int32)
    w_np = np.eye(4, dtype=np.int32) * 256
    with self.assertRaisesRegex(ValueError, "weight values must fit in signed int8"):
      (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()

  def test_relu_error_reports_missing_instructions(self):
    with self.assertRaisesRegex(NotImplementedError, "SXU_DISPATCH_VPU"):
      Tensor([[-1, 2, -3, 4]], dtype="int32", device="TINYTPU").relu().numpy()


if __name__ == "__main__":
  unittest.main()
