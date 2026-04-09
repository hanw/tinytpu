from __future__ import annotations
import os, sys, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tinygrad"))
os.environ["TINYTPU_SIM"] = str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")

import numpy as np
from tinygrad import Tensor
from tinygrad.runtime.ops_tinytpu import _infer_tiling, _tiling_failure_note


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

  def test_random_signed_deep_and_wide_gemm_matches_numpy(self):
    rng = np.random.default_rng(0)
    a_np = rng.integers(-8, 9, size=(2, 8), dtype=np.int32)
    w_np = rng.integers(-8, 9, size=(8, 8), dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_batched_random_signed_deep_and_wide_gemm_matches_numpy(self):
    rng = np.random.default_rng(1)
    a_np = rng.integers(-8, 9, size=(2, 1, 8), dtype=np.int32)
    w_np = rng.integers(-8, 9, size=(8, 8), dtype=np.int32)
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

  def test_zero_row_gemm_matches_numpy_shape(self):
    a = Tensor.empty(0, 4, dtype="int32", device="TINYTPU")
    w = Tensor(np.eye(4, dtype=np.int32), dtype="int32", device="TINYTPU")
    result = (a @ w).numpy()
    self.assertEqual(result.shape, (0, 4))

  def test_zero_column_gemm_matches_numpy_shape(self):
    a = Tensor.empty(2, 4, dtype="int32", device="TINYTPU")
    w = Tensor.empty(4, 0, dtype="int32", device="TINYTPU")
    result = (a @ w).numpy()
    self.assertEqual(result.shape, (2, 0))

  def test_relu_error_reports_missing_instructions(self):
    with self.assertRaisesRegex(NotImplementedError, "SXU_DISPATCH_VPU"):
      Tensor([[-1, 2, -3, 4]], dtype="int32", device="TINYTPU").relu().numpy()

  def test_relu_error_reports_uop_mix(self):
    with self.assertRaisesRegex(NotImplementedError, "op_counts: .*CMPLT="):
      Tensor([[-1, 2, -3, 4]], dtype="int32", device="TINYTPU").relu().numpy()

  def test_unsupported_width_reports_tiling_constraint(self):
    a_np = np.arange(4, dtype=np.int32).reshape(1, 4)
    w_np = np.arange(24, dtype=np.int32).reshape(4, 6)
    with self.assertRaisesRegex(NotImplementedError, "output size 6 is not divisible by 4"):
      (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()

  def test_zero_k_reports_zero_sized_gemm(self):
    a = Tensor.empty(2, 0, dtype="int32", device="TINYTPU")
    w = Tensor.empty(0, 4, dtype="int32", device="TINYTPU")
    with self.assertRaisesRegex(NotImplementedError, "zero-sized gemm"):
      (a @ w).numpy()


class TestTinyTPUTilingInference(unittest.TestCase):
  def test_infers_single_tile_shape(self):
    self.assertEqual(_infer_tiling(out_size=4, act_size=4, weight_size=16), (1, 1, 1))

  def test_infers_deep_and_wide_shape(self):
    self.assertEqual(_infer_tiling(out_size=16, act_size=16, weight_size=64), (2, 2, 2))

  def test_rejects_non_square_vector_factor(self):
    self.assertIsNone(_infer_tiling(out_size=8, act_size=4, weight_size=64))

  def test_failure_note_reports_divisibility_issue(self):
    note = _tiling_failure_note(out_size=6, act_size=4, weight_size=24)
    self.assertIn("output size 6 is not divisible by 4", note)
    self.assertIn("weight size 24 is not divisible by 16", note)

  def test_failure_note_reports_zero_sized_case(self):
    self.assertIn("zero-sized GEMM", _tiling_failure_note(out_size=8, act_size=0, weight_size=0))


if __name__ == "__main__":
  unittest.main()
