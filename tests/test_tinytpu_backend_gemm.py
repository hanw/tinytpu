from __future__ import annotations
import json, os, stat, subprocess, sys, tempfile, textwrap, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tinygrad"))
os.environ["TINYTPU_SIM"] = str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")

import numpy as np
from tinygrad import Tensor
from tinygrad.runtime.ops_tinytpu import _VPU_BOOL_OPS, _VPU_OPS, _infer_tiling, _parse_sim_output, _parse_vmem_output, _run_gemm_vec, _tiling_failure_note


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

  def test_relu_matches_reference(self):
    result = Tensor([[-1, 2, -3, 4]], dtype="int32", device="TINYTPU").relu().numpy()
    np.testing.assert_array_equal(result, np.array([[0, 2, 0, 4]], dtype=np.int32))

  def test_relu_full_tile_matches_reference(self):
    a_np = np.arange(-8, 8, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np, 0))

  def test_relu_multi_tile_matches_reference(self):
    a_np = np.arange(-16, 16, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np, 0))

  def test_relu_multi_tile_tail_matches_reference(self):
    a_np = np.arange(-8, 25, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np, 0))

  def test_sum4_matches_reference(self):
    result = Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, np.array(10, dtype=np.int32))

  def test_sum16_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a_np.sum())

  def test_sum32_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a_np.sum())

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

  def test_tiny_plus_int_matches_reference(self):
    # Mirrors tinygrad/test/test_tiny.py::TestTiny::test_plus_int.
    result = (
      Tensor([1, 2, 3], dtype="int32", device="TINYTPU") +
      Tensor([4, 5, 6], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([5, 7, 9], dtype=np.int32))

  def test_vpu_add_scalar_const_matches_reference(self):
    result = (Tensor([1, 2, 3], dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, np.array([2, 3, 4], dtype=np.int32))

  def test_vpu_add_negative_scalar_const_matches_reference(self):
    result = (Tensor([1, 2, 3], dtype="int32", device="TINYTPU") + -2).numpy()
    np.testing.assert_array_equal(result, np.array([-1, 0, 1], dtype=np.int32))

  def test_vpu_mul_scalar_const_matches_reference(self):
    result = (Tensor([1, 2, 3], dtype="int32", device="TINYTPU") * 2).numpy()
    np.testing.assert_array_equal(result, np.array([2, 4, 6], dtype=np.int32))

  def test_vpu_mul_negative_scalar_const_matches_reference(self):
    result = (Tensor([1, -2, 3], dtype="int32", device="TINYTPU") * -2).numpy()
    np.testing.assert_array_equal(result, np.array([-2, 4, -6], dtype=np.int32))

  def test_vpu_max_scalar_const_matches_reference(self):
    result = Tensor([-1, 2, -3], dtype="int32", device="TINYTPU").maximum(0).numpy()
    np.testing.assert_array_equal(result, np.array([0, 2, 0], dtype=np.int32))

  def test_vpu_max_negative_scalar_const_matches_reference(self):
    result = Tensor([-3, -1, 2], dtype="int32", device="TINYTPU").maximum(-2).numpy()
    np.testing.assert_array_equal(result, np.array([-2, -1, 2], dtype=np.int32))

  def test_vpu_cmplt_matches_reference(self):
    result = (
      Tensor([1, 4], dtype="int32", device="TINYTPU") <
      Tensor([2, 3], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([True, False], dtype=np.bool_))

  def test_vpu_cmpne_matches_reference(self):
    result = (
      Tensor([1, 2, 3], dtype="int32", device="TINYTPU") !=
      Tensor([1, 0, 3], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([False, True, False], dtype=np.bool_))

  def test_vpu_cmpeq_matches_reference(self):
    result = (
      Tensor([1, 2, 3], dtype="int32", device="TINYTPU") ==
      Tensor([1, 0, 3], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([True, False, True], dtype=np.bool_))

  def test_vpu_cmplt_scalar_const_matches_reference(self):
    result = (Tensor([1, 2, 3], dtype="int32", device="TINYTPU") < 3).numpy()
    np.testing.assert_array_equal(result, np.array([True, True, False], dtype=np.bool_))

  def test_vpu_cmpne_scalar_const_matches_reference(self):
    result = (Tensor([1, 2, 3], dtype="int32", device="TINYTPU") != 2).numpy()
    np.testing.assert_array_equal(result, np.array([True, False, True], dtype=np.bool_))

  def test_vpu_cmpeq_scalar_const_matches_reference(self):
    result = (Tensor([1, 2, 3], dtype="int32", device="TINYTPU") == 2).numpy()
    np.testing.assert_array_equal(result, np.array([False, True, False], dtype=np.bool_))

  def test_tiny_mul_int_matches_reference(self):
    result = (
      Tensor([1, 2, 3], dtype="int32", device="TINYTPU") *
      Tensor([4, 5, 6], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([4, 10, 18], dtype=np.int32))

  def test_tiny_sub_int_matches_reference(self):
    result = (
      Tensor([5, 7, -2], dtype="int32", device="TINYTPU") -
      Tensor([2, 3, 4], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([3, 4, -6], dtype=np.int32))

  def test_vpu_sub_scalar_const_matches_reference(self):
    result = (Tensor([5, 7, -2], dtype="int32", device="TINYTPU") - 1).numpy()
    np.testing.assert_array_equal(result, np.array([4, 6, -3], dtype=np.int32))

  def test_vpu_reverse_sub_scalar_const_matches_reference(self):
    result = (1 - Tensor([5, 7, -2], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([-4, -6, 3], dtype=np.int32))

  def test_tiny_neg_int_matches_reference(self):
    result = (-Tensor([1, -2, 3], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([-1, 2, -3], dtype=np.int32))

  def test_tiny_max_int_matches_reference(self):
    result = Tensor([1, 7, 3], dtype="int32", device="TINYTPU").maximum(
      Tensor([4, 5, 6], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([4, 7, 6], dtype=np.int32))

  def test_vpu_add_full_tile_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    b_np = np.arange(100, 116, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") + Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np + b_np)

  def test_vpu_mul_full_tile_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    b_np = np.arange(1, 17, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") * Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np * b_np)

  def test_vpu_max_full_tile_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    b_np = np.arange(15, -1, -1, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").maximum(Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np, b_np))

  def test_vpu_binary_signed_full_tile_matches_reference(self):
    a_np = np.arange(-8, 8, dtype=np.int32)
    b_np = np.arange(8, -8, -1, dtype=np.int32)
    add = (Tensor(a_np, dtype="int32", device="TINYTPU") + Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    mul = (Tensor(a_np, dtype="int32", device="TINYTPU") * Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    sub = (Tensor(a_np, dtype="int32", device="TINYTPU") - Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    mx = Tensor(a_np, dtype="int32", device="TINYTPU").maximum(Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(add, a_np + b_np)
    np.testing.assert_array_equal(mul, a_np * b_np)
    np.testing.assert_array_equal(sub, a_np - b_np)
    np.testing.assert_array_equal(mx, np.maximum(a_np, b_np))

  def test_vpu_compare_full_tile_matches_reference(self):
    a_np = np.arange(-8, 8, dtype=np.int32)
    b_np = np.arange(8, -8, -1, dtype=np.int32)
    lt = (Tensor(a_np, dtype="int32", device="TINYTPU") < Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    ne = (Tensor(a_np, dtype="int32", device="TINYTPU") != Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    eq = (Tensor(a_np, dtype="int32", device="TINYTPU") == Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(lt, a_np < b_np)
    np.testing.assert_array_equal(ne, a_np != b_np)
    np.testing.assert_array_equal(eq, a_np == b_np)

  def test_vpu_scalar_compare_full_tile_matches_reference(self):
    a_np = np.arange(-8, 8, dtype=np.int32)
    lt = (Tensor(a_np, dtype="int32", device="TINYTPU") < 0).numpy()
    ne = (Tensor(a_np, dtype="int32", device="TINYTPU") != 0).numpy()
    eq = (Tensor(a_np, dtype="int32", device="TINYTPU") == 0).numpy()
    np.testing.assert_array_equal(lt, a_np < 0)
    np.testing.assert_array_equal(ne, a_np != 0)
    np.testing.assert_array_equal(eq, a_np == 0)

  def test_vpu_shape_preserved_for_small_matrix_add(self):
    a_np = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b_np = np.array([[5, 6], [7, 8]], dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") + Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np + b_np)
    self.assertEqual(result.shape, (2, 2))

  def test_vpu_add_multi_tile_32_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(100, 132, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") + Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np + b_np)

  def test_vpu_add_multi_tile_tail_matches_reference(self):
    a_np = np.arange(17, dtype=np.int32)
    b_np = np.arange(100, 117, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") + Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np + b_np)

  def test_vpu_mul_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(1, 33, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") * Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np * b_np)

  def test_vpu_sub_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(100, 132, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") - Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np - b_np)

  def test_vpu_max_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(31, -1, -1, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").maximum(Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np, b_np))

  def test_vpu_cmplt_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(31, -1, -1, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") < Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np < b_np)

  def test_vpu_cmpne_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(31, -1, -1, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") != Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np != b_np)

  def test_vpu_cmpeq_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(31, -1, -1, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") == Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np == b_np)

  def test_vpu_add_scalar_const_multi_tile_matches_reference(self):
    a_np = np.arange(17, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a_np + 5)

  def test_vpu_sub_scalar_const_multi_tile_matches_reference(self):
    a_np = np.arange(17, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") - 1).numpy()
    np.testing.assert_array_equal(result, a_np - 1)

  def test_vpu_cmplt_scalar_const_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") < 16).numpy()
    np.testing.assert_array_equal(result, a_np < 16)

  def test_vpu_cmpeq_scalar_const_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") == 10).numpy()
    np.testing.assert_array_equal(result, a_np == 10)

  def test_vpu_reverse_sub_const_multi_tile_matches_reference(self):
    a_np = np.arange(17, dtype=np.int32)
    result = (10 - Tensor(a_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, 10 - a_np)

  def test_vpu_reverse_sub_const_multi_tile_32_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = (10 - Tensor(a_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, 10 - a_np)

  def test_where_matches_reference(self):
    cond = Tensor([True, False, True, False], device="TINYTPU")
    lhs = Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU")
    rhs = Tensor([5, 6, 7, 8], dtype="int32", device="TINYTPU")
    result = Tensor.where(cond, lhs, rhs).numpy()
    np.testing.assert_array_equal(result, np.array([1, 6, 3, 8], dtype=np.int32))

  def test_where_full_tile_matches_reference(self):
    cond_np = np.array([i % 2 == 0 for i in range(16)], dtype=np.bool_)
    lhs_np = np.arange(16, dtype=np.int32)
    rhs_np = np.arange(100, 116, dtype=np.int32)
    result = Tensor.where(Tensor(cond_np, device="TINYTPU"),
                          Tensor(lhs_np, dtype="int32", device="TINYTPU"),
                          Tensor(rhs_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.where(cond_np, lhs_np, rhs_np))

  def test_where_multi_tile_matches_reference(self):
    cond_np = np.array([i % 3 == 0 for i in range(32)], dtype=np.bool_)
    lhs_np = np.arange(32, dtype=np.int32)
    rhs_np = np.arange(100, 132, dtype=np.int32)
    result = Tensor.where(Tensor(cond_np, device="TINYTPU"),
                          Tensor(lhs_np, dtype="int32", device="TINYTPU"),
                          Tensor(rhs_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.where(cond_np, lhs_np, rhs_np))

  def test_and_matches_reference(self):
    a = Tensor([True, False, True, True], device="TINYTPU")
    b = Tensor([True, True, False, True], device="TINYTPU")
    result = (a & b).numpy()
    np.testing.assert_array_equal(result, np.array([True, False, False, True], dtype=np.bool_))

  def test_xor_matches_reference(self):
    a = Tensor([True, False, True, False], device="TINYTPU")
    b = Tensor([True, True, False, False], device="TINYTPU")
    result = (a ^ b).numpy()
    np.testing.assert_array_equal(result, np.array([False, True, True, False], dtype=np.bool_))

  def test_xor_multi_tile_matches_reference(self):
    a = np.array([True, False, True, False]*8, dtype=np.bool_)
    b = np.array([True, True, False, False]*8, dtype=np.bool_)
    result = (Tensor(a, device="TINYTPU") ^ Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a ^ b)

  def test_and_multi_tile_matches_reference(self):
    a = np.array([True, False]*16, dtype=np.bool_)
    b = np.array([True, True, False, False]*8, dtype=np.bool_)
    result = (Tensor(a, device="TINYTPU") & Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a & b)

  def test_or_matches_reference(self):
    a = Tensor([True, False, True, False], device="TINYTPU")
    b = Tensor([False, False, True, True], device="TINYTPU")
    result = (a | b).numpy()
    np.testing.assert_array_equal(result, np.array([True, False, True, True], dtype=np.bool_))

  def test_or_multi_tile_matches_reference(self):
    a = np.array([True, False]*16, dtype=np.bool_)
    b = np.array([False, False, True, True]*8, dtype=np.bool_)
    result = (Tensor(a, device="TINYTPU") | Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a | b)

  def test_not_matches_reference(self):
    result = (~Tensor([True, False, True, False], device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([False, True, False, True], dtype=np.bool_))

  def test_not_multi_tile_matches_reference(self):
    a = np.array([True, False]*16, dtype=np.bool_)
    result = (~Tensor(a, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, ~a)

  def test_permute_reports_movement_lowering_gap(self):
    with self.assertRaisesRegex(NotImplementedError, "General VMEM<->VReg movement kernels are not lowered yet"):
      Tensor([[1, 2], [3, 4]], dtype="int32", device="TINYTPU").permute(1, 0).numpy()

  def test_reshape_reports_movement_lowering_gap(self):
    with self.assertRaisesRegex(NotImplementedError, "General VMEM<->VReg movement kernels are not lowered yet"):
      Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU").reshape(2, 2).numpy()

  def test_vpu_opcode_table_marks_bool_results(self):
    self.assertEqual(_VPU_OPS["CMPEQ"], 8)
    self.assertEqual(_VPU_BOOL_OPS, {_VPU_OPS["CMPLT"], _VPU_OPS["CMPNE"], _VPU_OPS["CMPEQ"]})

  def test_lowering_dump_records_scalar_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((Tensor([1, 2, 3], dtype="int32", device="TINYTPU") + 1).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") == "VPU_BINARY" and r.get("vpu_op") == _VPU_OPS["ADD"] and r.get("rhs_const") == 1 for r in records))

  def test_lowering_dump_records_equality_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((Tensor([1, 2, 3], dtype="int32", device="TINYTPU") == Tensor([1, 0, 3], dtype="int32", device="TINYTPU")).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") == "VPU_BINARY" and r.get("vpu_op") == _VPU_OPS["CMPEQ"] for r in records))


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


class TestTinyTPUSimOutputParsing(unittest.TestCase):
  def test_parses_indented_mxu_result(self):
    self.assertEqual(_parse_sim_output("  mxu_result 1 -2 3 -4\nstatus ok\n"), [1, -2, 3, -4])

  def test_reports_bad_mxu_result_integer(self):
    with self.assertRaisesRegex(ValueError, "invalid mxu_result integer 'bad'"):
      _parse_sim_output("mxu_result 1 bad 3 4\nstatus ok\n")

  def test_rejects_wrong_mxu_result_width(self):
    with self.assertRaisesRegex(ValueError, "mxu_result expects 4 values, got 3"):
      _parse_sim_output("mxu_result 1 2 3\nstatus ok\n")

  def test_parses_vmem_result(self):
    result = _parse_vmem_output("vmem_result 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15\nstatus ok\n")
    self.assertEqual(result, list(range(16)))

  def test_rejects_wrong_vmem_result_width(self):
    with self.assertRaisesRegex(ValueError, "vmem_result expects 16 values, got 3"):
      _parse_vmem_output("vmem_result 1 2 3\nstatus ok\n")

  def test_reports_bad_vmem_result_integer(self):
    with self.assertRaisesRegex(ValueError, "invalid vmem_result integer 'bad'"):
      _parse_vmem_output("vmem_result 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 bad\nstatus ok\n")

  def test_run_gemm_rejects_sim_error_line(self):
    with tempfile.TemporaryDirectory() as td:
      sim = Path(td) / "fake_sim.py"
      sim.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        print("ERROR: injected failure")
        print("mxu_result 1 2 3 4")
        print("status ok")
      """), encoding="utf-8")
      sim.chmod(sim.stat().st_mode | stat.S_IEXEC)
      with self.assertRaisesRegex(RuntimeError, "simulator reported failure: ERROR: injected failure"):
        _run_gemm_vec(str(sim), np.eye(4, dtype=np.int8), np.arange(4, dtype=np.int8))

  def test_run_gemm_requires_ok_status(self):
    with tempfile.TemporaryDirectory() as td:
      sim = Path(td) / "fake_sim.py"
      sim.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        print("mxu_result 1 2 3 4")
        print("status busy")
      """), encoding="utf-8")
      sim.chmod(sim.stat().st_mode | stat.S_IEXEC)
      with self.assertRaisesRegex(RuntimeError, "simulator did not report `status ok`"):
        _run_gemm_vec(str(sim), np.eye(4, dtype=np.int8), np.arange(4, dtype=np.int8))


if __name__ == "__main__":
  unittest.main()
