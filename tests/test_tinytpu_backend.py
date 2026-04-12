from __future__ import annotations
import json, os, stat, subprocess, sys, tempfile, textwrap, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tinygrad"))
os.environ["TINYTPU_SIM"] = str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")
os.environ["DISABLE_COMPILER_CACHE"] = "1"

import numpy as np
from tinygrad import Tensor
from tinygrad.runtime.ops_tinytpu import _VPU_BOOL_OPS, _VPU_OPS, _infer_tiling, _parse_sim_output, _parse_vmem_output, _tiling_failure_note, _run_bundle, _build_full_gemm_bundle


@unittest.skipUnless((REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe").exists(), "runtime binary not built")
class TestTinyTPUBackend(unittest.TestCase):
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

  def test_fused_gemm_bias_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(4, 4)
    w_np = np.arange(16, dtype=np.int32).reshape(4, 4)
    b_np = np.array([1, 2, 3, 4], dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @
              Tensor(w_np, dtype="int32", device="TINYTPU") +
              Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np + b_np)

  def test_fused_gemm_relu_matches_numpy(self):
    a_np = np.array([[1, -2, 3, -4],
                     [5, -6, 7, -8],
                     [1, 1, 1, 1],
                     [-2, -2, -2, -2]], dtype=np.int32)
    w_np = np.array([[1, 2, -3, 4],
                     [5, -6, 7, -8],
                     [9, 10, -11, 12],
                     [13, -14, 15, -16]], dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np @ w_np, 0))

  def test_fused_gemm_bias_relu_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(4, 4) - 6
    w_np = np.arange(16, dtype=np.int32).reshape(4, 4) - 5
    b_np = np.array([1, -20, 3, -40], dtype=np.int32)
    result = ((Tensor(a_np, dtype="int32", device="TINYTPU") @
               Tensor(w_np, dtype="int32", device="TINYTPU")) +
              Tensor(b_np, dtype="int32", device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np @ w_np + b_np, 0))

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

  def test_min4_matches_reference(self):
    result = Tensor([3, 7, 1, 5], dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, np.array(1, dtype=np.int32))

  def test_min16_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, a_np.min())

  def test_min32_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, a_np.min())

  def test_max4_matches_reference(self):
    result = Tensor([3, 7, 1, 5], dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, np.array(7, dtype=np.int32))

  def test_max16_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a_np.max())

  def test_max32_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a_np.max())

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

  def test_sum_negative_matches_reference(self):
    a_np = np.array([-3, 5, -7, 2], dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a_np.sum())

  def test_max_negative_matches_reference(self):
    a_np = np.array([-5, -1, -8, -3], dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a_np.max())

  def test_min_negative_matches_reference(self):
    a_np = np.array([-5, -1, -8, -3], dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, a_np.min())

  def test_unsupported_width_reports_tiling_constraint(self):
    a_np = np.arange(4, dtype=np.int32).reshape(1, 4)
    w_np = np.arange(24, dtype=np.int32).reshape(4, 6)
    with self.assertRaises(NotImplementedError):
      (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()

  def test_zero_k_reports_zero_sized_gemm(self):
    a = Tensor.empty(2, 0, dtype="int32", device="TINYTPU")
    w = Tensor.empty(0, 4, dtype="int32", device="TINYTPU")
    with self.assertRaises(NotImplementedError):
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

  def test_fadd_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fmul_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_fsub_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

  def test_fsub_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 0.5).numpy()
    np.testing.assert_allclose(result, a - 0.5, rtol=1e-5)

  def test_frev_sub_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (1.5 - Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 1.5 - a, rtol=1e-5)

  def test_fneg_matches_reference(self):
    a = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fneg_full_tile_matches_reference(self):
    a = (np.arange(16, dtype=np.float32) - 8)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fneg_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fadd_signed_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    b = (16 - np.arange(32, dtype=np.float32)).astype(np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_signed_matches_reference(self):
    a = np.array([-1.5, 2.0, -3.0, 4.5], dtype=np.float32)
    b = np.array([0.5, -0.5, 1.0, -2.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fmul_signed_matches_reference(self):
    a = np.array([-1.5, 2.0, -3.0, 4.5], dtype=np.float32)
    b = np.array([2.0, -3.0, 0.5, -1.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_frecip_matches_reference(self):
    a = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_fdiv_multi_tile_matches_reference(self):
    a = np.arange(1, 33, dtype=np.float32)
    b = np.full(32, 2.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-3)

  def test_frecip_multi_tile_matches_reference(self):
    a = np.arange(1, 33, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_fmaximum_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(0.0).numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_fminimum_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(0.0).numpy()
    np.testing.assert_allclose(result, np.minimum(a, 0.0), rtol=1e-5)

  def test_fmaximum_negative_scalar_const_matches_reference(self):
    a = np.array([-5.0, -1.0, 2.0, -3.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(-2.0).numpy()
    np.testing.assert_allclose(result, np.maximum(a, -2.0), rtol=1e-5)

  def test_fminimum_negative_scalar_const_matches_reference(self):
    a = np.array([-5.0, -1.0, 2.0, -3.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(-2.0).numpy()
    np.testing.assert_allclose(result, np.minimum(a, -2.0), rtol=1e-5)

  def test_fminimum_scalar_const_matches_reference(self):
    a = np.array([1.0, -2.0, 3.0, 5.0, -1.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(2.5).numpy()
    np.testing.assert_allclose(result, np.minimum(a, 2.5), rtol=1e-5)

  def test_fwhere_matches_reference(self):
    cond = np.array([True, False, True, False])
    lhs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    rhs = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    result = Tensor.where(Tensor(cond, device="TINYTPU"),
                          Tensor(lhs, dtype="float", device="TINYTPU"),
                          Tensor(rhs, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.where(cond, lhs, rhs), rtol=1e-5)

  def test_frelu_full_tile_matches_reference(self):
    a = (np.arange(16, dtype=np.float32) - 8) * 0.5
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_multi_tile_matches_reference(self):
    a = (np.arange(32, dtype=np.float32) - 16) * 0.5
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_matches_reference(self):
    a = np.array([-1.5, 2.0, -3.0, 4.5, -0.0, 0.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_fminimum_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16
    b = 16 - np.arange(32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.minimum(a, b), rtol=1e-5)

  def test_fmaximum_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16
    b = 16 - np.arange(32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.maximum(a, b), rtol=1e-5)

  def test_fminimum_matches_reference(self):
    a = np.array([1.0, 5.0, 3.0, -2.0], dtype=np.float32)
    b = np.array([4.0, 2.0, 6.0, -5.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.minimum(a, b), rtol=1e-5)

  def test_int_to_float_cast_matches_reference(self):
    result = Tensor([1, 2, 3, -5], dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0, -5.0], dtype=np.float32), rtol=1e-5)

  def test_float_to_int_cast_matches_reference(self):
    result = Tensor([1.5, 2.7, -3.2, 0.0], dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, np.array([1, 2, -3, 0], dtype=np.int32))

  def test_int_float_int_roundtrip_matches_reference(self):
    a = np.array([1, 2, 3, -5, 10], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").cast("int32").numpy()
    np.testing.assert_array_equal(result, a)

  def test_int_to_float_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, a.astype(np.float32), rtol=1e-5)

  def test_float_to_int_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16
    result = Tensor(a, dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_float_sum_reduce_reports_unsupported(self):
    with self.assertRaises(NotImplementedError):
      Tensor([1.0, 2.0, 3.0, 4.0], dtype="float", device="TINYTPU").sum().numpy()

  def test_float_max_reduce_reports_unsupported(self):
    with self.assertRaises(NotImplementedError):
      Tensor([1.0, 2.0, 3.0, 4.0], dtype="float", device="TINYTPU").max().numpy()

  def test_float_min_reduce_reports_unsupported(self):
    with self.assertRaises(NotImplementedError):
      Tensor([1.0, 2.0, 3.0, 4.0], dtype="float", device="TINYTPU").min().numpy()

  def test_sqrt_reports_unsupported(self):
    with self.assertRaises(NotImplementedError):
      Tensor([4.0, 9.0, 16.0], dtype="float", device="TINYTPU").sqrt().numpy()

  def test_log2_reports_unsupported(self):
    with self.assertRaises(NotImplementedError):
      Tensor([2.0, 4.0, 8.0], dtype="float", device="TINYTPU").log2().numpy()

  def test_exp2_reports_unsupported(self):
    with self.assertRaises(NotImplementedError):
      Tensor([1.0, 2.0, 3.0], dtype="float", device="TINYTPU").exp2().numpy()

  def test_sin_reports_unsupported(self):
    with self.assertRaises(NotImplementedError):
      Tensor([0.0, 1.0, 2.0], dtype="float", device="TINYTPU").sin().numpy()

  def test_fdiv_tensor_tensor_matches_reference(self):
    a = np.array([4.0, 6.0, 8.0, 10.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 4.0, 5.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-3)

  def test_fdiv_scalar_const_matches_reference(self):
    a = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / 2.0).numpy()
    np.testing.assert_allclose(result, a / 2.0, rtol=1e-3)

  def test_fdiv_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(1, 33, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / 4.0).numpy()
    np.testing.assert_allclose(result, a / 4.0, rtol=1e-3)

  def test_fadd_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) * 0.5
    b = np.arange(32, dtype=np.float32) * 0.25
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fmul_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) * 0.5
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_fadd_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 2.5).numpy()
    np.testing.assert_allclose(result, a + 2.5, rtol=1e-5)

  def test_fmul_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 3.0).numpy()
    np.testing.assert_allclose(result, a * 3.0, rtol=1e-5)

  def test_fcmpne_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") != Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a != b)

  def test_fcmpeq_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a == b)

  def test_fcmpgt_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a > b)

  def test_fcmpgt_scalar_const_matches_reference(self):
    a = np.array([-1.0, 0.5, 2.0, 3.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > 1.0).numpy()
    np.testing.assert_array_equal(result, a > 1.0)

  def test_fcmpne_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") != 3.0).numpy()
    np.testing.assert_array_equal(result, a != 3.0)

  def test_fcmplt_scalar_const_matches_reference(self):
    a = np.array([-1.0, 0.5, 2.0, 3.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 2.0).numpy()
    np.testing.assert_array_equal(result, a < 2.0)

  def test_fcmplt_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 0.0).numpy()
    np.testing.assert_array_equal(result, a < 0.0)

  def test_fcmplt_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0, 2.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a < b)

  def test_fmax_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([1.5, 1.5, 1.5], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.maximum(a, b), rtol=1e-5)

  def test_fwhere_multi_tile_matches_reference(self):
    cond = np.array([True, False] * 16)
    lhs = np.arange(32, dtype=np.float32)
    rhs = np.arange(32, dtype=np.float32) + 100.0
    result = Tensor.where(Tensor(cond, device="TINYTPU"),
                          Tensor(lhs, dtype="float", device="TINYTPU"),
                          Tensor(rhs, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.where(cond, lhs, rhs), rtol=1e-5)

  def test_fmul_signed_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    b = np.arange(16, -16, -1, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_fsub_tensor_tensor_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16
    b = 16 - np.arange(32, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

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

  def test_bitwise_not_int32_matches_reference(self):
    a_np = np.array([1, 2, 3, 4], dtype=np.int32)
    result = (~Tensor(a_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, ~a_np)

  def test_bitwise_not_int32_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = (~Tensor(a_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, ~a_np)

  def test_minimum_matches_reference(self):
    result = Tensor([1, 5, 3], dtype="int32", device="TINYTPU").minimum(
      Tensor([4, 2, 6], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.int32))

  def test_minimum_signed_matches_reference(self):
    a_np = np.array([-8, -3, 5, -1], dtype=np.int32)
    b_np = np.array([-2, -5, 3, 7], dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").minimum(Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.minimum(a_np, b_np))

  def test_minimum_scalar_const_matches_reference(self):
    result = Tensor([1, 5, 3, 10], dtype="int32", device="TINYTPU").minimum(5).numpy()
    np.testing.assert_array_equal(result, np.array([1, 5, 3, 5], dtype=np.int32))

  def test_maximum_scalar_const_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").maximum(10).numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np, 10))

  def test_minimum_negative_scalar_const_matches_reference(self):
    result = Tensor([-5, -1, -8, -3], dtype="int32", device="TINYTPU").minimum(-2).numpy()
    np.testing.assert_array_equal(result, np.array([-5, -2, -8, -3], dtype=np.int32))

  def test_minimum_scalar_const_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").minimum(10).numpy()
    np.testing.assert_array_equal(result, np.minimum(a_np, 10))

  def test_minimum_full_tile_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    b_np = np.arange(15, -1, -1, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").minimum(Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.minimum(a_np, b_np))

  def test_minimum_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    b_np = np.arange(31, -1, -1, dtype=np.int32)
    result = Tensor(a_np, dtype="int32", device="TINYTPU").minimum(Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.minimum(a_np, b_np))

  def test_shl_scalar_const_matches_reference(self):
    result = (Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU") << 2).numpy()
    np.testing.assert_array_equal(result, np.array([4, 8, 12, 16], dtype=np.int32))

  def test_shl_full_tile_matches_reference(self):
    a_np = np.arange(16, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") << 3).numpy()
    np.testing.assert_array_equal(result, a_np << 3)

  def test_shl_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") << 1).numpy()
    np.testing.assert_array_equal(result, a_np << 1)

  def test_shr_full_tile_matches_reference(self):
    a_np = np.arange(16, 32, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") >> 1).numpy()
    np.testing.assert_array_equal(result, a_np >> 1)

  def test_shr_scalar_const_matches_reference(self):
    result = (Tensor([16, 32, 48, 64], dtype="int32", device="TINYTPU") >> 2).numpy()
    np.testing.assert_array_equal(result, np.array([4, 8, 12, 16], dtype=np.int32))

  def test_shr_multi_tile_matches_reference(self):
    a_np = np.arange(32, 64, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") >> 2).numpy()
    np.testing.assert_array_equal(result, a_np >> 2)

  def test_vpu_cmpne_scalar_const_multi_tile_matches_reference(self):
    a_np = np.arange(32, dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") != 10).numpy()
    np.testing.assert_array_equal(result, a_np != 10)

  def test_where_multi_tile_tail_matches_reference(self):
    cond_np = np.array([i % 2 == 0 for i in range(17)], dtype=np.bool_)
    lhs_np = np.arange(17, dtype=np.int32)
    rhs_np = np.arange(100, 117, dtype=np.int32)
    result = Tensor.where(Tensor(cond_np, device="TINYTPU"),
                          Tensor(lhs_np, dtype="int32", device="TINYTPU"),
                          Tensor(rhs_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.where(cond_np, lhs_np, rhs_np))

  def test_fused_add_relu_matches_reference(self):
    result = (Tensor([-3, 1, -1, 5], dtype="int32", device="TINYTPU") + Tensor([1, 1, 1, 1], dtype="int32", device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.array([0, 2, 0, 6], dtype=np.int32))

  def test_idiv_scalar_matches_reference(self):
    result = (Tensor([10, 20, 30, 40], dtype="int32", device="TINYTPU") // 3).numpy()
    np.testing.assert_array_equal(result, np.array([3, 6, 10, 13], dtype=np.int32))

  def test_idiv_negative_matches_reference(self):
    result = (Tensor([-10, -9, 9, 10], dtype="int32", device="TINYTPU") // 3).numpy()
    np.testing.assert_array_equal(result, np.array([-3, -3, 3, 3], dtype=np.int32))

  def test_mod_scalar_matches_reference(self):
    result = (Tensor([10, 20, 30, 40], dtype="int32", device="TINYTPU") % 3).numpy()
    np.testing.assert_array_equal(result, np.array([1, 2, 0, 1], dtype=np.int32))

  def test_mod_negative_matches_reference(self):
    result = (Tensor([-10, -9, 9, 10], dtype="int32", device="TINYTPU") % 3).numpy()
    np.testing.assert_array_equal(result, np.array([-1, 0, 0, 1], dtype=np.int32))

  def test_idiv_full_tile_matches_reference(self):
    data = list(range(3, 19))
    result = (Tensor(data, dtype="int32", device="TINYTPU") // 3).numpy()
    np.testing.assert_array_equal(result, np.array(data, dtype=np.int32) // 3)

  def test_idiv_multi_tile_matches_reference(self):
    data = list(range(3, 35))
    result = (Tensor(data, dtype="int32", device="TINYTPU") // 3).numpy()
    np.testing.assert_array_equal(result, np.array(data, dtype=np.int32) // 3)

  def test_idiv_negative_multi_tile_matches_reference(self):
    # TinyTPU VPU_DIV truncates toward zero (C-style), not floor (Python-style).
    # Build expected values using truncation so the test matches hardware semantics.
    import math
    data = list(range(-16, 16))
    result = (Tensor(data, dtype="int32", device="TINYTPU") // 3).numpy()
    expected = np.array([math.trunc(x / 3) for x in data], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)

  def test_mod_full_tile_matches_reference(self):
    data = list(range(3, 19))
    result = (Tensor(data, dtype="int32", device="TINYTPU") % 7).numpy()
    np.testing.assert_array_equal(result, np.array(data, dtype=np.int32) % 7)

  def test_mod_multi_tile_matches_reference(self):
    data = list(range(3, 35))
    result = (Tensor(data, dtype="int32", device="TINYTPU") % 7).numpy()
    np.testing.assert_array_equal(result, np.array(data, dtype=np.int32) % 7)

  def test_mod_negative_multi_tile_matches_reference(self):
    # TinyTPU MOD uses truncation-based remainder: a % b = a - b*(trunc(a/b)).
    import math
    data = list(range(-16, 16))
    result = (Tensor(data, dtype="int32", device="TINYTPU") % 3).numpy()
    expected = np.array([x - 3 * math.trunc(x / 3) for x in data], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)

  def test_idiv_tensor_tensor_matches_reference(self):
    import math
    a = np.array([10, 21, 35, 42], dtype=np.int32)
    b = np.array([3, 7, 5, 6], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") // Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    expected = np.array([math.trunc(x / y) for x, y in zip(a, b)], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)

  def test_idiv_tensor_tensor_multi_tile_matches_reference(self):
    import math
    a = np.arange(3, 35, dtype=np.int32)
    b = np.arange(2, 34, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") // Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    expected = np.array([math.trunc(x / y) for x, y in zip(a, b)], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)

  def test_mod_tensor_tensor_matches_reference(self):
    import math
    a = np.array([10, 21, 35, 42], dtype=np.int32)
    b = np.array([3, 7, 5, 6], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") % Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    expected = np.array([x - y * math.trunc(x / y) for x, y in zip(a, b)], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)

  def test_rowsum_keepdim_matches_reference(self):
    data = np.arange(8, dtype=np.int32).reshape(2, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1, keepdims=True))

  def test_rowmax_3x4_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_rowmin_3x4_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1))

  def test_relu_2d_3x4_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, device="TINYTPU").relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(data, 0))

  def test_abs_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(data))

  def test_clip_2d_3x4_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, device="TINYTPU").clip(-2, 3).numpy()
    np.testing.assert_array_equal(result, np.clip(data, -2, 3))

  def test_sum_2d_all_elements_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, int(data.sum()))

  def test_rowsum_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_cmplt_2d_3x4_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    b = (np.arange(12, dtype=np.int32) + 2).reshape(3, 4)
    result = (Tensor(a, device="TINYTPU") < Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a < b)

  def test_cmpeq_2d_3x4_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    b = (np.arange(12, dtype=np.int32) + 2).reshape(3, 4)
    result = (Tensor(a, device="TINYTPU") == Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a == b)

  def test_cmpne_2d_3x4_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    b = (np.arange(12, dtype=np.int32) + 2).reshape(3, 4)
    result = (Tensor(a, device="TINYTPU") != Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a != b)

  def test_bool_and_2d_3x4_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4) % 2 == 0
    b = (np.arange(12, dtype=np.int32) + 1).reshape(3, 4) % 2 == 0
    result = (Tensor(a, device="TINYTPU") & Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a & b)

  def test_bool_or_2d_3x4_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4) % 2 == 0
    b = (np.arange(12, dtype=np.int32) + 1).reshape(3, 4) % 2 == 0
    result = (Tensor(a, device="TINYTPU") | Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a | b)

  def test_add_32elem_large_matches_reference(self):
    a = np.arange(32, dtype=np.int32)
    b = np.arange(32, dtype=np.int32) + 1
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_colsum_4x4_single_tile_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colsum_3x4_single_tile_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colmax_4x4_single_tile_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmax_2x4_single_tile_matches_reference(self):
    data = np.array([[-5, -10, -3, -8], [-6, -2, -4, -9]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmin_4x4_single_tile_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_colmin_3x4_single_tile_matches_reference(self):
    data = np.array([[5, 10, 3, 8], [6, 2, 4, 9], [7, 1, 11, 4]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_colsum_4x3_narrow_single_tile_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(4, 3)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colmax_3x2_narrow_single_tile_matches_reference(self):
    data = np.array([[-5, -10], [-6, -2], [-7, -1]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmin_2x3_narrow_single_tile_matches_reference(self):
    data = np.array([[5, 10, 3], [6, 2, 4]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_colsum_6x6_multi_both_matches_reference(self):
    data = (np.arange(36, dtype=np.int32) - 18).reshape(6, 6)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colmax_5x7_multi_both_matches_reference(self):
    rng = np.random.default_rng(123)
    data = rng.integers(-50, 50, size=(5, 7), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmin_7x5_multi_both_matches_reference(self):
    rng = np.random.default_rng(456)
    data = rng.integers(-50, 50, size=(7, 5), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_colsum_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colmax_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmin_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_rowmax_4x8_keepdim_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1, keepdims=True))

  def test_rowmin_3x8_keepdim_matches_reference(self):
    data = (np.arange(24, dtype=np.int32) - 12).reshape(3, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1, keepdims=True))

  def test_colsum_5x3_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(5, 3)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colmax_6x3_matches_reference(self):
    data = (np.arange(18, dtype=np.int32) - 9).reshape(6, 3)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmin_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_add_4x8_tensor_tensor_neg_values_matches_reference(self):
    a = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    b = (np.arange(32, dtype=np.int32) + 1).reshape(4, 8)
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_mul_3x5_tensor_tensor_matches_reference(self):
    a = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    b = (np.arange(15, dtype=np.int32) + 1).reshape(3, 5)
    result = (Tensor(a, device="TINYTPU") * Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_sub_4x8_tensor_tensor_matches_reference(self):
    a = (np.arange(32, dtype=np.int32)).reshape(4, 8)
    b = np.full((4, 8), 5, dtype=np.int32)
    result = (Tensor(a, device="TINYTPU") - Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_max_3x8_tensor_tensor_matches_reference(self):
    a = (np.arange(24, dtype=np.int32) - 12).reshape(3, 8)
    b = np.zeros((3, 8), dtype=np.int32)
    result = Tensor(a, device="TINYTPU").maximum(Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.maximum(a, b))

  def test_random_where_3x4_matches_reference(self):
    rng = np.random.default_rng(11)
    cond = rng.integers(0, 2, size=(3, 4), dtype=np.bool_)
    lhs  = rng.integers(-50, 50, size=(3, 4), dtype=np.int32)
    rhs  = rng.integers(-50, 50, size=(3, 4), dtype=np.int32)
    result = Tensor(cond, device="TINYTPU").where(
      Tensor(lhs, dtype="int32", device="TINYTPU"),
      Tensor(rhs, dtype="int32", device="TINYTPU"),
    ).numpy()
    np.testing.assert_array_equal(result, np.where(cond, lhs, rhs))

  def test_random_abs_4x4_matches_reference(self):
    rng = np.random.default_rng(22)
    data = rng.integers(-200, 200, size=(4, 4), dtype=np.int32)
    result = Tensor(data, device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(data))

  def test_random_clip_3x8_matches_reference(self):
    rng = np.random.default_rng(33)
    data = rng.integers(-20, 20, size=(3, 8), dtype=np.int32)
    result = Tensor(data, device="TINYTPU").clip(-5, 10).numpy()
    np.testing.assert_array_equal(result, np.clip(data, -5, 10))

  def test_random_rowsum_4x8_matches_reference(self):
    rng = np.random.default_rng(44)
    data = rng.integers(-100, 100, size=(4, 8), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_random_colmin_4x4_matches_reference(self):
    rng = np.random.default_rng(55)
    data = rng.integers(-100, 100, size=(4, 4), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_random_neg_3x8_matches_reference(self):
    rng = np.random.default_rng(66)
    data = rng.integers(-1000, 1000, size=(3, 8), dtype=np.int32)
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_realize_breaks_fusion_for_chained_ops(self):
    """Without .realize(), tinygrad fuses relu+add into one kernel our backend
    may not recognize. With .realize() the ops execute separately."""
    data = np.arange(-4, 4, dtype=np.int32)
    # Without realize: fused into one kernel (may fail with NotImplementedError)
    # With realize: relu executes on TINYTPU, then add executes on TINYTPU
    relu_result = Tensor(data, device="TINYTPU").relu().realize()
    final = (relu_result + 1).numpy()
    np.testing.assert_array_equal(final, np.maximum(data, 0) + 1)

  def test_mul_then_relu_with_realize_matches_reference(self):
    data = np.arange(-8, 8, dtype=np.int32)
    mul_result = (Tensor(data, device="TINYTPU") * 2).realize()
    result = mul_result.relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(data * 2, 0))

  def test_sum_reduce_all_positive_4x4_matches_reference(self):
    """Sum reduction of all-positive 4x4 to scalar via VPU_SUM_REDUCE."""
    data = np.arange(1, 17, dtype=np.int32).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, int(data.sum()))

  def test_sum_reduce_all_negative_4x4_matches_reference(self):
    data = -np.arange(1, 17, dtype=np.int32).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, int(data.sum()))

  def test_max_reduce_all_negative_matches_reference(self):
    data = -np.arange(1, 17, dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, int(data.max()))

  def test_min_reduce_all_positive_matches_reference(self):
    data = np.arange(1, 17, dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, int(data.min()))

  def test_relu_then_mul_scalar_matches_reference(self):
    """relu -> mul scalar: realize between steps to prevent fusion."""
    data = np.arange(-4, 4, dtype=np.int32)
    r1 = Tensor(data, device="TINYTPU").relu().realize()
    result = (r1 * 2).numpy()
    np.testing.assert_array_equal(result, np.maximum(data, 0) * 2)

  def test_abs_then_sum_axis1_matches_reference(self):
    """abs on 4x4 then rowsum: realize abs first."""
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    abs_t = Tensor(data, device="TINYTPU").abs().realize()
    result = abs_t.sum(axis=1).numpy()
    expected = np.abs(data).sum(axis=1)
    np.testing.assert_array_equal(result, expected)

  def test_neg_then_max_scalar_matches_reference(self):
    """neg -> maximum(0): realize neg first."""
    data = np.arange(-4, 4, dtype=np.int32)
    neg_t = (-Tensor(data, device="TINYTPU")).realize()
    result = neg_t.maximum(0).numpy()
    np.testing.assert_array_equal(result, np.maximum(-data, 0))

  def test_where_all_true_matches_reference(self):
    cond = np.ones(16, dtype=np.bool_)
    lhs = np.arange(16, dtype=np.int32)
    rhs = -np.arange(16, dtype=np.int32)
    result = Tensor(cond, device="TINYTPU").where(
      Tensor(lhs, dtype="int32", device="TINYTPU"),
      Tensor(rhs, dtype="int32", device="TINYTPU"),
    ).numpy()
    np.testing.assert_array_equal(result, lhs)

  def test_where_all_false_matches_reference(self):
    cond = np.zeros(16, dtype=np.bool_)
    lhs = np.arange(16, dtype=np.int32)
    rhs = -np.arange(16, dtype=np.int32)
    result = Tensor(cond, device="TINYTPU").where(
      Tensor(lhs, dtype="int32", device="TINYTPU"),
      Tensor(rhs, dtype="int32", device="TINYTPU"),
    ).numpy()
    np.testing.assert_array_equal(result, rhs)

  def test_gemm_negative_values_matches_reference(self):
    a = np.array([[-1, -2, -3, -4]], dtype=np.int32)
    w = np.eye(4, dtype=np.int32) * (-1)
    result = (Tensor(a, dtype="int32", device="TINYTPU") @ Tensor(w, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a @ w)

  def test_gemm_zero_weight_matches_reference(self):
    a = np.array([[1, 2, 3, 4]], dtype=np.int32)
    w = np.zeros((4, 4), dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") @ Tensor(w, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.zeros((1, 4), dtype=np.int32))

  def test_zeros_4x4_add_matches_reference(self):
    z = np.zeros((4, 4), dtype=np.int32)
    result = (Tensor(z, device="TINYTPU") + Tensor(z, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, z)

  def test_sum_zeros_4x4_matches_reference(self):
    z = np.zeros((4, 4), dtype=np.int32)
    result = Tensor(z, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, 0)

  def test_max_zeros_matches_reference(self):
    z = np.zeros((4, 4), dtype=np.int32)
    result = Tensor(z, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, 0)

  def test_neg_zeros_matches_reference(self):
    z = np.zeros(16, dtype=np.int32)
    result = (-Tensor(z, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, z)

  def test_abs_zeros_4x4_matches_reference(self):
    z = np.zeros((4, 4), dtype=np.int32)
    result = Tensor(z, device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, z)

  def test_large_values_add_matches_reference(self):
    # int32 arithmetic with large values (no overflow)
    a = np.array([100000, -100000, 50000, -50000], dtype=np.int32)
    b = np.array([200000, 200000, -50000, 50000], dtype=np.int32)
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_abs_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(data))

  def test_clip_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, device="TINYTPU").clip(-3, 3).numpy()
    np.testing.assert_array_equal(result, np.clip(data, -3, 3))

  def test_min_4x8_tensor_tensor_matches_reference(self):
    a = np.arange(32, dtype=np.int32).reshape(4, 8)
    b = (np.arange(32, dtype=np.int32) + 1).reshape(4, 8)
    result = Tensor(a, device="TINYTPU").minimum(Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.minimum(a, b))

  def test_max_4x8_tensor_tensor_matches_reference(self):
    a = np.arange(32, dtype=np.int32).reshape(4, 8)
    b = (np.arange(32, dtype=np.int32) - 1).reshape(4, 8)
    result = Tensor(a, device="TINYTPU").maximum(Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.maximum(a, b))

  def test_sub_3x5_tensor_tensor_matches_reference(self):
    a = np.arange(15, dtype=np.int32).reshape(3, 5)
    b = (np.arange(15, dtype=np.int32) + 1).reshape(3, 5)
    result = (Tensor(a, device="TINYTPU") - Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_colsum_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colmax_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_rowmax_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_rowmin_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1))

  def test_sum_axis0_3x8_matches_reference(self):
    data = (np.arange(24, dtype=np.int32) - 12).reshape(3, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_max_axis0_3x8_matches_reference(self):
    data = (np.arange(24, dtype=np.int32) - 12).reshape(3, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_min_axis0_3x8_matches_reference(self):
    data = (np.arange(24, dtype=np.int32) - 12).reshape(3, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_rowsum_2x16_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(2, 16)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_shl_2d_4x8_matches_reference(self):
    data = np.arange(32, dtype=np.int32).reshape(4, 8)
    result = (Tensor(data, device="TINYTPU") << 2).numpy()
    np.testing.assert_array_equal(result, data << 2)

  def test_shr_2d_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) * 4).reshape(4, 8)
    result = (Tensor(data, device="TINYTPU") >> 2).numpy()
    np.testing.assert_array_equal(result, data >> 2)

  def test_not_2d_4x4_matches_reference(self):
    data = np.tile([True, False, True, True], 4).reshape(4, 4)
    result = (~Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, ~data)

  def test_idiv_2d_4x8_matches_reference(self):
    import math
    data = (np.arange(1, 33, dtype=np.int32)).reshape(4, 8)
    result = (Tensor(data, device="TINYTPU") // 3).numpy()
    expected = np.array([math.trunc(x / 3) for x in range(1, 33)], dtype=np.int32).reshape(4, 8)
    np.testing.assert_array_equal(result, expected)

  def test_mod_2d_3x8_matches_reference(self):
    import math
    data = np.arange(3, 27, dtype=np.int32).reshape(3, 8)
    result = (Tensor(data, device="TINYTPU") % 7).numpy()
    expected = np.array([x - 7 * math.trunc(x / 7) for x in range(3, 27)], dtype=np.int32).reshape(3, 8)
    np.testing.assert_array_equal(result, expected)

  def test_random_3x4_rowsum_matches_reference(self):
    rng = np.random.default_rng(123)
    data = rng.integers(-100, 100, size=(3, 4), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_random_4x4_colmin_matches_reference(self):
    rng = np.random.default_rng(456)
    data = rng.integers(-100, 100, size=(4, 4), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_random_4x8_colmax_matches_reference(self):
    rng = np.random.default_rng(789)
    data = rng.integers(-100, 100, size=(4, 8), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_random_8x4_rowmax_matches_reference(self):
    rng = np.random.default_rng(321)
    data = rng.integers(-100, 100, size=(8, 4), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_random_gemm_4x4_matches_reference(self):
    rng = np.random.default_rng(42)
    a = rng.integers(-8, 8, size=(2, 4), dtype=np.int32)
    w = rng.integers(-8, 8, size=(4, 4), dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") @ Tensor(w, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a @ w)

  def test_random_2d_add_matches_reference(self):
    rng = np.random.default_rng(42)
    a = rng.integers(-50, 50, size=(4, 4), dtype=np.int32)
    b = rng.integers(-50, 50, size=(4, 4), dtype=np.int32)
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_random_2d_mul_matches_reference(self):
    rng = np.random.default_rng(7)
    a = rng.integers(-10, 10, size=(3, 4), dtype=np.int32)
    b = rng.integers(-10, 10, size=(3, 4), dtype=np.int32)
    result = (Tensor(a, device="TINYTPU") * Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_random_rowsum_matches_reference(self):
    rng = np.random.default_rng(99)
    data = rng.integers(-100, 100, size=(8, 4), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_random_colsum_matches_reference(self):
    rng = np.random.default_rng(13)
    data = rng.integers(-100, 100, size=(4, 8), dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_random_neg_matches_reference(self):
    rng = np.random.default_rng(55)
    data = rng.integers(-1000, 1000, size=(4, 4), dtype=np.int32)
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_random_scalar_add_matches_reference(self):
    rng = np.random.default_rng(77)
    data = rng.integers(-500, 500, size=(3, 4), dtype=np.int32)
    result = (Tensor(data, device="TINYTPU") + 42).numpy()
    np.testing.assert_array_equal(result, data + 42)

  def test_fused_add_relu_4x8_matches_reference(self):
    a = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    b = np.full((4, 8), 3, dtype=np.int32)
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a + b, 0))

  def test_abs_3x8_matches_reference(self):
    data = (np.arange(24, dtype=np.int32) - 12).reshape(3, 8)
    result = Tensor(data, device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(data))

  def test_clip_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, device="TINYTPU").clip(-5, 5).numpy()
    np.testing.assert_array_equal(result, np.clip(data, -5, 5))

  def test_where_4x4_matches_reference(self):
    cond = np.tile([True, False], 8).reshape(4, 4)
    lhs = np.arange(16, dtype=np.int32).reshape(4, 4)
    rhs = np.zeros((4, 4), dtype=np.int32)
    result = Tensor(cond, device="TINYTPU").where(
      Tensor(lhs, dtype="int32", device="TINYTPU"),
      Tensor(rhs, dtype="int32", device="TINYTPU"),
    ).numpy()
    np.testing.assert_array_equal(result, np.where(cond, lhs, rhs))

  def test_rowsum_3x5_matches_reference(self):
    data = (np.arange(15, dtype=np.int32) - 7).reshape(3, 5)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_add5_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = (Tensor(data, device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, data + 5)

  def test_neg_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_neg_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_rowsum_3x8_matches_reference(self):
    data = (np.arange(24, dtype=np.int32) - 12).reshape(3, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_rowmax_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_rowmin_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1))

  def test_rowsum_4x3_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(4, 3)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_colmax_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmin_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_colsum_3x4_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_rowsum_3x4_with_neg_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_rowmin_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1))

  def test_sum_all_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, int(data.sum()))

  def test_sum_all_2d_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, int(data.sum()))

  def test_max_all_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, int(data.max()))

  def test_min_all_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, int(data.min()))

  def test_abs_2d_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(data))

  def test_not_2d_3x4_matches_reference(self):
    data = np.tile([True, False], 6).reshape(3, 4)
    result = (~Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, ~data)

  def test_xor_2d_3x4_matches_reference(self):
    a = np.tile([True, False], 6).reshape(3, 4)
    b = np.tile([False, True], 6).reshape(3, 4)
    result = (Tensor(a, device="TINYTPU") ^ Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a ^ b)

  def test_shr_2d_3x4_matches_reference(self):
    data = (np.arange(3, 15, dtype=np.int32) * 4).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") >> 2).numpy()
    np.testing.assert_array_equal(result, data >> 2)

  def test_colsum_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_rowmax_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_broadcast_add_1_to_4x4_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor([5], dtype="int32", device="TINYTPU") + Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, 5 + data)

  def test_broadcast_mul_3x4_by_scalar_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") * Tensor([3], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, data * 3)

  def test_broadcast_sub_scalar_from_4x4_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor([10], dtype="int32", device="TINYTPU") - Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, 10 - data)

  def test_mul_2d_4x4_tensor_tensor_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    b = (np.arange(16, dtype=np.int32) + 1).reshape(4, 4)
    result = (Tensor(a, device="TINYTPU") * Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_sub_2d_8x4_tensor_tensor_matches_reference(self):
    a = np.arange(32, dtype=np.int32).reshape(8, 4)
    b = (np.arange(32, dtype=np.int32) + 1).reshape(8, 4)
    result = (Tensor(a, device="TINYTPU") - Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_add_2d_4x8_tensor_tensor_matches_reference(self):
    a = np.arange(32, dtype=np.int32).reshape(4, 8)
    b = np.ones((4, 8), dtype=np.int32)
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_mul_neg_scalar_2d_4x4_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(data, device="TINYTPU") * (-2)).numpy()
    np.testing.assert_array_equal(result, data * (-2))

  def test_add_neg_scalar_2d_3x4_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") + (-5)).numpy()
    np.testing.assert_array_equal(result, data + (-5))

  def test_min_scalar_2d_4x4_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(data, device="TINYTPU").minimum(5).numpy()
    np.testing.assert_array_equal(result, np.minimum(data, 5))

  def test_max_scalar_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, device="TINYTPU").maximum(0).numpy()
    np.testing.assert_array_equal(result, np.maximum(data, 0))

  def test_shl_scalar_2d_3x4_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") << 1).numpy()
    np.testing.assert_array_equal(result, data << 1)

  def test_idiv_scalar_2d_3x4_matches_reference(self):
    import math
    data = np.arange(3, 15, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") // 2).numpy()
    expected = np.array([math.trunc(x / 2) for x in range(3, 15)], dtype=np.int32).reshape(3, 4)
    np.testing.assert_array_equal(result, expected)

  def test_colsum_keepdim_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0, keepdims=True))

  def test_rowmax_keepdim_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1, keepdims=True))

  def test_colmin_keepdim_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0, keepdims=True))

  def test_relu_2d_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, device="TINYTPU").relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(data, 0))

  def test_abs_2d_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(data))

  def test_clip_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, device="TINYTPU").clip(-3, 3).numpy()
    np.testing.assert_array_equal(result, np.clip(data, -3, 3))

  def test_fused_add_relu_2d_4x4_matches_reference(self):
    a = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    b = np.full((4, 4), 5, dtype=np.int32)
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a + b, 0))

  def test_gemm_4x4_identity_matches_reference(self):
    a = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    w = np.eye(4, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") @ Tensor(w, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a @ w)

  def test_gemm_2x4_times_4x4_matches_reference(self):
    a = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.int32)
    w = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") @ Tensor(w, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a @ w)

  def test_gemm_1x4_times_4x8_matches_reference(self):
    a = np.array([[1, 2, 3, 4]], dtype=np.int32)
    w = np.arange(32, dtype=np.int32).reshape(4, 8)
    result = (Tensor(a, dtype="int32", device="TINYTPU") @ Tensor(w, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a @ w)

  def test_neg_2d_8x4_matches_reference(self):
    data = np.arange(32, dtype=np.int32).reshape(8, 4) - 16
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_where_2d_3x4_matches_reference(self):
    cond = np.tile([True, False], 6).reshape(3, 4)
    lhs = np.arange(12, dtype=np.int32).reshape(3, 4)
    rhs = np.zeros((3, 4), dtype=np.int32)
    result = Tensor(cond, device="TINYTPU").where(
      Tensor(lhs, dtype="int32", device="TINYTPU"),
      Tensor(rhs, dtype="int32", device="TINYTPU"),
    ).numpy()
    np.testing.assert_array_equal(result, np.where(cond, lhs, rhs))

  def test_shl_2d_3x4_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") << 2).numpy()
    np.testing.assert_array_equal(result, data << 2)

  def test_shr_2d_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) * 4).reshape(4, 4)
    result = (Tensor(data, device="TINYTPU") >> 2).numpy()
    np.testing.assert_array_equal(result, data >> 2)

  def test_idiv_2d_3x4_matches_reference(self):
    import math
    data = np.arange(3, 15, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") // 3).numpy()
    expected = np.array([math.trunc(x / 3) for x in range(3, 15)], dtype=np.int32).reshape(3, 4)
    np.testing.assert_array_equal(result, expected)

  def test_mod_2d_4x4_matches_reference(self):
    import math
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(data, device="TINYTPU") % 5).numpy()
    expected = np.array([x - 5 * math.trunc(x / 5) for x in range(16)], dtype=np.int32).reshape(4, 4)
    np.testing.assert_array_equal(result, expected)

  def test_colsum_2d_3x4_matches_reference(self):
    data = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_add_2d_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    b = (np.arange(12, dtype=np.int32) + 1).reshape(3, 4)
    result = (Tensor(a, device="TINYTPU") + Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_sub_2d_4x4_matches_reference(self):
    a = (np.arange(16, dtype=np.int32)).reshape(4, 4)
    b = (np.arange(16, dtype=np.int32) + 1).reshape(4, 4)
    result = (Tensor(a, device="TINYTPU") - Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_mul_2d_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.int32) - 3).reshape(3, 4)
    b = (np.arange(12, dtype=np.int32) + 1).reshape(3, 4)
    result = (Tensor(a, device="TINYTPU") * Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_max_2d_4x4_matches_reference(self):
    a = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    b = (np.arange(16, dtype=np.int32) - 4).reshape(4, 4)
    result = Tensor(a, device="TINYTPU").maximum(Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.maximum(a, b))

  def test_min_2d_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.int32) - 6).reshape(3, 4)
    b = (np.arange(12, dtype=np.int32) - 3).reshape(3, 4)
    result = Tensor(a, device="TINYTPU").minimum(Tensor(b, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.minimum(a, b))

  def test_neg_full_tile_matches_reference(self):
    data = np.arange(16, dtype=np.int32) - 8
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_neg_2d_3x4_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4) - 6
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_neg_2d_4x4_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4) - 8
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_neg_multi_tile_matches_reference(self):
    data = np.arange(32, dtype=np.int32) - 16
    result = (-Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, -data)

  def test_mul_scalar_2d_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") * 3).numpy()
    np.testing.assert_array_equal(result, data * 3)

  def test_add_scalar_2d_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(data, device="TINYTPU") + 7).numpy()
    np.testing.assert_array_equal(result, data + 7)

  def test_sub_scalar_2d_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = (Tensor(data, device="TINYTPU") - 2).numpy()
    np.testing.assert_array_equal(result, data - 2)

  def test_abs_matches_reference(self):
    result = Tensor([-1, 2, -3, 4], dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.array([1, 2, 3, 4], dtype=np.int32))

  def test_abs_full_tile_matches_reference(self):
    data = list(range(-8, 8))
    result = Tensor(data, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(np.array(data, dtype=np.int32)))

  def test_abs_multi_tile_matches_reference(self):
    data = list(range(-16, 16))
    result = Tensor(data, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(np.array(data, dtype=np.int32)))

  def test_abs_all_positive_matches_reference(self):
    data = list(range(0, 16))
    result = Tensor(data, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.array(data, dtype=np.int32))

  def test_abs_all_negative_matches_reference(self):
    data = list(range(-16, 0))
    result = Tensor(data, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(np.array(data, dtype=np.int32)))

  def test_minimum_int_reports_correct_not_max(self):
    """Regression: minimum(a,b) previously misidentified as MAX."""
    result = Tensor([1, 5, 3], dtype="int32", device="TINYTPU").minimum(
      Tensor([4, 2, 6], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([1, 2, 3], dtype=np.int32))

  def test_clip_matches_reference(self):
    result = Tensor([-5, 0, 3, 10], dtype="int32", device="TINYTPU").clip(0, 5).numpy()
    np.testing.assert_array_equal(result, np.array([0, 0, 3, 5], dtype=np.int32))

  def test_clip_full_tile_matches_reference(self):
    data = list(range(-8, 8))
    result = Tensor(data, dtype="int32", device="TINYTPU").clip(-3, 3).numpy()
    np.testing.assert_array_equal(result, np.clip(np.array(data, dtype=np.int32), -3, 3))

  def test_clip_multi_tile_matches_reference(self):
    data = list(range(-10, 22))
    result = Tensor(data, dtype="int32", device="TINYTPU").clip(0, 5).numpy()
    np.testing.assert_array_equal(result, np.clip(np.array(data, dtype=np.int32), 0, 5))

  def test_fused_add_relu_full_tile_matches_reference(self):
    a = list(range(-8, 8))
    b = list(range(0, 16))
    result = (Tensor(a, dtype="int32", device="TINYTPU") + Tensor(b, dtype="int32", device="TINYTPU")).relu().numpy()
    expected = np.maximum(np.array(a, dtype=np.int32) + np.array(b, dtype=np.int32), 0)
    np.testing.assert_array_equal(result, expected)

  def test_fused_add_relu_multi_tile_matches_reference(self):
    a = list(range(-8, 24))
    b = list(range(0, 32))
    result = (Tensor(a, dtype="int32", device="TINYTPU") + Tensor(b, dtype="int32", device="TINYTPU")).relu().numpy()
    expected = np.maximum(np.array(a, dtype=np.int32) + np.array(b, dtype=np.int32), 0)
    np.testing.assert_array_equal(result, expected)

  def test_rowsum_2x4_matches_reference(self):
    data = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_rowsum_3x4_matches_reference(self):
    data = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_rowsum_4x4_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_rowsum_negative_values_matches_reference(self):
    data = np.array([[-1, 2, -3, 4], [5, -6, 7, -8]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_rowmax_2x4_matches_reference(self):
    data = np.array([[1, -2, 3, -4], [5, 6, -7, 8]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_rowmax_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_rowmin_2x4_matches_reference(self):
    data = np.array([[1, -2, 3, -4], [5, 6, -7, 8]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1))

  def test_rowmin_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1))

  def test_colsum_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colmax_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0))

  def test_colmin_4x4_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, data.min(axis=0))

  def test_colsum_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_colsum_4x8_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(4, 8)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_rowsum_8x4_matches_reference(self):
    """Row-wise sum for 8 rows (two 4-row hardware tiles)."""
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_rowmax_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, data.max(axis=1))

  def test_rowmin_8x4_matches_reference(self):
    data = (np.arange(32, dtype=np.int32) - 16).reshape(8, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, data.min(axis=1))

  def test_rowsum_5x4_matches_reference(self):
    """Odd row count: tile boundary at row 4, one 1-row partial tile."""
    data = (np.arange(20, dtype=np.int32) - 10).reshape(5, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_bool_to_int32_cast_matches_reference(self):
    result = Tensor([True, False, True, False], device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, np.array([1, 0, 1, 0], dtype=np.int32))

  def test_int32_to_bool_cast_matches_reference(self):
    result = Tensor([0, 1, 2, -1], dtype="int32", device="TINYTPU").cast("bool").numpy()
    np.testing.assert_array_equal(result, np.array([False, True, True, True]))

  def test_int32_to_bool_cast_multi_tile_matches_reference(self):
    data = list(range(-8, 24))
    result = Tensor(data, dtype="int32", device="TINYTPU").cast("bool").numpy()
    expected = np.array(data, dtype=np.int32) != 0
    np.testing.assert_array_equal(result, expected)

  def test_scalar_broadcast_mul_matches_reference(self):
    result = (Tensor([2], dtype="int32", device="TINYTPU") * Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([2, 4, 6, 8], dtype=np.int32))

  def test_scalar_broadcast_mul_reverse_matches_reference(self):
    result = (Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU") * Tensor([3], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([3, 6, 9, 12], dtype=np.int32))

  def test_scalar_broadcast_sub_matches_reference(self):
    result = (Tensor([10], dtype="int32", device="TINYTPU") - Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([9, 8, 7, 6], dtype=np.int32))

  def test_scalar_broadcast_max_matches_reference(self):
    result = Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU").maximum(
      Tensor([3], dtype="int32", device="TINYTPU")
    ).numpy()
    np.testing.assert_array_equal(result, np.array([3, 3, 3, 4], dtype=np.int32))

  def test_scalar_broadcast_add_multi_tile_matches_reference(self):
    data = list(range(1, 33))
    result = (Tensor(data, dtype="int32", device="TINYTPU") + Tensor([10], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array(data, dtype=np.int32) + 10)

  def test_scalar_broadcast_mul_multi_tile_matches_reference(self):
    data = list(range(1, 33))
    result = (Tensor(data, dtype="int32", device="TINYTPU") * Tensor([2], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array(data, dtype=np.int32) * 2)

  def test_scalar_broadcast_add_matches_reference(self):
    result = (Tensor([1], dtype="int32", device="TINYTPU") + Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([2, 3, 4, 5], dtype=np.int32))

  def test_scalar_broadcast_add_reverse_matches_reference(self):
    result = (Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU") + Tensor([1], dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.array([2, 3, 4, 5], dtype=np.int32))

  def test_trunc_float32_matches_reference(self):
    result = Tensor([1.2, -1.7, 0.0, 3.9], dtype="float32", device="TINYTPU").trunc().numpy()
    np.testing.assert_allclose(result, np.array([1.0, -1.0, 0.0, 3.0], dtype=np.float32))

  def test_reciprocal_float32_matches_reference(self):
    result = Tensor([1.0, 2.0, -4.0, 0.5], dtype="float32", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, np.array([1.0, 0.5, -0.25, 2.0], dtype=np.float32))

  def test_permute_reports_movement_lowering_gap(self):
    with self.assertRaises(NotImplementedError):
      Tensor([[1, 2], [3, 4]], dtype="int32", device="TINYTPU").permute(1, 0).numpy()

  def test_prod4_matches_reference(self):
    result = Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU").prod().numpy()
    np.testing.assert_array_equal(result, 24)

  def test_prod_full_tile_matches_reference(self):
    a = np.ones(16, dtype=np.int32)
    a[0] = 2; a[3] = 3; a[7] = 5
    result = Tensor(a, dtype="int32", device="TINYTPU").prod().numpy()
    np.testing.assert_array_equal(result, a.prod())

  def test_rowprod_3x4_matches_reference(self):
    data = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").prod(axis=1).numpy()
    np.testing.assert_array_equal(result, data.prod(axis=1))

  def test_colprod_3x4_matches_reference(self):
    data = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").prod(axis=0).numpy()
    np.testing.assert_array_equal(result, data.prod(axis=0))

  def test_rowprod_keepdim_matches_reference(self):
    data = np.arange(1, 13, dtype=np.int32).reshape(3, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").prod(axis=1, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.prod(axis=1, keepdims=True))

  def test_rowprod_wide_matches_reference(self):
    # Keep values small to avoid int32 overflow
    data = np.ones((3, 8), dtype=np.int32); data[0, 0] = 2; data[1, 3] = 3; data[2, 7] = 5
    result = Tensor(data, dtype="int32", device="TINYTPU").prod(axis=1).numpy()
    np.testing.assert_array_equal(result, data.prod(axis=1))

  def test_colprod_multi_row_matches_reference(self):
    data = np.ones((5, 3), dtype=np.int32); data[0, 0] = 2; data[2, 1] = 3; data[4, 2] = 5
    result = Tensor(data, dtype="int32", device="TINYTPU").prod(axis=0).numpy()
    np.testing.assert_array_equal(result, data.prod(axis=0))

  def test_reshape_1d_to_2d_matches_reference(self):
    a = np.arange(16, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").reshape(4, 4).numpy()
    np.testing.assert_array_equal(result, a.reshape(4, 4))

  def test_reshape_2d_to_1d_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").reshape(12).numpy()
    np.testing.assert_array_equal(result, a.reshape(12))

  def test_reshape_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").reshape(4, 8).numpy()
    np.testing.assert_array_equal(result, a.reshape(4, 8))

  def test_slice_prefix_matches_reference(self):
    a = np.arange(8, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU")[:4].numpy()
    np.testing.assert_array_equal(result, a[:4])

  def test_slice_suffix_matches_reference(self):
    a = np.arange(8, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU")[4:].numpy()
    np.testing.assert_array_equal(result, a[4:])

  def test_slice_middle_matches_reference(self):
    a = np.arange(16, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU")[4:12].numpy()
    np.testing.assert_array_equal(result, a[4:12])

  def test_slice_negative_suffix_matches_reference(self):
    a = np.arange(8, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU")[-4:].numpy()
    np.testing.assert_array_equal(result, a[-4:])

  def test_slice_negative_stop_matches_reference(self):
    a = np.arange(8, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU")[:-2].numpy()
    np.testing.assert_array_equal(result, a[:-2])

  def test_slice_2d_rows_matches_reference(self):
    a = np.arange(32, dtype=np.int32).reshape(8, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU")[2:6].numpy()
    np.testing.assert_array_equal(result, a[2:6])

  def test_expand_scalar_to_tile_matches_reference(self):
    result = Tensor([[7]], dtype="int32", device="TINYTPU").expand(4, 4).numpy()
    np.testing.assert_array_equal(result, np.full((4, 4), 7, dtype=np.int32))

  def test_expand_scalar_to_multi_tile_matches_reference(self):
    result = Tensor([[5]], dtype="int32", device="TINYTPU").expand(4, 8).numpy()
    np.testing.assert_array_equal(result, np.full((4, 8), 5, dtype=np.int32))

  def test_expand_row_to_4x4_matches_reference(self):
    result = Tensor([[10, 20, 30, 40]], dtype="int32", device="TINYTPU").expand(4, 4).numpy()
    np.testing.assert_array_equal(result, np.array([[10, 20, 30, 40]] * 4, dtype=np.int32))

  # Note: expand-row through a RANGE-loop (e.g. shape (6,3)) not yet supported —
  # requires detecting LOAD depending on inner RANGE only.

  def test_sum_axis0_keepdim_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0, keepdims=True))

  def test_max_axis0_keepdim_matches_reference(self):
    data = (np.arange(16, dtype=np.int32) - 8).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").max(axis=0, keepdim=True).numpy()
    np.testing.assert_array_equal(result, data.max(axis=0, keepdims=True))

  def test_sum_all_axes_2d_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, data.sum())

  def test_max_all_axes_2d_matches_reference(self):
    data = np.arange(16, dtype=np.int32).reshape(4, 4) - 8
    result = Tensor(data, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, data.max())

  def test_prod_all_axes_2d_matches_reference(self):
    data = np.ones((3, 4), dtype=np.int32); data[0,0] = 2; data[1,1] = 3; data[2,2] = 5
    result = Tensor(data, dtype="int32", device="TINYTPU").prod().numpy()
    np.testing.assert_array_equal(result, data.prod())

  def test_reshape_16_to_4x4_then_add_matches_reference(self):
    a = np.arange(16, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU").reshape(4, 4) +
              Tensor(np.ones(16, dtype=np.int32).reshape(4,4), dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a.reshape(4, 4) + 1)

  # Note: slice-then-sum currently fuses the sum over the unsliced buffer —
  # investigate schedule ordering before re-enabling.

  def test_vpu_opcode_table_marks_bool_results(self):
    self.assertEqual(_VPU_OPS["CMPEQ"], 8)
    self.assertEqual(_VPU_OPS["AND"], 15)
    self.assertEqual(_VPU_OPS["OR"], 16)
    self.assertEqual(_VPU_OPS["XOR"], 17)
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
      self.assertTrue(any(r.get("op") in ("VPU_BINARY", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_wmma_gemm_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          import numpy as np
          from tinygrad import Tensor
          a = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          b = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          print((a @ b).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("GEMM4x4", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_wmma_bias_relu_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          import numpy as np
          from tinygrad import Tensor
          a = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          b = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          bias = Tensor(np.array([1, 2, 3, 4], dtype=np.int32), dtype="int32", device="TINYTPU")
          print(((a @ b) + bias).relu().numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("GEMM4x4", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_row_broadcast_bias_as_sxu_program(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          import numpy as np
          from tinygrad import Tensor
          a = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          b = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          bias = Tensor(np.array([[1, 2, 3, 4]], dtype=np.int32), dtype="int32", device="TINYTPU")
          out = (a @ b).realize()
          print((out + bias).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") == "SXU_PROGRAM" and r.get("primitive") == "BROADCAST_ROW" and any(instr.startswith("2 10 ") for instr in r.get("instructions", [])) for r in records), records)

  def test_lowering_dump_records_minimum_scalar_const_as_sxu_program(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print(Tensor([1, 5, 3, 10], dtype="int32", device="TINYTPU").minimum(5).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") == "SXU_PROGRAM" and r.get("num_output_tiles") == 1 for r in records), records)

  def test_lowering_dump_records_where_as_select_primitive(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          import numpy as np
          from tinygrad import Tensor
          cond = Tensor(np.array([1, 0, 1, 0], dtype=np.int32), dtype="bool", device="TINYTPU")
          lhs = Tensor(np.array([10, 20, 30, 40], dtype=np.int32), dtype="int32", device="TINYTPU")
          rhs = Tensor(np.array([5, 6, 7, 8], dtype=np.int32), dtype="int32", device="TINYTPU")
          print(Tensor.where(cond, lhs, rhs).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") == "SXU_PROGRAM" and r.get("primitive") == "SELECT" and any(instr.startswith("2 8 ") for instr in r.get("instructions", [])) for r in records), records)

  def test_multi_wmma_4x4_at_4x8_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(4, 4)
    w_np = np.arange(32, dtype=np.int32).reshape(4, 8)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_multi_wmma_8x4_at_4x4_matches_numpy(self):
    a_np = np.arange(32, dtype=np.int32).reshape(8, 4)
    w_np = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_multi_wmma_8x8_at_8x8_matches_numpy(self):
    a_np = np.arange(64, dtype=np.int32).reshape(8, 8)
    w_np = np.arange(64, dtype=np.int32).reshape(8, 8)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np)

  def test_multi_wmma_4x4_at_4x8_bias_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(4, 4)
    w_np = np.arange(32, dtype=np.int32).reshape(4, 8)
    b_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @
              Tensor(w_np, dtype="int32", device="TINYTPU") +
              Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np + b_np)

  def test_multi_wmma_8x4_at_4x4_bias_matches_numpy(self):
    a_np = np.arange(32, dtype=np.int32).reshape(8, 4)
    w_np = np.arange(16, dtype=np.int32).reshape(4, 4)
    b_np = np.array([10, 20, 30, 40], dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @
              Tensor(w_np, dtype="int32", device="TINYTPU") +
              Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np + b_np)

  def test_multi_wmma_8x8_at_8x8_bias_matches_numpy(self):
    a_np = np.arange(64, dtype=np.int32).reshape(8, 8)
    w_np = np.arange(64, dtype=np.int32).reshape(8, 8)
    b_np = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int32)
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @
              Tensor(w_np, dtype="int32", device="TINYTPU") +
              Tensor(b_np, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a_np @ w_np + b_np)

  def test_multi_wmma_4x4_at_4x8_relu_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(4, 4) - 5
    w_np = np.arange(32, dtype=np.int32).reshape(4, 8) - 10
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np @ w_np, 0))

  def test_multi_wmma_8x8_at_8x8_relu_matches_numpy(self):
    a_np = np.arange(64, dtype=np.int32).reshape(8, 8) - 21
    w_np = np.arange(64, dtype=np.int32).reshape(8, 8) - 21
    result = (Tensor(a_np, dtype="int32", device="TINYTPU") @ Tensor(w_np, dtype="int32", device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np @ w_np, 0))

  def test_multi_wmma_4x4_at_4x8_bias_relu_matches_numpy(self):
    a_np = np.arange(16, dtype=np.int32).reshape(4, 4) - 5
    w_np = np.arange(32, dtype=np.int32).reshape(4, 8) - 10
    b_np = np.arange(8, dtype=np.int32) - 4
    result = ((Tensor(a_np, dtype="int32", device="TINYTPU") @
               Tensor(w_np, dtype="int32", device="TINYTPU")) +
              Tensor(b_np, dtype="int32", device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np @ w_np + b_np, 0))

  def test_multi_wmma_8x8_at_8x8_bias_relu_matches_numpy(self):
    a_np = np.arange(64, dtype=np.int32).reshape(8, 8) - 21
    w_np = np.arange(64, dtype=np.int32).reshape(8, 8) - 21
    b_np = np.arange(8, dtype=np.int32) - 4
    result = ((Tensor(a_np, dtype="int32", device="TINYTPU") @
               Tensor(w_np, dtype="int32", device="TINYTPU")) +
              Tensor(b_np, dtype="int32", device="TINYTPU")).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np @ w_np + b_np, 0))

  def test_lowering_dump_records_multi_wmma_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          import numpy as np
          from tinygrad import Tensor
          a = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          b = Tensor(np.arange(32, dtype=np.int32).reshape(4, 8), dtype="int32", device="TINYTPU")
          print((a @ b).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("GEMM4x4", "SXU_PROGRAM") and
                          r.get("num_weight_tiles") == 2 for r in records))

  def test_lowering_dump_records_multi_wmma_bias_relu_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          import numpy as np
          from tinygrad import Tensor
          a = Tensor(np.arange(16, dtype=np.int32).reshape(4, 4), dtype="int32", device="TINYTPU")
          b = Tensor(np.arange(32, dtype=np.int32).reshape(4, 8), dtype="int32", device="TINYTPU")
          bias = Tensor(np.arange(8, dtype=np.int32), dtype="int32", device="TINYTPU")
          print(((a @ b) + bias).relu().numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("GEMM4x4", "SXU_PROGRAM") and
                          r.get("num_weight_tiles") == 2
                          for r in records))

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
      self.assertTrue(any(r.get("op") in ("VPU_BINARY", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_div_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((Tensor([10, 20, 30], dtype="int32", device="TINYTPU") // 3).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("VPU_BINARY", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_mod_program_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((Tensor([10, 20, 30], dtype="int32", device="TINYTPU") % 3).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("VPU_PROGRAM", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_native_bool_and_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((Tensor([True, False, True], device="TINYTPU") & Tensor([True, True, False], device="TINYTPU")).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("VPU_BINARY", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_bool_not_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((~Tensor([True, False, True], device="TINYTPU")).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("VPU_BINARY", "SXU_PROGRAM") and r.get("bool_out") for r in records))

  def test_lowering_dump_records_int32_not_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((~Tensor([1, 2, 3], dtype="int32", device="TINYTPU")).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") in ("VPU_BINARY", "SXU_PROGRAM") for r in records))

  def test_lowering_dump_records_scalar_broadcast_descriptor(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          from tinygrad import Tensor
          print((Tensor([1], dtype="int32", device="TINYTPU") + Tensor([1, 2, 3], dtype="int32", device="TINYTPU")).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") == "SXU_PROGRAM" and r.get("primitive") == "BROADCAST_SCALAR" and any(instr.startswith("2 9 ") for instr in r.get("instructions", [])) for r in records), records)


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

  def test_run_bundle_rejects_sim_error_line(self):
    with tempfile.TemporaryDirectory() as td:
      sim = Path(td) / "fake_sim.py"
      sim.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        print("ERROR: injected failure")
        print("status ok")
      """), encoding="utf-8")
      sim.chmod(sim.stat().st_mode | stat.S_IEXEC)
      with self.assertRaisesRegex(RuntimeError, "simulator reported failure: ERROR: injected failure"):
        _run_bundle(str(sim), "4\n")

  def test_run_bundle_requires_ok_status(self):
    with tempfile.TemporaryDirectory() as td:
      sim = Path(td) / "fake_sim.py"
      sim.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        print("status busy")
      """), encoding="utf-8")
      sim.chmod(sim.stat().st_mode | stat.S_IEXEC)
      with self.assertRaisesRegex(RuntimeError, "simulator did not report `status ok`"):
        _run_bundle(str(sim), "4\n")


if __name__ == "__main__":
  unittest.main()
