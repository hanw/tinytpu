from __future__ import annotations
import json, os, stat, subprocess, sys, tempfile, textwrap, unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tinygrad"))
os.environ["TINYTPU_SIM"] = str(REPO_ROOT / "build" / "mkTbTinyTPURuntime.bexe")
os.environ["DISABLE_COMPILER_CACHE"] = "1"

import numpy as np
from tinygrad import Tensor
from tinygrad.runtime.ops_tinytpu import _VPU_BOOL_OPS, _VPU_OPS, _infer_tiling, _parse_sim_output, _parse_vmem_output, _tiling_failure_note, _run_bundle, _build_full_gemm_bundle, _vmem, _wmem, _amem, _mxu_psum_write, _mxu_psum_acc, _mxu_os, _mxu_clear, _psum_read, _psum_read_row, _psum_clear, _wait_mxu, _load, _store, _halt, _output_vmem, _end, _bundle, _vpu, _vpu_exp2, _vpu_log2, _load_vpu_result, _load_xlu_result, _set_pred_if_zero, _skip_if_pred, _psum_accumulate_row


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

  def test_multi_k_tile_gemm_bias_relu_through_psum(self):
    # Stress the multi-K-tile GEMM hardware epilogue path introduced
    # by iters 11-13: randomized int8 weights + int32 activations with
    # num_k_tiles>1 and both bias and relu in the epilogue. All tiles
    # go through SXU_PSUM_CLEAR -> N MXU-psum_acc -> PSUM_READ_ROW ->
    # VPU_ADD(bias) -> VPU_RELU -> STORE in hardware.
    rng = np.random.default_rng(42)
    a_np = rng.integers(-8, 9, size=(4, 8), dtype=np.int32)
    w_np = rng.integers(-8, 9, size=(8, 8), dtype=np.int32)
    b_np = rng.integers(-30, 30, size=(8,), dtype=np.int32)
    a_t = Tensor(a_np, dtype="int32", device="TINYTPU")
    w_t = Tensor(w_np, dtype="int32", device="TINYTPU")
    b_t = Tensor(b_np, dtype="int32", device="TINYTPU")
    result = ((a_t @ w_t) + b_t).relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a_np @ w_np + b_np, 0))

  def test_deep_k4_gemm_through_psum_matches_numpy(self):
    # num_k_tiles = 4 (16-wide K) pushes the PSUM accumulator chain
    # further than the deep-K existing tests. Without bias/relu so
    # just the pure accumulate path is exercised.
    rng = np.random.default_rng(7)
    a_np = rng.integers(-4, 5, size=(4, 16), dtype=np.int32)
    w_np = rng.integers(-4, 5, size=(16, 4), dtype=np.int32)
    a_t = Tensor(a_np, dtype="int32", device="TINYTPU")
    w_t = Tensor(w_np, dtype="int32", device="TINYTPU")
    result = (a_t @ w_t).numpy()
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

  def test_fsub_scalar_const_four_tile_matches_reference(self):
    a = np.arange(64, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 1.0).numpy()
    np.testing.assert_allclose(result, a - 1.0, rtol=1e-5)

  def test_fsub_scalar_const_4x4_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 1.0).numpy()
    np.testing.assert_allclose(result, a - 1.0, rtol=1e-5)

  def test_fsub_scalar_const_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 2.5).numpy()
    np.testing.assert_allclose(result, a - 2.5, rtol=1e-5)

  def test_fsub_scalar_const_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 0.5).numpy()
    np.testing.assert_allclose(result, a - 0.5, rtol=1e-5)

  def test_fsub_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 0.5).numpy()
    np.testing.assert_allclose(result, a - 0.5, rtol=1e-5)

  def test_frev_sub_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (1.5 - Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 1.5 - a, rtol=1e-5)

  def test_frev_sub_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = (10.0 - Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 10.0 - a, rtol=1e-5)

  def test_fneg_matches_reference(self):
    a = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fabs_matches_reference(self):
    a = np.array([-1.5, 2.0, -3.0, 4.5, -0.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_full_tile_matches_reference(self):
    a = np.arange(-8, 8, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_2d_matches_reference(self):
    a = np.array([[-1.5, 2.5], [3.5, -4.5]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_three_tile_matches_reference(self):
    a = np.arange(-24, 24, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_four_tile_matches_reference(self):
    a = np.arange(-32, 32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_five_tile_matches_reference(self):
    a = np.arange(-40, 40, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_scaled_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32) * 0.5
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_zero_mix_matches_reference(self):
    a = np.array([0.0, -0.0, 1.0, -1.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(np.abs(result), np.abs(a), rtol=1e-5)

  def test_fabs_five_elem_negative_matches_reference(self):
    a = -np.arange(1, 6, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_two_elem_negative_matches_reference(self):
    a = np.array([-1.5, -2.5], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_three_elem_mixed_matches_reference(self):
    a = np.array([1.0, -2.0, 3.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_single_elem_matches_reference(self):
    a = np.array([-5.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_negative_only_multi_tile_matches_reference(self):
    a = -np.arange(1, 33, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_fractional_matches_reference(self):
    a = np.array([-1.5, 2.5, -3.5, 4.5], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_4x4_matches_reference(self):
    a = np.arange(-8, 8, dtype=np.float32).reshape(4, 4)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_3x4_matches_reference(self):
    a = np.arange(-6, 6, dtype=np.float32).reshape(3, 4)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_5x3_matches_reference(self):
    a = np.arange(-7, 8, dtype=np.float32).reshape(5, 3)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_2d_3x2_matches_reference(self):
    a = np.array([[-1.0, 2.0], [-3.0, 4.0], [-5.0, 6.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fabs_3d_matches_reference(self):
    a = np.arange(-4, 4, dtype=np.float32).reshape(2, 2, 2)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_fneg_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fneg_2d_fractional_matches_reference(self):
    a = np.array([[1.5, -2.5], [3.5, -4.5]], dtype=np.float32)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fneg_2d_matches_reference(self):
    a = np.array([[1.0, -2.0], [3.0, -4.0]], dtype=np.float32)
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

  def test_fneg_four_tile_matches_reference(self):
    a = np.arange(64, dtype=np.float32) - 32
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fneg_4x4_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fneg_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32) - 24
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_fadd_signed_three_tile_matches_reference(self):
    a = np.arange(-24, 24, dtype=np.float32)
    b = (24 - np.arange(48, dtype=np.float32)).astype(np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_signed_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    b = (16 - np.arange(32, dtype=np.float32)).astype(np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_signed_full_tile_matches_reference(self):
    a = np.arange(-8, 8, dtype=np.float32)
    b = np.arange(8, -8, -1, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_signed_matches_reference(self):
    a = np.array([-1.5, 2.0, -3.0, 4.5], dtype=np.float32)
    b = np.array([0.5, -0.5, 1.0, -2.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fmul_signed_full_tile_matches_reference(self):
    a = np.arange(-8, 8, dtype=np.float32)
    b = np.arange(8, -8, -1, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_fmul_signed_matches_reference(self):
    a = np.array([-1.5, 2.0, -3.0, 4.5], dtype=np.float32)
    b = np.array([2.0, -3.0, 0.5, -1.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_exp2_matches_reference(self):
    a = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").exp2().numpy()
    # Remez minimax 1 + 0.7344x + 0.25x² on [-1,1]: exact at 0, 0.8%
    # low at 1, 13% low at 2 (poly error grows outside fit range),
    # 3% high at -1. Peak absolute error 0.53 at x=2.
    np.testing.assert_allclose(result, 2.0 ** a, atol=0.6)

  def test_exp2_full_tile_matches_reference(self):
    # Full 16-lane tile over [-1.5, 1.5] through one VPU_EXP2 dispatch.
    # Remez peak absolute error on this range ~ 0.17 at ±1.5 (vs ~0.25
    # for the earlier Taylor-form polynomial).
    a = np.linspace(-1.5, 1.5, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").exp2().numpy()
    expected = 2.0 ** a
    np.testing.assert_allclose(result, expected, atol=0.2)

  def test_exp2_tight_band_inside_fit_range(self):
    # Inside the Remez [-1, 1] fit range, max |err| should be < 0.02
    # absolute. Locks the coefficient upgrade so any future coefficient
    # regression (e.g. reverting to Taylor) fails here loudly.
    a = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").exp2().numpy()
    expected = 2.0 ** a
    np.testing.assert_allclose(result, expected, atol=0.02)

  def test_exp2_two_tile_matches_reference(self):
    # Two tiles (32 elements) to exercise the renderer's per-tile
    # dispatch loop through the multi-cycle TranscUnit.
    a = np.linspace(-1.5, 1.5, 32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").exp2().numpy()
    expected = 2.0 ** a
    np.testing.assert_allclose(result, expected, atol=0.3)

  def test_log2_powers_of_two_matches_reference(self):
    # Exact integer outputs: range reduction lands m=1.0 for all inputs,
    # so the polynomial contributes 0 and the exponent-as-float dominates.
    a = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").log2().numpy()
    np.testing.assert_allclose(result, np.log2(a), atol=0.05)

  def test_log_matches_reference(self):
    # tinygrad lowers log(x) = log2(x) · ln(2). scaled_log2 renderer
    # picks that up — LOG2 + FMUL. Pow-of-2 inputs are exact (LOG2
    # range reduction nails them); others carry Remez error.
    a = np.array([1.0, 2.0, 4.0, 8.0, 0.5, 0.25], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").log().numpy()
    np.testing.assert_allclose(result, np.log(a), atol=0.05)

  def test_log2_non_powers_matches_reference(self):
    # Non-power inputs hit the degree-2 polynomial in u=m-1 ∈ (0,1).
    # Remez peak error ~0.034 (vs Taylor's ~0.28). Tight band locks
    # the coefficient win.
    a = np.array([1.1, 1.25, 1.5, 1.75, 2.5, 3.0, 3.5, 5.0,
                  0.25, 0.4, 0.6, 0.8, 10.0, 12.0, 16.0, 20.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").log2().numpy()
    np.testing.assert_allclose(result, np.log2(a), atol=0.04)

  def test_frecip_2x3_matches_reference(self):
    a = np.array([[1.0, 2.0, 4.0], [0.5, 8.0, 16.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_frecip_2d_matches_reference(self):
    a = np.array([[1.0, 2.0], [4.0, 0.5]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_frecip_full_tile_matches_reference(self):
    a = np.arange(1, 17, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_frecip_matches_reference(self):
    a = np.array([2.0, 4.0, 8.0, 16.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_fdiv_multi_tile_matches_reference(self):
    a = np.arange(1, 33, dtype=np.float32)
    b = np.full(32, 2.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-3)

  def test_fdiv_four_tile_matches_reference(self):
    a = np.arange(1, 65, dtype=np.float32)
    b = np.full(64, 2.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-3)

  def test_frecip_multi_tile_negative_matches_reference(self):
    a = np.arange(-16, -4, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_frecip_multi_tile_matches_reference(self):
    a = np.arange(1, 33, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_frecip_four_tile_matches_reference(self):
    a = np.arange(1, 65, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-3)

  def test_fmaximum_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(0.0).numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_fmaximum_positive_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(8.0).numpy()
    np.testing.assert_allclose(result, np.maximum(a, 8.0), rtol=1e-5)

  def test_fminimum_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(0.0).numpy()
    np.testing.assert_allclose(result, np.minimum(a, 0.0), rtol=1e-5)

  def test_fminimum_positive_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(5.0).numpy()
    np.testing.assert_allclose(result, np.minimum(a, 5.0), rtol=1e-5)

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

  def test_frelu_3x3_matches_reference(self):
    a = np.arange(-4, 5, dtype=np.float32).reshape(3, 3)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_2d_matches_reference(self):
    a = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_five_elem_matches_reference(self):
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_positive_only_full_tile_matches_reference(self):
    a = np.arange(1, 17, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, a, rtol=1e-5)

  def test_frelu_full_tile_matches_reference(self):
    a = (np.arange(16, dtype=np.float32) - 8) * 0.5
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_multi_tile_matches_reference(self):
    a = (np.arange(32, dtype=np.float32) - 16) * 0.5
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_four_tile_matches_reference(self):
    a = np.arange(64, dtype=np.float32) - 32
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_4x4_matches_reference(self):
    a = np.arange(-8, 8, dtype=np.float32).reshape(4, 4)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32) - 24
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_frelu_matches_reference(self):
    a = np.array([-1.5, 2.0, -3.0, 4.5, -0.0, 0.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_fminimum_2d_matches_reference(self):
    a = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    b = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.minimum(a, b), rtol=1e-5)

  def test_fminimum_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32)
    b = np.full(48, 20.0, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.minimum(a, b), rtol=1e-5)

  def test_fminimum_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16
    b = 16 - np.arange(32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.minimum(a, b), rtol=1e-5)

  def test_fmaximum_2d_matches_reference(self):
    a = np.array([[1.0, -2.0], [-3.0, 4.0]], dtype=np.float32)
    b = np.array([[-1.0, 2.0], [3.0, -4.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.maximum(a, b), rtol=1e-5)

  def test_fmaximum_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32)
    b = np.full(48, 20.0, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(
      Tensor(b, dtype="float", device="TINYTPU")
    ).numpy()
    np.testing.assert_allclose(result, np.maximum(a, b), rtol=1e-5)

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

  def test_int_to_float_3_elem_matches_reference(self):
    result = Tensor([1, 2, 3], dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, np.array([1.0, 2.0, 3.0], dtype=np.float32), rtol=1e-5)

  def test_float_to_int_cast_matches_reference(self):
    result = Tensor([1.5, 2.7, -3.2, 0.0], dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, np.array([1, 2, -3, 0], dtype=np.int32))

  def test_float_to_int_2d_matches_reference(self):
    a = np.array([[1.5, 2.7], [3.1, -4.9]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_int_to_float_2d_matches_reference(self):
    a = np.array([[1, 2], [3, -4]], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, a.astype(np.float32), rtol=1e-5)

  def test_float_to_int_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32) - 24
    result = Tensor(a, dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_float_to_int_all_negative_matches_reference(self):
    a = np.array([-1.5, -2.7, -0.3, -9.8], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_int_float_int_roundtrip_matches_reference(self):
    a = np.array([1, 2, 3, -5, 10], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").cast("int32").numpy()
    np.testing.assert_array_equal(result, a)

  def test_int_to_float_signed_full_tile_matches_reference(self):
    a = np.arange(-8, 8, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, a.astype(np.float32), rtol=1e-5)

  def test_int_to_float_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, a.astype(np.float32), rtol=1e-5)

  def test_int_to_float_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, a.astype(np.float32), rtol=1e-5)

  def test_int_to_float_multi_tile_signed_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, a.astype(np.float32), rtol=1e-5)

  def test_float_to_int_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16
    result = Tensor(a, dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_float_sum_reduce_matches_reference(self):
    # Previously unsupported; now lowers through VPU_FSUM_REDUCE_TILE.
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sum().numpy()
    np.testing.assert_allclose(result, a.sum(), rtol=1e-6)

  def test_float_max_reduce_matches_reference(self):
    # Previously unsupported; now lowers through VPU_FMAX_REDUCE_TILE.
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").max().numpy()
    np.testing.assert_allclose(result, a.max(), rtol=1e-6)

  def test_float_max_reduce_full_tile_matches_reference(self):
    a = (np.arange(16, dtype=np.float32) - 8.0)
    result = Tensor(a, dtype="float32", device="TINYTPU").max().numpy()
    np.testing.assert_allclose(result, a.max(), rtol=1e-6)

  def test_float_max_reduce_multi_tile_matches_reference(self):
    a = (np.arange(32, dtype=np.float32) - 16.0) * 0.5
    result = Tensor(a, dtype="float32", device="TINYTPU").max().numpy()
    np.testing.assert_allclose(result, a.max(), rtol=1e-6)

  def test_float_max_reduce_all_negative_matches_reference(self):
    a = np.array([-4.0, -1.5, -3.0, -2.25], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").max().numpy()
    np.testing.assert_allclose(result, a.max(), rtol=1e-6)

  def test_float_min_reduce_matches_reference(self):
    # Previously unsupported; now lowers through VPU_FMIN_REDUCE_TILE
    # (single-tile only until VPU_FMIN ALU opcode is added).
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").min().numpy()
    np.testing.assert_allclose(result, a.min(), rtol=1e-6)

  def test_float_min_reduce_full_tile_matches_reference(self):
    a = (np.arange(16, dtype=np.float32) - 8.0)
    result = Tensor(a, dtype="float32", device="TINYTPU").min().numpy()
    np.testing.assert_allclose(result, a.min(), rtol=1e-6)

  def test_float_min_reduce_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) - 16.0
    result = Tensor(a, dtype="float32", device="TINYTPU").min().numpy()
    np.testing.assert_allclose(result, a.min(), rtol=1e-6)

  def test_float_min_reduce_all_negative_matches_reference(self):
    a = np.array([-0.5, -3.5, -1.0, -2.0], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").min().numpy()
    np.testing.assert_allclose(result, a.min(), rtol=1e-6)

  def test_float_rowsum_2x4_matches_reference(self):
    a = np.array([[1.0, 2.0, 3.0, 4.0], [0.5, 1.5, 2.5, 3.5]], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_allclose(result, a.sum(axis=1), rtol=1e-6)

  def test_float_rowsum_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.float32) - 4.0).reshape(3, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_allclose(result, a.sum(axis=1), rtol=1e-6)

  def test_float_rowmax_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.float32) - 5.0).reshape(3, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_allclose(result, a.max(axis=1), rtol=1e-6)

  def test_float_colsum_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.float32) - 4.0).reshape(3, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_allclose(result, a.sum(axis=0), rtol=1e-6)

  def test_float_colsum_4x4_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_allclose(result, a.sum(axis=0), rtol=1e-6)

  def test_float_colmax_4x4_matches_reference(self):
    a = (np.arange(16, dtype=np.float32) - 8.0).reshape(4, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_allclose(result, a.max(axis=0), rtol=1e-6)

  def test_float_rowmin_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.float32) - 5.0).reshape(3, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_allclose(result, a.min(axis=1), rtol=1e-6)

  def test_float_rowmin_2x4_matches_reference(self):
    a = np.array([[3.0, 1.0, 4.0, 2.0], [0.5, -1.0, 5.0, 0.0]], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_allclose(result, a.min(axis=1), rtol=1e-6)

  def test_float_colmin_3x4_matches_reference(self):
    a = (np.arange(12, dtype=np.float32) - 4.0).reshape(3, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_allclose(result, a.min(axis=0), rtol=1e-6)

  def test_float_colmin_4x4_matches_reference(self):
    a = (np.arange(16, dtype=np.float32) - 8.0).reshape(4, 4)
    result = Tensor(a, dtype="float32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_allclose(result, a.min(axis=0), rtol=1e-6)

  def test_float_prod_4elem_matches_reference(self):
    a = np.array([2.0, 3.0, 4.0, 0.5], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").prod().numpy()
    np.testing.assert_allclose(result, a.prod(), rtol=1e-6)

  def test_float_prod_full_tile_matches_reference(self):
    a = np.array([1.1]*8 + [0.9]*8, dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").prod().numpy()
    np.testing.assert_allclose(result, a.prod(), rtol=1e-5)

  def test_float_rowprod_3x4_matches_reference(self):
    a = np.array([[1.5, 2.0, 0.5, 3.0],
                  [1.0, 2.5, 0.8, 1.25],
                  [0.5, 2.0, 4.0, 0.25]], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").prod(axis=1).numpy()
    np.testing.assert_allclose(result, a.prod(axis=1), rtol=1e-5)

  def test_float_colprod_3x4_matches_reference(self):
    a = np.array([[1.5, 2.0, 0.5, 3.0],
                  [1.0, 2.5, 0.8, 1.25],
                  [0.5, 2.0, 4.0, 0.25]], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").prod(axis=0).numpy()
    np.testing.assert_allclose(result, a.prod(axis=0), rtol=1e-5)

  def test_int32_pad_1d_matches_reference(self):
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").pad((1, 1)).numpy()
    np.testing.assert_array_equal(result, np.pad(a, (1, 1)))

  def test_int32_pad_2_1_matches_reference(self):
    # Tinygrad unrolls small pads to scalar stores.
    a = np.array([5, 6, 7], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").pad((2, 1)).numpy()
    np.testing.assert_array_equal(result, np.pad(a, (2, 1)))

  def test_int32_flip_1d_matches_reference(self):
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").flip(0).numpy()
    np.testing.assert_array_equal(result, np.flip(a))


  def test_cos_matches_reference(self):
    # tinygrad lowers cos(x) = sin(-x + π/2), which the scaled_sin
    # renderer picks up (FMUL + FADD + SIN chain). Remez SIN peak
    # error is 1.2e-4 so the overall kernel is tight.
    a = np.array([0.0, np.pi/6, np.pi/4, np.pi/3], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").cos().numpy()
    expected = np.cos(a)
    np.testing.assert_allclose(result, expected, atol=1e-3)

  def test_cos_full_tile_matches_reference(self):
    # 16-lane tile over [0, π/2]: after the -x + π/2 shift the SIN
    # argument stays in [0, π/2] ⊂ the Remez fit range. Tight band.
    # Wider cos ranges need SIN range reduction (tracked separately).
    a = np.linspace(0.0, np.pi / 2, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").cos().numpy()
    expected = np.cos(a)
    np.testing.assert_allclose(result, expected, atol=2e-3)

  def test_tanh_small_inputs_matches_reference(self):
    # With EXP2 range reduction tanh is accurate across the whole real
    # line. |x| ≤ 2 (where tanh saturates) stays within 0.01 absolute.
    a = np.array([-2.0, -1.0, -0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 2.0],
                 dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").tanh().numpy()
    expected = np.tanh(a)
    np.testing.assert_allclose(result, expected, atol=0.01)

  def test_tanh_full_tile_matches_reference(self):
    # 16-lane tile sweep over [-2, 2]. Tight atol locks the range-reduced
    # EXP2 accuracy win for tanh.
    a = np.linspace(-2.0, 2.0, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").tanh().numpy()
    expected = np.tanh(a)
    np.testing.assert_allclose(result, expected, atol=0.02)

  def test_tanh_multi_tile_matches_reference(self):
    # 32 elements crossing a tile boundary to exercise the renderer's
    # per-tile replication of the tanh pipeline with EXP2 range
    # reduction engaged at each tile.
    a = np.linspace(-2.5, 2.5, 32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").tanh().numpy()
    expected = np.tanh(a)
    np.testing.assert_allclose(result, expected, atol=0.01)

  def test_sigmoid_matches_reference(self):
    # sigmoid(x) = 1/(1+exp(-x)); lowers to FMUL+EXP2+FADD+FRECIP chain.
    # After Remez EXP2 the compounded error is much smaller than with
    # Taylor — |x| ≤ 0.5 stays within 0.01 absolute.
    a = np.array([0.0, 0.3, -0.3, 0.5, -0.5], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sigmoid().numpy()
    expected = 1.0 / (1.0 + np.exp(-a))
    np.testing.assert_allclose(result, expected, atol=0.01)

  def test_sigmoid_wide_range_matches_reference(self):
    # With EXP2 range reduction sigmoid stays accurate across a wide
    # range — previously only |x| ≤ 0.7 worked.
    a = np.linspace(-4.0, 4.0, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sigmoid().numpy()
    expected = 1.0 / (1.0 + np.exp(-a))
    np.testing.assert_allclose(result, expected, atol=0.01)

  def test_sigmoid_multi_tile_matches_reference(self):
    # 32 elements crossing a tile boundary to exercise the renderer's
    # per-tile replication of the FMUL+EXP2+FADD+FRECIP chain.
    # EXP2 argument is -1/ln2·x ≈ -1.44·x; for |x| ≤ 0.694 it stays in
    # the Remez fit range. Inputs cover [-1, 1] so the outer |x| pay a
    # larger error — keep a modest band.
    a = np.linspace(-1.0, 1.0, 32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sigmoid().numpy()
    expected = 1.0 / (1.0 + np.exp(-a))
    np.testing.assert_allclose(result, expected, atol=0.05)

  def test_exp_wide_range_matches_reference(self):
    # With EXP2 range reduction Tensor.exp() stays within 2% relative
    # error across a wide range — previously only |x| ≤ 0.7 worked.
    a = np.linspace(-3.0, 3.0, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").exp().numpy()
    expected = np.exp(a)
    np.testing.assert_allclose(result, expected, rtol=0.02)

  def test_exp_small_inputs_matches_reference(self):
    # Tensor.exp(x) lowers to exp2(x * log2e); hardware runs EXP2 via
    # TranscUnit degree-2 Taylor, so exp(0)=1 exact, exp(1)≈2.66 (vs 2.718).
    # Loose rtol to match the approximation envelope.
    a = np.array([0.0, 0.3, 0.5, -0.3, -0.5], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").exp().numpy()
    np.testing.assert_allclose(result, np.exp(a), rtol=0.1, atol=0.02)

  def test_exp_multi_tile_matches_reference(self):
    # 32-element tile crossing exercises the scaled-exp2 renderer's
    # per-tile replication of FMUL+EXP2.
    a = np.linspace(-0.5, 0.5, 32, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").exp().numpy()
    np.testing.assert_allclose(result, np.exp(a), rtol=0.1, atol=0.05)

  def test_sqrt_of_one_is_exact(self):
    # sqrt(1) through LOG2→MUL→EXP2: log2(1)=0 exact, 0.5*0=0, exp2(0)=1
    # exact. So this is the one case that survives the compound Taylor
    # errors.
    a = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sqrt().numpy()
    np.testing.assert_allclose(result, np.sqrt(a), atol=0.05)

  def test_sqrt_dispatches_without_error(self):
    # After Remez LOG2/EXP2 + EXP2 range reduction the compound error
    # drops dramatically: sqrt of perfect powers of 4 is exact, and
    # other inputs are within ~3% relative.
    a = np.array([4.0, 16.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sqrt().numpy()
    self.assertTrue(np.all(np.isfinite(result)))
    self.assertTrue(np.all(result > 0))
    np.testing.assert_allclose(result, np.sqrt(a), rtol=0.03)

  def test_square_matches_reference(self):
    # Tensor(x) ** 2 lowers as MUL(x, x) with a single PARAM. The
    # elementwise renderer expects two PARAMs so it rejects; dedicated
    # self-square renderer emits LOAD + FMUL(v, v) + STORE.
    a = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") ** 2.0).numpy()
    np.testing.assert_array_equal(result, a ** 2.0)

  def test_silu_matches_reference(self):
    # swish / silu = x * sigmoid(x). tinygrad's UOp graph chains a
    # self-multiply around the sigmoid pattern, which the sigmoid
    # renderer alone won't match. Dedicated swish renderer emits the
    # full FMUL/EXP2/FADD/FRECIP/FMUL microprogram per tile.
    a = np.array([-3., -2., -1., 0., 1., 2., 3.], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").silu().numpy()
    expected = a / (1.0 + np.exp(-a))
    np.testing.assert_allclose(result, expected, atol=0.005)

  def test_cube_matches_reference(self):
    # Tensor(x) ** 3 lowers as MUL(x, MUL(x, x)) — a three-way self-
    # multiply that neither the elementwise nor self-square renderer
    # matches. Dedicated self-cube renderer emits two FMULs per tile.
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") ** 3.0).numpy()
    np.testing.assert_allclose(result, a ** 3.0, atol=1e-4)

  def test_log_compound_input_not_silently_log(self):
    # Regression: log(x+5) had the scaled-log2 renderer match on
    # MUL(LOG2(...), ln(2)) and silently emit LOG2(x) — dropping the
    # "+5" shift. Renderer now requires the LOG2 input to terminate at a
    # plain LOAD; compound expressions route to UNSUPPORTED.
    a = np.array([-3., -2., -1., 0., 1., 2., 3.], dtype=np.float32)
    try:
      got = (Tensor(a, dtype="float", device="TINYTPU") + 5).log().numpy()
    except NotImplementedError:
      return
    np.testing.assert_allclose(got, np.log(a + 5), atol=0.02)

  def test_sum_x_times_x_not_silently_sum(self):
    # Regression: sum(x*x) / sum(-x) / max(-x) used to silently return
    # sum(x) / max(x) because the scalar-reduce renderer ignored
    # pre-reduction data-path MULs. Guard now rejects kernels with
    # extra data-path MUL unless a post_op detection fires (sum(x*c)
    # still works).
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    for fn, ref in [(lambda t: (t*t).sum(), (a*a).sum()),
                    (lambda t: (-t).sum(), (-a).sum()),
                    (lambda t: (-t).max(), (-a).max())]:
      try:
        got = fn(Tensor(a, dtype="int32", device="TINYTPU")).numpy()
      except NotImplementedError:
        continue
      np.testing.assert_array_equal(got, ref)

  def test_abs_sum_not_silently_sum(self):
    # Regression: sum(abs(x)) was silently returning sum(x) because the
    # reducer ignored the WHERE+CMPLT-based abs subtree. The renderer now
    # refuses to lower when conditional selection ops appear inside the
    # reduce chain.
    a = np.array([-3., -2., -1., 0., 1., 2., 3.], dtype=np.float32)
    try:
      got = Tensor(a, dtype="float", device="TINYTPU").abs().sum().numpy()
    except NotImplementedError:
      return
    np.testing.assert_allclose(got, np.abs(a).sum(), rtol=1e-5)

  def test_reciprocal_sum_not_silently_sum(self):
    # Regression: `x.reciprocal().sum()` fused as a scalar SUM kernel
    # with a pre-reduction RECIPROCAL in the data path used to silently
    # drop the RECIPROCAL and sum the raw input. The reduction renderer
    # now rejects kernels carrying a pre-reduction transcendental or
    # RECIPROCAL so fused graphs fall through to UNSUPPORTED instead of
    # returning a wrong result.
    a = np.array([1., 2., 3., 4.], dtype=np.float32)
    try:
      got = Tensor(a, dtype="float", device="TINYTPU").reciprocal().sum().numpy()
    except NotImplementedError:
      return
    np.testing.assert_allclose(got, (1.0 / a).sum(), rtol=1e-5)

  def test_exp_sum_not_silently_sum(self):
    # Regression counterpart: exp().sum() must not silently skip the exp.
    a = np.array([0., 1., 2.], dtype=np.float32)
    try:
      got = Tensor(a, dtype="float", device="TINYTPU").exp().sum().numpy()
    except NotImplementedError:
      return
    np.testing.assert_allclose(got, np.exp(a).sum(), rtol=0.02)

  def test_float_mean_axis1_matches_reference(self):
    # Row-reduce mean: Tensor.mean(axis=1) fuses FSUM_REDUCE + FMUL(1/ncols).
    a = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=np.float32)
    got = Tensor(a, dtype="float", device="TINYTPU").mean(axis=1).numpy()
    np.testing.assert_allclose(got, a.mean(axis=1), rtol=1e-5)

  def test_float_mean_axis0_matches_reference(self):
    # Regression: Tensor.mean(axis=0) fuses into a single col-reduce
    # kernel: FSUM_REDUCE_COL followed by a FMUL against the scalar
    # 1/nrows. The col-reduce renderer used to silently ignore the
    # post-op MUL and emit pure sum. Now it detects the float CONST
    # factor and appends an FMUL.
    a = np.array([[1., 2., 3., 4.], [5., 6., 7., 8.]], dtype=np.float32)
    got = Tensor(a, dtype="float", device="TINYTPU").mean(axis=0).numpy()
    np.testing.assert_allclose(got, a.mean(axis=0), rtol=1e-5)

  def test_float_mean_matches_reference(self):
    # Regression: Tensor.mean() decomposes as sum * (1/N). The scalar
    # reduction renderer's post-op used integer MUL / ADD even when the
    # reduction was float, so mean() silently produced 0 (int reinterpret
    # of the float constant). Now it remaps to FMUL / FADD for float
    # reductions.
    for a in ([1., 2., 3., 4.],
              [1., 2., 3., 4., 5., 6., 7., 8.],
              [-2., 0., 2., 4.]):
      a_np = np.array(a, dtype=np.float32)
      got = Tensor(a_np, dtype="float", device="TINYTPU").mean().numpy()
      np.testing.assert_allclose(got, a_np.mean(), rtol=1e-5)

  def test_hardtanh_not_silently_relu(self):
    # Regression: hardtanh = clip(x, -1, 1) has the WHERE+CMPLT shape
    # of RELU but with two comparisons per element (one for min, one
    # for max). Without the per-element CMPLT guard the elementwise
    # renderer silently emitted RELU, returning max(x, 0) instead of
    # clip. Renderer now either returns a correct clip result or
    # UNSUPPORTED — never RELU.
    a = np.array([-3., -2., -1., 0., 1., 2., 3.], dtype=np.float32)
    try:
      result = Tensor(a, dtype="float", device="TINYTPU").hardtanh().numpy()
    except NotImplementedError:
      return
    np.testing.assert_allclose(result, np.clip(a, -1., 1.), atol=1e-5,
        err_msg="hardtanh must not silently lower as RELU")

  def test_softsign_matches_reference(self):
    # softsign = x / (1 + |x|). Dedicated renderer composes the abs
    # pattern with FADD(1) + FRECIP + FMUL so the full expression
    # lowers instead of the abs subtree being returned as "softsign".
    a = np.array([-3., -2., -1., 0., 1., 2., 3.], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").softsign().numpy()
    np.testing.assert_allclose(result, a / (1 + np.abs(a)), atol=1e-5)

  def test_self_square_rejects_power_of_four(self):
    # Regression guard: x**4 lowers as MUL(MUL(x,x), MUL(x,x)) — both
    # outer MUL operands are the SAME MUL(x,x) UOp after dedup, so the
    # original self-square pattern would false-match and silently return
    # x**2 as the "x**4" result. The renderer now verifies the factor
    # terminates at a LOAD (not a compound MUL), so x**4 must either
    # be rendered by a cube-or-higher renderer or explicitly fall
    # through to UNSUPPORTED — never silently down-grade to x**2.
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    try:
      result = (Tensor(a, dtype="float", device="TINYTPU") ** 4.0).numpy()
    except NotImplementedError:
      return   # acceptable — no higher-power renderer yet
    np.testing.assert_allclose(result, a ** 4.0, atol=1e-4,
        err_msg="x**4 must not silently lower as x**2")

  def test_rsqrt_matches_reference(self):
    # Tensor.rsqrt() = RECIPROCAL(SQRT(x)). The dedicated rsqrt renderer
    # emits a LOG2 -> FMUL(-0.5) -> EXP2 microprogram, saving one
    # divide vs naive 1/sqrt(x). Powers of 4 are exact.
    a = np.array([1.0, 4.0, 16.0, 0.25], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").rsqrt().numpy()
    np.testing.assert_allclose(result, 1.0 / np.sqrt(a), rtol=0.03)

  def test_sqrt_wide_range_matches_reference(self):
    # Spot-check a wider set: perfect squares exact, others within 3%.
    a = np.array([1.0, 4.0, 9.0, 16.0, 0.25, 0.5, 2.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sqrt().numpy()
    np.testing.assert_allclose(result, np.sqrt(a), rtol=0.03)

  def test_sin_small_angle_matches_reference(self):
    # |x| <= π/2 stays inside Remez degree-5 accuracy band (<5e-4 abs).
    a = np.array([-1.4, -0.7, 0.0, 0.7, 1.4], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sin().numpy()
    np.testing.assert_allclose(result, np.sin(a), atol=1e-3)

  def test_sin_full_tile_matches_reference(self):
    # Wider sweep still under |x| <= π/2; single-tile through VPU_SIN.
    # Remez peak error on [-π/2, π/2] is 1.2e-4 — tight band locks it.
    a = np.linspace(-1.5, 1.5, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sin().numpy()
    np.testing.assert_allclose(result, np.sin(a), atol=5e-4)

  def test_sin_wide_angle_matches_reference(self):
    # With mod-2π + quadrant-fold range reduction sin is accurate
    # outside the Remez fit window too. Peak absolute error on [-3π, 3π]
    # is ~5e-3 (compounded fold + Remez).
    a = np.linspace(-3 * np.pi, 3 * np.pi, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").sin().numpy()
    np.testing.assert_allclose(result, np.sin(a), atol=5e-3)

  def test_cos_wide_range_matches_reference(self):
    # Wide cos via scaled_sin(-x + π/2) also benefits from SIN's range
    # reduction. Previously restricted to [0, π/2]; now accurate out to
    # [-3π, 3π].
    a = np.linspace(-3 * np.pi, 3 * np.pi, 16, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").cos().numpy()
    np.testing.assert_allclose(result, np.cos(a), atol=5e-3)

  def test_fdiv_3x3_matches_reference(self):
    a = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    b = np.full((3, 3), 2.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-3)

  def test_fdiv_3x3_scalar_matches_reference(self):
    a = np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    result = (Tensor(a, dtype="float", device="TINYTPU") / 2.0).numpy()
    np.testing.assert_allclose(result, a / 2.0, rtol=1e-3)

  def test_fdiv_tensor_tensor_signed_matches_reference(self):
    a = np.array([8.0, -6.0, 12.0, -4.0], dtype=np.float32)
    b = np.array([2.0, -3.0, -4.0, 2.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-3)

  def test_fdiv_tensor_tensor_matches_reference(self):
    a = np.array([4.0, 6.0, 8.0, 10.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 4.0, 5.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-3)

  def test_frev_fdiv_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
    result = (4.0 / Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 4.0 / a, rtol=1e-3)

  def test_frev_fdiv_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(1, 17, dtype=np.float32)
    result = (4.0 / Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 4.0 / a, rtol=1e-3)

  def test_fdiv_scalar_const_matches_reference(self):
    a = np.array([2.0, 4.0, 8.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / 2.0).numpy()
    np.testing.assert_allclose(result, a / 2.0, rtol=1e-3)

  def test_fdiv_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(1, 33, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / 4.0).numpy()
    np.testing.assert_allclose(result, a / 4.0, rtol=1e-3)

  def test_fadd_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    b = np.full((3, 3), 2.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_2d_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_five_tile_matches_reference(self):
    a = np.arange(80, dtype=np.float32)
    b = np.full(80, 2.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_four_tile_matches_reference(self):
    a = np.arange(64, dtype=np.float32)
    b = np.full(64, 1.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32)
    b = np.full(48, 20.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_fadd_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) * 0.5
    b = np.arange(32, dtype=np.float32) * 0.25
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_frev_mul_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = (3.0 * Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 3.0 * a, rtol=1e-5)

  def test_fmul_scalar_const_four_tile_matches_reference(self):
    a = np.arange(64, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_fmul_scalar_const_4x4_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_fmul_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) * 0.5
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_fadd_positive_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 5.0).numpy()
    np.testing.assert_allclose(result, a + 5.0, rtol=1e-5)

  def test_fsub_rev_negative_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - (-3.0)).numpy()
    np.testing.assert_allclose(result, a + 3.0, rtol=1e-5)

  def test_fadd_scalar_const_five_tile_matches_reference(self):
    a = np.arange(80, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 1.0).numpy()
    np.testing.assert_allclose(result, a + 1.0, rtol=1e-5)

  def test_fadd_scalar_const_four_tile_matches_reference(self):
    a = np.arange(64, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 1.0).numpy()
    np.testing.assert_allclose(result, a + 1.0, rtol=1e-5)

  def test_fadd_scalar_const_4x4_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 2.0).numpy()
    np.testing.assert_allclose(result, a + 2.0, rtol=1e-5)

  def test_fadd_scalar_const_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 2.5).numpy()
    np.testing.assert_allclose(result, a + 2.5, rtol=1e-5)

  def test_frev_add_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = (2.5 + Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 2.5 + a, rtol=1e-5)

  def test_fadd_scalar_const_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 0.5).numpy()
    np.testing.assert_allclose(result, a + 0.5, rtol=1e-5)

  def test_fadd_scalar_const_2d_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 1.5).numpy()
    np.testing.assert_allclose(result, a + 1.5, rtol=1e-5)

  def test_fadd_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 2.5).numpy()
    np.testing.assert_allclose(result, a + 2.5, rtol=1e-5)

  def test_fadd_negative_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + -2.0).numpy()
    np.testing.assert_allclose(result, a + -2.0, rtol=1e-5)

  def test_fmul_scalar_const_3d_matches_reference(self):
    a = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 3.0).numpy()
    np.testing.assert_allclose(result, a * 3.0, rtol=1e-5)

  def test_fmul_scalar_const_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 0.5).numpy()
    np.testing.assert_allclose(result, a * 0.5, rtol=1e-5)

  def test_fmul_scalar_const_2d_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.5).numpy()
    np.testing.assert_allclose(result, a * 2.5, rtol=1e-5)

  def test_fmul_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 3.0).numpy()
    np.testing.assert_allclose(result, a * 3.0, rtol=1e-5)

  def test_fmul_negative_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * -2.0).numpy()
    np.testing.assert_allclose(result, a * -2.0, rtol=1e-5)

  def test_fcmpne_2d_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") != Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a != b)

  def test_fcmpne_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    b = np.arange(32, dtype=np.float32).copy()
    b[::2] += 1.0
    result = (Tensor(a, dtype="float", device="TINYTPU") != Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a != b)

  def test_fcmpne_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") != Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a != b)

  def test_fcmpeq_2d_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a == b)

  def test_fcmpeq_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    b = a.copy()
    b[::2] = 99.0
    result = (Tensor(a, dtype="float", device="TINYTPU") == Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a == b)

  def test_fcmpeq_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a == b)

  def test_fcmpgt_2d_matches_reference(self):
    a = np.array([[1.0, 5.0], [3.0, 0.0]], dtype=np.float32)
    b = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a > b)

  def test_fcmpgt_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    b = np.full(32, 15.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a > b)

  def test_fcmpgt_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    b = np.array([1.0, 0.0, 3.0, 0.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a > b)

  def test_fcmpgt_scalar_const_matches_reference(self):
    a = np.array([-1.0, 0.5, 2.0, 3.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > 1.0).numpy()
    np.testing.assert_array_equal(result, a > 1.0)

  def test_fcmpgt_fractional_scalar_const_matches_reference(self):
    a = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > 1.5).numpy()
    np.testing.assert_array_equal(result, a > 1.5)

  def test_fcmpgt_negative_scalar_const_matches_reference(self):
    a = np.array([-1.0, 0.5, 2.0, 3.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") > -0.5).numpy()
    np.testing.assert_array_equal(result, a > -0.5)

  def test_fcmpeq_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == 3.0).numpy()
    np.testing.assert_array_equal(result, a == 3.0)

  def test_fcmpeq_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == 5.0).numpy()
    np.testing.assert_array_equal(result, a == 5.0)

  def test_fcmpeq_2d_scalar_const_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 1.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == 1.0).numpy()
    np.testing.assert_array_equal(result, a == 1.0)

  def test_fcmpeq_fractional_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) * 0.5
    result = (Tensor(a, dtype="float", device="TINYTPU") == 4.0).numpy()
    np.testing.assert_array_equal(result, a == 4.0)

  def test_fcmpeq_negative_scalar_const_matches_reference(self):
    a = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == -2.0).numpy()
    np.testing.assert_array_equal(result, a == -2.0)

  def test_fcmpeq_fractional_scalar_const_matches_reference(self):
    a = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") == 2.5).numpy()
    np.testing.assert_array_equal(result, a == 2.5)

  def test_fcmpne_scalar_const_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") != 3.0).numpy()
    np.testing.assert_array_equal(result, a != 3.0)

  def test_fcmpne_fractional_scalar_const_matches_reference(self):
    a = np.array([1.5, 2.5, 3.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") != 2.5).numpy()
    np.testing.assert_array_equal(result, a != 2.5)

  def test_fcmpne_negative_scalar_const_matches_reference(self):
    a = np.array([-1.0, 0.5, 2.0, -0.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") != -0.5).numpy()
    np.testing.assert_array_equal(result, a != -0.5)

  def test_fcmplt_scalar_const_matches_reference(self):
    a = np.array([-1.0, 0.5, 2.0, 3.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 2.0).numpy()
    np.testing.assert_array_equal(result, a < 2.0)

  def test_fcmplt_fractional_scalar_const_matches_reference(self):
    a = np.array([1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 1.5).numpy()
    np.testing.assert_array_equal(result, a < 1.5)

  def test_fcmplt_negative_scalar_const_matches_reference(self):
    a = np.array([-1.0, 0.5, 2.0, 3.5], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < -0.5).numpy()
    np.testing.assert_array_equal(result, a < -0.5)

  def test_fcmplt_scalar_const_four_tile_matches_reference(self):
    a = np.arange(64, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 32.0).numpy()
    np.testing.assert_array_equal(result, a < 32.0)

  def test_fcmplt_scalar_const_4x4_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 8.0).numpy()
    np.testing.assert_array_equal(result, a < 8.0)

  def test_fcmpeq_scalar_const_4x4_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    result = (Tensor(a, dtype="float", device="TINYTPU") == 5.0).numpy()
    np.testing.assert_array_equal(result, a == 5.0)

  def test_fcmplt_scalar_const_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 0.0).numpy()
    np.testing.assert_array_equal(result, a < 0.0)

  def test_fcmplt_scalar_const_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 4.0).numpy()
    np.testing.assert_array_equal(result, a < 4.0)

  def test_fcmplt_2d_matches_reference(self):
    a = np.array([[1.0, 5.0], [3.0, 0.0]], dtype=np.float32)
    b = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a < b)

  def test_fcmplt_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32)
    b = np.full(32, 15.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") < Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a < b)

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

  def test_fwhere_derived_condition_matches_reference(self):
    lhs = np.arange(16, dtype=np.float32)
    rhs = np.arange(16, dtype=np.float32) + 100.0
    cond = np.arange(16) > 8
    result = Tensor.where(Tensor(cond, device="TINYTPU"),
                          Tensor(lhs, dtype="float", device="TINYTPU"),
                          Tensor(rhs, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.where(cond, lhs, rhs), rtol=1e-5)

  def test_fwhere_2d_matches_reference(self):
    cond = np.array([[True, False], [False, True]])
    lhs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    rhs = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    result = Tensor.where(Tensor(cond, device="TINYTPU"),
                          Tensor(lhs, dtype="float", device="TINYTPU"),
                          Tensor(rhs, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.where(cond, lhs, rhs), rtol=1e-5)

  def test_fwhere_three_tile_matches_reference(self):
    cond = np.array([True, False] * 24)
    lhs = np.arange(48, dtype=np.float32) - 24
    rhs = np.arange(48, dtype=np.float32)
    result = Tensor.where(Tensor(cond, device="TINYTPU"),
                          Tensor(lhs, dtype="float", device="TINYTPU"),
                          Tensor(rhs, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.where(cond, lhs, rhs), rtol=1e-5)

  def test_fwhere_signed_rhs_multi_tile_matches_reference(self):
    cond = np.array([True, False, True, False] * 8)
    lhs = np.arange(32, dtype=np.float32)
    rhs = -np.arange(32, dtype=np.float32)
    result = Tensor.where(Tensor(cond, device="TINYTPU"),
                          Tensor(lhs, dtype="float", device="TINYTPU"),
                          Tensor(rhs, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.where(cond, lhs, rhs), rtol=1e-5)

  def test_fwhere_multi_tile_matches_reference(self):
    cond = np.array([True, False] * 16)
    lhs = np.arange(32, dtype=np.float32)
    rhs = np.arange(32, dtype=np.float32) + 100.0
    result = Tensor.where(Tensor(cond, device="TINYTPU"),
                          Tensor(lhs, dtype="float", device="TINYTPU"),
                          Tensor(rhs, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.where(cond, lhs, rhs), rtol=1e-5)

  def test_fmul_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    b = np.full((3, 3), 2.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_fmul_2d_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    b = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_fmul_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32)
    b = np.full(48, 0.5, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_fmul_signed_multi_tile_matches_reference(self):
    a = np.arange(-16, 16, dtype=np.float32)
    b = np.arange(16, -16, -1, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_fsub_3x3_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3)
    b = np.full((3, 3), 1.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

  def test_fsub_2d_matches_reference(self):
    a = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    b = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

  def test_fsub_three_tile_matches_reference(self):
    a = np.arange(48, dtype=np.float32)
    b = np.full(48, 20.0, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

  def test_fsub_2x3_matches_reference(self):
    a = np.zeros((2, 3), dtype=np.float32)
    b = np.ones((2, 3), dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

  def test_fsub_signed_full_tile_matches_reference(self):
    a = np.arange(-8, 8, dtype=np.float32)
    b = np.arange(8, -8, -1, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

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

  def test_cmplt_column_broadcast_4x1_to_4x4_matches_reference(self):
    lhs = (np.arange(16, dtype=np.int32) - 4).reshape(4, 4)
    rhs = np.array([[-2], [0], [2], [4]], dtype=np.int32)
    result = (Tensor(lhs, dtype="int32", device="TINYTPU") <
              Tensor(rhs, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, lhs < rhs)

  def test_post_run_review_model_matches_reference(self):
    np.random.seed(7)
    x = np.random.randint(-3, 4, size=(4, 4), dtype=np.int32)
    w = np.random.randint(-2, 3, size=(4, 4), dtype=np.int32)
    b = np.random.randint(-1, 2, size=(4,), dtype=np.int32)
    th = np.random.randint(-2, 3, size=(4, 1), dtype=np.int32)

    y = (Tensor(x, dtype="int32", device="TINYTPU") @
         Tensor(w, dtype="int32", device="TINYTPU") +
         Tensor(b, dtype="int32", device="TINYTPU")).relu().realize()
    mask = (y < Tensor(th, dtype="int32", device="TINYTPU"))
    sel = mask.where(y, y * 2).realize()
    row_sum = sel.sum(axis=1).numpy()
    col_max = sel.max(axis=0).numpy()
    tile_min = sel.min().numpy()

    y_np = np.maximum(x @ w + b, 0)
    mask_np = y_np < th
    sel_np = np.where(mask_np, y_np, y_np * 2)
    np.testing.assert_array_equal(y.numpy(), y_np)
    np.testing.assert_array_equal(mask.numpy(), mask_np)
    np.testing.assert_array_equal(sel.numpy(), sel_np)
    np.testing.assert_array_equal(row_sum, sel_np.sum(axis=1))
    np.testing.assert_array_equal(col_max, sel_np.max(axis=0))
    np.testing.assert_array_equal(tile_min, int(sel_np.min()))

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

  def test_lowering_dump_records_column_broadcast_compare_as_sxu_program(self):
    with tempfile.TemporaryDirectory() as td:
      dump = Path(td) / "lowering.jsonl"
      env = {**os.environ, "PYTHONPATH": str(REPO_ROOT / "tinygrad"), "TINYTPU_DUMP_LOWERING": str(dump)}
      proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent("""\
          import numpy as np
          from tinygrad import Tensor
          lhs = Tensor((np.arange(16, dtype=np.int32) - 4).reshape(4, 4), dtype="int32", device="TINYTPU")
          rhs = Tensor(np.array([[-2], [0], [2], [4]], dtype=np.int32), dtype="int32", device="TINYTPU")
          print((lhs < rhs).numpy())
        """)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        env=env,
        check=False,
      )
      self.assertEqual(proc.returncode, 0, msg=proc.stdout + "\n" + proc.stderr)
      records = [json.loads(line) for line in dump.read_text(encoding="utf-8").splitlines()]
      self.assertTrue(any(r.get("op") == "SXU_PROGRAM" and r.get("primitive") == "BROADCAST_COL" and any(instr.startswith("2 11 ") for instr in r.get("instructions", [])) for r in records), records)

  def test_int32_3d_add_matches_reference(self):
    a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    b = np.ones((2, 3, 4), dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_fneg_unary_matches_reference(self):
    a = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_int32_shl_scalar_matches_reference(self):
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") << 1).numpy()
    np.testing.assert_array_equal(result, a << 1)

  def test_int32_rowmin_3x4_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, a.min(axis=1))

  def test_f2i_three_tile_mixed_matches_reference(self):
    a = np.arange(-5, 43, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_bool_xor_2d_matches_reference(self):
    a = np.array([[True, False], [True, True]])
    b = np.array([[False, False], [True, False]])
    result = (Tensor(a, dtype="bool", device="TINYTPU") ^ Tensor(b, dtype="bool", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a ^ b)

  def test_fmin_three_tile_tt_48elem_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 8, dtype=np.float32)
    b = np.full(48, 3.0, dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.minimum(a, b), rtol=1e-5)

  def test_int32_3x3_mul_tt_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * Tensor(a, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * a)

  def test_int32_3x3_add_tt_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    b = a + 1
    result = (Tensor(a, dtype="int32", device="TINYTPU") + Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a + b)

  def test_int32_3x3_sub_tt_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    b = a + 5
    result = (Tensor(b, dtype="int32", device="TINYTPU") - Tensor(a, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, b - a)

  def test_int32_5x5_add_scalar_matches_reference(self):
    a = np.arange(25, dtype=np.int32).reshape(5, 5)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 10).numpy()
    np.testing.assert_array_equal(result, a + 10)

  def test_float32_5x5_relu_matches_reference(self):
    a = np.arange(-10, 15, dtype=np.float32).reshape(5, 5)
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0), rtol=1e-5)

  def test_float32_4d_fmul_scalar_matches_reference(self):
    a = np.arange(24, dtype=np.float32).reshape(2, 3, 4, 1)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 3.0).numpy()
    np.testing.assert_allclose(result, a * 3.0, rtol=1e-5)

  def test_int32_three_tile_max_scalar_const_matches_reference(self):
    a = np.arange(-20, 28, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").maximum(0).numpy()
    np.testing.assert_array_equal(result, np.maximum(a, 0))

  def test_int32_three_tile_min_scalar_const_matches_reference(self):
    a = np.arange(-20, 28, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").minimum(0).numpy()
    np.testing.assert_array_equal(result, np.minimum(a, 0))

  def test_int32_2x2_cmplt_tt_matches_reference(self):
    a = np.array([[1, 2], [3, 4]], dtype=np.int32)
    b = np.array([[2, 2], [2, 2]], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") < Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a < b)

  def test_int32_5x5_cmpeq_scalar_const_matches_reference(self):
    a = np.arange(25, dtype=np.int32).reshape(5, 5)
    result = (Tensor(a, dtype="int32", device="TINYTPU") == 12).numpy()
    np.testing.assert_array_equal(result, a == 12)

  def test_int32_shr_by_two_matches_reference(self):
    a = np.array([16, 32, 64], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") >> 2).numpy()
    np.testing.assert_array_equal(result, a >> 2)

  def test_int32_2x4_idiv_scalar_const_matches_reference(self):
    a = np.arange(8, dtype=np.int32).reshape(2, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") // 3).numpy()
    np.testing.assert_array_equal(result, a // 3)

  def test_int32_2x4_imod_scalar_const_matches_reference(self):
    a = np.arange(8, dtype=np.int32).reshape(2, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") % 3).numpy()
    np.testing.assert_array_equal(result, a % 3)

  def test_float32_5elem_fneg_multi_tile_matches_reference(self):
    a = np.arange(5, dtype=np.float32)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_int32_3x3_cast_bool_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("bool").numpy()
    np.testing.assert_array_equal(result, a.astype(bool))

  def test_bool_3x3_cast_int32_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3).astype(bool)
    result = Tensor(a, dtype="bool", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_int32_17elem_cmplt_scalar_const_matches_reference(self):
    a = np.arange(17, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") < 8).numpy()
    np.testing.assert_array_equal(result, a < 8)

  def test_int32_17elem_cmpne_scalar_const_matches_reference(self):
    a = np.arange(17, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") != 5).numpy()
    np.testing.assert_array_equal(result, a != 5)

  def test_int32_3x3_rowmax_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, a.max(axis=1))

  def test_int32_3x3_colmin_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, a.min(axis=0))

  def test_int32_2x3_rowprod_matches_reference(self):
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").prod(axis=1).numpy()
    np.testing.assert_array_equal(result, a.prod(axis=1))

  def test_int32_2x3_colprod_matches_reference(self):
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").prod(axis=0).numpy()
    np.testing.assert_array_equal(result, a.prod(axis=0))

  def test_int32_2x2x2_scalar_sum_matches_reference(self):
    a = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_float32_2x2_negative_scalar_const_fmul_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * -3.0).numpy()
    np.testing.assert_allclose(result, a * -3.0, rtol=1e-5)

  def test_float32_6elem_rev_fsub_matches_reference(self):
    a = np.arange(6, dtype=np.float32)
    result = (2.0 - Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 2.0 - a, rtol=1e-5)

  def test_float32_7elem_rev_fadd_matches_reference(self):
    a = np.arange(7, dtype=np.float32)
    result = (5.0 + Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, 5.0 + a, rtol=1e-5)

  def test_float32_2x3_fcmplt_scalar_const_matches_reference(self):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = (Tensor(a, dtype="float", device="TINYTPU") < 3.0).numpy()
    np.testing.assert_array_equal(result, a < 3.0)

  def test_int32_40elem_and_tt_matches_reference(self):
    a = np.arange(40, dtype=np.int32)
    b = np.full(40, 0xF, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") & Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a & b)

  def test_int32_3elem_shl_by_three_matches_reference(self):
    a = np.array([1, 2, 3], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") << 3).numpy()
    np.testing.assert_array_equal(result, a << 3)

  def test_int32_1x16_add_const_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(1, 16)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_16x1_add_const_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(16, 1)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_4x4_xor_tt_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") ^ Tensor(a, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a ^ a)

  def test_int32_4x4_or_const_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") | 0xFF).numpy()
    np.testing.assert_array_equal(result, a | 0xFF)

  def test_int32_32elem_sum_matches_reference(self):
    a = np.arange(32, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_int32_scalar_plus_tensor_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = (5 + Tensor(a, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, 5 + a)

  def test_int32_slice_matches_reference(self):
    a = np.arange(8, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU")[2:6].numpy()
    np.testing.assert_array_equal(result, a[2:6])

  def test_int32_reshape_2d_to_1d_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").reshape(12).numpy()
    np.testing.assert_array_equal(result, a.reshape(12))

  def test_int32_reshape_1d_to_4d_matches_reference(self):
    a = np.arange(24, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").reshape(2, 3, 4, 1).numpy()
    np.testing.assert_array_equal(result, a.reshape(2, 3, 4, 1))

  def test_int32_chained_add_const_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = ((Tensor(a, dtype="int32", device="TINYTPU") + 1) + 2).numpy()
    np.testing.assert_array_equal(result, a + 3)

  def test_int32_chained_mul_const_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = ((Tensor(a, dtype="int32", device="TINYTPU") * 2) * 3).numpy()
    np.testing.assert_array_equal(result, a * 6)

  def test_float32_5elem_scalar_fclip_max0_matches_reference(self):
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(0.0).numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0.0), rtol=1e-5)

  def test_float32_5elem_scalar_fmin0_matches_reference(self):
    a = np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(0.0).numpy()
    np.testing.assert_allclose(result, np.minimum(a, 0.0), rtol=1e-5)

  def test_bool_3elem_cast_int_add_matches_reference(self):
    a = np.array([True, False, True], dtype=bool)
    result = (Tensor(a, dtype="bool", device="TINYTPU").cast("int32") + 1).numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32) + 1)

  def test_int32_bool_where_4elem_matches_reference(self):
    cond = np.array([True, False, True, False])
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([10, 20, 30, 40], dtype=np.int32)
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(
      Tensor(a, dtype="int32", device="TINYTPU"),
      Tensor(b, dtype="int32", device="TINYTPU"),
    ).numpy()
    np.testing.assert_array_equal(result, np.where(cond, a, b))

  def test_float32_fabs_three_negative_only_matches_reference(self):
    a = np.array([-3.0, -2.0, -1.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_int32_iabs_6elem_signed_matches_reference(self):
    a = np.array([-3, -2, -1, 1, 2, 3], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_float32_frecip_2x2_matches_reference(self):
    a = np.array([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").reciprocal().numpy()
    np.testing.assert_allclose(result, 1.0 / a, rtol=1e-5)

  def test_float32_fneg_2x2_matches_reference(self):
    a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_int32_iabs_8elem_mixed_signs_matches_reference(self):
    a = np.array([-4, -3, -2, -1, 0, 1, 2, 3], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_int32_3x3_iabs_matches_reference(self):
    a = -np.arange(1, 10, dtype=np.int32).reshape(3, 3)
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_float32_3x3_fabs_matches_reference(self):
    a = -np.arange(1, 10, dtype=np.float32).reshape(3, 3)
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_int32_4x4_and_const_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") & 0xF).numpy()
    np.testing.assert_array_equal(result, a & 0xF)

  def test_int32_5elem_or_tt_matches_reference(self):
    a = np.array([1, 2, 4, 8, 16], dtype=np.int32)
    b = np.array([16, 8, 4, 2, 1], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") | Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a | b)

  def test_int32_4elem_shr_tt_matches_reference(self):
    a = np.array([16, 32, 64, 128], dtype=np.int32)
    b = np.array([1, 2, 1, 3], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") >> Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a >> b)

  def test_int32_4elem_shl_tt_matches_reference(self):
    a = np.array([1, 2, 3, 4], dtype=np.int32)
    b = np.array([3, 2, 1, 0], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") << Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a << b)

  def test_int32_4x8_cmplt_scalar_const_matches_reference(self):
    a = np.arange(32, dtype=np.int32).reshape(4, 8)
    result = (Tensor(a, dtype="int32", device="TINYTPU") < 10).numpy()
    np.testing.assert_array_equal(result, a < 10)

  def test_int32_4x4_cmpne_scalar_const_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") != 5).numpy()
    np.testing.assert_array_equal(result, a != 5)

  def test_int32_6elem_cmpeq_tt_matches_reference(self):
    a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    b = np.array([1, 1, 1, 4, 5, 5], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") == Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a == b)

  def test_int32_4elem_idiv_tt_matches_reference(self):
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    b = np.array([2, 4, 5, 8], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") // Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a // b)

  def test_int32_4elem_imod_tt_matches_reference(self):
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    b = np.array([3, 4, 7, 9], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") % Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a % b)

  def test_float32_5x5_fneg_matches_reference(self):
    a = np.arange(25, dtype=np.float32).reshape(5, 5)
    result = (-Tensor(a, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, -a, rtol=1e-5)

  def test_float32_2x2_fmax_tt_matches_reference(self):
    a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.maximum(a, b), rtol=1e-5)

  def test_float32_2x2_fmin_tt_matches_reference(self):
    a = np.array([[1.0, 5.0], [3.0, 2.0]], dtype=np.float32)
    b = np.array([[3.0, 2.0], [1.0, 4.0]], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").minimum(Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, np.minimum(a, b), rtol=1e-5)

  def test_float32_3elem_fadd_tt_matches_reference(self):
    a = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_float32_3elem_fsub_tt_matches_reference(self):
    a = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a - b, rtol=1e-5)

  def test_float32_3elem_fmul_tt_matches_reference(self):
    a = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a * b, rtol=1e-5)

  def test_float32_3elem_fdiv_tt_matches_reference(self):
    a = np.array([6.0, 8.0, 10.0], dtype=np.float32)
    b = np.array([2.0, 2.0, 2.0], dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") / Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a / b, rtol=1e-5)

  def test_int32_6elem_i2f_matches_reference(self):
    a = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").cast("float").numpy()
    np.testing.assert_allclose(result, a.astype(np.float32), rtol=1e-5)

  def test_float32_6elem_f2i_matches_reference(self):
    a = np.array([1.4, 2.6, 3.1, 4.8, 5.5, 6.9], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").cast("int32").numpy()
    np.testing.assert_array_equal(result, a.astype(np.int32))

  def test_int32_2x5_rowsum_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(2, 5)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, a.sum(axis=1))

  def test_int32_5x2_colsum_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(5, 2)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, a.sum(axis=0))

  def test_int32_4x4_rowsum_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, a.sum(axis=1))

  def test_int32_4x4_colsum_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, a.sum(axis=0))

  def test_int32_2elem_add_const_matches_reference(self):
    a = np.arange(2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_7elem_add_const_matches_reference(self):
    a = np.arange(7, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_11elem_add_const_matches_reference(self):
    a = np.arange(11, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_17elem_add_const_matches_reference(self):
    a = np.arange(17, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_20elem_add_const_matches_reference(self):
    a = np.arange(20, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_31elem_add_const_matches_reference(self):
    a = np.arange(31, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_float32_5elem_add_const_matches_reference(self):
    a = np.arange(5, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 1.0).numpy()
    np.testing.assert_allclose(result, a + 1.0, rtol=1e-5)

  def test_float32_11elem_add_const_matches_reference(self):
    a = np.arange(11, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 1.0).numpy()
    np.testing.assert_allclose(result, a + 1.0, rtol=1e-5)

  def test_float32_20elem_add_const_matches_reference(self):
    a = np.arange(20, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") + 1.0).numpy()
    np.testing.assert_allclose(result, a + 1.0, rtol=1e-5)

  def test_int32_2x3_add_const_matches_reference(self):
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_3x2_add_const_matches_reference(self):
    a = np.arange(6, dtype=np.int32).reshape(3, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_4x2_add_const_matches_reference(self):
    a = np.arange(8, dtype=np.int32).reshape(4, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_2x4_add_const_matches_reference(self):
    a = np.arange(8, dtype=np.int32).reshape(2, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_3x4_add_const_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 4)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_4x3_add_const_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(4, 3)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_5x3_add_const_matches_reference(self):
    a = np.arange(15, dtype=np.int32).reshape(5, 3)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_3x5_add_const_matches_reference(self):
    a = np.arange(15, dtype=np.int32).reshape(3, 5)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_int32_6x2_add_const_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(6, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 5).numpy()
    np.testing.assert_array_equal(result, a + 5)

  def test_float32_2x3_mul_const_matches_reference(self):
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_float32_4x2_mul_const_matches_reference(self):
    a = np.arange(8, dtype=np.float32).reshape(4, 2)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_float32_3x4_mul_const_matches_reference(self):
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_float32_3x5_mul_const_matches_reference(self):
    a = np.arange(15, dtype=np.float32).reshape(3, 5)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_int32_6elem_imul_tt_full_matches_reference(self):
    a = np.full(6, 3, dtype=np.int32)
    b = np.full(6, 4, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_int32_relu_5elem_matches_reference(self):
    a = np.array([-5, -1, 0, 1, 5], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").relu().numpy()
    np.testing.assert_array_equal(result, np.maximum(a, 0))

  def test_int32_isub_tt_2elem_matches_reference(self):
    a = np.arange(2, dtype=np.int32); b = np.full(2, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_int32_isub_tt_3elem_matches_reference(self):
    a = np.arange(3, dtype=np.int32); b = np.full(3, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_int32_isub_tt_5elem_matches_reference(self):
    a = np.arange(5, dtype=np.int32); b = np.full(5, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_int32_isub_tt_7elem_matches_reference(self):
    a = np.arange(7, dtype=np.int32); b = np.full(7, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_int32_isub_tt_11elem_matches_reference(self):
    a = np.arange(11, dtype=np.int32); b = np.full(11, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a - b)

  def test_int32_imul_tt_2elem_matches_reference(self):
    a = np.arange(1, 3, dtype=np.int32); b = np.full(2, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_int32_imul_tt_3elem_matches_reference(self):
    a = np.arange(1, 4, dtype=np.int32); b = np.full(3, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_int32_imul_tt_5elem_matches_reference(self):
    a = np.arange(1, 6, dtype=np.int32); b = np.full(5, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_int32_imul_tt_7elem_matches_reference(self):
    a = np.arange(1, 8, dtype=np.int32); b = np.full(7, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_int32_imul_tt_11elem_matches_reference(self):
    a = np.arange(1, 12, dtype=np.int32); b = np.full(11, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * Tensor(b, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a * b)

  def test_float32_frelu_3elem_matches_reference(self):
    a = np.arange(3, dtype=np.float32) - 3
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0), rtol=1e-5)

  def test_float32_frelu_5elem_matches_reference(self):
    a = np.arange(5, dtype=np.float32) - 3
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0), rtol=1e-5)

  def test_float32_frelu_7elem_matches_reference(self):
    a = np.arange(7, dtype=np.float32) - 3
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0), rtol=1e-5)

  def test_float32_frelu_11elem_matches_reference(self):
    a = np.arange(11, dtype=np.float32) - 3
    result = Tensor(a, dtype="float", device="TINYTPU").relu().numpy()
    np.testing.assert_allclose(result, np.maximum(a, 0), rtol=1e-5)

  def test_int32_iabs_3elem_matches_reference(self):
    a = np.arange(3, dtype=np.int32) - 10
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_int32_iabs_5elem_matches_reference(self):
    a = np.arange(5, dtype=np.int32) - 10
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_int32_iabs_7elem_matches_reference(self):
    a = np.arange(7, dtype=np.int32) - 10
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_int32_iabs_11elem_matches_reference(self):
    a = np.arange(11, dtype=np.int32) - 10
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_int32_iabs_20elem_matches_reference(self):
    a = np.arange(20, dtype=np.int32) - 10
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_int32_iabs_31elem_matches_reference(self):
    a = np.arange(31, dtype=np.int32) - 10
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_int32_cmplt0_3elem_matches_reference(self):
    a = np.arange(-3, 0, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") < 0).numpy()
    np.testing.assert_array_equal(result, a < 0)

  def test_int32_cmplt0_5elem_matches_reference(self):
    a = np.arange(-3, 2, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") < 0).numpy()
    np.testing.assert_array_equal(result, a < 0)

  def test_int32_cmplt0_7elem_matches_reference(self):
    a = np.arange(-3, 4, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") < 0).numpy()
    np.testing.assert_array_equal(result, a < 0)

  def test_int32_cmplt0_11elem_matches_reference(self):
    a = np.arange(-3, 8, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") < 0).numpy()
    np.testing.assert_array_equal(result, a < 0)

  def test_bool_band_3elem_matches_reference(self):
    a = np.array([True, True, False])
    b = np.array([False, True, False])
    result = (Tensor(a, dtype="bool", device="TINYTPU") & Tensor(b, dtype="bool", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a & b)

  def test_bool_band_5elem_matches_reference(self):
    a = np.array([True, True, True, True, False])
    b = np.array([False, True, False, False, False])
    result = (Tensor(a, dtype="bool", device="TINYTPU") & Tensor(b, dtype="bool", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a & b)

  def test_bool_band_7elem_matches_reference(self):
    a = np.array([True]*6 + [False])
    b = np.array([False, True] + [False]*5)
    result = (Tensor(a, dtype="bool", device="TINYTPU") & Tensor(b, dtype="bool", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, a & b)

  def test_int32_2x3_mul_const_matches_reference(self):
    a = np.arange(6, dtype=np.int32).reshape(2, 3)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * 3).numpy()
    np.testing.assert_array_equal(result, a * 3)

  def test_int32_3x2_mul_const_matches_reference(self):
    a = np.arange(6, dtype=np.int32).reshape(3, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * 3).numpy()
    np.testing.assert_array_equal(result, a * 3)

  def test_int32_3d_222_add_const_matches_reference(self):
    a = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_3d_123_add_const_matches_reference(self):
    a = np.arange(6, dtype=np.int32).reshape(1, 2, 3)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_3d_223_add_const_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(2, 2, 3)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_3d_232_add_const_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(2, 3, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_3d_322_add_const_matches_reference(self):
    a = np.arange(12, dtype=np.int32).reshape(3, 2, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_3x3_rowmax_generic_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, a.max(axis=1))

  def test_int32_3x3_colmax_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, a.max(axis=0))

  def test_int32_4x4_rowmax_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, a.max(axis=1))

  def test_int32_4x4_colmax_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, a.max(axis=0))

  def test_int32_2x5_rowmax_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(2, 5)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, a.max(axis=1))

  def test_int32_2x5_rowmin_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(2, 5)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, a.min(axis=1))

  def test_int32_2x5_colmax_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(2, 5)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, a.max(axis=0))

  def test_int32_2x5_colmin_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(2, 5)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, a.min(axis=0))

  def test_int32_5x2_rowmax_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(5, 2)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=1).numpy()
    np.testing.assert_array_equal(result, a.max(axis=1))

  def test_int32_5x2_rowmin_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(5, 2)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, a.min(axis=1))

  def test_int32_5x2_colmax_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(5, 2)
    result = Tensor(a, dtype="int32", device="TINYTPU").max(axis=0).numpy()
    np.testing.assert_array_equal(result, a.max(axis=0))

  def test_int32_5x2_colmin_matches_reference(self):
    a = np.arange(10, dtype=np.int32).reshape(5, 2)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, a.min(axis=0))

  def test_int32_3x3_rowmin_matches_reference(self):
    a = np.arange(9, dtype=np.int32).reshape(3, 3)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, a.min(axis=1))

  def test_int32_4x4_rowmin_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=1).numpy()
    np.testing.assert_array_equal(result, a.min(axis=1))

  def test_int32_4x4_colmin_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").min(axis=0).numpy()
    np.testing.assert_array_equal(result, a.min(axis=0))

  def test_int32_3elem_scalar_sum_matches_reference(self):
    a = np.arange(3, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_int32_5elem_scalar_sum_matches_reference(self):
    a = np.arange(5, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_int32_7elem_scalar_sum_matches_reference(self):
    a = np.arange(7, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_int32_11elem_scalar_sum_matches_reference(self):
    a = np.arange(11, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_int32_3elem_scalar_max_matches_reference(self):
    a = np.arange(3, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a.max())

  def test_int32_5elem_scalar_max_matches_reference(self):
    a = np.arange(5, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a.max())

  def test_int32_7elem_scalar_max_matches_reference(self):
    a = np.arange(7, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a.max())

  def test_int32_11elem_scalar_max_matches_reference(self):
    a = np.arange(11, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a.max())

  def test_int32_3elem_scalar_min_matches_reference(self):
    a = np.arange(3, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, a.min())

  def test_int32_5elem_scalar_min_matches_reference(self):
    a = np.arange(5, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, a.min())

  def test_int32_7elem_scalar_min_matches_reference(self):
    a = np.arange(7, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, a.min())

  def test_int32_11elem_scalar_min_matches_reference(self):
    a = np.arange(11, dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").min().numpy()
    np.testing.assert_array_equal(result, a.min())

  def test_float32_3elem_mul_const_matches_reference(self):
    a = np.arange(3, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_float32_7elem_mul_const_matches_reference(self):
    a = np.arange(7, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_float32_31elem_mul_const_matches_reference(self):
    a = np.arange(31, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_float32_47elem_mul_const_matches_reference(self):
    a = np.arange(47, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") * 2.0).numpy()
    np.testing.assert_allclose(result, a * 2.0, rtol=1e-5)

  def test_float32_3elem_fadd_tt_matches_reference(self):
    a = np.arange(3, dtype=np.float32); b = a + 1
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_float32_7elem_fadd_tt_matches_reference(self):
    a = np.arange(7, dtype=np.float32); b = a + 1
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_float32_11elem_fadd_tt_matches_reference(self):
    a = np.arange(11, dtype=np.float32); b = a + 1
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_float32_20elem_fadd_tt_matches_reference(self):
    a = np.arange(20, dtype=np.float32); b = a + 1
    result = (Tensor(a, dtype="float", device="TINYTPU") + Tensor(b, dtype="float", device="TINYTPU")).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_float32_5elem_fabs_matches_reference(self):
    a = np.arange(5, dtype=np.float32) - 2
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_float32_7elem_fabs_matches_reference(self):
    a = np.arange(7, dtype=np.float32) - 2
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_float32_11elem_fabs_matches_reference(self):
    a = np.arange(11, dtype=np.float32) - 2
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_int32_2x3_iabs_matches_reference(self):
    a = np.arange(6, dtype=np.int32).reshape(2, 3) - 3
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_float32_2x3_fabs_matches_reference(self):
    a = np.arange(6, dtype=np.float32).reshape(2, 3) - 3
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_int32_4x4_iabs_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4) - 8
    result = Tensor(a, dtype="int32", device="TINYTPU").abs().numpy()
    np.testing.assert_array_equal(result, np.abs(a))

  def test_float32_3x3_fabs_shape_matches_reference(self):
    a = np.arange(9, dtype=np.float32).reshape(3, 3) - 5
    result = Tensor(a, dtype="float", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a), rtol=1e-5)

  def test_int32_5elem_or_c_matches_reference(self):
    a = np.arange(5, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") | 1).numpy()
    np.testing.assert_array_equal(result, a | 1)

  def test_int32_7elem_and_c_matches_reference(self):
    a = np.arange(7, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") & 3).numpy()
    np.testing.assert_array_equal(result, a & 3)

  def test_int32_11elem_xor_c_matches_reference(self):
    a = np.arange(11, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") ^ 5).numpy()
    np.testing.assert_array_equal(result, a ^ 5)

  def test_int32_3elem_or_c_matches_reference(self):
    a = np.arange(3, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") | 4).numpy()
    np.testing.assert_array_equal(result, a | 4)

  def test_int32_3elem_and_c_matches_reference(self):
    a = np.arange(3, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") & 3).numpy()
    np.testing.assert_array_equal(result, a & 3)

  def test_int32_3elem_xor_c_matches_reference(self):
    a = np.arange(3, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") ^ 1).numpy()
    np.testing.assert_array_equal(result, a ^ 1)

  def test_int32_17elem_or_c_matches_reference(self):
    a = np.arange(17, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") | 1).numpy()
    np.testing.assert_array_equal(result, a | 1)

  def test_int32_17elem_and_c_matches_reference(self):
    a = np.arange(17, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") & 15).numpy()
    np.testing.assert_array_equal(result, a & 15)

  def test_int32_17elem_xor_c_matches_reference(self):
    a = np.arange(17, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") ^ 7).numpy()
    np.testing.assert_array_equal(result, a ^ 7)

  def test_int32_7elem_isub_c_matches_reference(self):
    a = np.arange(7, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - 2).numpy()
    np.testing.assert_array_equal(result, a - 2)

  def test_int32_11elem_isub_c_matches_reference(self):
    a = np.arange(11, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - 2).numpy()
    np.testing.assert_array_equal(result, a - 2)

  def test_int32_20elem_isub_c_matches_reference(self):
    a = np.arange(20, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - 2).numpy()
    np.testing.assert_array_equal(result, a - 2)

  def test_int32_31elem_isub_c_matches_reference(self):
    a = np.arange(31, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") - 2).numpy()
    np.testing.assert_array_equal(result, a - 2)

  def test_float32_3elem_fsub_c_matches_reference(self):
    a = np.arange(3, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 1.0).numpy()
    np.testing.assert_allclose(result, a - 1.0, rtol=1e-5)

  def test_float32_7elem_fsub_c_matches_reference(self):
    a = np.arange(7, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 1.0).numpy()
    np.testing.assert_allclose(result, a - 1.0, rtol=1e-5)

  def test_float32_20elem_fsub_c_matches_reference(self):
    a = np.arange(20, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 1.0).numpy()
    np.testing.assert_allclose(result, a - 1.0, rtol=1e-5)

  def test_float32_31elem_fsub_c_matches_reference(self):
    a = np.arange(31, dtype=np.float32)
    result = (Tensor(a, dtype="float", device="TINYTPU") - 1.0).numpy()
    np.testing.assert_allclose(result, a - 1.0, rtol=1e-5)

  def test_int32_add_then_mul_const_chain_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = ((Tensor(a, dtype="int32", device="TINYTPU") + 1) * 2).numpy()
    np.testing.assert_array_equal(result, (a + 1) * 2)

  def test_int32_mul_then_add_const_chain_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = ((Tensor(a, dtype="int32", device="TINYTPU") * 2) + 1).numpy()
    np.testing.assert_array_equal(result, a * 2 + 1)

  def test_int32_mul_then_max_const_chain_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * 2).maximum(3).numpy()
    np.testing.assert_array_equal(result, np.maximum(a * 2, 3))

  def test_int32_mul_then_cmpne_chain_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = ((Tensor(a, dtype="int32", device="TINYTPU") * 2) != 4).numpy()
    np.testing.assert_array_equal(result, (a * 2) != 4)

  def test_int32_add_then_cmplt_chain_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = ((Tensor(a, dtype="int32", device="TINYTPU") + 1) < 5).numpy()
    np.testing.assert_array_equal(result, (a + 1) < 5)

  def test_int32_three_op_chain_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = (((Tensor(a, dtype="int32", device="TINYTPU") + 1) * 2) + 3).numpy()
    np.testing.assert_array_equal(result, (a + 1) * 2 + 3)

  def test_int32_four_op_chain_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = ((((Tensor(a, dtype="int32", device="TINYTPU") + 1) * 2) + 3) * 2).numpy()
    np.testing.assert_array_equal(result, ((a + 1) * 2 + 3) * 2)

  def test_float32_three_op_chain_matches_reference(self):
    a = np.arange(4, dtype=np.float32)
    result = (((Tensor(a, dtype="float", device="TINYTPU") + 1.0) * 2.0) + 3.0).numpy()
    np.testing.assert_allclose(result, (a + 1.0) * 2.0 + 3.0, rtol=1e-5)

  def test_float32_fclip_chain_matches_reference(self):
    a = np.array([-3.0, -1.0, 0.5, 2.0, 5.0], dtype=np.float32)
    result = Tensor(a, dtype="float", device="TINYTPU").maximum(0.0).minimum(1.0).numpy()
    np.testing.assert_allclose(result, np.clip(a, 0.0, 1.0), rtol=1e-5)

  def test_int32_4x4_transpose_matches_reference(self):
    a = np.arange(16, dtype=np.int32).reshape(4, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").permute(1, 0).numpy()
    np.testing.assert_array_equal(result, a.T)

  def test_int32_zeros_matches_reference(self):
    np.testing.assert_array_equal(Tensor.zeros(4, dtype="int32", device="TINYTPU").numpy(), np.zeros(4, dtype=np.int32))

  def test_int32_ones_matches_reference(self):
    np.testing.assert_array_equal(Tensor.ones(4, dtype="int32", device="TINYTPU").numpy(), np.ones(4, dtype=np.int32))

  def test_float32_ones_5elem_matches_reference(self):
    np.testing.assert_allclose(Tensor.ones(5, dtype="float", device="TINYTPU").numpy(), np.ones(5, dtype=np.float32), rtol=1e-5)

  def test_int32_full_7_matches_reference(self):
    np.testing.assert_array_equal(Tensor.full((3,), 7, dtype="int32", device="TINYTPU").numpy(), np.full(3, 7, dtype=np.int32))

  def test_int32_2x3_zeros_matches_reference(self):
    np.testing.assert_array_equal(Tensor.zeros((2, 3), dtype="int32", device="TINYTPU").numpy(), np.zeros((2, 3), dtype=np.int32))

  def test_float32_4x4_transpose_matches_reference(self):
    a = np.arange(16, dtype=np.float32).reshape(4, 4)
    np.testing.assert_allclose(Tensor(a, dtype="float", device="TINYTPU").permute(1, 0).numpy(), a.T, rtol=1e-5)

  def test_bool_zeros_matches_reference(self):
    np.testing.assert_array_equal(Tensor.zeros(4, dtype="bool", device="TINYTPU").numpy(), np.zeros(4, dtype=bool))

  def test_bool_ones_matches_reference(self):
    np.testing.assert_array_equal(Tensor.ones(4, dtype="bool", device="TINYTPU").numpy(), np.ones(4, dtype=bool))

  def test_bool_where_scalar_consts_matches_reference(self):
    cond = np.array([True, False, True, False])
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(1, 0).numpy()
    np.testing.assert_array_equal(result, np.where(cond, 1, 0))

  def test_bool_where_tensor_lhs_const_rhs_matches_reference(self):
    cond = np.array([True, False, True, False])
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(
      Tensor(a, dtype="int32", device="TINYTPU"), 99).numpy()
    np.testing.assert_array_equal(result, np.where(cond, a, 99))

  def test_bool_where_const_lhs_tensor_rhs_matches_reference(self):
    cond = np.array([True, False, True, False])
    a = np.array([10, 20, 30, 40], dtype=np.int32)
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(
      99, Tensor(a, dtype="int32", device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, np.where(cond, 99, a))

  def test_bool_where_consts_6elem_matches_reference(self):
    cond = np.array([True, False, True, False, True, False])
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(1, 0).numpy()
    np.testing.assert_array_equal(result, np.where(cond, 1, 0))

  def test_bool_where_float_consts_matches_reference(self):
    cond = np.array([True, False, True, False])
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(1.5, 2.5).numpy()
    np.testing.assert_allclose(result, np.where(cond, 1.5, 2.5), rtol=1e-5)

  def test_bool_where_consts_2d_matches_reference(self):
    cond = np.array([[True, False], [False, True]])
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(5, 10).numpy()
    np.testing.assert_array_equal(result, np.where(cond, 5, 10))

  def test_bool_where_consts_16elem_matches_reference(self):
    cond = np.array([True]*8 + [False]*8)
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(1, 0).numpy()
    np.testing.assert_array_equal(result, np.where(cond, 1, 0))

  def test_int32_size1_add_const_matches_reference(self):
    a = np.array([5], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") + 1).numpy()
    np.testing.assert_array_equal(result, a + 1)

  def test_int32_size1_mul_const_matches_reference(self):
    a = np.array([5], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * 3).numpy()
    np.testing.assert_array_equal(result, a * 3)

  def test_int32_size1_sum_matches_reference(self):
    a = np.array([5], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_int32_max_reduce_all_negative_matches_reference(self):
    a = np.array([-3, -1, -5, -2], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").max().numpy()
    np.testing.assert_array_equal(result, a.max())

  def test_int32_3d_sum_all_matches_reference(self):
    a = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    result = Tensor(a, dtype="int32", device="TINYTPU").sum().numpy()
    np.testing.assert_array_equal(result, a.sum())

  def test_int32_sum_then_add_const_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU").sum() + 10).numpy()
    np.testing.assert_array_equal(result, a.sum() + 10)

  def test_int32_clip_chain_matches_reference(self):
    a = np.array([-3, 0, 3, 10], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").maximum(0).minimum(5).numpy()
    np.testing.assert_array_equal(result, np.clip(a, 0, 5))

  def test_bool_not_2x2_matches_reference(self):
    data = np.array([[True, False], [False, True]])
    result = (~Tensor(data, device="TINYTPU")).numpy()
    np.testing.assert_array_equal(result, ~data)

  def test_int32_3d_222_mul_const_matches_reference(self):
    a = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") * 3).numpy()
    np.testing.assert_array_equal(result, a * 3)

  def test_colsum_1x4_matches_reference(self):
    data = np.array([[1, 2, 3, 4]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=0).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=0))

  def test_rowsum_4x1_matches_reference(self):
    data = np.array([[1], [2], [3], [4]], dtype=np.int32)
    result = Tensor(data, dtype="int32", device="TINYTPU").sum(axis=1).numpy()
    np.testing.assert_array_equal(result, data.sum(axis=1))

  def test_prod5_matches_reference(self):
    a = np.array([1, 2, 3, 4, 5], dtype=np.int32)
    result = Tensor(a, dtype="int32", device="TINYTPU").prod().numpy()
    np.testing.assert_array_equal(result, a.prod())

  def test_float32_9elem_fabs_matches_reference(self):
    a = np.array([-4.5, -3.0, -1.5, 0.0, 1.5, 3.0, 4.5, -6.0, 7.0], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").abs().numpy()
    np.testing.assert_allclose(result, np.abs(a))

  def test_int32_3d_222_shr_matches_reference(self):
    a = (np.arange(8, dtype=np.int32) * 4).reshape(2, 2, 2)
    result = (Tensor(a, dtype="int32", device="TINYTPU") >> 2).numpy()
    np.testing.assert_array_equal(result, a >> 2)

  def test_bool_where_negative_consts_matches_reference(self):
    cond = np.array([True, False, True, False])
    result = Tensor(cond, dtype="bool", device="TINYTPU").where(-3, -7).numpy()
    np.testing.assert_array_equal(result, np.where(cond, -3, -7))

  def test_float32_3d_222_fadd_tt_matches_reference(self):
    a = (np.arange(8, dtype=np.float32) - 4.0).reshape(2, 2, 2)
    b = (np.arange(8, dtype=np.float32) * 0.5).reshape(2, 2, 2)
    ta = Tensor(a, dtype="float32", device="TINYTPU")
    tb = Tensor(b, dtype="float32", device="TINYTPU")
    result = (ta + tb).numpy()
    np.testing.assert_allclose(result, a + b, rtol=1e-5)

  def test_int32_max_then_add_const_matches_reference(self):
    a = np.array([-1, 3, 0, 5], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU").max() + 2).numpy()
    np.testing.assert_array_equal(result, a.max() + 2)

  def test_int32_min_then_add_const_matches_reference(self):
    a = np.array([4, -2, 9, 1], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU").min() + 7).numpy()
    np.testing.assert_array_equal(result, a.min() + 7)

  def test_int32_sum_then_mul_const_matches_reference(self):
    a = np.arange(4, dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU").sum() * 5).numpy()
    np.testing.assert_array_equal(result, a.sum() * 5)

  def test_int32_prod_then_add_const_matches_reference(self):
    a = np.array([2, 3, 4], dtype=np.int32)
    result = (Tensor(a, dtype="int32", device="TINYTPU").prod() + 1).numpy()
    np.testing.assert_array_equal(result, a.prod() + 1)

  def test_float32_sum_4elem_matches_reference(self):
    a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").sum().numpy()
    np.testing.assert_allclose(result, a.sum(), rtol=1e-6)

  def test_float32_sum_full_tile_matches_reference(self):
    a = np.arange(1, 17, dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").sum().numpy()
    np.testing.assert_allclose(result, a.sum(), rtol=1e-6)

  def test_float32_sum_multi_tile_matches_reference(self):
    a = np.arange(32, dtype=np.float32) * 0.5
    result = Tensor(a, dtype="float32", device="TINYTPU").sum().numpy()
    np.testing.assert_allclose(result, a.sum(), rtol=1e-5)

  def test_float32_sum_signed_matches_reference(self):
    a = np.array([-1.5, 2.5, -0.25, 3.25], dtype=np.float32)
    result = Tensor(a, dtype="float32", device="TINYTPU").sum().numpy()
    np.testing.assert_allclose(result, a.sum(), rtol=1e-6)


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

  def test_mxu_os_dispatches_equivalent_to_ws(self):
    # End-to-end sim check that SXU_DISPATCH_MXU_OS drives the Controller
    # through the OS path. Today the OS path still runs WS behavior (PE
    # accumulator-hold + operand-swap land in later iters), so identity
    # weights × [1,2,3,4] must still produce [1,2,3,4] in the MXU output
    # register. The test's purpose is to prove the opcode is wired
    # through the SXU fetch → Controller.startOS → dispatch chain.
    sim = os.environ["TINYTPU_SIM"]
    ident_weights = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    bundle = _bundle(
      _wmem(0, ident_weights),
      _amem(0, [1, 2, 3, 4]),
      _mxu_os(0, 0, 1),                         # DISPATCH_MXU_OS
      _wait_mxu(),
      _halt(),
      "3 1",                                    # OUTPUT_MXU
      _end(),
    )
    out = _run_bundle(sim, bundle)
    self.assertEqual(_parse_sim_output(out), [1, 2, 3, 4])

  def test_back_to_back_xlu_broadcast_chain(self):
    # Exercise the dual-issue XLU path end to end: LOAD a tile into v0,
    # then fire two DISPATCH_XLU_BROADCAST ops back to back (v1 := v0
    # broadcast of lane 1, v2 := v1 broadcast of lane 0), STORE v2.
    # Each XLU dispatch advances pc immediately; the second dispatch must
    # wait for the structural-hazard guard (!xlu_busy) to clear. Final
    # tile in VMEM[2] must be the scalar broadcast of v0[0][1] = 20.
    sim = os.environ["TINYTPU_SIM"]
    src_tile = [10, 20, 30, 40] + [0] * 12
    bundle = _bundle(
      _vmem(0, src_tile),
      _load(0, 0),
      "2 3 0 1 0 0 1 0 0 0",                    # XLU_BROADCAST v1 := v0 lane=1
      "2 3 0 2 1 0 0 0 0 0",                    # XLU_BROADCAST v2 := v1 lane=0
      _store(2, 2),                             # VMEM[2] := v2
      _halt(),
      _output_vmem(2),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tile = _parse_vmem_output(out)
    # Every lane/sublane must carry the broadcast scalar 20.
    self.assertEqual(tile[0:4], [20, 20, 20, 20])

  def test_mxu_os_accumulator_hold_across_dispatches(self):
    # Two back-to-back OS dispatches must accumulate in the PE
    # accumulators. Identity weights * [1,2,3,4] = [1,2,3,4] per dispatch,
    # so after two dispatches the drained result is [2,4,6,8].
    sim = os.environ["TINYTPU_SIM"]
    ident_weights = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    bundle = _bundle(
      _wmem(0, ident_weights),
      _amem(0, [1, 2, 3, 4]),
      _mxu_os(0, 0, 1),                         # OS dispatch #1 -> [1,2,3,4]
      _wait_mxu(),
      _mxu_os(0, 0, 1),                         # OS dispatch #2 accumulates
      _wait_mxu(),
      _halt(),
      "3 1",                                    # OUTPUT_MXU
      _end(),
    )
    out = _run_bundle(sim, bundle)
    self.assertEqual(_parse_sim_output(out), [2, 4, 6, 8])

  def test_mxu_clear_resets_os_accumulator(self):
    # OS dispatch, MXU_CLEAR, OS dispatch -> fresh [1,2,3,4]. Verifies
    # SXU_MXU_CLEAR reaches ctrl.clearArray and zeroes PE accumulators
    # between OS-mode accumulation epochs.
    sim = os.environ["TINYTPU_SIM"]
    ident_weights = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    bundle = _bundle(
      _wmem(0, ident_weights),
      _amem(0, [1, 2, 3, 4]),
      _mxu_os(0, 0, 1),                         # OS dispatch #1 -> [1,2,3,4]
      _wait_mxu(),
      _mxu_clear(),                             # zero PE accumulators
      _mxu_os(0, 0, 1),                         # OS dispatch #2 fresh
      _wait_mxu(),
      _halt(),
      "3 1",                                    # OUTPUT_MXU
      _end(),
    )
    out = _run_bundle(sim, bundle)
    self.assertEqual(_parse_sim_output(out), [1, 2, 3, 4])

  def test_mxu_psum_accumulate_via_bundle(self):
    # End-to-end hardware test of the PSUM accumulator path: two MXU
    # dispatches (WRITE then ACCUMULATE) with identity weights and
    # [1,2,3,4] activations should land [2,4,6,8] in bucket 0 row 0.
    # Other rows are covered by the preceding SXU_PSUM_WRITE that
    # clears the tile from a zero vreg.
    sim = os.environ["TINYTPU_SIM"]
    zero_tile = [0] * 16
    ident_weights = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    bundle = _bundle(
      _vmem(0, zero_tile),
      _wmem(0, ident_weights),
      _amem(0, [1, 2, 3, 4]),
      _load(0, 0),                              # v0 := VMEM[0] (zeros)
      "2 15 0 0 0 0 0 0 0 0",                   # PSUM_WRITE psum[0] := v0
      _mxu_psum_acc(0, 0, 1, 0, 0),             # MXU psum[0].row=0 += act*W
      _wait_mxu(),
      _mxu_psum_acc(0, 0, 1, 0, 0),             # second dispatch same target
      _wait_mxu(),
      _psum_read(1, 0),                         # v1 := psum[0]
      _store(5, 1),                             # VMEM[5] := v1
      _halt(),
      _output_vmem(5),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tile = _parse_vmem_output(out)
    # Row 0 doubled; other rows remain zero (cleared via SXU_PSUM_WRITE).
    self.assertEqual(tile[0:4], [2, 4, 6, 8])
    self.assertEqual(tile[4:16], [0] * 12)

  def test_load_vpu_result_copies_linger_register(self):
    # Exercise SXU_LOAD_VPU_RESULT end-to-end: run a single VPU ADD
    # (v0 + v1 -> v2), then LOAD_VPU_RESULT into v3. v2 and v3 should
    # hold identical tiles because vpu.resultReg still contains the
    # last dispatch's output.
    sim = os.environ["TINYTPU_SIM"]
    lhs = [i for i in range(16)]
    rhs = [100 + i for i in range(16)]
    bundle = _bundle(
      _vmem(0, lhs),
      _vmem(1, rhs),
      _load(0, 0),
      _load(1, 1),
      _vpu(2, 0, _VPU_OPS["ADD"], 1),      # v2 := v0 + v1
      _load_vpu_result(3),                 # v3 := vpu.result
      _store(4, 2),
      _store(5, 3),
      _halt(),
      _output_vmem(4),
      _output_vmem(5),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tiles = [line for line in out.splitlines() if line.startswith("vmem_result")]
    self.assertEqual(len(tiles), 2)
    # Both tiles are the same element-wise sum.
    expected = [lhs[i] + rhs[i] for i in range(16)]
    got0 = [int(x) for x in tiles[0].split()[1:]]
    got1 = [int(x) for x in tiles[1].split()[1:]]
    self.assertEqual(got0, expected)
    self.assertEqual(got1, expected)

  def test_load_xlu_result_copies_linger_register(self):
    # Drive one XLU broadcast (row 2 of v0 broadcast to a tile),
    # collect into v1 via the existing BROADCAST opcode, then ALSO
    # copy xlu.resultReg into v2 via LOAD_XLU_RESULT. Both vregs
    # should STORE the same tile.
    sim = os.environ["TINYTPU_SIM"]
    # VMEM[0]: row 2 = [10,20,30,40]; other rows arbitrary but distinct.
    tile = [0] * 16
    tile[0:4]   = [1, 2, 3, 4]
    tile[4:8]   = [5, 6, 7, 8]
    tile[8:12]  = [10, 20, 30, 40]
    tile[12:16] = [13, 14, 15, 16]
    bundle = _bundle(
      _vmem(0, tile),
      _load(0, 0),
      # BROADCAST_ROW opcode 10: vregDst=1, vregSrc=0, vregSrc2=row
      f"2 10 0 1 0 0 2 0 0 0",
      _load_xlu_result(2),
      _store(1, 1),
      _store(2, 2),
      _halt(),
      _output_vmem(1),
      _output_vmem(2),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tiles = [line for line in out.splitlines() if line.startswith("vmem_result")]
    self.assertEqual(len(tiles), 2)
    got0 = [int(x) for x in tiles[0].split()[1:]]
    got1 = [int(x) for x in tiles[1].split()[1:]]
    # Row 2 broadcast -> every row equals [10,20,30,40].
    expected = [10, 20, 30, 40] * 4
    self.assertEqual(got0, expected)
    self.assertEqual(got1, expected)

  def test_skip_if_pred_skips_store_when_pred_set(self):
    # VMEM[0] = zero tile (pred becomes True after SET_PRED_IF_ZERO v0).
    # Program intends to skip the first STORE under the predicate and
    # execute the second unconditionally. Expected: VMEM[3] unchanged
    # (garbage or prior state), VMEM[4] = 42-tile.
    sim = os.environ["TINYTPU_SIM"]
    zero_tile = [0] * 16
    forty_two = [42] * 16
    sentinel   = [99] * 16
    bundle = _bundle(
      _vmem(0, zero_tile),     # source of zeros -> v0
      _vmem(1, forty_two),     # source of 42s   -> v1
      _vmem(3, sentinel),      # initial VMEM[3] = 99; skip path must leave it
      _load(0, 0),             # v0 := zeros
      _load(1, 1),             # v1 := 42s
      _set_pred_if_zero(0),    # pred := (v0[0][0] == 0) -> True
      _skip_if_pred(),         # pred is True -> skip next instruction
      _store(3, 1),            # SKIPPED: would write VMEM[3] = 42s
      _store(4, 1),            # executed: VMEM[4] = 42s
      _halt(),
      _output_vmem(3),
      _output_vmem(4),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tiles = [line for line in out.splitlines() if line.startswith("vmem_result")]
    self.assertEqual(len(tiles), 2)
    got_skipped = [int(x) for x in tiles[0].split()[1:]]
    got_unskipped = [int(x) for x in tiles[1].split()[1:]]
    self.assertEqual(got_skipped, sentinel,
                     "SKIP_IF_PRED did not skip the protected STORE")
    self.assertEqual(got_unskipped, forty_two)

  def test_skip_if_pred_auto_resets_after_consume(self):
    # Two SKIP_IF_PRED ops in sequence: the first should skip (pred=True),
    # but pred auto-resets so the second should NOT skip. Expected:
    # VMEM[3] unchanged (skipped), VMEM[4] updated, VMEM[5] also updated.
    sim = os.environ["TINYTPU_SIM"]
    zero_tile = [0] * 16
    forty_two = [42] * 16
    sentinel_a = [11] * 16
    sentinel_b = [22] * 16
    bundle = _bundle(
      _vmem(0, zero_tile),
      _vmem(1, forty_two),
      _vmem(3, sentinel_a),    # skipped STORE target
      _vmem(5, sentinel_b),    # unskipped STORE target, pre-populated
      _load(0, 0),
      _load(1, 1),
      _set_pred_if_zero(0),    # pred := True
      _skip_if_pred(),         # consume pred (skip next)
      _store(3, 1),            # SKIPPED
      _skip_if_pred(),         # pred auto-reset -> False, does not skip
      _store(5, 1),            # NOT SKIPPED: VMEM[5] := 42s
      _halt(),
      _output_vmem(3),
      _output_vmem(5),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tiles = [line for line in out.splitlines() if line.startswith("vmem_result")]
    self.assertEqual(len(tiles), 2)
    got3 = [int(x) for x in tiles[0].split()[1:]]
    got5 = [int(x) for x in tiles[1].split()[1:]]
    self.assertEqual(got3, sentinel_a, "first SKIP_IF_PRED should skip")
    self.assertEqual(got5, forty_two,
                     "second SKIP_IF_PRED should NOT skip (pred auto-reset)")

  def test_skip_if_pred_does_not_skip_when_pred_clear(self):
    # Same shape but v0 holds a nonzero — pred stays False and the
    # STORE after SKIP_IF_PRED must execute.
    sim = os.environ["TINYTPU_SIM"]
    nonzero = [7] + [0] * 15
    forty_two = [42] * 16
    sentinel = [99] * 16
    bundle = _bundle(
      _vmem(0, nonzero),
      _vmem(1, forty_two),
      _vmem(3, sentinel),
      _load(0, 0),
      _load(1, 1),
      _set_pred_if_zero(0),  # pred := (7 == 0) -> False
      _skip_if_pred(),       # no skip
      _store(3, 1),          # executed: VMEM[3] = 42s
      _halt(),
      _output_vmem(3),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tiles = [line for line in out.splitlines() if line.startswith("vmem_result")]
    self.assertEqual(len(tiles), 1)
    got = [int(x) for x in tiles[0].split()[1:]]
    self.assertEqual(got, forty_two)

  def test_psum_accumulate_row_from_vpu_side(self):
    # Two PSUM_ACCUMULATE_ROW ops against a freshly-cleared bucket,
    # both sourcing row 0 of distinct vregs, must leave the sum in the
    # target row. Other rows of the bucket stay zero.
    sim = os.environ["TINYTPU_SIM"]
    tile_a = [1, 2, 3, 4] + [0] * 12
    tile_b = [10, 20, 30, 40] + [0] * 12
    bundle = _bundle(
      _vmem(0, tile_a),
      _vmem(1, tile_b),
      _load(0, 0),
      _load(1, 1),
      _psum_clear(0),
      _psum_accumulate_row(0, 0, 2),       # psum[0].row[2] += v0.row[0]
      _psum_accumulate_row(1, 0, 2),       # psum[0].row[2] += v1.row[0]
      _psum_read_row(2, 0, 2),             # v2.row[0] := psum[0].row[2]
      _store(5, 2),
      _halt(),
      _output_vmem(5),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tile = _parse_vmem_output(out)
    self.assertEqual(tile[0:4], [11, 22, 33, 44])
    self.assertEqual(tile[4:16], [0] * 12)

  def test_vpu_exp2_bundle_matches_approx(self):
    # Drive a VMEM preload of four Float32 inputs [0.0, 1.0, 2.0, -1.0]
    # (remaining lanes filled with 0.0), run VPU_EXP2 end-to-end through
    # the TranscUnit multi-cycle walker, and store the result back.
    # TranscUnit implements degree-2 Taylor of e^(x*ln2); acceptance bands
    # match the VPU TB coverage.
    import struct
    sim = os.environ["TINYTPU_SIM"]
    def _f(x: float) -> int:
      return int.from_bytes(struct.pack("<f", x), "little", signed=False)
    inputs = [0.0, 1.0, 2.0, -1.0] + [0.0] * 12
    tile_bits = [_f(v) for v in inputs]
    bundle = _bundle(
      _vmem(0, tile_bits),
      _load(0, 0),
      _vpu_exp2(1, 0),
      _store(1, 1),
      _halt(),
      _output_vmem(1),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tile = _parse_vmem_output(out)
    as_floats = [struct.unpack("<f", int(u).to_bytes(4, "little", signed=False))[0]
                 for u in (t & 0xFFFFFFFF for t in tile)]
    # Range-reduced Remez EXP2: x = n + f, 2^x = poly(f) * 2^n via
    # exponent manipulation. Integer inputs are exact.
    self.assertAlmostEqual(as_floats[0], 1.0, delta=0.01)
    self.assertAlmostEqual(as_floats[1], 2.0, delta=0.02)  # exact at x=1
    self.assertAlmostEqual(as_floats[2], 4.0, delta=0.04)  # exact at x=2
    self.assertAlmostEqual(as_floats[3], 0.5, delta=0.01)  # exact at x=-1

  def test_psum_read_row_extracts_single_row(self):
    # Accumulate into psum[0] row 2, then PSUM_READ_ROW that into v1
    # row 0 (mirroring LOAD_MXU_RESULT shape). STORE should produce
    # a VMEM tile with row 0 = [2,4,6,8] and rows 1..3 = 0.
    sim = os.environ["TINYTPU_SIM"]
    zero_tile = [0] * 16
    ident_weights = [1, 0, 0, 0,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    bundle = _bundle(
      _vmem(0, zero_tile),
      _wmem(0, ident_weights),
      _amem(0, [1, 2, 3, 4]),
      _load(0, 0),                              # v0 := zeros
      "2 15 0 0 0 0 0 0 0 0",                   # PSUM_WRITE psum[0] := v0
      _mxu_psum_acc(0, 0, 1, 0, 2),             # psum[0].row=2 += [1,2,3,4]
      _wait_mxu(),
      _mxu_psum_acc(0, 0, 1, 0, 2),             # same target again
      _wait_mxu(),
      _psum_read_row(1, 0, 2),                  # v1[0] := psum[0].row[2]
      _store(5, 1),                             # VMEM[5] := v1
      _halt(),
      _output_vmem(5),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    tile = _parse_vmem_output(out)
    self.assertEqual(tile[0:4], [2, 4, 6, 8])
    self.assertEqual(tile[4:16], [0] * 12)


if __name__ == "__main__":
  unittest.main()
