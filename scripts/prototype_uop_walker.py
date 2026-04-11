"""
Prototype: UOp-walking renderer for TinyTPU.

Instead of analyze_tinytpu_uops (pattern-match entire kernel) → JSON descriptor →
bundle builder → sim, this walks UOps linearly and emits bundle text directly.

Run: PYTHONPATH=tinygrad python scripts/prototype_uop_walker.py
"""
import sys, os, struct, json, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tinygrad"))
os.environ["TINYTPU_SIM"] = os.path.join(os.path.dirname(__file__), "..", "build", "mkTbTinyTPURuntime.bexe")
os.environ["DISABLE_COMPILER_CACHE"] = "1"

from tinygrad import Tensor, Device
from tinygrad.engine.realize import get_program
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import PtrDType
from tinygrad.runtime.ops_tinytpu import (
    _run_bundle, _parse_vmem_output, _parse_multi_vmem_output, _parse_sim_output,
    _VPU_OPS, _ROWS, _COLS, _TILE_ELEMS,
    _vmem, _wmem, _amem, _load, _store, _vpu, _mxu, _wait_mxu, _load_mxu_result,
    _halt, _output_vmem, _output_mxu, _end, _bundle, _broadcast, _require_int8_range,
    _sim_path,
)

# ---------------------------------------------------------------------------
# UOp-walking bundle emitter
# ---------------------------------------------------------------------------

def render_bundle(uops: list[UOp], bufs: tuple[bytearray, ...]) -> tuple[str, list[int]]:
    """Walk UOps and emit a TinyTPU bundle string + list of output VMEM addresses.

    Returns (bundle_text, output_vmem_addrs).
    """
    # Collect params
    params = {}  # arg → UOp
    for u in uops:
        if u.op is Ops.PARAM and isinstance(u.dtype, PtrDType):
            params[u.arg] = u

    # Check for WMMA
    wmmas = [u for u in uops if u.op is Ops.WMMA]
    if wmmas:
        return _render_wmma_bundle(uops, wmmas, params, bufs)

    # Check for simple elementwise (VPU)
    stores = [u for u in uops if u.op is Ops.STORE]
    loads = [u for u in uops if u.op is Ops.LOAD]
    if stores and loads and not wmmas:
        return _render_elementwise_bundle(uops, params, bufs)

    raise NotImplementedError(f"UOp walker: unsupported kernel pattern")


def _find_param_arg(u: UOp) -> int | None:
    """Find the unique PARAM arg reachable from u."""
    ps = {n.arg for n in u.toposort() if n.op is Ops.PARAM}
    return next(iter(ps)) if len(ps) == 1 and isinstance(next(iter(ps)), int) else None


def _render_wmma_bundle(uops, wmmas, params, bufs):
    """Render a WMMA kernel as a bundle."""
    wmma = wmmas[0]
    out_params = {_find_param_arg(s.src[0]) for s in uops if s.op is Ops.STORE}
    out_params.discard(None)
    out_arg = next(iter(out_params))
    act_arg = _find_param_arg(wmma.src[0])
    weight_arg = _find_param_arg(wmma.src[1])

    out_size = params[out_arg].dtype.size
    act_size = params[act_arg].dtype.size
    weight_size = params[weight_arg].dtype.size

    # Infer tiling
    import math
    num_vecs = max(1, act_size // _ROWS)
    # Use the existing tiling inference
    from tinygrad.runtime.ops_tinytpu import _infer_tiling
    tiling = _infer_tiling(out_size, act_size, weight_size)
    if tiling is None:
        raise RuntimeError(f"WMMA tiling failed: out={out_size} act={act_size} weight={weight_size}")
    num_vecs, num_k_tiles, num_weight_tiles = tiling
    out_cols = num_weight_tiles * _COLS
    k_cols = num_k_tiles * _ROWS

    # Detect epilogue
    from collections import Counter
    op_counts = Counter(u.op.name for u in uops)
    extra_params = sorted(k for k in params if k not in {out_arg, act_arg, weight_arg})
    has_bias = len(extra_params) == 1 and op_counts.get("ADD", 0) > 0
    has_relu = op_counts.get("WHERE", 0) > 0 and op_counts.get("CMPLT", 0) > 0
    bias_arg = extra_params[0] if has_bias else None

    # Read buffer data
    act_i32 = np.frombuffer(bytes(bufs[act_arg]), dtype="<i4")
    weight_i32 = np.frombuffer(bytes(bufs[weight_arg]), dtype="<i4")
    _require_int8_range("weight", weight_i32)
    _require_int8_range("activation", act_i32)

    weight_matrix = weight_i32.reshape(k_cols, out_cols).astype(np.int8)
    act_rows = act_i32.reshape(num_vecs, k_cols).astype(np.int8)

    bias_i32 = None
    if has_bias:
        bias_i32 = np.frombuffer(bytes(bufs[bias_arg]), dtype="<i4")

    # --- Emit bundle directly ---
    data_lines = []

    # Preload weight tiles into WMEM
    for k in range(num_k_tiles):
        for t in range(num_weight_tiles):
            w_tile = weight_matrix[k * _ROWS : (k + 1) * _ROWS, t * _COLS : (t + 1) * _COLS]
            data_lines.append(_wmem(k * num_weight_tiles + t, [int(x) for x in w_tile.flatten()]))

    # Preload activation rows into AMEM
    for row in range(num_vecs):
        for k in range(num_k_tiles):
            a_tile = act_rows[row, k * _ROWS : (k + 1) * _ROWS]
            data_lines.append(_amem(row * num_k_tiles + k, [int(x) for x in a_tile]))

    # Preload bias into VMEM if needed
    bias_vmem_base = 0
    if bias_i32 is not None:
        for t in range(num_weight_tiles):
            bias_tile = [0] * _TILE_ELEMS
            for i in range(_COLS):
                bias_tile[i] = int(bias_i32[t * _COLS + i])
            data_lines.append(_vmem(bias_vmem_base + t, bias_tile))

    out_vmem_base = num_weight_tiles if bias_i32 is not None else 0

    # Emit SXU program
    prog_lines = []
    for row in range(num_vecs):
        for tile_idx in range(num_weight_tiles):
            # MXU dispatches for K-tile accumulation
            for k in range(num_k_tiles):
                wmem_addr = k * num_weight_tiles + tile_idx
                amem_addr = row * num_k_tiles + k
                prog_lines.append(_mxu(wmem_addr, amem_addr, 1))
                prog_lines.append(_wait_mxu())
                prog_lines.append(_load_mxu_result(k))

            # Accumulate
            if num_k_tiles == 1:
                cur = 0
            else:
                acc = num_k_tiles
                prog_lines.append(_vpu(acc, 0, _VPU_OPS["ADD"], 1))
                cur = acc
                for k in range(2, num_k_tiles):
                    nxt = cur + 1
                    prog_lines.append(_vpu(nxt, cur, _VPU_OPS["ADD"], k))
                    cur = nxt

            # Bias epilogue
            if bias_i32 is not None:
                bv = cur + 1
                prog_lines.append(_load(bv, bias_vmem_base + tile_idx))
                rv = bv + 1
                prog_lines.append(_vpu(rv, cur, _VPU_OPS["ADD"], bv))
                cur = rv

            # ReLU epilogue
            if has_relu:
                nxt = cur + 1
                prog_lines.append(_vpu(nxt, cur, 2))  # VPU_RELU
                cur = nxt

            out_addr = out_vmem_base + row * num_weight_tiles + tile_idx
            prog_lines.append(_store(out_addr, cur))

    prog_lines.append(_halt())

    # Output records
    output_addrs = []
    output_lines = []
    for row in range(num_vecs):
        for tile_idx in range(num_weight_tiles):
            addr = out_vmem_base + row * num_weight_tiles + tile_idx
            output_addrs.append(addr)
            output_lines.append(_output_vmem(addr))
    output_lines.append(_end())

    return _bundle(*(data_lines + prog_lines + output_lines)), output_addrs


def _render_elementwise_bundle(uops, params, bufs):
    """Render a simple elementwise kernel as a bundle."""
    # For now, fall back to the existing path
    raise NotImplementedError("elementwise UOp walker not yet implemented")


# ---------------------------------------------------------------------------
# Test: run a WMMA kernel through the UOp walker
# ---------------------------------------------------------------------------

def test_walker():
    sim = _sim_path()
    passed, failed = 0, 0

    def check(name, got, expected):
        nonlocal passed, failed
        ok = np.array_equal(got, expected)
        print(f"  {'PASS' if ok else 'FAIL'} {name}")
        if ok: passed += 1
        else: failed += 1; print(f"    got={got.flatten()[:8]}\n    exp={expected.flatten()[:8]}")

    print("=" * 50)
    print("UOp-walker prototype tests")
    print("=" * 50)

    for desc, a_shape, w_shape in [
        ("4x4@4x4", (4,4), (4,4)),
        ("4x4@4x8", (4,4), (4,8)),
        ("8x4@4x4", (8,4), (4,4)),
        ("8x8@8x8", (8,8), (8,8)),
    ]:
        a_np = np.arange(a_shape[0]*a_shape[1], dtype=np.int32).reshape(a_shape) % 5 - 2
        w_np = np.arange(w_shape[0]*w_shape[1], dtype=np.int32).reshape(w_shape) % 5 - 2

        # Get UOps from tinygrad
        x = Tensor(a_np, dtype="int32", device="TINYTPU")
        W = Tensor(w_np, dtype="int32", device="TINYTPU")
        sched = (x @ W).schedule()
        prog = get_program(sched[-1].ast, Device['TINYTPU'].renderer)

        # Build buffers in the same order tinygrad would
        out_buf = bytearray(a_shape[0] * w_shape[1] * 4)
        act_buf = bytearray(a_np.astype("<i4").tobytes())
        weight_buf = bytearray(w_np.astype("<i4").tobytes())
        sim_bufs = (out_buf, act_buf, weight_buf)

        # Run through UOp walker
        bundle, out_addrs = render_bundle(prog.uops, sim_bufs)
        stdout = _run_bundle(sim, bundle)
        vmem_results = _parse_multi_vmem_output(stdout)

        # Assemble output
        num_vecs = a_shape[0]
        num_weight_tiles = w_shape[1] // _COLS
        out_cols = w_shape[1]
        result = np.zeros((num_vecs, out_cols), dtype=np.int32)
        for row in range(num_vecs):
            for t in range(num_weight_tiles):
                tile = vmem_results[row * num_weight_tiles + t]
                result[row, t*_COLS:(t+1)*_COLS] = tile[:_COLS]

        expected = a_np @ w_np
        check(desc, result, expected)

    # Test with bias + relu epilogue
    for desc, a_shape, w_shape in [("4x4+bias+relu", (4,4), (4,4)), ("4x4@4x8+bias+relu", (4,4), (4,8))]:
        a_np = np.arange(16, dtype=np.int32).reshape(4,4) % 7 - 3
        w_np = np.arange(a_shape[1]*w_shape[1], dtype=np.int32).reshape(a_shape[1], w_shape[1]) % 5 - 2
        b_np = np.arange(w_shape[1], dtype=np.int32) - w_shape[1]//2

        x = Tensor(a_np, dtype="int32", device="TINYTPU")
        W = Tensor(w_np, dtype="int32", device="TINYTPU")
        B = Tensor(b_np, dtype="int32", device="TINYTPU")
        sched = ((x @ W + B).relu()).schedule()
        # Find the WMMA kernel (skip COPY items)
        for s in sched:
            if hasattr(s, 'ast') and s.ast.op != Ops.COPY:
                prog = get_program(s.ast, Device['TINYTPU'].renderer)
                wmmas = [u for u in prog.uops if u.op is Ops.WMMA]
                if wmmas:
                    break

        out_buf = bytearray(4 * w_shape[1] * 4)
        act_buf = bytearray(a_np.astype("<i4").tobytes())
        weight_buf = bytearray(w_np.astype("<i4").tobytes())
        bias_buf = bytearray(b_np.astype("<i4").tobytes())
        sim_bufs = (out_buf, act_buf, weight_buf, bias_buf)

        bundle, out_addrs = render_bundle(prog.uops, sim_bufs)
        stdout = _run_bundle(sim, bundle)
        vmem_results = _parse_multi_vmem_output(stdout)

        num_weight_tiles = w_shape[1] // _COLS
        result = np.zeros((4, w_shape[1]), dtype=np.int32)
        for row in range(4):
            for t in range(num_weight_tiles):
                tile = vmem_results[row * num_weight_tiles + t]
                result[row, t*_COLS:(t+1)*_COLS] = tile[:_COLS]

        expected = np.maximum(a_np @ w_np + b_np, 0)
        check(desc, result, expected)

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if test_walker() else 1)
