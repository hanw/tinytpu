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
    """Render an elementwise kernel by mapping scalar UOps to VPU tile ops.

    The UOp graph has the pattern:
      PARAM (out, src1, src2) → [RANGE →] LOAD×N + OP×N + STORE×N [→ END]
    For each RANGE iteration, we emit one VPU tile dispatch.
    Without RANGE (fully unrolled), we emit one tile for all ops.
    """
    from collections import Counter
    op_counts = Counter(u.op.name for u in uops)

    # Detect ReLU pattern: CMPLT + WHERE (unary, no binary ALU)
    is_relu = op_counts.get("WHERE", 0) > 0 and op_counts.get("CMPLT", 0) > 0
    out_arg = 0
    src_args = sorted(k for k in params if k != out_arg)

    if is_relu and len(src_args) == 1:
        return _render_relu_bundle(uops, params, bufs)

    if len(src_args) < 2:
        raise NotImplementedError(f"elementwise UOp walker: expected ≥2 source params, got {len(src_args)}")

    # Find the binary ALU op
    alu_ops_map = {
        "ADD": "ADD", "MUL": "MUL", "SUB": "SUB", "MAX": "MAX", "MIN": "MIN",
        "CMPLT": "CMPLT", "CMPNE": "CMPNE", "CMPEQ": "CMPEQ",
        "AND": "AND", "OR": "OR", "XOR": "XOR",
        "SHL": "SHL", "SHR": "SHR", "IDIV": "DIV",
    }
    alu_name = None
    for op_name, vpu_name in alu_ops_map.items():
        if op_counts.get(op_name, 0) > 0:
            alu_name = vpu_name
            break
    if alu_name is None:
        raise NotImplementedError(f"elementwise UOp walker: no recognized ALU op in {dict(op_counts)}")

    # Trace param ordering from the UOp graph
    alu_op_enum = getattr(Ops, [k for k,v in alu_ops_map.items() if v == alu_name][0])
    alu_uop = next(u for u in uops if u.op is alu_op_enum)
    lhs_arg = _find_param_arg(alu_uop.src[0])
    rhs_arg = _find_param_arg(alu_uop.src[1])
    if lhs_arg is None or rhs_arg is None:
        raise NotImplementedError(f"elementwise UOp walker: could not trace ALU operand params")

    lhs_i32 = np.frombuffer(bytes(bufs[lhs_arg]), dtype="<i4")
    rhs_i32 = np.frombuffer(bytes(bufs[rhs_arg]), dtype="<i4")
    num_elems = min(params[out_arg].dtype.size, lhs_i32.size, rhs_i32.size)
    vpu_op = _VPU_OPS[alu_name]

    # Emit tiled bundles, one per VMEM tile
    data_lines = []
    prog_lines = []
    out_addrs = []

    for chunk_start in range(0, num_elems, _TILE_ELEMS):
        chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
        chunk_size = chunk_end - chunk_start

        lhs_tile = np.zeros(_TILE_ELEMS, dtype=np.int32)
        rhs_tile = np.zeros(_TILE_ELEMS, dtype=np.int32)
        lhs_tile[:chunk_size] = lhs_i32[chunk_start:chunk_end]
        rhs_tile[:chunk_size] = rhs_i32[chunk_start:chunk_end]

        tile_idx = chunk_start // _TILE_ELEMS
        vmem_lhs = tile_idx * 3
        vmem_rhs = tile_idx * 3 + 1
        vmem_out = tile_idx * 3 + 2

        data_lines.append(_vmem(vmem_lhs, [int(x) for x in lhs_tile]))
        data_lines.append(_vmem(vmem_rhs, [int(x) for x in rhs_tile]))

        prog_lines.append(_load(0, vmem_lhs))
        prog_lines.append(_load(1, vmem_rhs))
        prog_lines.append(_vpu(2, 0, vpu_op, 1))
        prog_lines.append(_store(vmem_out, 2))
        out_addrs.append(vmem_out)

    prog_lines.append(_halt())
    output_lines = [_output_vmem(a) for a in out_addrs] + [_end()]

    return _bundle(*(data_lines + prog_lines + output_lines)), out_addrs


def _render_relu_bundle(uops, params, bufs):
    """Render a ReLU (max(x, 0)) kernel as VPU RELU ops."""
    out_arg = 0
    src_arg = next(k for k in params if k != out_arg)
    src_i32 = np.frombuffer(bytes(bufs[src_arg]), dtype="<i4")
    num_elems = params[out_arg].dtype.size

    data_lines = []
    prog_lines = []
    out_addrs = []

    for chunk_start in range(0, num_elems, _TILE_ELEMS):
        chunk_end = min(chunk_start + _TILE_ELEMS, num_elems)
        chunk_size = chunk_end - chunk_start
        tile = np.zeros(_TILE_ELEMS, dtype=np.int32)
        tile[:chunk_size] = src_i32[chunk_start:chunk_end]

        tile_idx = chunk_start // _TILE_ELEMS
        vmem_src = tile_idx * 2
        vmem_out = tile_idx * 2 + 1

        data_lines.append(_vmem(vmem_src, [int(x) for x in tile]))
        prog_lines.append(_load(0, vmem_src))
        prog_lines.append(_vpu(1, 0, 2))  # VPU_RELU = opcode 2
        prog_lines.append(_store(vmem_out, 1))
        out_addrs.append(vmem_out)

    prog_lines.append(_halt())
    output_lines = [_output_vmem(a) for a in out_addrs] + [_end()]

    return _bundle(*(data_lines + prog_lines + output_lines)), out_addrs


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

    # Test elementwise ops via walker
    print("\n--- Elementwise via UOp walker ---")
    for desc, op_fn, np_fn in [
        ("4-elem ADD", lambda a,b: a+b, lambda a,b: a+b),
        ("4-elem MUL", lambda a,b: a*b, lambda a,b: a*b),
        ("4-elem SUB", lambda a,b: a-b, lambda a,b: a-b),
        ("16-elem ADD", lambda a,b: a+b, lambda a,b: a+b),
    ]:
        if "16" in desc:
            a_np = np.arange(16, dtype=np.int32)
            b_np = np.arange(16, 32, dtype=np.int32)
        else:
            a_np = np.array([1,2,3,4], dtype=np.int32)
            b_np = np.array([5,6,7,8], dtype=np.int32)

        x = Tensor(a_np, dtype="int32", device="TINYTPU")
        y = Tensor(b_np, dtype="int32", device="TINYTPU")
        sched = op_fn(x, y).schedule()
        for s in sched:
            if hasattr(s, 'ast') and s.ast.op != Ops.COPY:
                p = get_program(s.ast, Device['TINYTPU'].renderer)
                if any(u.op is Ops.WMMA for u in p.uops):
                    continue  # skip WMMA
                out_buf = bytearray(len(a_np) * 4)
                lhs_buf = bytearray(a_np.astype("<i4").tobytes())
                rhs_buf = bytearray(b_np.astype("<i4").tobytes())
                try:
                    bundle, out_addrs = render_bundle(p.uops, (out_buf, lhs_buf, rhs_buf))
                    stdout = _run_bundle(sim, bundle)
                    vmem_results = _parse_multi_vmem_output(stdout)
                    result = np.concatenate([np.array(t[:min(_TILE_ELEMS, len(a_np) - i*_TILE_ELEMS)], dtype=np.int32)
                                            for i, t in enumerate(vmem_results)])
                    expected = np_fn(a_np, b_np)
                    check(desc, result, expected)
                except NotImplementedError as e:
                    print(f"  SKIP {desc}: {e}")

    # Test ReLU
    a_relu = np.array([1, -2, 3, -4], dtype=np.int32)
    x_relu = Tensor(a_relu, dtype="int32", device="TINYTPU")
    sched_r = x_relu.relu().schedule()
    for s in sched_r:
        if hasattr(s, 'ast') and s.ast.op != Ops.COPY:
            p = get_program(s.ast, Device['TINYTPU'].renderer)
            if not any(u.op is Ops.WMMA for u in p.uops):
                out_buf = bytearray(4 * 4)
                src_buf = bytearray(a_relu.astype("<i4").tobytes())
                try:
                    bundle, out_addrs = render_bundle(p.uops, (out_buf, src_buf))
                    stdout = _run_bundle(sim, bundle)
                    result = np.array(_parse_vmem_output(stdout)[:4], dtype=np.int32)
                    check("4-elem RELU", result, np.maximum(a_relu, 0))
                except NotImplementedError as e:
                    print(f"  SKIP RELU: {e}")

    print(f"\n{passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if test_walker() else 1)
