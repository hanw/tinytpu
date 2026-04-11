"""
MNIST GAN forward pass on TinyTPU.

Adapted from tinygrad/examples/mnist_gan.py. Uses quantized int8 GEMM on
the MXU with float32 leaky_relu activations on the VPU — the same mixed
quantized-inference pattern used on real TPUs.

Architecture (scaled down from original 128→256→512→1024→784):
  Generator:     4 → 8 → 8 → 4  (3 layers, leaky_relu between)
  Discriminator: 4 → 8 → 4      (2 layers, leaky_relu + argmax)

Profile:
    PYTHONPATH=tinygrad python scripts/profile_tpu.py \\
        --from-tinygrad scripts/models/mnist_gan.py

Visualize:
    open scripts/viz_pipeline.html   # Load JSON → trace.json
"""
import os, sys, struct, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("TINYTPU_SIM", os.path.join(REPO, "build", "mkTbTinyTPURuntime.bexe"))
os.environ["DISABLE_COMPILER_CACHE"] = "1"
sys.path.insert(0, os.path.join(REPO, "tinygrad"))

from tinygrad.runtime.ops_tinytpu import (
    _vmem, _load, _vpu, _store, _halt, _output_vmem, _end, _bundle,
    _run_bundle, _parse_vmem_output, _parse_multi_vmem_output,
    _build_full_gemm_bundle, _VPU_OPS, _TILE_ELEMS
)

sim = os.environ["TINYTPU_SIM"]
_COLS = 4

def f2i(f): return struct.unpack('<i', struct.pack('<f', f))[0]
def i2f(i): return struct.unpack('<f', struct.pack('<i', i))[0]
def float_tile(vals):
    t = [0] * _TILE_ELEMS
    for i, v in enumerate(vals): t[i] = f2i(float(v))
    return t

def vpu_float_binary(a, b, op):
    n = len(a)
    stdout = _run_bundle(sim, _bundle(
        _vmem(0, float_tile(a)), _vmem(1, float_tile(b)),
        _load(0, 0), _load(1, 1),
        _vpu(2, 0, _VPU_OPS[op], 1),
        _store(2, 2), _halt(), _output_vmem(2), _end()))
    return np.array([i2f(v) for v in _parse_vmem_output(stdout)[:n]], dtype=np.float32)

def vpu_i2f(int_vals):
    n = len(int_vals)
    tile = [0]*_TILE_ELEMS; tile[:n] = [int(v) for v in int_vals]
    stdout = _run_bundle(sim, _bundle(
        _vmem(0, tile), _load(0, 0), _vpu(1, 0, _VPU_OPS["I2F"]),
        _store(2, 1), _halt(), _output_vmem(2), _end()))
    return np.array([i2f(v) for v in _parse_vmem_output(stdout)[:n]], dtype=np.float32)

def vpu_f2i(float_vals):
    n = len(float_vals)
    stdout = _run_bundle(sim, _bundle(
        _vmem(0, float_tile(float_vals)), _load(0, 0), _vpu(1, 0, _VPU_OPS["F2I"]),
        _store(2, 1), _halt(), _output_vmem(2), _end()))
    return np.array(_parse_vmem_output(stdout)[:n], dtype=np.int32)

def gemm(act_i8, weight_i8, nv, nk, nw):
    bundle = _build_full_gemm_bundle(act_i8, weight_i8, nv, nk, nw)
    stdout = _run_bundle(sim, bundle)
    tiles = _parse_multi_vmem_output(stdout)
    oc = nw * _COLS
    out = np.zeros((nv, oc), dtype=np.int32)
    for r in range(nv):
        for t in range(nw):
            out[r, t*_COLS:(t+1)*_COLS] = tiles[r*nw+t][:_COLS]
    return out

def leaky_relu_row(x_f, alpha=0.2):
    scaled = vpu_float_binary(list(x_f), [alpha]*len(x_f), "FMUL")
    return vpu_float_binary(list(x_f), list(scaled), "FMAX")

def linear_lrelu(act_i8, w_i8, nv, nk, nw, alpha=0.2):
    g = gemm(act_i8, w_i8, nv, nk, nw)
    oc = nw * _COLS
    result = np.zeros((nv, oc), dtype=np.int8)
    for r in range(nv):
        rf = vpu_i2f(g[r])
        ra = leaky_relu_row(rf, alpha)
        result[r] = np.clip(vpu_f2i(ra), -127, 127).astype(np.int8)
    return result


if __name__ == "__main__":
    np.random.seed(42)
    print("MNIST GAN on TinyTPU (int8 MXU + float32 VPU)")

    W_g1 = np.random.randint(-3, 4, size=(4,8), dtype=np.int32).astype(np.int8)
    W_g2 = np.random.randint(-3, 4, size=(8,8), dtype=np.int32).astype(np.int8)
    W_g3 = np.random.randint(-3, 4, size=(8,4), dtype=np.int32).astype(np.int8)
    W_d1 = np.random.randint(-3, 4, size=(4,8), dtype=np.int32).astype(np.int8)
    W_d2 = np.random.randint(-3, 4, size=(8,4), dtype=np.int32).astype(np.int8)
    noise = np.random.randint(-2, 3, size=(4,4), dtype=np.int32).astype(np.int8)

    # Generator
    g1 = linear_lrelu(noise, W_g1, 4, 1, 2)
    g2 = linear_lrelu(g1, W_g2, 4, 2, 2)
    g3 = gemm(g2, W_g3, 4, 2, 1)
    gen_out = np.array([vpu_i2f(row) for row in g3])
    print(f"Generator output: {gen_out.shape}\n{gen_out}")

    # Discriminator
    d_in = np.array([np.clip(vpu_f2i(row), -127, 127).astype(np.int8) for row in gen_out])
    d1 = linear_lrelu(d_in, W_d1, 4, 1, 2)
    d2 = gemm(d1, W_d2, 4, 2, 1)
    disc_out = np.array([vpu_i2f(row) for row in d2])
    print(f"Discriminator logits: {disc_out.shape}\n{disc_out}")
    print(f"Predictions: {disc_out.argmax(axis=1)}")
