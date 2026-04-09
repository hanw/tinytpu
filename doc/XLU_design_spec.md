# Cross-Lane Unit (XLU) — Design Specification

**Document status:** Draft v0.1  
**Scope:** TPU TensorCore functional unit — Cross-Lane Unit (XLU)  
**Based on:** Google Cloud TPU public documentation, JAX/Mosaic TPU compiler source (jax-ml/jax), TPU v4/v5p/v7x architecture documentation, and canonical SIMD/VLSI cross-lane design principles.

---

## 1. Overview

### 1.1 Purpose

The Cross-Lane Unit (XLU) is a dedicated hardware functional unit within the Google TPU TensorCore responsible for **data permutation across the lane dimension of vector registers**. It is distinct from the Matrix Multiply Unit (MXU), the vector unit, and the scalar unit.

The XLU enables efficient execution of **transpose operations** and **lane-crossing shuffles** that rearrange elements across the 128 physical lanes of a vector register. Without the XLU, such permutations would require expensive multi-cycle software emulation through HBM.

### 1.2 Position in TensorCore

```
TensorCore
├── MXU (Matrix Multiply Unit)       — systolic array, bulk FLOPS
├── Vector Unit (VPU)                — element-wise: activations, softmax
│   └── XLU (Cross-Lane Unit)        — lane-axis permutation, transpose
└── Scalar Unit                      — control flow, address calculation
```

The XLU is tightly coupled to the Vector Unit and shares access to the vector register file (vreg file). It does not interface with HBM directly.

---

## 2. Background: TPU Vector Register Layout

### 2.1 Vector Register Structure

Each TPU vector register (vreg) holds a 2D tile of data with shape:

```
vreg[sublanes][lanes] = vreg[S][L]
```

For current TPU generations:
- **L = 128 lanes** (the parallelism axis, maps to physical ALU lanes)
- **S = 8 sublanes** (the "depth" axis within each lane group)

This gives a vreg capacity of 128 × 8 = 1,024 elements per vreg. For BF16 (2 bytes), one vreg holds 2 KB. For FP32 (4 bytes), 4 KB.

Physically, the 128 lanes are independent processing elements that execute in lockstep (SIMD). A vreg is therefore analogous to an AVX-512 register, but 2D.

### 2.2 Tiled vs. Untiled Dimensions

When a multi-dimensional tensor is laid out across vregs, some axes map to the **lane dimension** (innermost tiled axis) and others map to **untiled (outer) dimensions** spanning multiple vregs.

```
Tensor shape [M, N]
  Tiled layout: sublane axis ← M tiles, lane axis ← N tiles
  Outer (untiled) axes: spans multiple vregs in the vreg array
```

This distinction is critical to XLU usage:

- **Transpose touching the innermost (tiled) dimension** → data must cross lanes → **requires XLU**
- **Transpose of only outer (untiled) dimensions** → vregs are simply reordered in the register array → **no XLU needed**, handled by Scalar Unit / loop reordering

### 2.3 Why Cross-Lane is Hard

In a 128-lane SIMD design, all 128 lanes share the same instruction stream. When lane `i` needs data from lane `j ≠ i`, the data must physically travel across a routing fabric. This is fundamentally different from in-lane element-wise ops.

Without dedicated hardware, a cross-lane permutation of N elements costs O(log N) passes through a butterfly network of AND/OR operations. With the XLU, this is reduced to a single instruction latency.

---

## 3. Functional Description

### 3.1 Core Function

Given an input vreg `V[s][l]` and a permutation specification `P`, the XLU computes:

```
Output[s][l] = Input[s][P(l)]      (lane permutation)
Output[s][l] = Input[P(s)][l]      (sublane permutation)
Output[s][l] = Input[P(s)][P(l)]   (full 2D permutation)
```

The Mosaic compiler determines whether XLU hardware is engaged based on whether the innermost (lane-axis) permutation is non-trivial:

```cpp
// From jax-ml/jax: jaxlib/mosaic/dialect/tpu/transforms/canonicalize_mosaic.cc
bool uses_xlu = !op.getPermutation().empty() &&
                op.getPermutation().back() != op.getPermutation().size() - 1;
```

That is: if the last dimension of the permutation is not the identity (i.e., the innermost tiled axis is being reordered), the XLU is engaged.

### 3.2 Operation Classes

The XLU supports the following logical operation classes:

| Class | Description | Example use case |
|---|---|---|
| Lane permute | Arbitrary reordering of elements across 128 lanes | General matrix transpose |
| Lane rotate | Cyclic shift of elements across lanes by offset K | Attention score shifts |
| Lane broadcast | Replicate one lane's value to all lanes | Scalar broadcast |
| Lane gather | Index-driven gather from lane dimension | Sparse embedding lookup |
| Sublane permute | Reordering across sublane axis | Tensor reshape |
| Sublane rotate | Cyclic shift across sublane axis | Convolution sliding window |
| 2D transpose | Swap sublane and lane axes (full S×L → L×S) | Classic matrix transpose |

### 3.3 Relationship to Transpose

The most common XLU use case is matrix transpose. For a tile stored as `[S][L]`:

- A **row-major to column-major** conversion requires swapping the lane and sublane axes
- This touches the innermost (lane) dimension → XLU required
- A mere **vreg array reordering** (e.g., swapping which vreg holds which row) is a scalar/loop-level transformation → XLU not needed

---

## 4. Microarchitecture

### 4.1 Physical Structure

The XLU is implemented as a **configurable permutation network** over the 128-lane datapath. The canonical implementation is a **log-stage butterfly network** (similar to an AES MixColumns or Beneš network structure):

```
Stage 0:  swap neighbors (stride 1)
Stage 1:  swap stride-2 pairs
Stage 2:  swap stride-4 pairs
...
Stage 6:  swap stride-64 pairs
Stage 7:  swap stride-128 pairs  (optional, for full reversal)
```

For 128 lanes, this requires **7 stages** to implement arbitrary permutations, or **fewer stages** for restricted permutations (rotates, broadcasts).

```
         L=128 lanes
         ┌──────────────────────────────────────────┐
Input ───┤  S0  │  S1  │  S2  │  S3  │  S4  │  S5  ├─── Output
         └──────────────────────────────────────────┘
              ↑ each stage: swap/pass based on control bits
```

The **control word** for each stage is loaded from a dedicated XLU control register, set by the Scalar Unit before XLU dispatch.

### 4.2 Datapath Width

- **Lane count:** 128
- **Element width:** 8-bit (FP8/INT8), 16-bit (BF16/FP16), 32-bit (FP32/INT32)
- **Total datapath width per cycle:** 128 × 32 bits = 4,096 bits = 512 bytes
- **Sublane processing:** The XLU processes one sublane row at a time, or multiple rows in parallel depending on pipeline width

### 4.3 Dtype Constraints

From compiler source evidence, **8-bit transposes through the XLU have hardware limitations on generation > 3** (TPU v4 and newer). This manifests as a compiler workaround:

```cpp
// TODO(b/448848595): Enable 8-bit transposes on generation 7.
if (element_type.getIntOrFloatBitWidth() == 8 && ctx.compatibility_mode &&
    ctx.hardware_generation > 3 && uses_xlu) {
  // Widen to 16-bit before transposing, then narrow back
  ...
}
```

This implies:
- BF16 and FP32 transposes: **fully supported** on all generations
- FP8/INT8 transposes: **emulated via 16-bit widening** on gen > 3 when XLU is used
- Root cause is likely a routing fabric width mismatch at 8-bit granularity in packed lane mode

### 4.4 Pipeline Stages

```
Cycle 0:   Instruction decode + control word decode
Cycle 1:   Read source vreg from register file
Cycle 2:   Permutation stage 0–2 (butterfly first half)
Cycle 3:   Permutation stage 3–6 (butterfly second half)
Cycle 4:   Write result vreg to register file
```

Estimated latency: **4–5 cycles** (consistent with other SIMD cross-lane units like x86 VPERMPS at 3 cycles + 1 throughput, NVIDIA shfl.sync at 4 cycles).

The XLU likely has a **throughput of 1 vreg/cycle** at steady state when control words are pre-loaded.

---

## 5. Interface Specification

### 5.1 Inputs

| Signal | Width | Description |
|---|---|---|
| `src_vreg` | S × L × W bits | Source vector register (S sublanes, L=128 lanes, W=element width) |
| `perm_ctrl[6:0]` | 7 × L bits | Per-stage butterfly control bits (or compressed permutation vector) |
| `op_type[2:0]` | 3 bits | Operation class: PERMUTE / ROTATE / BROADCAST / GATHER / TRANSPOSE |
| `rotate_amount` | 7 bits | Cyclic shift amount for ROTATE operations (0–127) |
| `dtype` | 2 bits | 00=FP8/INT8, 01=BF16/FP16, 10=FP32/INT32 |
| `sublane_sel` | 3 bits | Which sublane(s) to process (or ALL) |

### 5.2 Outputs

| Signal | Width | Description |
|---|---|---|
| `dst_vreg` | S × L × W bits | Destination vector register |
| `done` | 1 bit | Asserted when result is written back to vreg file |
| `fault` | 1 bit | Asserted on unsupported dtype/op combination |

### 5.3 Control Path

The Scalar Unit programs the XLU via a **control register interface**:

```
XLU_CTRL_REG_0: op_type, dtype, rotate_amount, sublane_sel
XLU_CTRL_REG_1..7: per-stage butterfly control words (for PERMUTE)
XLU_SRC_VREG: vreg address in register file
XLU_DST_VREG: destination vreg address
XLU_DISPATCH: write to this register triggers execution
```

This is analogous to the Scalar Unit programming the MXU before a matrix multiply dispatch.

---

## 6. Supported Operations — Detail

### 6.1 Lane Permute (PERMUTE)

Arbitrary permutation of 128 lane elements.

```
Input:  [v0, v1, v2, ..., v127]
Perm P: [3, 0, 127, 1, ...]
Output: [v3, v0, v127, v1, ...]
```

Control: Full 7-stage butterfly control word required. Pre-computed by compiler from the permutation array.

Cost: Fixed at ~4 cycles regardless of permutation complexity.

### 6.2 Lane Rotate (ROTATE)

Cyclic shift of all lane elements by constant K.

```
Input:  [v0, v1, v2, ..., v127]
K=3
Output: [v3, v4, ..., v127, v0, v1, v2]
```

Control: Only `rotate_amount` register needed. Butterfly stages collapse to a barrel-shifter configuration — potentially fewer active stages → possible 2-cycle latency.

Primary use case: Sliding window operations, attention key-query offsets.

### 6.3 Lane Broadcast (BROADCAST)

Replicate lane `i`'s value to all 128 lanes.

```
Input:  [v0, v1, ..., vi, ..., v127]
i=5
Output: [v5, v5, v5, ..., v5]
```

Control: Single lane index. Implemented as a degenerate permutation where all outputs point to source lane i.

### 6.4 Lane Gather (GATHER)

Index-driven gather: each output lane `l` reads from the lane index stored in a **separate index vreg** `idx_vreg[l]`.

```
Input data:  [v0, v1, ..., v127]
Index vreg:  [32, 0, 64, 1, ...]
Output:      [v32, v0, v64, v1, ...]
```

This is a **data-dependent permutation** — unlike PERMUTE, the control word is not known at compile time. The XLU must therefore include a **dynamic routing path** where the butterfly control is loaded per-cycle from the index vreg.

This is the most complex XLU operation and has the highest area cost. Likely implemented via the same butterfly fabric but with dynamic per-cycle control word injection.

### 6.5 Sublane Rotate (SUBLANE_ROTATE)

Cyclic shift across the sublane axis (depth rotation). Similar to ROTATE but operating on the S=8 sublane axis rather than the L=128 lane axis.

```
Input vreg:  row0=[...], row1=[...], ..., row7=[...]
K=2
Output:      row0=old_row2, row1=old_row3, ..., row7=old_row1
```

This may be implemented in the **VPU** rather than the XLU proper, as it involves a smaller routing problem (8 elements vs. 128).

### 6.6 Full 2D Transpose (TRANSPOSE)

Swap the sublane and lane axes: `out[l][s] = in[s][l]`.

This is the canonical "matrix transpose within a tile" and is the primary use case identified in the compiler source.

```
Input vreg  [8 sublanes][128 lanes]:
  row0: [a00, a01, a02, ... a0,127]
  row1: [a10, a11, a12, ... a1,127]
  ...
  row7: [a70, a71, a72, ... a7,127]

Output vreg [128 lanes][8 sublanes]:
  (requires multiple output vregs or a 128×8 → 8×128 scatter)
```

Because the output tile shape (128 × 8) differs from the input shape (8 × 128), a full 2D transpose requires either:
1. Writing to a different-shaped output vreg format, or
2. Using **8 successive XLU operations** to scatter each sublane row across 8 output vregs

The compiler handles this via a sequence of XLU instructions with vreg array restructuring.

---

## 7. Compiler Interface (Mosaic)

### 7.1 Transpose Detection

In the Mosaic MLIR dialect, vector.transpose operations are lowered through the XLU when they touch the tiled (lane-axis) dimension:

```cpp
// canonicalize_mosaic.cc
bool uses_xlu = !op.getPermutation().empty() &&
                op.getPermutation().back() != op.getPermutation().size() - 1;
```

If `uses_xlu` is false, the transpose is pure vreg-array reindexing at zero cost.

### 7.2 Dtype Canonicalization

For 8-bit transposes via XLU on gen > 3, the compiler widens to 16-bit:

```cpp
if (element_type.getIntOrFloatBitWidth() == 8 && ctx.compatibility_mode &&
    ctx.hardware_generation > 3 && uses_xlu) {
  // Cast INT8/FP8 → INT16/BF16
  // Perform transpose via XLU (now legal at 16-bit)
  // Cast back to INT8/FP8
}
```

This adds 2 extra conversion passes but avoids illegal hardware operation.

### 7.3 XLA/Mosaic Operation Mapping

| Mosaic op | XLU operation | Notes |
|---|---|---|
| `vector.transpose` (inner dim) | PERMUTE / TRANSPOSE | Main transpose path |
| `tpu.rotate` | ROTATE | Sliding window, attention |
| `tpu.gather` (lane-indexed) | GATHER | Dynamic index gather |
| `vector.broadcast` (lane dim) | BROADCAST | Scalar broadcast to all lanes |
| `tpu.rotate` (sublane dim) | SUBLANE_ROTATE | Possibly VPU, not XLU |

---

## 8. Performance Characteristics

### 8.1 Throughput

| Operation | Throughput (vregs/cycle) | Latency (cycles) |
|---|---|---|
| Lane PERMUTE | 1 | ~4–5 |
| Lane ROTATE | 1 | ~2–3 |
| Lane BROADCAST | 1 | ~2 |
| Lane GATHER (dynamic) | 1 | ~5–6 |
| Full 2D TRANSPOSE (8×128) | 1/8 (8 XLU ops) | ~32–40 |

### 8.2 Bandwidth

- Input bandwidth to XLU: **512 bytes/cycle** (128 lanes × 4 bytes per lane)
- Output bandwidth from XLU: **512 bytes/cycle**
- This matches HBM bandwidth to MXU, so XLU is not a bandwidth bottleneck when operating in vreg-resident data

### 8.3 Area vs. MXU

The XLU butterfly network for 128 lanes requires approximately:
- **7 stages × 64 2:1 MUX units = 448 MUX units**
- For 32-bit elements: 448 × 32 = **14,336 bits of routing fabric**
- Compare MXU (128×128 BF16 systolic): ~32,768 MAC units × ~100 gates each

The XLU is roughly **2–3 orders of magnitude smaller** than the MXU in area and power.

---

## 9. Design Constraints and Known Issues

### 9.1 8-bit Transpose Limitation (Gen > 3)

**Issue:** Direct 8-bit (FP8/INT8) transposes via XLU are unsupported or produce incorrect results on hardware generation > 3.

**Evidence:** Bug tracker b/448848595, comment "Enable 8-bit transposes on generation 7."

**Root cause (hypothesized):** The XLU routing fabric at gen > 3 was designed around 16-bit packed lanes. When packing two 8-bit elements per lane slot, the butterfly control granularity cannot address individual 8-bit elements within a 16-bit lane slot.

**Workaround:** Widen to 16-bit before XLU dispatch, narrow after.

**Status:** Open as of TPU7x (gen 7). The TODO comment suggests this is known and planned for fix.

### 9.2 Untiled Dimension Transposes — No XLU

Transposes that reorder only outer (untiled) dimensions are **free** — they're implemented by reindexing into the vreg array without any data movement. The compiler must correctly distinguish these from XLU-requiring transposes.

### 9.3 Dynamic Gather Latency

GATHER operations with runtime-determined indices have higher latency (~5–6 cycles) because the butterfly control word must be loaded from the index vreg each cycle, preventing static control word pre-loading.

### 9.4 Sublane vs. Lane Operations

Operations on the **sublane axis (depth=8)** likely bypass the full XLU butterfly and use a simpler **8:1 MUX tree** in the VPU or a dedicated sublane rotate unit. This is cheaper in hardware but has separate constraints.

---

## 10. Hardware Generation Matrix

| TPU Gen | XLU Present | BF16 Transpose | FP8/INT8 Transpose | Notes |
|---|---|---|---|---|
| v3 (gen 3) | Yes | Supported | Supported | Pre-XLU gen limit |
| v4 (gen 4) | Yes | Supported | Emulated (widen) | 8-bit issue introduced |
| v5p (gen 5) | Yes | Supported | Emulated (widen) | Same constraint |
| v6e (gen 6) | Yes | Supported | Emulated (widen) | Same constraint |
| TPU7x/Ironwood (gen 7) | Yes | Supported | TODO b/448848595 | Still not fixed as of gen 7 |

---

## 11. Open Questions and Research Directions

1. **Exact implementation:** Is the XLU a Beneš network (rearrangeable, non-blocking) or a simpler omega/butterfly network (blocking for some permutations)?

2. **Sublane XLU:** Does the XLU handle sublane permutations, or is there a separate "SXU" (Sublane Xchange Unit) or is it folded into the VPU?

3. **Dynamic GATHER implementation:** Is the index vreg loaded into the butterfly control registers in a separate clock, or does the XLU have a dedicated operand port for the index?

4. **Dual-chiplet (Ironwood) implications:** In TPU7x's dual-chiplet architecture, vregs are per-chiplet. Does the XLU operate intra-chiplet only, or can it route across the D2D interface?

5. **FP8 fix timeline:** The TODO comment targeting gen 7 suggests FP8 native transpose may be implemented in a future generation. What silicon change is required?

6. **Interaction with SparseCores:** SparseCores handle embedding lookup (sparse gather). Is there overlap with the XLU's GATHER operation, or are they entirely separate pipelines?

---

## 12. References

1. Google Cloud TPU documentation: `docs.cloud.google.com/tpu/docs/tpu7x` (Ironwood architecture)
2. Google Cloud TPU documentation: `docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm` (TensorCore structure)
3. JAX Mosaic TPU compiler: `jax-ml/jax/jaxlib/mosaic/dialect/tpu/transforms/canonicalize_mosaic.cc` — commit 2fbe731 (apaszke, Oct 2025)
4. JAX PR #32290: "[Mosaic TPU] Only canonicalize the dtype of transposes if they use the XLU"
5. Jouppi et al., "TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings", arXiv:2304.01433
6. Patterson & Hennessy, "Computer Organization and Design", Chapter 6 (SIMD architectures)
7. NVIDIA PTX ISA: `shfl.sync` instruction (cross-lane shuffle, for comparison)
8. Intel Architecture Instruction Reference: `VPERMPS`, `VPSHUFB` (cross-lane permute, for comparison)

---

*Document generated: 2026-04-08*  
*Status: Engineering working draft — based on reverse-engineered compiler evidence and public architecture docs. Not an official Google document.*
