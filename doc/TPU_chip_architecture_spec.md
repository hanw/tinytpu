# Google TPU — Chip-Level and System-on-Chip Architecture Specification

**Document status:** Engineering Reference Draft v1.0  
**Scope:** Full chip architecture — TensorCore internals (Scalar, Vector/XLU, MXU), SparseCore, memory subsystem, and SoC integration  
**Coverage:** TPU v1 through TPU7x (Ironwood), with emphasis on v4/v5p/v7x  
**Based on:** Google Cloud TPU public documentation, JAX/Mosaic compiler source (jax-ml/jax), TPU v4 paper (arXiv:2304.01433), original TPU paper (arXiv:1704.04760), and publicly disclosed architecture details

---

## Table of Contents

1. Executive Overview
2. Chip Block Diagram
3. TensorCore Architecture
   - 3.1 Scalar Unit
   - 3.2 Matrix Multiply Unit (MXU)
   - 3.3 Vector Processing Unit (VPU)
   - 3.4 Cross-Lane Unit (XLU)
   - 3.5 On-Chip Vector Memory (VMEM)
   - 3.6 TensorCore Execution Pipeline
4. SparseCore Architecture
5. Memory Subsystem
   - 5.1 Memory Hierarchy
   - 5.2 HBM Interface
   - 5.3 DMA Engine
6. System-on-Chip (SoC) Architecture
   - 6.1 Multi-TensorCore Integration
   - 6.2 Dual-Chiplet Architecture (TPU7x)
   - 6.3 Inter-Chip Interconnect (ICI)
   - 6.4 PCIe Host Interface
   - 6.5 Clock Architecture
   - 6.6 Power Domains
7. Generation Comparison Matrix
8. Execution Model and Programming Interface
9. Known Design Constraints and Pitfalls
10. References

---

## 1. Executive Overview

Google's Tensor Processing Unit (TPU) is a purpose-built ASIC for neural network workloads. Unlike general-purpose CPUs or GPUs, the TPU is a **domain-specific processor** — it forgoes hardware for rendering, floating-point division, caching hierarchies, and branch prediction in favor of maximizing throughput per watt for matrix multiply and element-wise neural network operations.

The fundamental architectural unit is the **TensorCore**, which contains four cooperating execution units:

```
TensorCore
├── Scalar Unit (SXU)        — control flow, address generation, scalar ops
├── Matrix Multiply Unit (MXU) — systolic array; dominates chip area and FLOPS
├── Vector Processing Unit (VPU) — 128-wide SIMD element-wise ops
│   └── Cross-Lane Unit (XLU)    — lane-axis permutation / transpose
└── VMEM                     — on-chip SRAM scratchpad
```

Multiple TensorCores are integrated into a single chip alongside **SparseCore** accelerators, **HBM** memory stacks, **ICI** fabric, and a **PCIe** host interface to form the SoC.

At the pod level, hundreds to thousands of chips are interconnected via ICI to form a **3D torus** supercomputer fabric.

---

## 2. Chip Block Diagram

### 2.1 Single-Chiplet View (v4 / v5p style)

```
╔═════════════════════════════════════════════════════════════════════════╗
║                          TPU CHIP (v4/v5p)                              ║
║                                                                         ║
║  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  ┌────┐ ║
║  │  TensorCore 0   │  │  TensorCore 1   │  │  TensorCore 2   │  │ T3 │ ║
║  │  ┌──────────┐   │  │  ┌──────────┐   │  │  ┌──────────┐   │  │    │ ║
║  │  │  MXU ×2  │   │  │  │  MXU ×2  │   │  │  │  MXU ×2  │   │  │... │ ║
║  │  └──────────┘   │  │  └──────────┘   │  │  └──────────┘   │  │    │ ║
║  │  ┌──────────┐   │  │  ┌──────────┐   │  │  ┌──────────┐   │  │    │ ║
║  │  │ VPU/XLU  │   │  │  │ VPU/XLU  │   │  │  │ VPU/XLU  │   │  │    │ ║
║  │  └──────────┘   │  │  └──────────┘   │  │  └──────────┘   │  │    │ ║
║  │  ┌──────────┐   │  │  ┌──────────┐   │  │  ┌──────────┐   │  │    │ ║
║  │  │  Scalar  │   │  │  │  Scalar  │   │  │  │  Scalar  │   │  │    │ ║
║  │  └──────────┘   │  │  └──────────┘   │  │  └──────────┘   │  │    │ ║
║  │  ┌──────────┐   │  │  ┌──────────┐   │  │  ┌──────────┐   │  │    │ ║
║  │  │  VMEM    │   │  │  │  VMEM    │   │  │  │  VMEM    │   │  │    │ ║
║  │  │ (16-64MB)│   │  │  │ (16-64MB)│   │  │  │ (16-64MB)│   │  │    │ ║
║  │  └──────────┘   │  │  └──────────┘   │  │  └──────────┘   │  │    │ ║
║  └─────────────────┘  └─────────────────┘  └─────────────────┘  └────┘ ║
║                                                                         ║
║  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐   ║
║  │   SparseCore 0    │  │   SparseCore 1    │  │ SparseCore 2,3... │   ║
║  └───────────────────┘  └───────────────────┘  └───────────────────┘   ║
║                                                                         ║
║  ┌─────────────────────────────────────────────────────────────────┐   ║
║  │                    HBM Controller + PHY                         │   ║
║  └─────────────────────────────────────────────────────────────────┘   ║
║                                                                         ║
║  ┌────────────┐   ┌────────────────────┐   ┌──────────────────────┐   ║
║  │ ICI Fabric │   │  PCIe Host I/F     │   │ Clocks / Power / PMU │   ║
║  │  (6 links) │   │  (Gen4/Gen5)       │   │                      │   ║
║  └────────────┘   └────────────────────┘   └──────────────────────┘   ║
╚═════════════════════════════════════════════════════════════════════════╝

    ↓↓↓↓↓↓↓↓↓↓↓
╔══════════════════╗   ╔══════════════════╗
║    HBM Stack 0   ║   ║    HBM Stack 1   ║   (×N stacks depending on gen)
╚══════════════════╝   ╚══════════════════╝
```

### 2.2 Dual-Chiplet View (TPU7x / Ironwood)

```
╔══════════════════════════════════════════════════════════╗
║                    TPU7x PHYSICAL CHIP                   ║
║  ┌────────────────────────┐  ┌────────────────────────┐  ║
║  │       CHIPLET A        │  │       CHIPLET B        │  ║
║  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐  │  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐  │  ║
║  │  │TC│ │TC│ │TC│ │TC│  │  │  │TC│ │TC│ │TC│ │TC│  │  ║
║  │  └──┘ └──┘ └──┘ └──┘  │  │  └──┘ └──┘ └──┘ └──┘  │  ║
║  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐  │  │  ┌──┐ ┌──┐ ┌──┐ ┌──┐  │  ║
║  │  │SC│ │SC│ │SC│ │SC│  │  │  │SC│ │SC│ │SC│ │SC│  │  ║
║  │  └──┘ └──┘ └──┘ └──┘  │  │  └──┘ └──┘ └──┘ └──┘  │  ║
║  │  ┌────────────────┐    │  │  ┌────────────────┐    │  ║
║  │  │  HBM (96 GB)   │    │  │  │  HBM (96 GB)   │    │  ║
║  │  └────────────────┘    │  │  └────────────────┘    │  ║
║  └────────────────────────┘  └────────────────────────┘  ║
║               ↕  D2D (Die-to-Die) Interface  ↕            ║
║             6× faster than 1D ICI link                    ║
╚══════════════════════════════════════════════════════════╝

TC = TensorCore  (4 per chiplet = 8 total per chip)
SC = SparseCore  (4 per chiplet = 8 total per chip)
```

---

## 3. TensorCore Architecture

The TensorCore is the fundamental compute unit of the TPU. All ML computation is expressed as a sequence of operations dispatched to four cooperating sub-units within each TensorCore.

### 3.1 Scalar Unit (SXU — Scalar eXecution Unit)

#### Purpose
The Scalar Unit provides sequential control flow and scalar computation. It is the "CPU" of the TensorCore — responsible for loop control, conditional branching, address arithmetic, and programming the other units.

#### Functional Description

```
┌────────────────────────────────────────────────────────────────┐
│                        SCALAR UNIT                              │
│                                                                  │
│  ┌────────────┐   ┌──────────────┐   ┌──────────────────────┐  │
│  │ Instruction│   │    Scalar    │   │    Memory Address    │  │
│  │   Fetch /  │──▶│  Register   │──▶│     Generation       │  │
│  │   Decode   │   │  File (32×  │   │    (VMEM + SMEM      │  │
│  └────────────┘   │   32-bit)   │   │     addressing)      │  │
│                   └──────────────┘   └──────────────────────┘  │
│                          │                      │               │
│                   ┌──────────────┐   ┌──────────────────────┐  │
│                   │  Scalar ALU  │   │   Control Register   │  │
│                   │  (int/fp32)  │   │   Programming Bus    │  │
│                   └──────────────┘   │  (MXU, VPU, DMA)     │  │
│                                      └──────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

#### Register File
- **32 scalar registers**, each 32-bit
- Supports INT32, FP32, and address values
- Separate scalar memory (SMEM): small on-chip buffer for scalar variables and loop state

#### ISA Characteristics
- **VLIW (Very Long Instruction Word)** style: a single instruction packet can dispatch simultaneously to Scalar, VPU, and MXU units
- Scalar instructions: load/store, branch, compare, arithmetic (INT32/FP32), shift
- All loops must be **statically unrolled or have statically-bounded trip counts** — no dynamic branch prediction
- Out-of-order execution within a single unit; inter-unit ordering enforced by compiler-inserted synchronization barriers

#### Control Register Bus
The Scalar Unit programs other units before dispatching them:
- **MXU control:** load weight tile address, output accumulator address, computation shape
- **VPU control:** element-wise operation type, reduction axes, activation function
- **XLU control:** permutation type, rotate amount, source/destination vreg addresses
- **DMA control:** VMEM ↔ HBM transfer descriptors, size, stride

#### Latency
- Scalar arithmetic: **1 cycle**
- VMEM scalar load: **3–5 cycles** (pipelined)
- Control register write + dispatch: **1 cycle** (write), **N cycles** (unit latency, can overlap)

---

### 3.2 Matrix Multiply Unit (MXU)

#### Purpose
The MXU is the primary compute engine. It implements a **systolic array** for matrix-multiply-accumulate operations. It contributes the vast majority (~85–90%) of peak FLOPS for training workloads.

#### Systolic Array Design

A systolic array is a grid of Processing Elements (PEs) connected to their neighbors. Data flows rhythmically through the array like a heartbeat:

```
  W₀₀  W₀₁  W₀₂  ...  W₀,127
   ↓    ↓    ↓          ↓
  ┌──┐ ┌──┐ ┌──┐       ┌──┐  ← Row 0 of PEs
  │PE│→│PE│→│PE│→ ... →│PE│
  └──┘ └──┘ └──┘       └──┘
   ↓    ↓    ↓          ↓
  ┌──┐ ┌──┐ ┌──┐       ┌──┐  ← Row 1 of PEs
  │PE│→│PE│→│PE│→ ... →│PE│
  └──┘ └──┘ └──┘       └──┘
   ...
   ↓    ↓    ↓          ↓
  ┌──┐ ┌──┐ ┌──┐       ┌──┐  ← Row N-1 of PEs
  │PE│→│PE│→│PE│→ ... →│PE│
  └──┘ └──┘ └──┘       └──┘
   ↓    ↓    ↓          ↓
  A₀   A₁   A₂   ...  A₁₂₇  (accumulator outputs)
```

- **Activations** (matrix A) flow from left to right
- **Weights** (matrix B) flow top to bottom — pre-loaded from VMEM weight FIFO
- **Partial sums** accumulate downward in each PE
- Each PE performs: `acc += a × w`

#### MXU Dimensions by Generation

| Generation | Array Size | MACs/cycle | Input dtype | Accum dtype |
|---|---|---|---|---|
| TPU v1 | 256 × 256 | 65,536 | INT8 | INT32 |
| TPU v2 | 128 × 128 | 16,384 | BF16 | FP32 |
| TPU v3 | 128 × 128 | 16,384 | BF16 | FP32 |
| TPU v4 | 128 × 128 | 16,384 | BF16 | FP32 |
| TPU v5p | 128 × 128 | 16,384 | BF16/FP8 | FP32 |
| TPU v6e | 256 × 256 | 65,536 | BF16/FP8 | FP32 |
| TPU7x | 256 × 256 | 65,536 | BF16/FP8 | FP32 |

Each TensorCore has **2 MXUs** in v4/v5p (×2 throughput), so the effective MAC rate per TensorCore is doubled.

#### MXU Interface

```
Inputs:
  activation_vreg[128]   — row of 128 BF16 values (from VMEM)
  weight_tile[128][128]  — weight matrix tile (pre-loaded from VMEM weight FIFO)
  accumulator_addr       — where to write 128×128 FP32 result tile

Outputs:
  result_tile[128][128]  — FP32 accumulated result, written to VMEM accumulator
```

#### Throughput and Latency

- **Throughput:** One 128×128 BF16 matrix tile per cycle (steady state)
- **Latency (systolic fill):** 128 cycles to fill the array with the first weight column
- **Startup latency:** ~200–256 cycles from dispatch to first output
- **Compute-bound efficiency:** >95% utilization on large GEMMs; drops sharply for small tiles (<32×32 inputs)

#### Memory Interface

The MXU reads weights from a dedicated **Weight FIFO** in VMEM. Activations are loaded directly from vector registers. Results are written to the **Accumulator buffer** in VMEM.

The Scalar Unit programs the MXU via control registers before dispatch. DMA transfers pre-stage weight tiles from HBM into VMEM while the MXU runs, enabling **double-buffering** (compute on tile N while loading tile N+1).

---

### 3.3 Vector Processing Unit (VPU)

#### Purpose
The VPU handles all element-wise and reduction operations that are not matrix multiplies. This includes:
- Activation functions (ReLU, GELU, SiLU, sigmoid, tanh)
- Element-wise arithmetic (add, multiply, divide, max, min)
- Reductions (sum, max across axes)
- Type conversions (BF16↔FP32↔FP8↔INT8)
- Normalization computations (softmax components, layer norm)
- Transcendentals (exp, log, sqrt — via polynomial approximation)

#### Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     VECTOR PROCESSING UNIT                        │
│                                                                    │
│   128 lanes — each lane operates on one element per cycle         │
│                                                                    │
│   Lane 0   Lane 1   Lane 2   ...   Lane 127                        │
│   ┌────┐   ┌────┐   ┌────┐         ┌────┐                        │
│   │ALU │   │ALU │   │ALU │   ...   │ALU │     ← element-wise     │
│   └──┬─┘   └──┬─┘   └──┬─┘         └──┬─┘                        │
│      │        │        │               │                           │
│   ┌──▼─┐   ┌──▼─┐   ┌──▼─┐         ┌──▼─┐                        │
│   │SREG│   │SREG│   │SREG│   ...   │SREG│     ← sublane regs     │
│   │×8  │   │×8  │   │×8  │         │×8  │      (8 per lane)      │
│   └────┘   └────┘   └────┘         └────┘                        │
│                                                                    │
│   Reduction tree:  128 → 64 → 32 → 16 → 8 → 4 → 2 → 1           │
│   (for sum/max reductions across the lane dimension)              │
│                                                                    │
│   XLU (Cross-Lane Unit):  ──────────────────────────────────┐    │
│   Butterfly permutation network across 128 lanes            │    │
│   └────────────────────────────────────────────────────────-┘    │
└──────────────────────────────────────────────────────────────────┘
```

#### Vector Register File (Vreg File)

This is the primary working memory for VPU and XLU operations.

```
Vreg structure:
  Sublane dimension:  8 rows  (for BF16, FP32, INT32)
                      16 rows (for INT8, FP8 — packed 2:1)
  Lane dimension:     128 columns

  Native tile shape (target_shape): (8, 128) = 1,024 elements for BF16
  Memory per vreg at BF16: 1,024 × 2 bytes = 2 KB
  Memory per vreg at FP32: 1,024 × 4 bytes = 4 KB
  Memory per vreg at INT8: 2,048 × 1 byte  = 2 KB (16×128 packed)

  Register file capacity:
  - ~128–256 architecturally visible vregs per TensorCore
  - Physical register file may be larger for renaming
```

#### Supported Data Types

| Type | Bits | Sublane packing | Elements per vreg |
|---|---|---|---|
| BF16 | 16 | 1 per sublane | 8 × 128 = 1,024 |
| FP16 | 16 | 1 per sublane | 8 × 128 = 1,024 |
| FP32 | 32 | 1 per sublane | 8 × 128 = 1,024 |
| INT32 | 32 | 1 per sublane | 8 × 128 = 1,024 |
| INT8 | 8 | 2 per sublane | 16 × 128 = 2,048 |
| FP8 | 8 | 2 per sublane | 16 × 128 = 2,048 |

#### VPU Throughput

- **Element-wise ops:** 1 vreg/cycle = 1,024 BF16 operations/cycle
- **Reduction (sum over 128 lanes):** 7-stage tree, ~7 cycles latency, 1 result/vreg/cycle throughput
- **Type conversion:** 1 vreg/cycle
- **Transcendentals (exp, log):** ~4–8 cycles via hardware polynomial unit

---

### 3.4 Cross-Lane Unit (XLU)

#### Purpose
The XLU is a sub-unit of the VPU that handles **lane-axis data permutation** — operations where element `i` in lane `j` needs to be routed to a different lane position. It is engaged for matrix transposes, gather operations, and lane rotations.

#### Trigger Condition (from Mosaic compiler)

```cpp
// XLU is used when the innermost (lane-axis) dimension is being permuted
bool uses_xlu = !permutation.empty() &&
                permutation.back() != permutation.size() - 1;
```

Transpositions of outer (untiled) dimensions do not require XLU — they are free loop reorderings.

#### Physical Implementation: 7-Stage Butterfly Network

```
128 lanes, 7 log₂(128) stages:

Stage 0 (stride 1):   [0↔1], [2↔3], ..., [126↔127]   — 64 switches
Stage 1 (stride 2):   [0↔2], [1↔3], ..., [124↔126]   — 64 switches
Stage 2 (stride 4):   ...                              — 64 switches
Stage 3 (stride 8):   ...                              — 64 switches
Stage 4 (stride 16):  ...                              — 64 switches
Stage 5 (stride 32):  ...                              — 64 switches
Stage 6 (stride 64):  [0↔64], [1↔65], ..., [63↔127]  — 64 switches

Total: 7 × 64 = 448 configurable 2:1 MUX switches
Routing fabric width: 448 × (element_width) bits
```

Each stage's 64 switches are controlled by a 64-bit control word loaded from control registers. For a **ROTATE** operation, all stages configure as a barrel-shifter pattern. For an arbitrary **PERMUTE**, the compiler pre-computes the 7 control words at compile time.

For **GATHER** (dynamic permutation driven by data), the index vector is fed into a control word generation unit that produces the 7 control words in a pipeline before the butterfly executes.

#### XLU Operation Latency

| Operation | Cycles |
|---|---|
| Lane ROTATE | 2–3 |
| Lane BROADCAST | 2 |
| Lane PERMUTE (static) | 4–5 |
| Lane GATHER (dynamic) | 5–7 |
| Full 2D TRANSPOSE (8×128 tile) | ~40 (8 sequential PERMUTE ops) |

### 3.5 On-Chip Vector Memory (VMEM)

#### Purpose
VMEM is the **local scratchpad SRAM** for each TensorCore. It serves as a high-bandwidth staging area between HBM and the execution units. Unlike a cache, VMEM is **explicitly managed** by the compiler — no hardware caching or eviction.

#### Structure

```
VMEM layout per TensorCore:
┌────────────────────────────────────────┐
│           VMEM (16–64 MB)              │
├────────────────┬───────────────────────┤
│ Weight Buffers │ Activation Buffers     │
│ (MXU weight    │ (MXU input/output,    │
│  staging)      │  VPU operands)        │
├────────────────┴───────────────────────┤
│ Accumulator Buffer                     │
│ (FP32 MXU output staging, before       │
│  VPU activation and writeback)        │
├───────────────────────────────────────┤
│ Scoped VMEM (tunable split)           │
│ ← used by XLU/VPU kernel tiling       │
│ → configurable via xla_tpu_scoped_    │
│   vmem_limit_kib flag (64 MB total)   │
└───────────────────────────────────────┘
```

#### Capacity by Generation

| Generation | VMEM per TensorCore |
|---|---|
| TPU v4 | 16 MB |
| TPU v5p | 32 MB |
| TPU v6e | ~32 MB |
| TPU7x | 64 MB |

#### Bandwidth

VMEM bandwidth to execution units must exceed HBM bandwidth to avoid becoming a bottleneck. For TPU7x:

- **VMEM → MXU:** Sufficient to feed the 256×256 systolic array at full rate; estimated **>10 TB/s** internal bandwidth
- **VMEM → VPU:** 128-wide reads each cycle; ~512 bytes/cycle per TensorCore
- **HBM → VMEM (DMA):** 7,380 GiB/s per chip (distributed across TensorCores)

---

### 3.6 TensorCore Execution Pipeline

The four units of a TensorCore operate largely **independently and in parallel**, synchronized only by explicit barrier instructions inserted by the compiler. This is a key source of efficiency:

```
Time →

Scalar:  [addr calc] [DMA setup] [loop ctrl]  [barrier]  [addr calc] ...
DMA:                 [HBM→VMEM tile load ──────────────]  [next load...]
MXU:                              [matmul on tile N ──] [tile N+1 ──]
VPU:                                               [activation] [norm] ...
XLU:                                                     [transpose]
                                   ↑ overlap ↑
```

The compiler (Mosaic) must explicitly schedule the inter-unit dependencies using this model. The programmer (via JAX/Pallas) writes high-level code; Mosaic lowers it to the VLIW packet stream.

---

## 4. SparseCore Architecture

### 4.1 Purpose

SparseCores are **dedicated accelerators for sparse lookup and embedding operations**, primarily used in recommendation models. They handle the high memory-bandwidth, low-compute embedding lookup pattern that would otherwise bottleneck the MXU.

### 4.2 Block Diagram

```
┌────────────────────────────────────────────────────────────────────┐
│                          SPARSECORE                                 │
│                                                                      │
│  ┌──────────────┐   ┌───────────────┐   ┌──────────────────────┐  │
│  │  Index FIFO  │   │ Embedding     │   │  Reduction / Pool    │  │
│  │  (sparse     │──▶│ Lookup Engine │──▶│  (sum/mean pooling   │  │
│  │   indices)   │   │ (random HBM   │   │   across looked-up   │  │
│  └──────────────┘   │  access)      │   │   embeddings)        │  │
│                     └───────────────┘   └──────────────────────┘  │
│                                                  │                  │
│  ┌──────────────────────────────────────────────▼──────────────┐  │
│  │              SparseCore ↔ TensorCore ICI                     │  │
│  │           (collected results → VPU for downstream ops)       │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 4.3 Operation Model

SparseCores execute **embedding bag** operations:
1. Accept a batch of sparse integer indices (e.g., user feature IDs)
2. Look up corresponding embedding vectors from HBM (random access pattern)
3. Pool (sum or mean) embedding vectors within each bag
4. Return dense output tensor to the TensorCore VPU for further processing

### 4.4 Counts and Integration

| Generation | SparseCores per chip |
|---|---|
| TPU v4 | 4 |
| TPU v5p | 4 |
| TPU7x (Ironwood) | 8 (4 per chiplet × 2) |

SparseCores are programmed via **XLA flags** (`xla_tpu_sparse_*`) and are transparent to JAX-level user code. The `SparseCore Collective Offloading` feature in TPU7x allows SparseCores to act as **independent communication threads**, overlapping All-Gather/Reduce-Scatter collectives with TensorCore compute — a key ICI latency-hiding mechanism.

---

## 5. Memory Subsystem

### 5.1 Memory Hierarchy

```
                    Latency         Bandwidth        Capacity
                    -------         ---------        --------
Scalar Regs          1 cycle         unlimited        32 × 32b
Vreg File            1 cycle         ~TB/s            ~256 vregs (~512 KB BF16)
VMEM (SRAM)          3–10 cycles     >10 TB/s/chip    16–64 MB/TC
   ↕ DMA
HBM (off-chip)       ~200 cycles     1–7 TB/s/chip    16–192 GB/chip
   ↕ PCIe
Host DDR (CPU RAM)   ~500 cycles     ~100 GB/s        100s of GB
   ↕ ICI
Remote HBM           ~1–5 μs         1–9 TB/s/chip    1000s of GB
```

### 5.2 HBM Interface

High Bandwidth Memory (HBM) is a **3D-stacked DRAM** package mounted adjacent to (or on the same interposer as) the TPU die.

#### Physical Interface

```
TPU Die ────── HBM PHY ────── Silicon Interposer ────── HBM Stack
                              (2.5D packaging)          (stacked DRAM dies)
```

#### Specifications by Generation

| Gen | HBM Type | Capacity | Bandwidth per chip |
|---|---|---|---|
| v1 | DDR3 (host) | 8 GB | 34 GB/s (PCIe constrained) |
| v2 | HBM2 | 16 GB | 600 GB/s |
| v3 | HBM2 | 32 GB | 900 GB/s |
| v4 | HBM2e | 32 GB | 1,200 GB/s |
| v5e | HBM2e | 16 GB | 819 GB/s |
| v5p | HBM3 | 95 GB | 2,765 GB/s |
| v6e | HBM3 | 32 GB | 1,638 GiB/s |
| TPU7x | HBM3e | 192 GB | 7,380 GiB/s |

#### HBM Controller

- Handles burst scheduling, bank interleaving, and refresh
- Multiple independent HBM channels (typically 4–8 stacks)
- Each TensorCore has an **HBM slice** allocated by the memory controller; capacity is evenly distributed (e.g., 8 GB per TensorCore for v4 with 4 TCs and 32 GB total)

### 5.3 DMA Engine

Each TensorCore has a dedicated **DMA (Direct Memory Access) engine** that moves data between HBM and VMEM independently of the Scalar Unit.

- Supports **strided transfers**: can load a non-contiguous slice of HBM into contiguous VMEM
- Supports **1D, 2D, and 3D strides** (essential for loading tensor tiles with padding)
- Overlap: DMA transfers can run **concurrently** with MXU/VPU execution on the previous tile
- The Scalar Unit programs the DMA via control registers and then issues a `wait` instruction when VMEM data is needed

---

## 6. System-on-Chip (SoC) Architecture

### 6.1 Multi-TensorCore Integration

Multiple TensorCores on a single chip share:
- **HBM** (via the HBM controller and memory fabric)
- **ICI interface** (a single chip-level ICI block, shared)
- **PCIe host interface** (one PCIe connection per chip to the host server)
- **On-chip interconnect (NoC):** a mesh or ring connecting all TensorCores and SparseCores

TensorCores do **not** share VMEM — each has private scratchpad. Shared state lives in HBM.

```
On-chip NOC (Network-on-Chip):

  TC0 ──── TC1
   │        │
  SC0      SC1      ← SparseCores on same ring/mesh
   │        │
  TC2 ──── TC3
   │
 ┌──────────────────────────────────┐
 │   HBM Controller (shared)        │
 │   ICI Fabric Interface (shared)  │
 │   PCIe PHY (shared)              │
 └──────────────────────────────────┘
```

### 6.2 Dual-Chiplet Architecture (TPU7x Ironwood)

TPU7x departs from the single-die model. Two **chiplets** are integrated on a single package, connected via a **Die-to-Die (D2D) interface**.

```
Physical Chip Package (TPU7x):

┌──────────────────────────────────────────────────────────────┐
│  Chiplet A (4 TC + 4 SC + 96 GB HBM)                        │
│  ────────────────────────────────────────────────────────    │
│                         ↕↕↕ D2D (×6 faster than 1D ICI) ↕↕↕│
│  ────────────────────────────────────────────────────────    │
│  Chiplet B (4 TC + 4 SC + 96 GB HBM)                        │
└──────────────────────────────────────────────────────────────┘
```

#### D2D Interface Properties
- Bandwidth: **6× faster than a single 1D ICI link**
- Latency: lower than inter-chip ICI (on-package)
- Protocol: custom high-density die-to-die signaling (not PCIe, not ICI)
- Purpose: enables intra-chip collective operations to span both chiplets

#### Framework View
JAX/XLA expose each Ironwood chip as **two separate "devices"** — one per chiplet. This allows existing single-chip sharding strategies to scale to both chiplets with minimal code changes. The XLA compiler adds a 4th topology dimension for chiplet selection.

#### Memory Addressing
Each chiplet has **its own 96 GB HBM**, giving 192 GB per chip total. Memory is **not unified** across chiplets — cross-chiplet access goes through the D2D interface. Compiler-driven data placement is critical for performance.

---

### 6.3 Inter-Chip Interconnect (ICI)

The ICI is a **custom, high-speed chip-to-chip network fabric** that connects multiple TPU chips into a slice or pod. It is the key enabler of multi-chip tensor parallelism.

#### Physical Characteristics

| Generation | Links per chip | Bandwidth per link | Total ICI BW per chip |
|---|---|---|---|
| v2 | 4 links | ~8 Gbps | ~32 Gbps (32 GB/s) |
| v3 | 4 links | ~164 Gbps | ~656 Gbps (82 GB/s) |
| v4 | 6 links | ~100 Gbps | ~600 Gbps (75 GB/s) |
| v5p | 6 links | ~800 Gbps | ~4,800 Gbps (600 GB/s) |
| v6e | 4 links | ~800 Gbps | ~3,200 Gbps (400 GB/s) |
| TPU7x | 6 links | ~1,200 Gbps | ~7,200 Gbps (~900 GB/s bidir) |

#### Topology

**Within a slice (pod):** chips are connected in a 3D torus (v4, v5p, v7x) or 2D torus (v5e):

```
3D Torus (v4/v5p/v7x):
Each chip connects to 2 neighbors along each of X, Y, Z axes:
  chip(x,y,z) connects to:
    (x±1, y, z)  — 2 links (X dimension)
    (x, y±1, z)  — 2 links (Y dimension)
    (x, y, z±1)  — 2 links (Z dimension)
  Total: 6 links per chip ✓

Wrap-around links at cube boundaries form the torus.
```

ICI uses **direct copper links** within a cube (4×4×4 = 64-chip unit); optical links via OCS between cubes.

#### Optical Circuit Switching (OCS) — v4 and later

Between cubes, **Optical Circuit Switches** allow the inter-cube topology to be dynamically reconfigured. This enables:
- Fault isolation (route around failed OCS ports)
- Multi-tenant pod partitioning (reconfigure topology for different customer shapes)
- Non-3D-torus topologies (e.g., butterfly, fat-tree) for specific workloads

OCS reconfiguration time is ~100ms (milliseconds) — sufficient for batch-to-batch reshaping but not within a training step.

#### ICI Resiliency

For slices ≥ 1 cube (≥64 chips for v5p, ≥64 chips for v7x), ICI resiliency is **enabled by default**. This allows ICI traffic to be rerouted around failed optical links at the cost of ~temporary ICI bandwidth degradation.

#### Data Center Network (DCN)

For **multi-slice training** (across pod boundaries), the standard data center Ethernet network (DCN) carries gradient synchronization traffic. The `Multislice` feature in JAX/XLA uses DCN for the slower inter-slice collectives.

```
Topology hierarchy:
  TensorCores within chip ──── on-chip NOC ──────────────── ~ns latency
  Chips within cube ──────────── ICI copper ─────────────── ~μs latency
  Cubes within pod ──────────── ICI optical (via OCS) ───── ~μs latency
  Pods within datacenter ─────── DCN Ethernet ──────────── ~10–100 μs latency
```

---

### 6.4 PCIe Host Interface

Each TPU chip connects to a host server via **PCIe**:

| Generation | PCIe Gen | Bandwidth (device→host) |
|---|---|---|
| v1 | Gen3 ×16 | ~16 GB/s |
| v2–v4 | Gen3 ×16 | ~16 GB/s |
| v5e | Gen4 ×16 | ~32 GB/s |
| v5p | Gen4/5 | ~32–64 GB/s |
| TPU7x | Gen5 ×16 | ~64 GB/s |

PCIe is used for:
- **Program loading:** XLA compiled binaries loaded from host to HBM at job start
- **Input data streaming:** inference inputs, initial training data
- **Result readback:** final outputs, gradients for host-side optimizer steps
- **Control plane:** job scheduling, health monitoring, error handling

PCIe is **not** in the critical path of step-to-step training (gradients exchange via ICI, not PCIe).

---

### 6.5 Clock Architecture

#### Clock Domains

```
┌─────────────────────────────────────────┐
│ Core clock domain (MXU, VPU, XLU, VMEM)│  ~700 MHz – 1.05 GHz
├─────────────────────────────────────────┤
│ HBM PHY clock domain                    │  ~2 GHz (tied to HBM spec)
├─────────────────────────────────────────┤
│ ICI SerDes clock domain                 │  ~25–50 GHz (high-speed serial)
├─────────────────────────────────────────┤
│ PCIe clock domain                       │  PCIe reference clock
└─────────────────────────────────────────┘
```

#### Core Clocks by Generation

| Generation | Core clock (est.) | Note |
|---|---|---|
| TPU v1 | 700 MHz | Disclosed |
| TPU v2 | 700 MHz | Estimated |
| TPU v3 | 940 MHz | Disclosed |
| TPU v4 | ~1.05 GHz | Estimated from FLOPS / MACs |
| TPU v5p | ~1.1 GHz | Estimated |
| TPU7x | ~1.0–1.2 GHz | Estimated |

The TPU runs at a **lower, more stable clock** than GPUs. This is intentional: deterministic latency is more important than peak frequency for the systolic array pipeline.

#### Clock Synchronization

TensorCores within a chip share a common core clock. Across chips, ICI uses **source-synchronous clocking** (embedded clock in SerDes) — there is no global synchronization across the pod.

---

### 6.6 Power Domains

```
┌──────────────────────────────────────────────────────┐
│             CHIP-LEVEL POWER DOMAINS                  │
│                                                        │
│  ┌────────────────────────────────────────────────┐   │
│  │  Compute domain (MXU + VPU + XLU)             │   │
│  │  (highest power, power-gated when idle)        │   │
│  └────────────────────────────────────────────────┘   │
│  ┌──────────────┐  ┌──────────────────────────────┐   │
│  │ Scalar domain │  │  VMEM domain                 │   │
│  │ (always on)  │  │  (SRAM, state retained)       │   │
│  └──────────────┘  └──────────────────────────────┘   │
│  ┌────────────────────────────────────────────────┐   │
│  │  Memory domain (HBM PHY)                       │   │
│  │  (separately power-managed)                    │   │
│  └────────────────────────────────────────────────┘   │
│  ┌────────────────────────────────────────────────┐   │
│  │  ICI domain (SerDes always on for heartbeat)   │   │
│  └────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────┘
```

#### TDP by Generation

| Generation | TDP per chip (est.) |
|---|---|
| TPU v1 | ~75 W |
| TPU v2 | ~280 W |
| TPU v3 | ~450 W |
| TPU v4 | ~170–200 W |
| TPU v5p | ~300–400 W |
| TPU7x | ~300–500 W (estimated) |

TPU7x uses **liquid cooling** for the chip package, enabling higher sustained TDP vs. air-cooled generations.

---

## 7. Generation Comparison Matrix

### 7.1 Chip-Level Compute

| Spec | v1 | v2 | v3 | v4 | v5e | v5p | v6e | TPU7x |
|---|---|---|---|---|---|---|---|---|
| TensorCores/chip | 1 | 2 | 2 | 4 | 4 | 4 | 4 | 8 |
| MXU/TC | 1 | 1 | 1 | 2 | 2 | 2 | 1 | 1 |
| MXU size | 256² | 128² | 128² | 128² | 128² | 128² | 256² | 256² |
| MACs/cycle/chip | 65,536 | 32,768 | 32,768 | 131,072 | 131,072 | 131,072 | 262,144 | 524,288 |
| BF16 TFLOPS/chip | — | ~45 | ~420 | 275 | 197 | 459 | 918 | 2,307 |
| FP8 TFLOPS/chip | — | — | — | — | 393 | 459 | 918 | 4,614 |
| VMEM/TC | 28 MB* | ~16 MB | ~16 MB | 16 MB | ~16 MB | 32 MB | ~32 MB | 64 MB |

*v1 "unified buffer" = activation buffer + weight FIFO

### 7.2 Memory and Interconnect

| Spec | v1 | v2 | v3 | v4 | v5e | v5p | v6e | TPU7x |
|---|---|---|---|---|---|---|---|---|
| HBM/chip | 8 GB DDR3 | 16 GB | 32 GB | 32 GB | 16 GB | 95 GB | 32 GB | 192 GB |
| HBM BW/chip | 34 GB/s | 600 GB/s | 900 GB/s | 1,200 GB/s | 819 GB/s | 2,765 GB/s | 1,638 GB/s | 7,380 GB/s |
| ICI BW/chip | — | ~32 Gbps | ~656 Gbps | ~600 Gbps | ~1,600 Gbps | 4,800 Gbps | ~3,200 Gbps | ~7,200 Gbps |
| ICI topology | — | 2D torus | 3D torus | 3D torus | 2D torus | 3D torus | 2D torus | 3D torus |
| Max pod chips | — | 512 | 1,024 | 4,096 | 256 | 8,960 | 4,096* | 9,216 |

### 7.3 SoC Features

| Feature | v1 | v2 | v3 | v4 | v5e | v5p | v6e | TPU7x |
|---|---|---|---|---|---|---|---|---|
| SparseCore | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ (4) | ✓ (2) | ✓ (8) |
| Optical ICI (OCS) | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |
| Dual-chiplet | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| FP8 native | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |
| XLU 8-bit support | N/A | ✓ | ✓ | ✗* | ✗* | ✗* | ✗* | ✗* |
| Liquid cooling | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ | ✓ | ✓ |

*8-bit XLU transpose limited from v4 onward — emulated via 16-bit widening

---

## 8. Execution Model and Programming Interface

### 8.1 Compilation Stack

```
User Python (JAX / PyTorch / TF)
         │
         ▼
   XLA (Accelerated Linear Algebra compiler)
   — Graph-level optimization, fusion, layout assignment
         │
         ▼
   Mosaic TPU (MLIR-based low-level compiler)
   — Tile sizing, VMEM allocation, DMA scheduling
   — VLIW packet generation (Scalar/VPU/MXU dispatch)
   — XLU control word pre-computation
         │
         ▼
   TPU binary (.tpubin)
   — Loaded via PCIe into HBM at job start
   — Executed autonomously; host CPU not in critical path
```

### 8.2 Execution Model

Once the binary is loaded, the TPU runs **autonomously**:

1. **Scalar Unit** fetches and decodes instructions
2. For matrix ops: Scalar programs DMA (HBM→VMEM), then dispatches MXU
3. For vector ops: Scalar dispatches VPU/XLU
4. **DMA** runs concurrently with MXU/VPU (double-buffering)
5. **MXU** runs at full throughput when fed properly
6. **ICI collectives** (All-Reduce, All-Gather) are interleaved with compute, overlapped via SparseCore Collective Offloading on v7x

The **host CPU** is not in the critical path during step execution — it issues the "step" command and waits for completion while the TPU runs independently.

### 8.3 VLIW Packet Format

A single TPU instruction packet contains slots for simultaneous dispatch to:

```
┌──────────────────────────────────────────────────────────┐
│                  VLIW Instruction Packet                   │
├────────────┬───────────────┬──────────────┬──────────────┤
│ Scalar op  │  VPU/XLU op  │   DMA op     │  MXU op      │
│ (control,  │  (element-   │  (HBM↔VMEM  │  (128×128    │
│  branch)   │   wise/perm) │   transfer)  │  matmul)     │
└────────────┴───────────────┴──────────────┴──────────────┘
        ↑ all four slots can be non-NOP simultaneously ↑
```

The Mosaic compiler manages slot scheduling to maximize packet utilization.

---

## 9. Known Design Constraints and Pitfalls

### 9.1 MXU Under-Utilization on Small Tiles
MXU efficiency drops sharply for tiles smaller than ~32×32. The systolic array takes 128 fill cycles before producing output. Workloads with small batch sizes (batch size < 128) will see <50% MXU utilization.

### 9.2 HBM Bandwidth vs. FLOPS Asymmetry
For memory-bound ops (element-wise, activations), the bottleneck is HBM bandwidth, not FLOPS. Example for TPU v5p:
- BF16 FLOPS: 459 TFLOPS/chip
- HBM bandwidth: 2,765 GB/s = 1.38 T BF16 elements/s
- Break-even arithmetic intensity: 459/1.38 ≈ **333 FLOPs per element loaded**
- Standard transformer: ~100–300 FLOPs/element → often memory-bound

### 9.3 VMEM Capacity Bottleneck
For long-sequence models, activation tensors grow quadratically with sequence length. VMEM may not hold an entire forward pass. Solutions: rematerialization (gradient checkpointing), context parallelism, FP8 to halve activation size.

### 9.4 ICI Latency vs. Compute Overlap
All-Reduce and All-Gather collectives over ICI add latency proportional to payload size. The compiler must schedule compute to overlap with ICI. Poorly scheduled sharding strategies can become ICI-latency-bound even with high BW.

### 9.5 XLU 8-bit Limitation (All gens > v3)
See §3.4 — FP8/INT8 transposes via XLU require software widening to 16-bit. This has a ~2× overhead on transpose throughput for 8-bit types. Bug b/448848595 remains open through TPU7x.

### 9.6 Dual-Chiplet Memory Locality (TPU7x)
On Ironwood, accessing memory from the other chiplet goes through the D2D interface. Tensors placed on chiplet A that need to be read by chiplet B incur D2D bandwidth. The compiler's placement decisions must account for chiplet locality.

### 9.7 No Hardware Cache Coherency
There is no coherency protocol between TensorCores. All data sharing goes through HBM. This avoids cache-coherency overhead but requires explicit DMA management.

---

## 10. References

1. Google Cloud TPU: *System Architecture*, docs.cloud.google.com/tpu/docs/system-architecture-tpu-vm
2. Google Cloud TPU: *TPU7x (Ironwood)*, docs.cloud.google.com/tpu/docs/tpu7x
3. Google Cloud TPU: *TPU v5p*, docs.cloud.google.com/tpu/docs/v5p
4. Google Cloud TPU: *Ironwood Performance Optimizations*, docs.cloud.google.com/tpu/docs/ironwood-performance
5. Jouppi et al., **"TPU v4: An Optically Reconfigurable Supercomputer for Machine Learning with Hardware Support for Embeddings"**, ISCA 2023. arXiv:2304.01433
6. Jouppi et al., **"In-Datacenter Performance Analysis of a Tensor Processing Unit"**, ISCA 2017. arXiv:1704.04760
7. JAX/Mosaic TPU Compiler: `jax-ml/jax/jaxlib/mosaic/dialect/tpu/` (GitHub)
8. JAX PR #32290: *[Mosaic TPU] Only canonicalize the dtype of transposes if they use the XLU*. jax-ml/jax@2fbe731
9. JAX Pallas TPU documentation: docs.jax.dev/en/latest/pallas/tpu/
10. Wikipedia: *Tensor Processing Unit*, en.wikipedia.org/wiki/Tensor_Processing_Unit
11. Intel Architecture Instruction Reference: VPERMPS, VPSHUFB (cross-lane comparison)
12. NVIDIA PTX ISA: shfl.sync, vote.sync (cross-lane comparison)

---

*Generated: 2026-04-08 | Engineering Reference Draft v1.0*  
*Status: Derived from public documentation and compiler source analysis. Not an official Google document.*  
*Companion document: XLU_design_spec.md (Cross-Lane Unit detailed spec)*
