# TinyTPU

A TPU prototype implemented in Bluespec SystemVerilog (BSV), with a working tinygrad co-simulation backend.

## Architecture

```
TinyTPUChip
├── TensorCore (TC0)
│   ├── ScalarUnit (SXU) — microprogram sequencer
│   ├── SystolicArray (MXU) — 4×4 matrix multiply
│   ├── Controller — drives MXU through WeightSRAM/ActivationSRAM
│   ├── VPU — element-wise ops (ADD, MUL, RELU, MAX, SUM_REDUCE)
│   ├── XLU — lane permutations (ROTATE, BROADCAST, PERMUTE, TRANSPOSE)
│   ├── VMEM — unified on-chip scratchpad (16 tiles × 4×4 × Int#(32))
│   └── VRegFile — 16 vector registers
├── SparseCore — embedding table lookup with sum-pooling
├── HBMModel — behavioral HBM with configurable read latency
└── ChipNoC — ring network-on-chip connecting all units
```

The SXU executes a fixed instruction set: `LOAD_VREG`, `STORE_VREG`, `DISPATCH_VPU`, `DISPATCH_MXU`, `WAIT_MXU`, `HALT`. This replaces the old FSM-based `mkController` with a fully programmable sequencer.

## Prerequisites

- [BSC compiler](https://github.com/B-Lang-org/bsc) (Bluesim mode)
- GNU Make
- Python ≥ 3.11 (for co-simulation)

## Build & Test

```sh
# Run all BSV hardware tests
make test

# Build the co-simulation runtime testbench
make runtime-tb

# Run the tinygrad co-simulation end-to-end test
python3 scripts/test_cosim.py

# Run individual test suites
make test-pe        # Processing element
make test-array     # Systolic array
make test-accel     # Original TensorAccelerator (legacy)
make test-4x4       # 4×4 parameterization
make test-xlu       # Cross-Lane Unit (ROTATE/BROADCAST/PERMUTE/TRANSPOSE)
make test-vmem      # Vector memory scratchpad
make test-vregfile  # Vector register file
make test-vpu       # Vector processing unit (5 ops)
make test-sxu       # Scalar unit microprogram
make test-tc        # TensorCore end-to-end GEMM
make test-sc        # SparseCore embedding lookup
make test-hbm       # HBM behavioral model
make test-noc       # On-chip ring NOC
make test-chip      # Full chip pipeline: TC0 GEMM -> SparseCore lookup

# Clean build artifacts
make clean
```

## Source Files

### Hardware (BSV)

| File | Description |
|---|---|
| `src/PE.bsv` | Systolic array processing element |
| `src/SystolicArray.bsv` | Parameterized systolic array `mkSystolicArray#(rows, cols)` |
| `src/WeightSRAM.bsv` | Weight tile SRAM for MXU |
| `src/ActivationSRAM.bsv` | Activation vector SRAM for MXU |
| `src/Controller.bsv` | MXU FSM controller (load weights, stream activations, drain) |
| `src/TensorAccelerator.bsv` | Legacy top-level integrating MXU+SRAMs+Controller |
| `src/XLU.bsv` | Cross-Lane Unit: barrel-shift rotate, butterfly permute, broadcast, transpose |
| `src/VMEM.bsv` | Unified scratchpad SRAM with 1-cycle read latency |
| `src/VRegFile.bsv` | Vector register file (combinatorial read, registered write) |
| `src/VPU.bsv` | Vector processing unit: ADD, MUL, RELU, MAX, SUM_REDUCE |
| `src/ScalarUnit.bsv` | Microprogram sequencer driving VMEM/VRF/VPU/XLU/MXU |
| `src/TensorCore.bsv` | Top-level integration of SXU + MXU + VPU + XLU + VMEM + VRegFile |
| `src/SparseCore.bsv` | Embedding lookup accelerator with sequential sum-pooling |
| `src/HBMModel.bsv` | Behavioral HBM model: pipelined reads with configurable latency |
| `src/ChipNoC.bsv` | Token-ring NOC: per-node inbox FIFOs, ring forwarding FIFO |
| `src/TinyTPUChip.bsv` | Chip top-level: TC0 + SparseCore + HBMModel + ChipNoC |

### Co-simulation

| File | Description |
|---|---|
| `test/TbTinyTPURuntime.bsv` | Generic BSV testbench: reads numeric text bundle, runs TensorCore, prints results |
| `bdpi/tinytpu_io.c` | BDPI C helper: `tinytpu_bundle_open` / `tinytpu_bundle_read_int` (reads `$TINYTPU_BUNDLE`) |
| `tinygrad/tinygrad/runtime/ops_tinytpu.py` | tinygrad `TINYTPU` device: allocator, UOp→SXU renderer, program driver |
| `scripts/test_cosim.py` | End-to-end co-simulation test: identity/scale/permutation GEMMs vs numpy |

## Tile Format

The fundamental data unit is a **vreg tile**: `Vector#(sublanes, Vector#(lanes, Int#(32)))`. For the 4×4 prototype: 4 sublanes × 4 lanes × 32-bit signed integer = 64 bytes per tile.

## SXU Instruction Set

| Opcode | Fields | Description |
|---|---|---|
| `SXU_LOAD_VREG` | `vmemAddr`, `vregDst` | VMEM[vmemAddr] → VRF[vregDst] |
| `SXU_STORE_VREG` | `vmemAddr`, `vregSrc` | VRF[vregSrc] → VMEM[vmemAddr] |
| `SXU_DISPATCH_VPU` | `vpuOp`, `vregSrc`, `vregSrc2`, `vregDst` | Element-wise VPU op, result to VRF |
| `SXU_DISPATCH_MXU` | `mxuWBase`, `mxuABase`, `mxuTLen` | Start MXU matrix multiply |
| `SXU_WAIT_MXU` | — | Stall until MXU done |
| `SXU_HALT` | — | Stop execution |

## Software Stack

The `tinygrad/` submodule (forked at [hanw/tinygrad](https://github.com/hanw/tinygrad)) provides the compiler backend. Registering `device="TINYTPU"` with tinygrad routes tensor operations through:

```
User Python (tinygrad Tensor ops)
        │
        ▼
TinyTPURenderer          ← detects 4×4 GEMM in UOps, emits JSON descriptor
        │
        ▼
TinyTPUProgram           ← builds numeric text bundle from buffer data,
        │                   invokes BSV simulator via subprocess,
        │                   copies Int32 results back to output buffer
        ▼
TbTinyTPURuntime.bexe    ← BSV sim reads bundle (via BDPI C helper),
                            loads TensorCore#(4,4,16), prints results
```

The co-simulation protocol (bundle format, result format) is documented in `doc/software-spec.md`.

**Quick start:**

```python
from tinygrad import Tensor

a = Tensor([[1, 2, 3, 4]], dtype='int32', device='TINYTPU')
w = Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], dtype='int32', device='TINYTPU')
print((a @ w).numpy())  # [[1 2 3 4]] — runs on BSV TensorCore simulation
```

Currently supports 4×4 GEMM. Other ops raise `NotImplementedError`.

## Documentation

| File | Description |
|---|---|
| `doc/TPU_chip_architecture_spec.md` | Full chip architecture specification |
| `doc/XLU_design_spec.md` | XLU design specification |
| `doc/software-spec.md` | tinygrad co-simulation software stack spec |
| `doc/plan-*.md` | TDD implementation plans for each hardware unit |
