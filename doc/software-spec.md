# TinyTPU Software Stack — Specification

**Document status:** Draft v0.1  
**Scope:** tinygrad backend for TinyTPU BSV simulation — from user tensor ops to hardware execution  
**Based on:** tinygrad device model (`tinygrad/runtime/ops_*.py`), TinyTPU hardware plans

---

## 1. Overview

The TinyTPU software stack connects tinygrad (the ML framework) to the TinyTPU BSV hardware simulation. A user writes standard tinygrad Python code; the stack compiles it to a TinyTPU SXU microprogram, drives the BSV simulator, and returns results as numpy arrays.

```
User Python (tinygrad tensors)
        │
        ▼
tinygrad IR (UOps)
        │
        ▼
TinyTPURenderer          ← translates UOps → SXU instruction list
        │
        ▼
TinyTPUProgram           ← serializes program + input data → .json bundle
        │                   invokes BSV simulator, reads output
        ▼
BSV TensorCore simulator ← generic testbench reads bundle, runs, writes results
        │
        ▼
TinyTPUAllocator         ← VMEM address management, copyin/copyout via bundle
```

---

## 2. Files

| File | Language | Responsibility |
|---|---|---|
| `tinygrad/tinygrad/runtime/ops_tinytpu.py` | Python | Device class, Allocator, Program, Renderer |
| `tinygrad/tinygrad/renderer/tinytpu.py` | Python | UOp → SXU instruction translation |
| `test/TbTinyTPURuntime.bsv` | BSV | Generic BSV testbench: reads bundle, runs TensorCore, writes results |
| `Makefile` | Make | `runtime-tb` target: build TbTinyTPURuntime executable |
| `scripts/run_tinytpu.py` | Python | CLI: invoke BSV simulator, parse results (used by Program.__call__) |

---

## 3. Execution Protocol (Co-Simulation)

### 3.1 Bundle Format

`TinyTPUProgram.__call__` writes a JSON **execution bundle** before invoking the simulator:

```json
{
  "version": 1,
  "sublanes": 4,
  "lanes": 4,
  "vmem": [
    { "addr": 0, "tile": [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]] },
    { "addr": 1, "tile": [[10,0,0,0],[0,10,0,0],[0,0,10,0],[0,0,0,10]] }
  ],
  "weight_tiles": [
    { "addr": 0, "tile": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] }
  ],
  "activation_tiles": [
    { "addr": 1, "data": [1, 2, 3, 4] }
  ],
  "program": [
    { "op": "SXU_DISPATCH_MXU", "mxuWBase": 0, "mxuABase": 1, "mxuTLen": 1 },
    { "op": "SXU_WAIT_MXU" },
    { "op": "SXU_HALT" }
  ],
  "output_vmem_addrs": [],
  "output_mxu_result": true
}
```

Fields:
- `vmem` — initial VMEM tile contents (written before program runs)
- `weight_tiles` — preloaded via `tc.loadWeightTile(addr, data)`
- `activation_tiles` — preloaded via `tc.loadActivationTile(addr, data)`
- `program` — SXU instruction sequence (decoded by TbTinyTPURuntime)
- `output_vmem_addrs` — VMEM addresses to read back after HALT
- `output_mxu_result` — if true, read back `getMxuResult` after HALT

### 3.2 Result Format

The BSV simulator writes a JSON **result file** to stdout (or a named file):

```json
{
  "mxu_result": [11, 22, 33, 44],
  "vmem_output": [
    { "addr": 2, "tile": [[11,22,33,44],[0,0,0,0],[0,0,0,0],[0,0,0,0]] }
  ],
  "cycles": 15,
  "status": "ok"
}
```

### 3.3 BSV Generic Testbench (`TbTinyTPURuntime.bsv`)

A single BSV testbench that:
1. Reads bundle path from `$test$plusargs("bundle=%s", path)`
2. Uses `$fopen` / `$fscanf` to parse the bundle (simple line-based text format, not JSON — see §3.4)
3. Preloads weight tiles and activation tiles into TensorCore
4. Loads the SXU program
5. Calls `tc.start(len)` and polls `tc.isDone`
6. Writes results to stdout as key=value lines

### 3.4 Text Bundle Format (BSV-parseable)

Since BSV `$fscanf` cannot parse JSON, the Python side generates a **text bundle** alongside the JSON:

```
# TinyTPU text bundle v1
SUBLANES 4
LANES 4
WEIGHT_TILE 0  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1
ACT_TILE    1  1 2 3 4
INSTR SXU_DISPATCH_MXU 0 1 1
INSTR SXU_WAIT_MXU 0 0 0
INSTR SXU_HALT 0 0 0
OUTPUT_MXU 1
END
```

The BSV testbench reads tokens line by line with `$fscanf`.

---

## 4. tinygrad Device Implementation

### 4.1 `TinytpuAllocator`

```python
class TinytpuAllocator(LRUAllocator):
    # VMEM is managed as a flat array of tile addresses
    # Each alloc returns a (base_addr, num_tiles) handle
    def _alloc(self, size:int, options:BufferSpec) -> tuple[int,int]:
        num_tiles = (size + TILE_BYTES - 1) // TILE_BYTES
        base = self._bump_ptr
        self._bump_ptr += num_tiles
        return (base, num_tiles)

    def _copyin(self, dest:tuple, src:memoryview): ...  # writes to bundle staging
    def _copyout(self, dest:memoryview, src:tuple): ...  # reads from last result
    def _free(self, opaque, options): pass  # no-op for simulation
```

`TILE_BYTES = sublanes × lanes × 4` (4 bytes per Int#(32) element).

### 4.2 `TinyTPURenderer`

Subclasses `Renderer`. Translates a UOp list to an `SxuProgram` (list of SXU instruction dicts).

**UOp → SXU mapping:**

| tinygrad UOp | SXU Instruction | Notes |
|---|---|---|
| `LOAD(buf, idx)` | `SXU_LOAD_VREG vmemAddr vregDst` | idx selects VMEM address |
| `STORE(buf, idx, val)` | `SXU_STORE_VREG vmemAddr vregSrc` | stores vreg to VMEM |
| `REDUCE(MUL+ADD, axes)` | `SXU_DISPATCH_MXU` + `SXU_WAIT_MXU` | matrix multiply (GEMM) |
| `ADD(a, b)` | `SXU_DISPATCH_VPU VPU_ADD vsrc1 vsrc2 vdst` | element-wise add |
| `MUL(a, b)` | `SXU_DISPATCH_VPU VPU_MUL vsrc1 vsrc2 vdst` | element-wise multiply |
| `MAX(a, 0)` | `SXU_DISPATCH_VPU VPU_RELU vsrc _ vdst` | ReLU activation |
| `MAX(a, b)` | `SXU_DISPATCH_VPU VPU_MAX vsrc1 vsrc2 vdst` | element-wise max |
| `REDUCE(ADD, lane_axis)` | `SXU_DISPATCH_VPU VPU_SUM_REDUCE vsrc _ vdst` | lane reduction |
| `PERMUTE / TRANSPOSE` | `SXU_DISPATCH_XLU ...` | routes through XLU |

**VReg allocation:** The renderer uses a simple linear scan over vreg indices (0–15). Each UOp result is assigned a vreg; registers are freed when the value is consumed.

### 4.3 `TinyTPUProgram`

```python
class TinyTPUProgram:
    def __init__(self, device, name:str, lib:bytes):
        self.program = json.loads(lib)  # lib = JSON-encoded SXU program

    def __call__(self, *bufs, global_size, local_size, vals, wait=False):
        bundle = build_bundle(self.program, bufs)
        write_text_bundle(bundle, "/tmp/tinytpu_bundle.txt")
        result = subprocess.run(
            [TINYTPU_SIM_PATH, "+bundle=/tmp/tinytpu_bundle.txt"],
            capture_output=True, text=True
        )
        return parse_result(result.stdout)
```

### 4.4 `TinytpuDevice`

```python
class TinytpuDevice(Compiled):
    def __init__(self, device:str):
        from tinygrad.renderer.tinytpu import TinyTPURenderer
        super().__init__(device,
            allocator=TinytpuAllocator(self),
            renderer=TinyTPURenderer(),
            compiler=TinyTPUCompiler(),   # JSON-encodes the rendered program
            runtime=TinyTPUProgram)
```

---

## 5. Simulator Build

The BSV testbench `TbTinyTPURuntime.bsv` is built once:

```makefile
runtime-tb: $(BUILDDIR)/mkTbTinyTPURuntime.bexe
TINYTPU_SIM := $(BUILDDIR)/mkTbTinyTPURuntime.bexe
```

`TinyTPUProgram.__call__` invokes `$(TINYTPU_SIM)` via subprocess.

The simulator path is configured via environment variable `TINYTPU_SIM` or defaults to `build/mkTbTinyTPURuntime.bexe`.

---

## 6. Development Sequence

Implement in this order (each depends on the previous):

1. **VMEM** (`plan-vmem.md`) — hardware prerequisite
2. **VRegFile** (`plan-vregfile.md`) — hardware prerequisite
3. **VPU** (`plan-vpu.md`) — hardware prerequisite
4. **ScalarUnit** (`plan-scalar-unit.md`) — hardware prerequisite
5. **TensorCore** (`plan-tensorcore.md`) — hardware prerequisite
6. **`TbTinyTPURuntime.bsv`** — generic BSV testbench (reads bundle, runs TC, writes results)
7. **`tinygrad/tinygrad/renderer/tinytpu.py`** — UOp → SXU renderer
8. **`tinygrad/tinygrad/runtime/ops_tinytpu.py`** — tinygrad device/allocator/program
9. **End-to-end test** — `Tensor([1,2,3,4], device="TINYTPU") @ weight_matrix`

Steps 1–5 are pure BSV (hardware). Steps 6–9 are pure Python (software). They can be developed in parallel once the hardware interface is stable.

---

## 7. Hardware/Software Interface Contract

The Python layer and BSV simulator agree on:

| Parameter | Value (TinyTPU prototype) |
|---|---|
| sublanes | 4 |
| lanes | 4 |
| VMEM depth | 16 tiles |
| VReg count | 16 |
| MXU size | 4×4 (rows=cols=4) |
| Tile element type | Int#(32) (32-bit signed integer) |
| Tile size (bytes) | 4×4×4 = 64 bytes |
| SXU program max length | 16 instructions |

These constants are defined in `tinygrad/tinygrad/runtime/ops_tinytpu.py` as `TINYTPU_CONFIG` and mirrored in `test/TbTinyTPURuntime.bsv` as BSV parameters.

---

## 8. Example: `a @ b` (Matrix Multiply)

```python
import tinygrad
from tinygrad import Tensor

# 4×4 matrix multiply on TinyTPU
a = Tensor([[1,2,3,4],[0,0,0,0],[0,0,0,0],[0,0,0,0]], device="TINYTPU")
w = Tensor([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], device="TINYTPU")
result = (a @ w).numpy()
# → [1,2,3,4,0,0,...] — identity transform
```

Internally this compiles to:
```
WEIGHT_TILE 0  1 0 0 0  0 1 0 0  0 0 1 0  0 0 0 1   # identity matrix
ACT_TILE    1  1 2 3 4                               # first row of a
INSTR SXU_DISPATCH_MXU 0 1 1
INSTR SXU_WAIT_MXU 0 0 0
INSTR SXU_HALT 0 0 0
OUTPUT_MXU 1
```

---

*Document generated: 2026-04-08*  
*Status: Draft — implementation pending hardware modules*
