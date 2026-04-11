# TinyTPU Assembly Language (TASM)

**Status:** Reference spec  
**Tooling:** `scripts/tasm.py` — assembler and disassembler  
**Wire format:** `doc/tinytpu_bundle_format.md`

---

## Overview

TASM is a human-readable assembly language for TinyTPU co-simulation bundles.
A bundle is a self-contained program loaded into the BSV testbench
(`TbTinyTPURuntime.bsv`) via the `$TINYTPU_BUNDLE` environment variable.

A bundle has two sections in order:

1. **Data declarations** — pre-load tiles into VMEM, WMEM, or AMEM before execution.
2. **Program** — SXU instruction sequence (LOAD / VPU / STORE / MXU / HALT).
3. **Output directives** — what to emit after HALT (`OUTPUT_MXU`, `OUTPUT_VMEM`).
4. **END** — signals the testbench to stop loading and begin execution.

The assembler (`tasm.assemble()`) converts TASM text to the numeric wire format.
The disassembler (`tasm.disassemble()`) converts wire format back to TASM for
debugging.

---

## Syntax

- One statement per line.
- Comments begin with `#` and extend to end of line.
- Keywords and mnemonic names are **case-insensitive**.
- Blank lines are ignored.
- Integers are decimal; negative values are written with a leading `-`.

---

## Memory model

| Space | Description | Element type | Capacity |
|---|---|---|---|
| `VMEM[N]` | Vector memory — holds 4×4 Int32 tiles | `int32` | up to 256 tiles |
| `WMEM[N]` | Weight SRAM — holds 4×4 Int8 weight tiles | `int8` | up to 256 tiles |
| `AMEM[N]` | Activation SRAM — holds 4-element Int8 vectors | `int8` | up to 256 slots |
| `v<N>` | Vector register file — 4×4 Int32 tiles | `int32` | 16 registers (v0–v15) |

Tiles are stored and transferred in **row-major** order: for a 4×4 tile the
16 values appear as `row0col0 row0col1 row0col2 row0col3 row1col0 ...`.

---

## Data declarations

Data declarations are loaded into the named memory before execution begins.
They must appear **before** any instructions.

### `VMEM[N] = v0 v1 ... v15`

Load a 4×4 Int32 tile into VMEM slot N.  Exactly 16 space-separated integers.

```
VMEM[0] = -1  2 -3  4   0  0  0  0   0  0  0  0   0  0  0  0
VMEM[1] =  0  0  0  0   0  0  0  0   0  0  0  0   0  0  0  0
```

### `WMEM[N] = w00 w01 ... w33`

Load a 4×4 Int8 weight tile into WMEM slot N.  Exactly 16 values, row-major,
clamped to `[-128, 127]`.

```
WMEM[0] = 1 0 0 0   0 1 0 0   0 0 1 0   0 0 0 1   # identity
```

### `AMEM[N] = a0 a1 a2 a3`

Load a 4-element Int8 activation vector into AMEM slot N.  Exactly 4 values.

```
AMEM[1] = 3 7 -2 5
```

---

## Instructions

Instructions are executed sequentially by the SXU (Scalar eXecution Unit).
Each instruction maps to one `SxuInstr` loaded into the SXU program memory.

### `LOAD vD, VMEM[S]`

Load VMEM tile at slot S into vector register vD.

```
LOAD v0, VMEM[0]
LOAD v1, VMEM[1]
```

**Wire:** `2 0 <S> <D> 0 0 0 0 0 0`  
(SxuOpCode=`SXU_LOAD_VREG`=0, vmemAddr=S, vregDst=D)

---

### `STORE VMEM[D], vS`

Store vector register vS into VMEM slot D.

```
STORE VMEM[2], v2
```

**Wire:** `2 1 <D> 0 <S> 0 0 0 0 0`  
(SxuOpCode=`SXU_STORE_VREG`=1, vmemAddr=D, vregSrc=S)

---

### `VPU vD = OP(vA [, vB])`

Dispatch a VPU operation.  Binary ops take two source registers; unary/reduce
ops take one (hardware still receives vB=v0 as src2 but ignores it).

```
VPU v2 = ADD(v0, v1)         # binary
VPU v3 = SUB(v0, v1)         # binary: v0 - v1
VPU v4 = MAX(v0, v1)         # binary elementwise max
VPU v5 = RELU(v0)            # unary: max(v0, 0)
VPU v6 = MAX_REDUCE(v0)      # reduce: every element = max across row
```

**Wire (binary):** `2 2 0 <D> <A> <op_int> <B> 0 0 0`  
**Wire (unary):** `2 2 0 <D> <A> <op_int> 0 0 0 0`  
(SxuOpCode=`SXU_DISPATCH_VPU`=2, vregDst=D, vregSrc=A, vpuOp=op_int, vregSrc2=B)

#### VPU operation table

| Mnemonic | Integer | Arity | Semantics |
|---|---|---|---|
| `ADD` | 0 | binary | `vD[i] = vA[i] + vB[i]` |
| `MUL` | 1 | binary | `vD[i] = vA[i] * vB[i]` |
| `RELU` | 2 | unary | `vD[i] = max(vA[i], 0)` |
| `MAX` | 3 | binary | `vD[i] = max(vA[i], vB[i])` |
| `SUM_REDUCE` | 4 | unary | `vD[i] = sum(vA[row])` (all lanes in row) |
| `CMPLT` | 5 | binary | `vD[i] = (vA[i] < vB[i]) ? 1 : 0` |
| `CMPNE` | 6 | binary | `vD[i] = (vA[i] != vB[i]) ? 1 : 0` |
| `SUB` | 7 | binary | `vD[i] = vA[i] - vB[i]` |
| `CMPEQ` | 8 | binary | `vD[i] = (vA[i] == vB[i]) ? 1 : 0` |
| `MAX_REDUCE` | 9 | unary | `vD[i] = max(vA[row])` (all lanes in row) |
| `SHL` | 10 | binary | `vD[i] = vA[i] << vB[i]` |
| `SHR` | 11 | binary | `vD[i] = vA[i] >> vB[i]` |
| `MIN` | 12 | binary | `vD[i] = min(vA[i], vB[i])` |
| `MIN_REDUCE` | 13 | unary | `vD[i] = min(vA[row])` (all lanes in row) |
| `DIV` | 14 | binary | `vD[i] = vA[i] / vB[i]` (0 if vB[i]=0) |
| `AND` | 15 | binary | `vD[i] = vA[i] & vB[i]` (bitwise) |
| `OR` | 16 | binary | `vD[i] = vA[i] \| vB[i]` (bitwise) |
| `XOR` | 17 | binary | `vD[i] = vA[i] ^ vB[i]` (bitwise) |

Indices `[i]` denote all elements of the 4×4 tile.  Reduce ops broadcast the
per-row result back to every element in that row.

---

### `BROADCAST vN [, lane=L]`

Broadcast lane L of vector register vN across all lanes of that register
(in-place), using the XLU (eXpand Lane Unit).  Default lane is 0.

```
BROADCAST v0            # broadcast lane 0 of v0 → v0
BROADCAST v1, lane=2    # broadcast lane 2 of v1 → v1
```

**Wire:** `2 3 0 <N> <N> 0 <L> 0 0 0`  
(SxuOpCode=`SXU_DISPATCH_XLU_BROADCAST`=3, vregDst=N, vregSrc=N, vregSrc2=L)

**Use case:** Scalar-broadcast a single element of a VMEM tile to fill the
entire register before an elementwise op.

---

### `MXU WMEM[W], AMEM[A], tiles=N`

Dispatch the MXU (Matrix eXecution Unit) to multiply weight tiles starting at
WMEM[W] against activation vectors starting at AMEM[A] for N consecutive tiles.
Execution is asynchronous; follow with `WAIT_MXU`.

```
MXU WMEM[0], AMEM[1], tiles=1
WAIT_MXU
```

**Wire:** `2 4 0 0 0 0 0 <W> <A> <N>`  
(SxuOpCode=`SXU_DISPATCH_MXU`=4, mxuWBase=W, mxuABase=A, mxuTLen=N)

---

### `WAIT_MXU`

Stall the SXU until the MXU signals completion.

**Wire:** `2 5 0 0 0 0 0 0 0 0`

---

### `HALT`

Stop SXU execution.  The testbench then emits any requested output records.

**Wire:** `2 6 0 0 0 0 0 0 0 0`

---

## Output directives

Output directives appear after all instructions (before `END`).

### `OUTPUT_MXU`

After HALT, print the 4-element MXU accumulator result as:

```
mxu_result v0 v1 v2 v3
```

**Wire:** `3 1`

### `OUTPUT_VMEM VMEM[N]`

After HALT, print the 16 elements of VMEM tile N as:

```
vmem_result e00 e01 e02 e03 e10 e11 e12 e13 e20 e21 e22 e23 e30 e31 e32 e33
```

**Wire:** `6 <N>`

---

## `END`

Signals the testbench to stop reading the bundle and begin execution.
Must be the last statement.

**Wire:** `4`

---

## Complete examples

### 1. Elementwise add

```tasm
# Inputs
VMEM[0] = 1 2 3 4  5  6  7  8  9 10 11 12 13 14 15 16
VMEM[1] = 1 1 1 1  1  1  1  1  1  1  1  1  1  1  1  1

# Program
LOAD  v0, VMEM[0]
LOAD  v1, VMEM[1]
VPU   v2 = ADD(v0, v1)
STORE VMEM[2], v2
HALT

OUTPUT_VMEM VMEM[2]
END
```

### 2. Elementwise abs (via SUB + MAX)

```tasm
# input in VMEM[0]; zeros in VMEM[1]
VMEM[0] = -1  2 -3  4  0  0  0  0  0  0  0  0  0  0  0  0
VMEM[1] =  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0

LOAD  v0, VMEM[0]          # v0 = input
LOAD  v1, VMEM[1]          # v1 = zeros
VPU   v2 = SUB(v1, v0)     # v2 = 0 - input  (negate)
VPU   v3 = MAX(v0, v2)     # v3 = max(input, -input) = abs
STORE VMEM[2], v3
HALT

OUTPUT_VMEM VMEM[2]
END
```

### 3. GEMM tile (identity weight × activation)

```tasm
WMEM[0] = 1 0 0 0   0 1 0 0   0 0 1 0   0 0 0 1
AMEM[1] = 3 7 -2 5

MXU   WMEM[0], AMEM[1], tiles=1
WAIT_MXU
HALT

OUTPUT_MXU
END
```

### 4. WHERE(cond, lhs, rhs) = cond*lhs + (1-cond)*rhs

```tasm
VMEM[0] =  1  0  1  0  1  0  1  0  1  0  1  0  1  0  1  0   # cond
VMEM[1] = 10 20 30 40 10 20 30 40 10 20 30 40 10 20 30 40   # lhs
VMEM[2] = -1 -2 -3 -4 -1 -2 -3 -4 -1 -2 -3 -4 -1 -2 -3 -4 # rhs
VMEM[3] =  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1  1   # ones

LOAD v0, VMEM[0]   # cond
LOAD v1, VMEM[1]   # lhs
LOAD v2, VMEM[2]   # rhs
LOAD v3, VMEM[3]   # ones

VPU v4 = MUL(v0, v1)   # cond * lhs
VPU v5 = SUB(v3, v0)   # 1 - cond
VPU v6 = MUL(v5, v2)   # (1-cond) * rhs
VPU v7 = ADD(v4, v6)   # result

STORE VMEM[4], v7
HALT

OUTPUT_VMEM VMEM[4]
END
```

---

## Wire format mapping summary

| TASM statement | Record type | Wire encoding |
|---|---|---|
| `WMEM[N] = ...` | 0 | `0 N w0 w1 ... w15` |
| `AMEM[N] = ...` | 1 | `1 N a0 a1 a2 a3` |
| `LOAD vD, VMEM[S]` | 2 | `2 0 S D 0 0 0 0 0 0` |
| `STORE VMEM[D], vS` | 2 | `2 1 D 0 S 0 0 0 0 0` |
| `VPU vD = OP(vA, vB)` | 2 | `2 2 0 D A <op> B 0 0 0` |
| `VPU vD = OP(vA)` | 2 | `2 2 0 D A <op> 0 0 0 0` |
| `BROADCAST vN [, lane=L]` | 2 | `2 3 0 N N 0 L 0 0 0` |
| `MXU WMEM[W], AMEM[A], tiles=N` | 2 | `2 4 0 0 0 0 0 W A N` |
| `WAIT_MXU` | 2 | `2 5 0 0 0 0 0 0 0 0` |
| `HALT` | 2 | `2 6 0 0 0 0 0 0 0 0` |
| `OUTPUT_MXU` | 3 | `3 1` |
| `END` | 4 | `4` |
| `VMEM[N] = ...` | 5 | `5 N v0 v1 ... v15` |
| `OUTPUT_VMEM VMEM[N]` | 6 | `6 N` |

The instruction record (type 2) field order is:  
`record_type sxu_opcode vmemAddr vregDst vregSrc vpuOp vregSrc2 mxuWBase mxuABase mxuTLen`

---

## Tooling

```bash
# Assemble TASM → wire format
python3 scripts/tasm.py assemble program.tasm

# Disassemble wire format → TASM (for debugging captured bundles)
python3 scripts/tasm.py disassemble bundle.txt

# Round-trip check
python3 scripts/tasm.py assemble program.tasm | python3 scripts/tasm.py disassemble -
```

The Python API:

```python
from scripts.tasm import assemble, disassemble

wire = assemble(tasm_text)    # str → str
tasm = disassemble(wire_text) # str → str
```
