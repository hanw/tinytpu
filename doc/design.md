# INT8 Tensor Accelerator — Bluespec Design Spec

**Date:** 2026-04-07
**Target:** ASIC / Bluesim simulation
**Dataflow:** Weight-stationary systolic array
**Precision:** INT8 (activations and weights), INT32 (accumulators)
**Array size:** Parameterized, targeting 4x4 to 8x8

---

## 1. Processing Element (PE)

Each PE performs one INT8 multiply-accumulate per cycle.

### Interface

```bsv
interface PE_IFC;
  method Action loadWeight(Int#(8) w);
  method Action feedActivation(Int#(8) a);
  method Int#(32) getAccum();
  method Action clearAccum();
  method Int#(8) passActivation();
endinterface
```

### Internals

- `Reg#(Int#(8)) weight` — loaded once per inference tile (stationary)
- `Reg#(Int#(32)) accum` — 32-bit accumulator. 8x8 multiply produces 16-bit results; 32 bits handles up to 65K accumulations without overflow.
- `Reg#(Int#(8)) act_pass` — registered activation output for systolic wavefront timing

### Behavior

Each cycle the PE:
1. Receives an activation on `feedActivation`
2. Multiplies by its stationary weight
3. Adds product to `accum`
4. Registers the activation for forwarding via `passActivation` (1-cycle delay)

---

## 2. Systolic Array

### Parameterization

```bsv
module mkSystolicArray(SystolicArray_IFC#(rows, cols))
  provisos(Add#(1, _, rows), Add#(1, _, cols));
```

### Structure

`Vector#(rows, Vector#(cols, PE_IFC))` — a 2D grid of PEs.

### Dataflow

- **Weights** load vertically: column `j` receives weights for output channel `j`. Loaded once before streaming.
- **Activations** flow horizontally: row `i` receives activation element `i`. Each PE in the row sees the same activation, delayed by one cycle per column (systolic skew).
- **Outputs** drain from accumulators: each column's accumulated values form one output element.

### Systolic Skew

Activations are staggered — row 0 starts at cycle 0, row 1 at cycle 1, etc. The array needs `rows + cols - 1` cycles to complete one matrix-vector multiply. Skew is managed by the controller, not the PEs.

### Interface

```bsv
interface SystolicArray_IFC#(numeric type rows, numeric type cols);
  method Action loadWeights(Vector#(rows, Vector#(cols, Int#(8))) w);
  method Action feedActivations(Vector#(rows, Int#(8)) a);
  method Vector#(cols, Int#(32)) getResults();
  method Action clearAll();
endinterface
```

---

## 3. SRAM Banks

### Weight SRAM (one logical bank per column)

```bsv
interface WeightSRAM_IFC#(numeric type depth, numeric type cols);
  method Action write(UInt#(TLog#(depth)) addr, Vector#(cols, Int#(8)) data);
  method Action readReq(UInt#(TLog#(depth)) addr);
  method Vector#(cols, Int#(8)) readResp();
endinterface
```

- `depth` parameterizes storage capacity (number of weight rows)
- `cols` parallel banks for single-cycle full-row reads
- Address width: `TLog#(depth)` bits

### Activation SRAM (one logical bank per row)

```bsv
interface ActivationSRAM_IFC#(numeric type depth, numeric type rows);
  method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
  method Action readReq(UInt#(TLog#(depth)) addr);
  method Vector#(rows, Int#(8)) readResp();
endinterface
```

- Same parameterization pattern as weight SRAM
- Sequential reads feed `feedActivations` each cycle

### Output Buffer

`Vector#(cols, Reg#(Int#(32)))` register file. Results drain after each tile computation. No SRAM — the controller reads `getResults()` and outputs or feeds downstream.

---

## 4. Controller FSM

### States

```bsv
typedef enum {
  Idle,
  LoadWeights,
  StreamActivations,
  Drain,
  Done
} ControlState deriving (Bits, Eq);
```

### Phase Sequencing

1. **Idle** — awaiting `start` with tile configuration
2. **LoadWeights** — reads one row/cycle from weight SRAM, loads into array. Takes `rows` cycles.
3. **StreamActivations** — reads one activation vector/cycle, feeds array with systolic skew (zeros for not-yet-active rows). Takes `tileLen + rows - 1` cycles.
4. **Drain** — final accumulations settle, reads `getResults()` into output buffer. Takes 1 cycle.
5. **Done** — signals completion, returns to Idle.

### Interface

```bsv
interface Controller_IFC#(numeric type rows, numeric type cols, numeric type depth);
  method Action start(UInt#(TLog#(depth)) weightBase,
                      UInt#(TLog#(depth)) actBase,
                      UInt#(TLog#(depth)) tileLen);
  method Bool done();
  method Vector#(cols, Int#(32)) results();
endinterface
```

The controller owns the skew pattern. PEs and SRAMs are stateless with respect to sequencing.

---

## 5. Top-Level Integration

### Interface

```bsv
interface TensorAccelerator_IFC#(numeric type rows, numeric type cols, numeric type depth);
  method Action loadWeightTile(UInt#(TLog#(depth)) addr, Vector#(rows, Vector#(cols, Int#(8))) wRow);
  method Action loadActivationTile(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) aRow);
  method Action startCompute(UInt#(TLog#(depth)) weightBase,
                             UInt#(TLog#(depth)) actBase,
                             UInt#(TLog#(depth)) tileLen);
  method Bool computeDone();
  method Vector#(cols, Int#(32)) getOutput();
endinterface
```

### Module

```bsv
module mkTensorAccelerator(TensorAccelerator_IFC#(rows, cols, depth))
  provisos(Add#(1, _, rows), Add#(1, _, cols),
           Add#(1, _, depth), Log#(depth, _));
```

### Internal Components

- `mkSystolicArray` — the PE grid
- `mkWeightSRAM` — weight storage
- `mkActivationSRAM` — activation storage
- `mkController` — FSM orchestration

Controller reads from SRAMs and drives the array. Host writes to SRAMs and triggers compute.

### Typical Instantiation

```bsv
TensorAccelerator_IFC#(4, 4, 256) accel <- mkTensorAccelerator;  // 4x4, 256-deep
TensorAccelerator_IFC#(8, 8, 512) accel <- mkTensorAccelerator;  // 8x8, 512-deep
```

### Type-Level Guarantees

- SRAM address widths are always `TLog#(depth)` — no off-by-one sizing
- Weight matrix dimensions match array dimensions at compile time
- Activation vector width matches row count
- Output vector width matches column count

---

## 6. Scope Boundaries

**In scope:**
- PE, systolic array, SRAM banks, controller, top-level module
- Parameterization via BSC numeric types
- Bluesim testbench

**Out of scope (intentionally):**
- Requantization (separate downstream block)
- DMA / AXI / host bus interface
- Multi-tile scheduling (software responsibility)
- Power/clock gating
