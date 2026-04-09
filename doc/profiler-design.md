# TinyTPU Profiler — Design Spec

**Date:** 2026-04-09
**Status:** Draft → pending user review
**Reference:** Modeled on `~/p/original_performance_takehome/profile_asm.py`

## 1. Goal

Build a profiler for TinyTPU SXU microprograms that:

1. Runs a bundle through the BSV TensorCore simulator with tracing enabled.
2. Prints a reference-style text report: summary, instruction mix, per-unit
   utilization, hotspots, MXU breakdown, bubble analysis.
3. Emits a Perfetto-compatible `trace.json` for timeline visualization.
4. Accepts either a hand-written bundle file or (optionally) a tinygrad
   program that lowers to a bundle via the `TINYTPU` device.

## 2. Scope

**In scope (TensorCore-level tracing):**

- `ScalarUnit` — per-rule state transitions (FETCH, LOAD_REQ/RESP, STORE,
  DISPATCH_VPU, VPU_COLLECT, DISPATCH_MXU, WAIT_MXU, HALT).
- `Controller` (MXU) — LOAD_W, STREAM_A, DRAIN, DONE.
- `VPU` — EXEC, RESULT.
- `VMEM` — READ_REQ, READ_RESP, WRITE.

**Out of scope:**

- Chip-level units (`ChipNoC`, `HBMModel`, `SparseCore`, `XLU`). The runtime
  testbench instantiates only a `TensorCore`, so there is nothing to trace
  beyond the four units above. A chip-level profiler would need a second
  traced testbench on top of `TinyTPUChip`; defer until chip-level programs
  exist.
- HTML viewer (`watch_trace.py` equivalent). User drags `trace.json` into
  ui.perfetto.dev manually.
- `--compare` mode, `--csv` output, section labels.

## 3. Architecture

```
  bundle.txt ─┐                          ┌── text report (stdout)
              ▼                          │
  profile_tpu.py ──► mkTbTinyTPURuntimeTrace.bexe
              ▲       env: TINYTPU_TRACE=1, TINYTPU_BUNDLE=/tmp/…
  tinygrad ───┘                          │
                        stdout: TRACE lines + result lines
                                         │
                          parse events ──┼──► trace.json (Perfetto)
                                         │
                                     reports
```

Two components:

### 3.1 Traced BSV binary

A new `make runtime-tb-trace` target produces
`build/mkTbTinyTPURuntimeTrace.bexe`, compiled with `-D TRACE`. The source is
the existing `test/TbTinyTPURuntime.bsv` plus conditional `$display` calls
added to rules in `src/ScalarUnit.bsv` and `src/Controller.bsv`. Every trace
call is wrapped in `` `ifdef TRACE … `endif ``, so the untraced `runtime-tb`
target stays byte-identical.

Note: `VMEM.bsv` and `VPU.bsv` are method-based modules with no rules of their
own, so VMEM and VPU events are emitted from the `ScalarUnit` rules that
invoke `vmem.readReq/readResp/write` and `vpu.execute/result`. This is
cleaner than adding trace-only rules to the units themselves.

There is no runtime toggle: the `-D TRACE` compile gate is sufficient. If
you want an untraced run, use the untraced `runtime-tb` binary; if you want
a traced run, use `runtime-tb-trace`. Keeping the two binaries separate is
simpler than a runtime switch.

Each traced module carries a local `UInt#(32) cycle` register incremented
every clock by a free-running rule (under `` `ifdef TRACE ``), so all
timestamps share the same clock and do not require plumbing a cycle signal
through interfaces.

### 3.2 Python profiler (`scripts/profile_tpu.py`)

Layout:

```
scripts/profile_tpu.py      — argparse CLI
scripts/profiler/
    __init__.py
    bundle.py               — read/write TbTinyTPURuntime numeric bundles
    tinygrad_bridge.py      — lazy: build bundle from a tinygrad script
    trace_parser.py         — parse TRACE lines → list[Event]
    perfetto_emitter.py     — Event list → trace.json
    reports.py              — text reports
```

CLI:

```
python scripts/profile_tpu.py <bundle.txt> [--trace-out trace.json]
python scripts/profile_tpu.py --from-tinygrad script.py
python scripts/profile_tpu.py --sample
```

`--sample` runs the sample program described in §7.

## 4. BSV trace event format

Every transition rule emits one line:

```
TRACE cycle=<N> unit=<SXU|MXU|VPU|VMEM> ev=<EVENT> [key=value …]
```

Events per unit:

| Unit | Event          | Extra fields      | Emitted by                 |
|------|----------------|-------------------|----------------------------|
| SXU  | `FETCH`        | `pc=<P>`          | `do_fetch`                 |
| SXU  | `LOAD_REQ`     | `pc=<P> addr=<A>` | `do_load_req`              |
| SXU  | `LOAD_RESP`    | `pc=<P>`          | `do_load_resp`             |
| SXU  | `STORE`        | `pc=<P> addr=<A>` | `do_store`                 |
| SXU  | `DISPATCH_VPU` | `pc=<P> op=<O>`   | `do_vpu`                   |
| SXU  | `VPU_COLLECT`  | `pc=<P>`          | `do_vpu_collect`           |
| SXU  | `DISPATCH_MXU` | `pc=<P>`          | `do_mxu`                   |
| SXU  | `WAIT_MXU`     | `pc=<P>`          | `do_wait_mxu` (per cycle)  |
| SXU  | `HALT`         | `pc=<P>`          | `do_fetch` when opcode=HALT |
| MXU  | `LOAD_W`       | `addr=<A>`        | `do_load_weights` in Controller |
| MXU  | `LOAD_W_RESP`  |                   | `do_load_weights_resp`     |
| MXU  | `STREAM_A`     | `cyc=<C>`         | `do_stream` in Controller  |
| MXU  | `DRAIN`        |                   | `do_drain` in Controller   |
| VPU  | `EXEC`         | `op=<O>`          | emitted from SXU `do_vpu`  |
| VPU  | `RESULT`       |                   | emitted from SXU `do_vpu_collect` |
| VMEM | `READ_REQ`     | `addr=<A>`        | emitted from SXU `do_load_req` |
| VMEM | `READ_RESP`    |                   | emitted from SXU `do_load_resp` |
| VMEM | `WRITE`        | `addr=<A>`        | emitted from SXU `do_store` |

The parser ignores unknown keys so the format can be extended without
breaking old traces.

## 5. Python profiler components

### 5.1 `bundle.py`

Reads/writes the numeric bundle format defined in
`test/TbTinyTPURuntime.bsv` (records: `0` weight tile, `1` act tile,
`2` instr, `3` output_mxu, `4` end). Exposes `Bundle` dataclass and
`parse_file(path) / write_file(path, bundle)` helpers. The profiler
can also read an existing bundle and extract the SXU program for
report annotation (opcode names per PC).

### 5.2 `trace_parser.py`

Parses lines matching `^TRACE ` into `Event(cycle, unit, ev, fields)`.
Non-TRACE lines (the usual `mxu_result`, `cycles`, `status` output)
are returned separately so the caller can still check correctness.

### 5.3 `perfetto_emitter.py`

Converts `list[Event]` to a Chrome tracing JSON document:

```json
{"traceEvents": [
  {"ph": "X", "name": "DISPATCH_VPU(ADD)", "cat": "SXU",
   "ts": 42, "dur": 1, "pid": 1, "tid": 1},
  …
]}
```

`pid=1` = TinyTPU. `tid` is one of `{1:SXU, 2:MXU, 3:VPU, 4:VMEM}`. `ts`
and `dur` are the BSV cycle count; Perfetto shows them as microseconds
but the display is fine for an RTL trace. Consecutive same-event slices
on the same tid are coalesced into a single duration event.

### 5.4 `reports.py`

Implements the text reports:

- **`print_summary`** — bundle path, correctness (`status ok` + optional
  numeric result check if `mxu_result` is present), total cycles,
  instruction count, per-unit busy/idle cycles.
- **`print_hotspots(top_n)`** — top-N SXU PCs by cycles spent at that
  PC (summed across all states for that instruction), with opcode name.
- **`print_instruction_mix`** — SxuOpCode frequency + %.
- **`print_utilization`** — for each unit (SXU/MXU/VPU/VMEM): busy
  cycles, idle cycles, util %. Overall utilization line.
- **`print_mxu_breakdown`** — cycles in LOAD_W / STREAM_A / DRAIN, plus
  SXU WAIT_MXU stall cycles. Useful for spotting dispatch overhead.
- **`print_bubbles(threshold=0)`** — cycles where zero units are busy,
  grouped by the SXU PC that was currently executing. Points at the
  instruction causing the bubble.

### 5.5 `tinygrad_bridge.py`

Lazy-imports the tinygrad `TINYTPU` device and runs a user-supplied
Python script that builds a tinygrad `Tensor` expression. The bridge
monkey-patches the `TinyTPUProgram` subprocess call to (a) capture the
bundle file and (b) skip executing the untraced binary. The profiler
then runs its own traced binary on the captured bundle.

If the tinygrad submodule is not populated, `--from-tinygrad` prints a
clear error pointing at `git submodule update --init`.

## 6. Build & test

### 6.1 Makefile

```makefile
runtime-tb-trace: $(BDPI_OBJ)
	bsc -bdir build -simdir build -D TRACE \
	    -p src:test:+ -u -sim test/TbTinyTPURuntime.bsv
	bsc -bdir build -simdir build -sim -e mkTbTinyTPURuntime \
	    -o build/mkTbTinyTPURuntimeTrace.bexe $(BDPI_OBJ)
```

The untraced `runtime-tb` target is unchanged.

### 6.2 Tests

- **`test-trace`** (new make target) — builds the traced binary, runs
  a hand-written tiny bundle, greps stdout for at least one
  `TRACE cycle=` line and the expected `status ok`.
- **`tests/test_profile_tpu.py`** (new Python test dir) — runs the
  profiler on the sample bundle, asserts: bundle parses, traced binary
  exits 0, `trace.json` is valid JSON with non-empty `traceEvents`,
  the text report contains `SXU`, `MXU`, `VPU`, `VMEM` utilization
  lines and non-zero total cycles.

## 7. Sample program

`scripts/profiler/sample_program.py` emits a bundle that exercises all
four traced units end-to-end:

1. Load weight tile + activation tile (setup)
2. `SXU_DISPATCH_MXU` — kicks MXU
3. `SXU_WAIT_MXU` — stalls until done
4. `SXU_LOAD_VREG` — pull MXU result staging into VRF via VMEM
5. `SXU_DISPATCH_VPU` (RELU) — exercise VPU
6. `SXU_STORE_VREG` — exercise VMEM write
7. `SXU_DISPATCH_MXU` + `SXU_WAIT_MXU` — second MXU dispatch
8. `SXU_HALT`

Invoked via `python scripts/profile_tpu.py --sample`. The resulting
report should show non-zero utilization on every traced unit and
non-zero MXU stall cycles.

## 8. YAGNI cuts (reasoning)

| Cut                       | Why                                           |
|---------------------------|-----------------------------------------------|
| HTML auto-forwarder       | Drag-and-drop to ui.perfetto.dev works fine   |
| `--compare` mode          | No current use case; add on demand            |
| `--csv` output            | No downstream consumer                        |
| Labeled sections          | Bundle format has no labels; add when asm lang lands |
| Chip-level tracing        | Runtime testbench is TensorCore-only          |
| Runtime `TINYTPU_TRACE` env gate | Compile-time `-D TRACE` already produces a separate binary |

## 9. Extension notes

- **Section labels.** Add a `5 <name_u32> <pc>` record to the bundle
  format. Profiler would map PCs to named sections and add
  `print_sections`. No BSV changes needed.
- **Chip-level tracing.** Mirror the `-D TRACE` pattern in a new
  `TbTinyTPUChipRuntime` testbench. Add NOC/HBM/SparseCore events.
- **`--compare` / `--csv`.** Trivial once reports exist.
