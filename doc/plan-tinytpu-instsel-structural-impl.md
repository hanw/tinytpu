# TinyTPU InstSel Structural Slice — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate the 12 remaining `_render_*_sxu_program` structural recognizers out of `ops_tinytpu.py` into a `tinytpu_lowering` package behind a positive kernel classifier.

**Architecture:** A classifier (`classify(uops) -> KernelClass`) dispatches each kernel to one focused lowerer (elementwise / reduction / broadcast / gemm, plus movement pending a spike). The single-file lowerer becomes a package. No try-chains, no fallback.

**Tech Stack:** Python, tinygrad UOps (`tinygrad/tinygrad/`), BSV simulator verification via `tests/test_tinytpu_backend.py`.

**Reference:** `doc/plan-tinytpu-instsel-structural.md` (design) and `doc/plan-tinytpu-instsel.md` (the completed first slice). The legacy `_render_*` recognizers in `ops_tinytpu.py` are the behavioral reference for each lowerer.

## Conventions for every task

- The lowering code lives in the `tinygrad/` submodule. Commit inside the submodule first, then commit the submodule pointer (plus `TODO.md`/`results.tsv`/tests) in the parent repo.
- Verification command (run from repo root):
  `PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py::TestTinyTPUBackend -q`
- A "clean" result = no test that passed in `/tmp/base_fail.txt` newly fails (the diff method is in Task 0). Hardware-exposure failures (`EXP2`/`LOG2`/`SIN`) are pre-accepted.
- Append one `results.tsv` row and update `TODO.md` per `AGENT.md`.
- A recognizer's function body and its dispatch-chain call are deleted in the same commit that lands its replacement.

---

## Task 0: Capture the verification baseline

**Files:**
- None (produces `/tmp/struct_base.txt`).

- [ ] **Step 1: Run the full backend suite at current HEAD and save the failure set**

Run:
```bash
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py::TestTinyTPUBackend -q 2>&1 \
  | grep -E "^(FAILED|[0-9]+ (passed|failed))" > /tmp/struct_base.txt
tail -1 /tmp/struct_base.txt
```
Expected: `88 failed, 810 passed` (the post-iteration-8 state).

- [ ] **Step 2: Define the regression-diff snippet for reuse**

After any later run saved to `/tmp/cur.txt`, regressions are:
```bash
comm -23 <(grep '^FAILED' /tmp/cur.txt | sort) <(grep '^FAILED' /tmp/struct_base.txt | sort)
```
An empty result = no regression. Record this; every task below uses it.

---

## Task 1: Split `tinytpu_lowering.py` into a package (behavior-neutral)

**Files:**
- Delete: `tinygrad/tinygrad/runtime/support/tinytpu_lowering.py`
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/__init__.py`
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/common.py`
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/elementwise.py`

- [ ] **Step 1: Create `common.py`** with everything not specific to elementwise lowering: `TpuInst`, `TpuKernel`, `_VPU`/`_ALU_TO_VPU`/`_FLOAT_VPU`/`_UNARY_VPU`/`_DATA_OPS` tables, `_NUM_VREGS`, `_ROWS`/`_COLS`/`_TILE_ELEMS`, `_canon`, `_unique_param`, `_data_dag`, `_store_lanes`, `_is_float`, `_float_operands`, `_const_bits`, `_expand_sqrt`/`_expand_mod`/`_INSTSEL`/`_INSTSEL_OPS`/`_run_instsel`.

- [ ] **Step 2: Create `elementwise.py`** with `can_lower` and `lower_kernel` (the current walker), importing its helpers from `.common`.

- [ ] **Step 3: Create `__init__.py`** that re-exports the existing public surface so `ops_tinytpu.py` keeps working unchanged:

```python
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower, lower_kernel
```

- [ ] **Step 4: Verify the import surface is unchanged**

Run: `PYTHONPATH=tinygrad .venv/bin/python3 -c "from tinygrad.runtime.support.tinytpu_lowering import can_lower, lower_kernel; print('ok')"`
Expected: `ok`

- [ ] **Step 5: Run the full suite**

Run the verification command into `/tmp/cur.txt`; run the Task 0 diff.
Expected: `88 failed, 810 passed`, empty diff (behavior-neutral).

- [ ] **Step 6: Commit** (submodule, then parent pointer + `results.tsv` + `TODO.md`).

```
tinytpu: split tinytpu_lowering into a package
```

---

## Task 2: Introduce the kernel classifier

**Files:**
- Modify: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/__init__.py`
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/classify.py`
- Modify: `tinygrad/tinygrad/runtime/ops_tinytpu.py` (the `render()` method)

- [ ] **Step 1: Create `classify.py`** with a `KernelClass` enum (`ELEMENTWISE`, `GEMM`, `UNSUPPORTED`; `REDUCTION`/`BROADCAST` added in later tasks) and `classify(uops) -> KernelClass`:

```python
from enum import Enum, auto
from tinygrad.uop.ops import Ops
from tinygrad.runtime.support.tinytpu_lowering.elementwise import can_lower

class KernelClass(Enum):
  ELEMENTWISE = auto()
  GEMM = auto()
  UNSUPPORTED = auto()

def classify(uops):
  if any(u.op is Ops.WMMA for u in uops): return KernelClass.GEMM
  if can_lower(uops): return KernelClass.ELEMENTWISE
  return KernelClass.UNSUPPORTED
```

- [ ] **Step 2: Wire `render()`** so the elementwise branch goes through `classify`. `render()` calls `classify`; on `ELEMENTWISE` it calls `lower_kernel`; on `GEMM`/`UNSUPPORTED` it falls to the existing `_render_sxu_program` path (still holding the not-yet-migrated structural recognizers).

- [ ] **Step 3: Run the full suite**; Task 0 diff.
Expected: `88 failed, 810 passed`, empty diff.

- [ ] **Step 4: Commit.**

```
tinytpu: add kernel classifier, route elementwise through it
```

---

## Task 3: Reduction lowerer

**Files:**
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/reduction.py`
- Modify: `.../tinytpu_lowering/classify.py`, `.../__init__.py`
- Modify: `tinygrad/tinygrad/runtime/ops_tinytpu.py` (delete 3 recognizers + dispatch lines)

**Reference:** `_render_reduction_sxu_program`, `_render_rowreduce_sxu_program`, `_render_colreduce_sxu_program` are the behavioral spec. Design §7.

- [ ] **Step 1: Add `is_reduction(uops)`** to `reduction.py` — positive predicate: exactly two `PtrDType` params, output size < the other param's size, the single STORE's value is an associative-op tree (`ADD`/`MAX`/`MUL`/`XOR`) over loads of the input, no transcendental/`WHERE`/compare ops on the data path. Add `REDUCTION` to `KernelClass` and `classify` (checked before `ELEMENTWISE`).

- [ ] **Step 2: Implement `lower_reduction(uops) -> dict`** emitting an `SXU_PROGRAM` descriptor. Port the legacy logic structurally: classify reduce op + axis (scalar `O==1` / row / col), emit `VPU_*_REDUCE_TILE` or `_COL` per tile with identity padding (`0`/`-inf`/`+inf`/`1.0`), multi-tile combine, post-op folding. Reuse `TpuInst`/`TpuKernel` from `.common`.

- [ ] **Step 3: Verify reductions route to the new lowerer**

Run: `PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py::TestTinyTPUBackend -q -k "sum or reduce or _max or _min or prod" `
Expected: same pass/fail as base for those tests.

- [ ] **Step 4: Delete** `_render_reduction_sxu_program`, `_render_rowreduce_sxu_program`, `_render_colreduce_sxu_program` and their three dispatch-chain lines in `_render_sxu_program`.

- [ ] **Step 5: Run the full suite**; Task 0 diff.
Expected: empty diff (no regression). Pass count may rise if a legacy reduction bug is fixed.

- [ ] **Step 6: Commit.**

```
tinytpu: reduction lowerer, delete reduction recognizers
```

---

## Task 4: Broadcast lowerer

**Files:**
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/broadcast.py`
- Modify: `.../classify.py`, `.../__init__.py`, `ops_tinytpu.py`

**Reference:** `_render_rowbc_sxu_program`, `_render_colbc_sxu_program`, `_render_colbc_where_sxu_program`. Design §8.

- [ ] **Step 1: Add `is_broadcast(uops)`** — a smaller operand replicated along an axis into a larger output; add `BROADCAST` to `KernelClass`/`classify`.

- [ ] **Step 2: Implement `lower_broadcast(uops) -> dict`** — classify the replication axis (scalar/row/col) from the operand's index relationship to the output ranges; emit `BROADCAST_SCALAR`/`BROADCAST_ROW`/`BROADCAST_COL`; for the `colbc_where` shape, emit the broadcast then defer the `WHERE` to the elementwise emit path.

- [ ] **Step 3: Verify** with `-k "broadcast or bcast or rowbc or colbc"`; same pass/fail as base.

- [ ] **Step 4: Delete** the three broadcast recognizers + dispatch lines.

- [ ] **Step 5: Run the full suite**; Task 0 diff — empty.

- [ ] **Step 6: Commit.**

```
tinytpu: broadcast lowerer, delete broadcast recognizers
```

---

## Task 5: Trivial kernels — cast, copy, const-fill

**Files:**
- Modify: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/elementwise.py`
- Modify: `.../classify.py` (if needed), `ops_tinytpu.py`

**Reference:** `_render_cast_sxu_program`, `_render_copy_sxu_program`, `_render_const_fill_sxu_program`.

- [ ] **Step 1: Extend `can_lower`/`lower_kernel`** so a kernel whose store value is a bare `LOAD` (copy), a bare `CONST` (const-fill), or a `CAST` chain (cast) is accepted. `_data_dag` already treats `LOAD`/`CONST` as leaves; allow a degenerate DAG with no interior node — emit `LOAD`→`STORE` (copy) or `CONST`-tile→`STORE` (fill). For cast, handle `I2F`/`F2I` (opcodes already in `_VPU`) as interior unary ops; keep the bool→int transparency in `_canon`.

- [ ] **Step 2: Verify** with `-k "cast or copy or contiguous or fill or expand"`; same pass/fail as base. Check `_render_cast` for any dtype-size *repacking* logic — if a repacking form exists, keep that narrow case and note it; only the per-element converts move to the walker.

- [ ] **Step 3: Delete** `_render_copy_sxu_program`, `_render_const_fill_sxu_program`, and the cast recognizer's now-dead cases + dispatch lines.

- [ ] **Step 4: Run the full suite**; Task 0 diff — empty.

- [ ] **Step 5: Commit.**

```
tinytpu: fold cast/copy/const-fill into the elementwise walker
```

---

## Task 6: Relocate GEMM into the package

**Files:**
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/gemm.py`
- Modify: `.../classify.py`, `.../__init__.py`, `ops_tinytpu.py`

**Reference:** the `WMMA` branch of `_render_sxu_program`, `_render_gemm_fallback_sxu_program`, `_generate_gemm_sxu_instructions`, `_extract_wmma_epilogue`, `_apply_gemm_epilogue`, `_infer_tiling`. Design §9.

- [ ] **Step 1: Move** the WMMA lowering path and `_render_gemm_fallback_sxu_program` into `gemm.py` as `lower_gemm(uops) -> dict`, carrying along the GEMM-only helpers they call. Do not change the tiling/PSUM/epilogue logic — relocate and adjust imports only.

- [ ] **Step 2: Wire `classify`** so `GEMM` dispatches to `lower_gemm`; `render()` calls it.

- [ ] **Step 3: Verify** with `-k "gemm or wmma or matmul or psum"`; identical pass/fail to base.

- [ ] **Step 4: Delete** the WMMA branch and `_render_gemm_fallback_sxu_program` from `ops_tinytpu.py`; check each moved helper for remaining `ops_tinytpu.py` callers before deleting it there.

- [ ] **Step 5: Run the full suite**; Task 0 diff — empty (behavior-neutral relocation).

- [ ] **Step 6: Commit.**

```
tinytpu: relocate GEMM lowering into the tinytpu_lowering package
```

---

## Task 7: Movement spike — pad/transpose

**Files:**
- None (investigation; produces a decision recorded in `TODO.md`).

- [ ] **Step 1: Dump the UOp graphs** for representative `pad` and `transpose` kernels (e.g. `Tensor(...).pad(...)`, `Tensor(...).transpose()` on `TINYTPU`), using the `get_program` pattern from `scripts/prototype_uop_walker.py`.

- [ ] **Step 2: Inspect `tinygrad/tinygrad/codegen/rangeify.py`** — determine whether movement ops are already lowered to index arithmetic for other backends and whether the TinyTPU renderer can consume that form (i.e. whether `pad`/`transpose` can stop producing a distinct render kernel).

- [ ] **Step 3: Record the decision** in `TODO.md`: either "movement consumable upstream — recognizers delete with no replacement" or "movement stays renderer-side — `movement.py` needed". Commit the `TODO.md` note.

```
tinytpu: movement-op rangeify spike — record decision
```

---

## Task 8: Resolve movement (branch on Task 7)

**Files (branch A — upstream):**
- Modify: `tinygrad/tinygrad/runtime/ops_tinytpu.py` (delete `_render_pad`/`_render_transpose` + dispatch); possibly `tinygrad/tinygrad/codegen/rangeify.py`.

**Files (branch B — renderer-side):**
- Create: `tinygrad/tinygrad/runtime/support/tinytpu_lowering/movement.py`
- Modify: `.../classify.py`, `.../__init__.py`, `ops_tinytpu.py`.

- [ ] **Step 1 (branch A):** apply the upstream change so movement ops lower to index arithmetic for TINYTPU; delete `_render_pad_sxu_program`/`_render_transpose_sxu_program` and their dispatch lines.

- [ ] **Step 1 (branch B):** implement `is_movement`/`lower_movement` in `movement.py` — pad as a scatter into a padded tile (`PAD_FILL`/`pad_map` data-plan mode), transpose via `SXU_DISPATCH_XLU_TRANSPOSE`; wire into `classify`; delete the two recognizers.

- [ ] **Step 2: Verify** with `-k "pad or transpose or permute"`; same pass/fail as base.

- [ ] **Step 3: Run the full suite**; Task 0 diff — empty.

- [ ] **Step 4: Commit.**

```
tinytpu: resolve movement ops (<upstream|movement lowerer>)
```

---

## Task 9: Final cleanup and exit-criteria check

**Files:**
- Modify: `tinygrad/tinygrad/runtime/ops_tinytpu.py`, `TODO.md`

- [ ] **Step 1: Confirm zero recognizers remain**

Run: `grep -c "_render_.*_sxu_program(uops: list" tinygrad/tinygrad/runtime/ops_tinytpu.py`
Expected: `0`.

- [ ] **Step 2: Remove orphaned helpers** — for every remaining `_` helper in `ops_tinytpu.py`, check for callers across `ops_tinytpu.py`, `tests/`, `scripts/`; delete any with none. Keep `analyze_tinytpu_uops` (external consumer) and the instruction-encoder helpers imported by the test file.

- [ ] **Step 3: Confirm `_render_sxu_program`** is gone or reduced to the `classify`+dispatch shim; `render()` is `classify` → lowerer → descriptor.

- [ ] **Step 4: Run the full suite**; Task 0 diff — empty.

- [ ] **Step 5: Update `TODO.md`** — mark the structural slice complete; record final `ops_tinytpu.py` line count.

- [ ] **Step 6: Commit.**

```
tinytpu: complete InstSel structural slice, final cleanup
```

---

## Self-Review Notes

- **Spec coverage:** design §5 → Task 2; §6 → Task 1; §7 → Task 3; §8 → Task 4; §9 → Task 6; §10 → Tasks 7–8; trivial kernels (§6) → Task 5; §11 order preserved; §12 verification embedded in every task; §14 exit criteria → Task 9.
- **Verification:** every code task ends with the full sim suite + the Task 0 regression diff; behavior-neutral tasks (1, 6) additionally require an unchanged count.
- **Movement branch:** Task 8 explicitly branches on the Task 7 spike outcome so the plan does not block on an unknown.
