# tinygrad Version Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the vendored tinygrad submodule by 475 upstream commits (to the core tree at `5de53f78f`) while keeping the TinyTPU backend working with zero new test failures.

**Architecture:** Keep `e0c5faa26` as the submodule trunk. Build one commit on top of it whose tree is `5de53f78f`'s core plus `e0c5faa26`'s backend, ported via enum renames. History stays a single linear line; `origin/master` fast-forwards. The orphaned merge `5de53f78f` is discarded — used only as a source tree.

**Tech Stack:** Python, tinygrad UOp graph / renderer, pytest, git submodule, BSV simulator.

**Spec:** `doc/tinygrad-upgrade-spec.md`

---

## Background an engineer needs

The TinyTPU backend lives in the tinygrad submodule (`tinygrad/` in the repo root):

- `tinygrad/tinygrad/runtime/ops_tinytpu.py` — device + `TinyTPURenderer` entry point (1260 lines).
- `tinygrad/tinygrad/renderer/tinytpu/` — the InstSel lowering package: `common.py`, `classify.py`, `elementwise.py`, `reduction.py`, `broadcast.py`, `gemm.py`, `movement.py`, `__init__.py`.

Tests are in the **parent** repo: `tests/test_tinytpu_backend.py`.

Three submodule commits matter:
- `705d7e08d` — InstSel-migration baseline (lowering package at the old path `runtime/support/tinytpu_lowering/`).
- `e0c5faa26` — current submodule master = `705d7e08d` + the renderer-move (package now at `renderer/tinytpu/`). **Good backend, old tinygrad core.**
- `5de53f78f` — orphaned botched merge = `705d7e08d` + 475 upstream commits + a merge that reverted `ops_tinytpu.py` to a 471-line GEMM-only stub. **Newer tinygrad core, broken backend, still carries the old `runtime/support/tinytpu_lowering/` package, no `renderer/tinytpu/`.**

The upgrade combines `5de53f78f`'s core with `e0c5faa26`'s backend.

### The breaking changes (from the breakage analysis)

The new tinygrad renamed three `Ops` enum members. The backend references them both as `Ops.<NAME>` attributes **and** as string literals (because `op_counts = Counter(u.op.name for u in uops)` keys on the op's `.name`, and `_find_scalar_const_binary(uops, op_name)` compares `op_name` against `u.op.name`):

| Old | New | Semantics |
|---|---|---|
| `Ops.IDIV` | `Ops.CDIV` | C-style truncating division (same as old `IDIV`) |
| `Ops.MOD` | `Ops.CMOD` | C-style modulo (same as old `MOD`) |
| `Ops.VECTORIZE` | `Ops.STACK` | vectorize/pack lanes |

Everything else the backend imports (`Renderer`, `Compiled`, `Allocator`, `BufferSpec`, `Compiler`, `UOp`, `UPat`, `PatternMatcher`, `graph_rewrite`, `PtrDType`, `dtypes`, `TensorCore`) is unchanged at the same import paths.

One test uses removed internals (`tinygrad.engine.realize.get_program`, `Tensor.schedule()`, `ScheduleItem.ast`, `prog.uops`). The new model is `tinygrad.codegen.to_program(ast, renderer)` returning an `Ops.PROGRAM` UOp; the linearized uops live at `prog.src[2].src` (the `Ops.LINEAR` child). Confirmed by upstream's own `test/backend/test_linearizer.py`, which does `to_program(...).src[2].src` and gets a kernel AST via `tensor.schedule_linear().src[-1].src[0]`.

### Verification environment

Run everything from the repo root `/Users/hanwang/p/tinytpu`. Use the repo-local venv:

```bash
cd /Users/hanwang/p/tinytpu
test -d .venv || (python3 -m venv .venv && .venv/bin/pip install --quiet pytest numpy)
```

Run tests with `PYTHONPATH=tinygrad .venv/bin/python3 -m pytest ...`.

---

## File Structure

| File | Repo | Change |
|---|---|---|
| `tinygrad/tinygrad/runtime/ops_tinytpu.py` | submodule | 7 rename edits |
| `tinygrad/tinygrad/renderer/tinytpu/common.py` | submodule | 5 rename edits + 1 docstring |
| `tinygrad/tinygrad/renderer/tinytpu/movement.py` | submodule | 4 rename edits |
| `tinygrad/` core (everything else) | submodule | swapped wholesale to `5de53f78f` |
| `tests/test_tinytpu_backend.py` | parent | rewrite one test (`test_instsel_walker_owns_int_elementwise`) |
| `tinygrad` submodule pointer | parent | advanced to the new submodule commit |

---

## Task 1: Capture the pre-upgrade baseline

**Files:** none modified. Produces `/Users/hanwang/p/tinytpu/upgrade-baseline-failures.txt`.

The acceptance gate is "the same failures, zero new ones." That requires the exact set of currently-failing test ids, captured **before** any change, while the submodule is still at `e0c5faa26`.

- [ ] **Step 1: Confirm the starting state**

```bash
cd /Users/hanwang/p/tinytpu
git -C tinygrad rev-parse HEAD          # expect e0c5faa26b0db05a8f59c0095ff97fceadf944e2
git -C tinygrad status --short          # expect clean
test -d .venv || (python3 -m venv .venv && .venv/bin/pip install --quiet pytest numpy)
```

- [ ] **Step 2: Run the full backend suite and record failing test ids**

```bash
cd /Users/hanwang/p/tinytpu
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py -q --tb=no -rf > /tmp/tinytpu-baseline-run.txt 2>&1; echo "pytest exit: $?"
grep '^FAILED ' /tmp/tinytpu-baseline-run.txt | cut -d' ' -f2 | sort > /Users/hanwang/p/tinytpu/upgrade-baseline-failures.txt
grep -cE '^FAILED ' /tmp/tinytpu-baseline-run.txt
tail -1 /tmp/tinytpu-baseline-run.txt
```

Expected: ≈88 failed, ≈810 passed (the 88 are the pre-existing broken BSV `EXP2`/`LOG2`/`SIN` hardware). The exact count is whatever this run reports — that becomes the baseline. `upgrade-baseline-failures.txt` now holds one sorted test id per line.

- [ ] **Step 3: Sanity-check the baseline file**

```bash
wc -l < /Users/hanwang/p/tinytpu/upgrade-baseline-failures.txt   # should match the FAILED count above, and be > 0
```

If the file is empty or the count is 0, something is wrong (the suite did not run) — stop and investigate before continuing.

**No commit.** `upgrade-baseline-failures.txt` is an untracked working artifact consumed by Task 4.

---

## Task 2: Upgrade the submodule — swap core, restore backend, apply renames

**Files (in the submodule `/Users/hanwang/p/tinytpu/tinygrad`):**
- Whole core tree: swapped to `5de53f78f`
- Modify: `tinygrad/runtime/ops_tinytpu.py`
- Modify: `tinygrad/renderer/tinytpu/common.py`
- Modify: `tinygrad/renderer/tinytpu/movement.py`
- Remove: `tinygrad/runtime/support/tinytpu_lowering/` (stale pre-move package)

The intermediate state between the tree swap and the renames does not import (the backend references `Ops.IDIV` etc., which no longer exist). The whole task is therefore **one submodule commit** made at the end.

- [ ] **Step 1: Build the upgraded tree**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad
git status --short                       # must be clean
git read-tree -m -u 5de53f78f            # working tree + index now match 5de53f78f; HEAD stays at e0c5faa26
git checkout e0c5faa26 -- tinygrad/runtime/ops_tinytpu.py tinygrad/renderer/tinytpu
git rm -r --quiet tinygrad/runtime/support/tinytpu_lowering
find . -name __pycache__ -type d -exec rm -rf {} +    # drop stale bytecode
```

- [ ] **Step 2: Verify the tree is exactly "5de53f78f core + e0c5faa26 backend"**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad
# Backend files must equal e0c5faa26's backend exactly (no diff yet — renames come next):
git diff --stat e0c5faa26 -- tinygrad/runtime/ops_tinytpu.py tinygrad/renderer/tinytpu
# Core files must equal 5de53f78f's core exactly:
git diff --stat 5de53f78f -- . ':!tinygrad/runtime/ops_tinytpu.py' ':!tinygrad/renderer/tinytpu' ':!tinygrad/runtime/support/tinytpu_lowering'
```

Both commands must print **nothing** (empty diff). If either prints output, the tree is wrong — stop and fix Step 1 before continuing.

- [ ] **Step 3: Rename in `tinygrad/runtime/ops_tinytpu.py`**

Apply these exact replacements:

```python
# line ~246
-    _has_complex_op = any(op_counts.get(x, 0) for x in ("IDIV", "MOD", "RECIP"))
+    _has_complex_op = any(op_counts.get(x, 0) for x in ("CDIV", "CMOD", "RECIP"))

# line ~255
-    if len(params) in {2, 3} and divmod_pattern is not None and divmod_pattern[0] == "IDIV":
+    if len(params) in {2, 3} and divmod_pattern is not None and divmod_pattern[0] == "CDIV":

# line ~277
-    elif len(params) in {2, 3} and divmod_pattern is not None and divmod_pattern[0] == "MOD":
+    elif len(params) in {2, 3} and divmod_pattern is not None and divmod_pattern[0] == "CMOD":

# lines ~794-797 (function _classify_divmod_pattern)
-    if any(u.op is Ops.IDIV for u in uops) and all(v.op is Ops.WHERE for v in values):
-        return "IDIV", _find_scalar_const_binary(uops, "IDIV")
-    if any(u.op is Ops.MOD for u in uops) and all(v.op is Ops.ADD for v in values):
-        return "MOD", _find_scalar_const_binary(uops, "MOD")
+    if any(u.op is Ops.CDIV for u in uops) and all(v.op is Ops.WHERE for v in values):
+        return "CDIV", _find_scalar_const_binary(uops, "CDIV")
+    if any(u.op is Ops.CMOD for u in uops) and all(v.op is Ops.ADD for v in values):
+        return "CMOD", _find_scalar_const_binary(uops, "CMOD")
```

The string-literal changes are required: `op_counts` keys on `u.op.name`, and `_find_scalar_const_binary` compares its `op_name` argument against `u.op.name`. The `divmod_pattern` label (`"CDIV"`/`"CMOD"`) is internal and must match the comparisons at lines ~255/~277, which is why all four move together.

- [ ] **Step 4: Rename in `tinygrad/renderer/tinytpu/common.py`**

```python
# line ~42 (opcode table)
-               Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.IDIV: "DIV"}
+               Ops.SHL: "SHL", Ops.SHR: "SHR", Ops.CDIV: "DIV"}

# line ~197 (docstring of _store_lanes — keep it accurate)
-  Float kernels store a VECTORIZE of N lane computations; int kernels store
+  Float kernels store a STACK of N lane computations; int kernels store

# line ~201
-  return list(v.src) if v.op is Ops.VECTORIZE else [v]
+  return list(v.src) if v.op is Ops.STACK else [v]

# line ~214 (inside _expand_mod)
-  return a.alu(Ops.SUB, a.alu(Ops.IDIV, b).alu(Ops.MUL, b))
+  return a.alu(Ops.SUB, a.alu(Ops.CDIV, b).alu(Ops.MUL, b))

# line ~218 (_INSTSEL pattern)
-  (UPat(Ops.MOD, name="x"), _expand_mod),
+  (UPat(Ops.CMOD, name="x"), _expand_mod),

# line ~220
-_INSTSEL_OPS = (Ops.SQRT, Ops.MOD)
+_INSTSEL_OPS = (Ops.SQRT, Ops.CMOD)
```

- [ ] **Step 5: Rename in `tinygrad/renderer/tinytpu/movement.py`**

There are two identical occurrences of the op-name tuple (lines ~53 and ~140) — change **both**:

```python
-  for n in ("WHERE", "MOD", "RECIP", "RECIPROCAL", "TRUNC", "MULACC",
+  for n in ("WHERE", "CMOD", "RECIP", "RECIPROCAL", "TRUNC", "MULACC",
```

```python
# line ~228
-                 and op_counts.get("VECTORIZE", 0) == 1 and op_counts.get("RANGE", 0) == 1
+                 and op_counts.get("STACK", 0) == 1 and op_counts.get("RANGE", 0) == 1

# line ~262
-    if val.op is Ops.VECTORIZE:
+    if val.op is Ops.STACK:
```

The comments at lines ~224 and ~249 mention "VECTORIZE" prose — update them to "STACK" so the comments stay accurate.

- [ ] **Step 6: Verify no stale references remain**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad
grep -rn "IDIV\|Ops\.MOD\b\|VECTORIZE" tinygrad/runtime/ops_tinytpu.py tinygrad/renderer/tinytpu/
```

Expected: **no output**. Any hit is a missed rename — fix it. (Note `RECIPROCAL` is fine and untouched; the pattern above does not match it.)

- [ ] **Step 7: Import + run smoke test (int and float paths)**

```bash
cd /Users/hanwang/p/tinytpu
PYTHONPATH=tinygrad .venv/bin/python3 -c "
from tinygrad import Tensor, Device
from tinygrad.uop.ops import Ops
assert hasattr(Ops, 'CDIV') and hasattr(Ops, 'CMOD') and hasattr(Ops, 'STACK')
assert not hasattr(Ops, 'IDIV') and not hasattr(Ops, 'MOD') and not hasattr(Ops, 'VECTORIZE')
import tinygrad.runtime.ops_tinytpu
from tinygrad.renderer.tinytpu import classify, can_lower, lower_kernel
# int elementwise: exercises the int unrolled-store path
a = Tensor([1,2,3,4], dtype='int32', device='TINYTPU')
b = Tensor([5,6,7,8], dtype='int32', device='TINYTPU')
print('int add result:', (a + b).numpy())
# float elementwise: float kernels store a STACK of lane computations, so
# this exercises _store_lanes() and confirms Ops.STACK still exposes its
# lanes via .src (the VECTORIZE->STACK rename in common.py:201).
xf = Tensor([1.0,2.0,3.0,4.0], dtype='float32', device='TINYTPU')
yf = Tensor([0.5,0.5,0.5,0.5], dtype='float32', device='TINYTPU')
print('float add result:', (xf + yf).numpy())
print('IMPORT+RUN OK')
"
```

Expected: prints `int add result: [ 6  8 10 12]`, `float add result: [1.5 2.5 3.5 4.5]`, and `IMPORT+RUN OK`. An `AttributeError` on `Ops.IDIV`/`MOD`/`VECTORIZE` means a rename was missed; an error inside `_store_lanes` means the `STACK` rename or its `.src` assumption is wrong.

- [ ] **Step 8: Validate div/mod op mapping with the divmod test subset**

```bash
cd /Users/hanwang/p/tinytpu
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py -q --tb=short -k "div or mod or Div or Mod" 2>&1; echo "exit: $?"
```

This is a focused gate on the rename hypothesis. The renames assume the new tinygrad's `//` and `%` still emit C-style div/mod (now `Ops.CDIV`/`Ops.CMOD`).

- If these tests pass at the **same rate as the baseline** for matching test ids (compare against `upgrade-baseline-failures.txt`), the mapping is correct — continue.
- If div/mod tests that passed in the baseline now fail, the new tinygrad likely emits `Ops.FLOORDIV`/`Ops.FLOORMOD` for `//`/`%` instead. **Report this as BLOCKED** with the actual op observed — run:
  ```bash
  PYTHONPATH=tinygrad .venv/bin/python3 -c "
  from tinygrad import Tensor
  from tinygrad.uop.ops import Ops
  a = Tensor([10,20,30,40], dtype='int32', device='TINYTPU')
  lin = (a % 7).schedule_linear()
  print('schedule_linear src ops:', [s.op for s in lin.src])
  "
  ```
  and stop — the backend would then need `FLOORDIV`/`FLOORMOD` handling, which is a scope change for the human to decide.

- [ ] **Step 9: Check `_expand_mod` reachability**

`_expand_mod` (`common.py`) rewrites `CMOD → SUB(a, MUL(CDIV(a,b), b))`. The identity is correct for C-style div/mod. Check whether a `CMOD` UOp actually reaches the backend's renderer:

```bash
cd /Users/hanwang/p/tinytpu
PYTHONPATH=tinygrad .venv/bin/python3 -c "
from tinygrad import Tensor, Device
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops
a = Tensor([10,20,30,40], dtype='int32', device='TINYTPU')
lin = (a % 7).schedule_linear()
asts = [s.src[0] for s in lin.src if s.src and s.src[0].op is Ops.SINK]
prog = to_program(asts[-1], Device['TINYTPU'].renderer)
uops = list(prog.src[2].src)
print('CMOD reaches renderer:', any(u.op is Ops.CMOD for u in uops))
"
```

- If it prints `True`: `_expand_mod` is live — keep it as renamed.
- If it prints `False`: the upstream pipeline decomposes `CMOD` before render, so `_expand_mod` is dead code. Per the project rule (no dead code), remove all three pieces in `common.py`: the `_expand_mod` function, its entry `(UPat(Ops.CMOD, name="x"), _expand_mod),` in `_INSTSEL`, and `Ops.CMOD` from `_INSTSEL_OPS` (leaving `_INSTSEL_OPS = (Ops.SQRT,)`). Then re-run Step 7's import smoke test to confirm nothing else referenced them.

- [ ] **Step 10: Commit the submodule**

```bash
cd /Users/hanwang/p/tinytpu/tinygrad
git add -A
git status --short      # review: ops_tinytpu.py + renderer/tinytpu/* modified/added, runtime/support/tinytpu_lowering/* deleted, core swapped
git commit -m "tinytpu: bump vendored tinygrad core (+475 commits) and port backend

Swap the tinygrad core to the 5de53f78f-era version and port the
TinyTPU backend: Ops.IDIV->CDIV, Ops.MOD->CMOD, Ops.VECTORIZE->STACK
(enum refs and the op-name string mirrors). Single linear history on
top of e0c5faa26."
git log --oneline -1
git log --oneline -1 --format='parent: %p'    # parent must be e0c5faa26
```

The new commit's parent **must** be `e0c5faa26` (linear history; `origin/master` will fast-forward).

---

## Task 3: Rewrite the broken test and bump the submodule pointer

**Files:**
- Modify: `tests/test_tinytpu_backend.py` (the method `test_instsel_walker_owns_int_elementwise`, ~lines 4231-4246)
- Modify: `tinygrad` submodule pointer (parent repo)

- [ ] **Step 1: Replace the test method**

Find `def test_instsel_walker_owns_int_elementwise` in `tests/test_tinytpu_backend.py`. Replace the whole method body with:

```python
  def test_instsel_walker_owns_int_elementwise(self):
    # doc/plan-tinytpu-instsel.md: int32/bool elementwise kernels are lowered
    # by the UOp-walking InstSel pass, not the legacy _render_* recognizers.
    from tinygrad import Device
    from tinygrad.codegen import to_program
    from tinygrad.uop.ops import Ops
    from tinygrad.renderer.tinytpu import can_lower, lower_kernel
    a = Tensor([1, 2, 3, 4], dtype="int32", device="TINYTPU")
    b = Tensor([5, 6, 7, 8], dtype="int32", device="TINYTPU")
    # schedule_linear() returns a linear UOp; its items wrap kernel ASTs.
    # The compute kernel's AST is an Ops.SINK (host->device COPYs are not).
    linear = (a + b).schedule_linear()
    kernel_asts = [s.src[0] for s in linear.src if s.src and s.src[0].op is Ops.SINK]
    self.assertTrue(kernel_asts, f"no compute kernel scheduled; got {[s.op for s in linear.src]}")
    prog = to_program(kernel_asts[-1], Device["TINYTPU"].renderer)
    # to_program returns an Ops.PROGRAM UOp; src[2] is the Ops.LINEAR child,
    # whose .src is the tuple of linearized uops.
    uops = list(prog.src[2].src)
    self.assertTrue(can_lower(uops), "InstSel walker did not claim int elementwise add")
    desc = lower_kernel(uops)
    self.assertEqual(desc["op"], "SXU_PROGRAM")
    self.assertTrue(any(i.startswith("2 2 ") for i in desc["instructions"]), desc["instructions"])
```

This replaces the removed `get_program`/`schedule()`/`ScheduleItem.ast`/`prog.uops` internals with the current `to_program` + `Ops.PROGRAM`/`Ops.LINEAR` model (the same pattern upstream uses in `test/backend/test_linearizer.py`).

- [ ] **Step 2: Run the rewritten test**

```bash
cd /Users/hanwang/p/tinytpu
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest "tests/test_tinytpu_backend.py" -q --tb=short -k "test_instsel_walker_owns_int_elementwise" 2>&1; echo "exit: $?"
```

Expected: 1 passed.

If it fails because `prog.src[2]` is not the `Ops.LINEAR` node or `linear.src` items are shaped differently, inspect the actual structure and adapt the two extraction lines:
```bash
PYTHONPATH=tinygrad .venv/bin/python3 -c "
from tinygrad import Tensor, Device
from tinygrad.codegen import to_program
from tinygrad.uop.ops import Ops
a = Tensor([1,2,3,4], dtype='int32', device='TINYTPU')
b = Tensor([5,6,7,8], dtype='int32', device='TINYTPU')
linear = (a + b).schedule_linear()
print('linear.src item ops:', [s.op for s in linear.src])
asts = [s.src[0] for s in linear.src if s.src and s.src[0].op is Ops.SINK]
prog = to_program(asts[-1], Device['TINYTPU'].renderer)
print('prog.src ops:', [s.op for s in prog.src])
"
```
Use the printed structure to locate the `Ops.LINEAR` child (`next(s for s in prog.src if s.op is Ops.LINEAR)` is a structure-independent fallback for the `prog.src[2]` access).

- [ ] **Step 3: Commit the pointer bump and the test**

```bash
cd /Users/hanwang/p/tinytpu
git add tinygrad tests/test_tinytpu_backend.py
git status --short      # expect: modified tinygrad (pointer), modified tests/test_tinytpu_backend.py
git commit -m "tinytpu: bump tinygrad submodule to upgraded core; port instsel test

Advance the tinygrad submodule pointer to the upgraded commit and port
test_instsel_walker_owns_int_elementwise from the removed get_program/
schedule() internals to the current to_program API."
git log --oneline -1
```

---

## Task 4: Full verification — zero regressions

**Files:** none modified. Compares against `upgrade-baseline-failures.txt` from Task 1.

- [ ] **Step 1: Run the full backend suite**

```bash
cd /Users/hanwang/p/tinytpu
find tinygrad -name __pycache__ -type d -exec rm -rf {} +
PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py -q --tb=short -rf > /tmp/tinytpu-upgrade-run.txt 2>&1; echo "pytest exit: $?"
tail -1 /tmp/tinytpu-upgrade-run.txt
grep '^FAILED ' /tmp/tinytpu-upgrade-run.txt | cut -d' ' -f2 | sort > /tmp/tinytpu-upgrade-failures.txt
wc -l < /tmp/tinytpu-upgrade-failures.txt
```

- [ ] **Step 2: Diff the failure set against the baseline**

```bash
cd /Users/hanwang/p/tinytpu
echo "=== NEW failures (in upgrade, not in baseline) — must be empty ==="
comm -13 upgrade-baseline-failures.txt /tmp/tinytpu-upgrade-failures.txt
echo "=== NEWLY FIXED (in baseline, not in upgrade) — informational ==="
comm -23 upgrade-baseline-failures.txt /tmp/tinytpu-upgrade-failures.txt
```

**Acceptance gate:** the "NEW failures" list must be **empty**. Any test id there is a regression introduced by the upgrade — it must be fixed before the task is complete. If a regression appears, dispatch a fix (the most likely cause is a missed rename or a `to_program`/structure assumption); do not accept it.

The rewritten `test_instsel_walker_owns_int_elementwise` must be in neither list (it passes in both — it passed before the upgrade as the old version, and passes now as the rewritten version). If it appears in "NEW failures", the Task 3 rewrite is wrong.

- [ ] **Step 3: Confirm the sim-backed GEMM tests passed**

```bash
cd /Users/hanwang/p/tinytpu
grep -E "TestTinyTPUBackendGemm" /tmp/tinytpu-upgrade-run.txt | grep -E "FAILED|ERROR" || echo "TestTinyTPUBackendGemm: no failures"
```

`TestTinyTPUBackendGemm` runs through the real BSV simulator (`build/mkTbTinyTPURuntime.bexe`). It is part of the full-file run in Step 1; this step just confirms none of its tests regressed. Expected: `TestTinyTPUBackendGemm: no failures`.

- [ ] **Step 4: End-to-end co-sim spot check**

```bash
cd /Users/hanwang/p/tinytpu
.venv/bin/python3 scripts/test_cosim.py 2>&1; echo "exit: $?"
```

Expected: exit 0. This exercises the backend end-to-end through the simulator and confirms the submodule swap did not break the cosim path. (The BSV `make test-*` unit tests are hardware-side and tinygrad-version-independent, so they are not a gate for this upgrade; the cosim run is the meaningful integration check.)

- [ ] **Step 5: Final state check**

```bash
cd /Users/hanwang/p/tinytpu
git -C tinygrad log --oneline -2
git -C tinygrad log -1 --format='submodule HEAD parent: %p (expect e0c5faa26...)'
git log --oneline -2
git status --short      # expect clean (upgrade-baseline-failures.txt is the only untracked artifact)
```

Confirm: the submodule has one new commit whose parent is `e0c5faa26` (linear history), and the parent repo has one new commit advancing the pointer.

**No commit.** Verification only. Do not push — pushing happens only on explicit user request (see the submodule push workflow).

---

## Notes for the executor

- **Do not push.** Both commits stay local until the user asks. The submodule push workflow (submodule first, then parent) is in the project memory.
- **Submodule commit order:** the submodule commit (Task 2) must exist before the parent pointer commit (Task 3) — Task 3 stages `tinygrad` which records the submodule's new HEAD.
- **BLOCKED conditions:** Task 2 Step 8 (div/mod tests regress → wrong op mapping) is the one place where the plan's assumption can fail. If it triggers, stop and surface it — do not guess at `FLOORDIV`/`FLOORMOD` handling without human direction.
- The `upgrade-baseline-failures.txt` artifact must survive from Task 1 to Task 4. It is untracked; do not `git clean` it away.
