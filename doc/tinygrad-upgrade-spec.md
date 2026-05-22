# TinyTPU Backend — tinygrad Version Upgrade

**Status:** design approved 2026-05-21

## Goal

Upgrade the vendored tinygrad submodule from the pre-bump version (`705d7e08d`,
the InstSel-migration baseline) to the bumped version already vendored in the
orphaned merge `5de53f78f` — 475 upstream commits ahead — while keeping the
TinyTPU backend fully working, with no new test failures.

## Background

The InstSel migration is complete: `ops_tinytpu.py` was reduced from 6984 to
1260 lines, and the lowering logic lives in the `tinygrad/renderer/tinytpu/`
package (`common.py`, `classify.py`, `elementwise.py`, `reduction.py`,
`broadcast.py`, `gemm.py`, `movement.py`). The verified migrated state is
submodule commit `e0c5faa26` (migration final `705d7e08d` + the renderer-move
commit).

A `hanw.refactor` merge once pulled the bumped tinygrad in but botched the
`ops_tinytpu.py` conflict (reverted it to the old GEMM-only renderer). That
merge is the orphaned commit `5de53f78f`. So `5de53f78f` contains the correct
**core tinygrad** of the target version, but a **broken backend** and no
`renderer/tinytpu/` package (the renderer-move commit is not in its history).

This upgrade was deliberately deferred after the migration; this spec executes it.

## Target version

The tree at commit `5de53f78f` (already vendored, breakage fully
characterized). Not the latest upstream — that would require re-characterizing
breakage against newer churn.

## Breakage analysis

The 475-commit core diff is large in absolute terms (≈133K line insertions),
but the backend's API contact surface barely moved.

### Compatible — no change needed

- `Renderer` base class (`tinygrad/renderer/__init__.py`): purely additive
  (`asm`, `supported_dtypes` added with defaults). `render(uops) -> str`
  contract unchanged. Attributes the backend sets (`compiler`, `has_local`,
  `code_for_op`, `tensor_cores`, etc.) all still exist.
- `Compiled`, `Allocator`, `BufferSpec`, `Compiler` (`tinygrad/device.py`):
  constructor signatures identical. New `Allocator._map` is a latent gap only
  for multi-device transfer; TinyTPU tests are single-device.
- `graph_rewrite`, `UPat`, `PatternMatcher` (`tinygrad.uop.ops`): signatures
  and semantics unchanged.
- All imported symbols (`Ops`, `UOp`, `PtrDType`, `dtypes`, `TensorCore`, …)
  still exist at the same import paths.
- ≈99% of tests drive the backend through the public `Tensor` API
  (`Tensor(...)`, `@`, `.relu()`, `.sum()`, `.realize()`, `.numpy()`), which
  is unchanged.

### Breaking — must change

**1. Ops enum renames (8 backend sites, 3 files).** Upstream split division
into C-style and floor variants and renamed the vectorize op:

- `Ops.IDIV` → `Ops.CDIV` (C-style truncating division — same semantics as
  old `IDIV`)
- `Ops.MOD` → `Ops.CMOD` (C-style modulo — same semantics as old `MOD`)
- `Ops.VECTORIZE` → `Ops.STACK`

Sites (line numbers approximate — verify by grep during implementation):

| Symbol | File |
|---|---|
| `Ops.IDIV` | `tinygrad/runtime/ops_tinytpu.py` (~794) |
| `Ops.MOD` | `tinygrad/runtime/ops_tinytpu.py` (~796) |
| `Ops.IDIV` ×2 | `tinygrad/renderer/tinytpu/common.py` (~42 opcode table, ~214 `_expand_mod`) |
| `Ops.MOD` ×2 | `tinygrad/renderer/tinytpu/common.py` (~218 `UPat`, ~220 `_INSTSEL_OPS`) |
| `Ops.VECTORIZE` | `tinygrad/renderer/tinytpu/common.py` (~201) |
| `Ops.VECTORIZE` | `tinygrad/renderer/tinytpu/movement.py` (~262) |

**2. One test uses removed internals.** `tests/test_tinytpu_backend.py:4234-4244`
uses `tinygrad.engine.realize.get_program`, `Tensor.schedule()`,
`ScheduleItem.ast`, and `prog.uops` — all removed. The new model:
`tinygrad.codegen.to_program(ast, renderer)` returns an `Ops.PROGRAM` UOp whose
linearized uops live in the `Ops.LINEAR` child, not a `.uops` attribute.

## Approach

Build on `5de53f78f` and restore the backend, rather than re-doing the merge.
`5de53f78f` is already a proven, buildable tree containing the exact target
core. Hand conflict-resolution is what got botched before; this avoids it.

### Work items

**1. Construct the upgraded tree.**
In the submodule:
- `git checkout 5de53f78f` (detached HEAD).
- Force-restore the backend from `e0c5faa26`:
  `git checkout e0c5faa26 -- tinygrad/runtime/ops_tinytpu.py tinygrad/renderer/tinytpu`.
- Remove the stale pre-move package location if present in `5de53f78f`:
  `git rm -r tinygrad/runtime/support/tinytpu_lowering`.
- Cross-check against `git diff --name-status 705d7e08d e0c5faa26` so that
  every backend file the migration created/moved/deleted is accounted for —
  no migration artifact is left behind and no stale file survives.

**2. Apply the 8 enum renames.** Edit the 3 files. Verify with
`grep -rn 'Ops\.\(IDIV\|MOD\|VECTORIZE\)' tinygrad/runtime/ops_tinytpu.py tinygrad/renderer/tinytpu`
returning zero hits.

**3. Validate the two semantic risks.**
- `_expand_mod` (`common.py`) rewrites modulo as
  `CMOD → SUB(a, MUL(CDIV(a,b), b))`. This identity holds for C-style div/mod,
  so the renamed rewrite is correct. Check whether upstream now auto-decomposes
  `CMOD`; if the backend pattern is unreachable dead code, remove it (the
  project rule is no fallbacks, no dead code).
- Confirm `STACK` UOps still expose their element list via `.src` (the
  `movement.py` and `common.py` sites assume this).

**4. Rewrite the broken test** (`tests/test_tinytpu_backend.py:4234-4244`).
Port to `tinygrad.codegen.to_program`; extract the uops from the new
`Ops.PROGRAM`/`Ops.LINEAR` UOp structure. Keep the same assertions
(`can_lower(uops)`, `lower_kernel(uops)` produces a descriptor).

**5. Commit.** One submodule commit on top of `5de53f78f`, then one
parent-repo commit advancing the submodule pointer. Follow the submodule
workflow (commit inside submodule first, then parent pointer).

## Verification

Run from the repo root using the repo-local venv (`.venv/bin/python3`).

1. **Capture the baseline failure set first.** Before any change, on
   `e0c5faa26`, run the full `tests/test_tinytpu_backend.py` and record the
   exact set of failing test ids. Expected: 810 pass / 88 fail — the 88 are
   the pre-existing broken BSV `EXP2`/`LOG2`/`SIN` hardware.
2. **After the upgrade**, run the full `tests/test_tinytpu_backend.py` again.
   Gate: the failure set is **exactly the same 88 tests** (same ids, not just
   the same count) and **zero new failures**; the rewritten test at item 4
   passes.
3. Run the sim-backed `TestTinyTPUBackendGemm` (real BSV simulator at
   `build/mkTbTinyTPURuntime.bexe`) — must pass; tinygrad-version-independent,
   so this confirms no harness breakage.
4. Spot-check `make test-<unit>` and `python3 scripts/test_cosim.py` — these
   are hardware-side and version-independent; they confirm nothing in the
   submodule swap broke the build/cosim path.

A regression is any test that passed on `e0c5faa26` and fails after the
upgrade. Zero regressions is the acceptance criterion.

## Risks

1. **The test rewrite (item 4)** is the only genuinely non-mechanical change.
   The new `Ops.PROGRAM`/`Ops.LINEAR` UOp structure must be inspected directly
   to find where the linearized uops live.
2. **`_expand_mod` semantics (item 3)** — must confirm `CDIV`/`CMOD` are the
   C-style (truncating) variants and that the rewrite is still reachable.
3. **`STACK` `.src` shape (item 3)** — verify the renamed op exposes elements
   the same way `VECTORIZE` did.
4. **Stale migration artifacts** — `5de53f78f` predates the renderer-move, so
   the pre-move `runtime/support/tinytpu_lowering/` directory must be removed
   explicitly; the `git diff --name-status` cross-check guards against this.

## Out of scope

- Upgrading beyond `5de53f78f` to the latest upstream tinygrad.
- Fixing the 88 broken `EXP2`/`LOG2`/`SIN` hardware tests (a separate `src/`
  BSV change — see `doc/plan-primitive-ops-handoff.md`).
- Adopting the new opt-in `ISARenderer` instruction-selection pipeline; the
  plain-`Renderer` `render(uops)` path the backend uses still works.
