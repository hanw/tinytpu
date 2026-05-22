# VPU_PAIR_ROTATE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `VPU_PAIR_ROTATE` opcode so a RoPE rotation over a tile lowers to a single VPU dispatch.

**Architecture:** A new `VpuOp` enum value, implemented in the existing per-row VPU case statement in `src/VPU.bsv`, reached through the existing `SXU_DISPATCH_VPU` opcode (no new SXU opcode). It is a two-source float32 op: `src1` = data tile, `src2` = coefficient tile (even lane = cos, odd lane = sin). Assembler support goes in `scripts/tasm.py`; verification is an assembler round-trip test plus a runtime numeric test through the rebuilt simulator.

**Tech Stack:** Bluespec SystemVerilog (`src/VPU.bsv`), Python assembler (`scripts/tasm.py`), pytest, the BSV simulator (`build/mkTbTinyTPURuntime.bexe`).

**Spec:** `doc/vpu-pair-rotate-spec.md`

---

## Background an engineer needs

This is sub-project 1 of the CODA-gap program. Scope: **RoPE only** — one VPU opcode.

**The operation.** `VPU_PAIR_ROTATE` rotates each adjacent lane pair `(2p, 2p+1)` within a row by an angle:
```
out[s][2p]   = data[s][2p]·cos − data[s][2p+1]·sin
out[s][2p+1] = data[s][2p]·sin + data[s][2p+1]·cos
```
`src1` is the data tile; `src2` is the coefficient tile, packed so the even lane of each pair holds `cos θ` and the odd lane holds `sin θ`. Float32 (IEEE-754 bits stored in each `Int#(32)` lane).

**How a VPU op is added (established 4-file pattern, from commits `d4f0fca`/`1f11ec3`).**
- `src/VPU.bsv` — add the `VpuOp` enum value and a case arm in `mkVPU`'s dispatch.
- `scripts/tasm.py` — add the opcode number to the `_VPU` dict.
- `tests/test_tasm.py` — bump the opcode-count assertion, add a round-trip test.
- `tests/test_tinytpu_backend.py` — add a runtime numeric test (bundle → simulator → compare).

**Opcode number.** `scripts/tasm.py`'s `_VPU` dict currently holds 84 ops, numbered 0–83 (max `MAX_U32: 83`). `VPU_PAIR_ROTATE` gets the next number: **84**.

**`_VPU_OPS` is intentionally NOT touched.** The submodule's `tinygrad/tinygrad/runtime/ops_tinytpu.py` has a mirror dict `_VPU_OPS`. Updating it is deferred to the follow-on backend-lowering task (the only consumer that needs it). This sub-project is **parent-repo only** — no submodule change, no submodule-pointer bump. The runtime test references the opcode as a local constant `PAIR_ROTATE = 84`, matching how `tests/test_tasm.py` already hardcodes opcode numbers.

**VPU.bsv float idioms** (from the existing `VPU_FADD`/`VPU_FMUL`/`VPU_FSUB` arms, ~line 791):
- `bits2fp(x)` — `Int#(32)` → `Float`; `fp2bits(f)` — `Float` → `Int#(32)`.
- `tpl_1(multFP(a, b, Rnd_Nearest_Even))` — float multiply, result `Float`.
- `tpl_1(addFP(a, b, Rnd_Nearest_Even))` — float add, result `Float`.
- Negation: `Float n = f; n.sign = !n.sign;` (used by `VPU_FSUB`).

**Verification env.** Repo root `/Users/hanwang/p/tinytpu`. venv: `.venv/bin/python3`. Tests: `PYTHONPATH=tinygrad .venv/bin/python3 -m pytest ...`. After any `src/*.bsv` change the simulator must be rebuilt with `make runtime-tb` — the `tests/conftest.py` staleness guard enforces this and will abort the suite otherwise. The full `tests/test_tinytpu_backend.py` baseline is **965 passed**.

Project rules: commit messages `subsystem: description`, no `Co-Authored-By`. Do not push. CLAUDE.md: do not pipe command output through `head`/`tail` for monitoring; the explicit `next(...)`/`grep` one-shot extractions below are fine.

---

## File Structure

| File | Repo | Change |
|---|---|---|
| `scripts/tasm.py` | parent | add `"PAIR_ROTATE": 84` to `_VPU` |
| `tests/test_tasm.py` | parent | bump count assertion 84→85; add round-trip test |
| `src/VPU.bsv` | parent | add `VPU_PAIR_ROTATE` enum value + dispatch case + even-lanes static assert |
| `tests/test_tinytpu_backend.py` | parent | add `test_vpu_pair_rotate_float` runtime numeric test |

Two tasks, two parent-repo commits. Task 1 (assembler) is pure Python — no simulator. Task 2 (hardware + runtime test) rebuilds the simulator.

---

## Task 1: Assembler support for PAIR_ROTATE

**Files:**
- Modify: `scripts/tasm.py` (the `_VPU` dict, ~lines 78–163)
- Modify: `tests/test_tasm.py` (`test_vpu_ops_cover_full_range` at lines 731/733; new test near the other `test_vpu_*_roundtrip` tests, ~line 752)

- [ ] **Step 1: Write the failing round-trip test**

In `tests/test_tasm.py`, add this function next to `test_vpu_rotl_roundtrip` (~line 752):

```python
def test_vpu_pair_rotate_roundtrip():
    prog = ("LOAD  v0, VMEM[0]\n"
            "LOAD  v1, VMEM[1]\n"
            "VPU   v2 = PAIR_ROTATE(v0, v1)\n"
            "STORE VMEM[2], v2\n"
            "HALT\nEND\n")
    wire = assemble(prog)
    vpu_line = next(ln for ln in wire.strip().splitlines() if ln.startswith("2 2 "))
    assert vpu_line.split()[5] == "84"
    assert "PAIR_ROTATE(v0, v1)" in disassemble(wire)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/hanwang/p/tinytpu && .venv/bin/python3 -m pytest tests/test_tasm.py::test_vpu_pair_rotate_roundtrip -q`
Expected: FAIL — `assemble` raises `unknown VPU op 'PAIR_ROTATE'`.

- [ ] **Step 3: Add the opcode to `_VPU`**

In `scripts/tasm.py`, in the `_VPU` dict, the last entry is `"MAX_U32": 83,`. Add a line after it:

```python
    "MAX_U32":           83,
    "PAIR_ROTATE":       84,
}
```

Do **not** add `PAIR_ROTATE` to `_VPU_UNARY` — it is a two-source (binary) op, so the assembler/disassembler must treat it as `OP(vA, vB)`, which is the default for ops not in `_VPU_UNARY`.

- [ ] **Step 4: Update the opcode-count assertion**

In `tests/test_tasm.py`, `test_vpu_ops_cover_full_range` (lines 731 and 733) currently reads:

```python
    assert len(_VPU) == 84
    codes = sorted(_VPU.values())
    assert codes == list(range(84))
```

Change both `84` to `85`:

```python
    assert len(_VPU) == 85
    codes = sorted(_VPU.values())
    assert codes == list(range(85))
```

- [ ] **Step 5: Run the assembler tests**

Run: `cd /Users/hanwang/p/tinytpu && .venv/bin/python3 -m pytest tests/test_tasm.py -q`
Expected: PASS — all of `test_tasm.py` green, including `test_vpu_pair_rotate_roundtrip` and `test_vpu_ops_cover_full_range`.

- [ ] **Step 6: Commit**

```bash
cd /Users/hanwang/p/tinytpu
git add scripts/tasm.py tests/test_tasm.py
git commit -m "tasm: assembler support for VPU PAIR_ROTATE opcode"
```

---

## Task 2: VPU_PAIR_ROTATE hardware + runtime numeric test

**Files:**
- Modify: `src/VPU.bsv` (the `VpuOp` enum ~lines 8–62; the `mkVPU` dispatch case ~line 791; the imports ~lines 3–6)
- Modify: `tests/test_tinytpu_backend.py` (new test in class `TestTinyTPUSimOutputParsing`, near `test_vpu_fabs_float`)

- [ ] **Step 1: Write the failing runtime numeric test**

In `tests/test_tinytpu_backend.py`, add this method to class `TestTinyTPUSimOutputParsing` (near `test_vpu_fabs_float`):

```python
  def test_vpu_pair_rotate_float(self):
    # VPU_PAIR_ROTATE: 2D rotation of each adjacent lane pair (2p, 2p+1).
    #   out[2p]   = d[2p]*cos - d[2p+1]*sin
    #   out[2p+1] = d[2p]*sin + d[2p+1]*cos
    # src2 is the coefficient tile: even lane = cos, odd lane = sin.
    import struct, math
    sim = os.environ["TINYTPU_SIM"]
    def f2i(x): return struct.unpack("<i", struct.pack("<f", x))[0]
    def i2f(x): return struct.unpack("<f", struct.pack("<i", x))[0]
    PAIR_ROTATE = 84  # VPU_PAIR_ROTATE — see scripts/tasm.py _VPU and src/VPU.bsv
    # 4 rows x 4 lanes; one rotation angle per row, both lane-pairs use it.
    data = [ 1.0,  2.0,  3.0,  4.0,
            -1.5,  0.5,  2.0, -3.0,
             0.0,  1.0, -1.0,  0.0,
             5.0, -5.0,  2.5,  2.5]
    angles = [0.0, math.pi / 2, math.pi / 4, math.pi / 6]
    coef = []
    for r in range(4):
      c, s = math.cos(angles[r]), math.sin(angles[r])
      coef += [c, s, c, s]  # pair 0 and pair 1 of this row share the angle
    bundle = _bundle(
      _vmem(0, [f2i(x) for x in data]),
      _vmem(1, [f2i(x) for x in coef]),
      _load(0, 0),
      _load(1, 1),
      _vpu(2, 0, PAIR_ROTATE, 1),
      _store(2, 2),
      _halt(),
      _output_vmem(2),
      _end(),
    )
    out = _run_bundle(sim, bundle)
    got = [i2f(x) for x in _parse_vmem_output(out)]
    expected = []
    for r in range(4):
      c, s = math.cos(angles[r]), math.sin(angles[r])
      row = data[r * 4:r * 4 + 4]
      for p in range(2):
        de, do = row[2 * p], row[2 * p + 1]
        expected.append(de * c - do * s)
        expected.append(de * s + do * c)
    np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-5)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd /Users/hanwang/p/tinytpu && PYTHONPATH=tinygrad .venv/bin/python3 -m pytest "tests/test_tinytpu_backend.py::TestTinyTPUSimOutputParsing::test_vpu_pair_rotate_float" -q --tb=short`
Expected: FAIL — the currently-built simulator has no `VPU_PAIR_ROTATE` (opcode 84), so the result does not match the rotation reference. (The simulator is currently fresh, so the staleness guard does not fire yet.)

- [ ] **Step 3: Add the `VPU_PAIR_ROTATE` enum value**

In `src/VPU.bsv`, the `VpuOp` enum currently ends:

```bsv
               // Unsigned-viewed 32-bit min/max for sort keys and
               // hashing. Semantically independent of signed MIN/MAX.
               VPU_MIN_U32, VPU_MAX_U32 }
   VpuOp deriving (Bits, Eq, FShow);
```

Change it to:

```bsv
               // Unsigned-viewed 32-bit min/max for sort keys and
               // hashing. Semantically independent of signed MIN/MAX.
               VPU_MIN_U32, VPU_MAX_U32,
               // Pairwise 2D rotation of adjacent lane pairs (RoPE):
               //   out[2p]   = d[2p]*cos - d[2p+1]*sin
               //   out[2p+1] = d[2p]*sin + d[2p+1]*cos
               // src2 packs cos in even lanes, sin in odd lanes.
               VPU_PAIR_ROTATE }
   VpuOp deriving (Bits, Eq, FShow);
```

- [ ] **Step 4: Add the dispatch case arm**

In `src/VPU.bsv`, in `mkVPU`'s per-row op case statement, add a `VPU_PAIR_ROTATE` arm among the float ops (e.g. directly after the `VPU_FSUB` arm, ~line 810):

```bsv
            VPU_PAIR_ROTATE: begin
               // Rotate each adjacent lane pair (2p, 2p+1) by the angle in
               // src2: even lane = cos, odd lane = sin.
               for (Integer p = 0; p < valueOf(lanes) / 2; p = p + 1) begin
                  Float d_e = bits2fp(src1[s][2 * p]);
                  Float d_o = bits2fp(src1[s][2 * p + 1]);
                  Float c   = bits2fp(src2[s][2 * p]);
                  Float sn  = bits2fp(src2[s][2 * p + 1]);
                  Float ec = tpl_1(multFP(d_e, c,  Rnd_Nearest_Even));
                  Float os = tpl_1(multFP(d_o, sn, Rnd_Nearest_Even));
                  Float es = tpl_1(multFP(d_e, sn, Rnd_Nearest_Even));
                  Float oc = tpl_1(multFP(d_o, c,  Rnd_Nearest_Even));
                  Float os_neg = os; os_neg.sign = !os_neg.sign;
                  row[2 * p]     = fp2bits(tpl_1(addFP(ec, os_neg, Rnd_Nearest_Even)));
                  row[2 * p + 1] = fp2bits(tpl_1(addFP(es, oc,     Rnd_Nearest_Even)));
               end
            end
```

(If the surrounding arms reference the row accumulator by a name other than `row`, or the float helpers by other names, match the existing `VPU_FSUB` arm exactly — it uses the same `bits2fp`/`multFP`/`addFP`/`fp2bits`/`tpl_1`/`Rnd_Nearest_Even` and the `.sign` negation idiom.)

- [ ] **Step 5: Add the even-lanes static assertion**

`VPU_PAIR_ROTATE` requires an even lane count. In `src/VPU.bsv`:
- If `Assert` is not already imported, add `import Assert :: *;` with the other imports near the top (next to `import Vector :: *;`).
- As the first statement inside `module mkVPU`, add:

```bsv
   staticAssert(valueOf(lanes) % 2 == 0,
                "VPU_PAIR_ROTATE requires an even lane count");
```

If `staticAssert` placement causes a Bluespec elaboration error, move it to just before the dispatch rule, or fall back to a `// REQUIRES: lanes is even` comment above the `VPU_PAIR_ROTATE` arm — the lane count is statically 4, so this is a guard against future change, not a correctness blocker.

- [ ] **Step 6: Rebuild the simulator**

Run: `cd /Users/hanwang/p/tinytpu && make runtime-tb`
Expected: build succeeds; `build/mkTbTinyTPURuntime.bexe.so` is regenerated. (Required: `src/VPU.bsv` changed; the `tests/conftest.py` guard will abort the suite against a stale simulator.)

- [ ] **Step 7: Run the runtime test to verify it passes**

Run: `cd /Users/hanwang/p/tinytpu && PYTHONPATH=tinygrad .venv/bin/python3 -m pytest "tests/test_tinytpu_backend.py::TestTinyTPUSimOutputParsing::test_vpu_pair_rotate_float" -q --tb=short`
Expected: PASS — 1 passed.

- [ ] **Step 8: Run the full backend suite (no regressions)**

Run: `cd /Users/hanwang/p/tinytpu && PYTHONPATH=tinygrad .venv/bin/python3 -m pytest tests/test_tinytpu_backend.py -q --tb=short`
Expected: **966 passed, 0 failed** (the prior 965 + `test_vpu_pair_rotate_float`).

- [ ] **Step 9: Commit**

```bash
cd /Users/hanwang/p/tinytpu
git add src/VPU.bsv tests/test_tinytpu_backend.py
git commit -m "vpu: VPU_PAIR_ROTATE — single-dispatch pairwise rotation for RoPE"
git log --oneline -2
```

---

## Notes for the executor

- **Parent repo only.** No submodule change, no `git add tinygrad`, no push.
- **Order.** Task 1 (assembler, pure Python) and Task 2 (hardware) are independent; do Task 1 first by convention.
- **The simulator rebuild (Task 2 Step 6) is mandatory** before the runtime test — the staleness guard enforces it.
- **Deferred follow-on (not this plan):** the tinygrad-backend RoPE lowering, which will also add `"PAIR_ROTATE": 84` to `_VPU_OPS` in the submodule's `ops_tinytpu.py`.
