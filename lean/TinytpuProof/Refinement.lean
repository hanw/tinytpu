/-
  Refinement proof: pipelined PE refines abstract PE
  
  Abstract PE:  accum' = accum + a * w          (1 cycle)
  Pipelined PE: stage1: product = a * w         (cycle 1)
                stage2: accum' = accum + product (cycle 2)
  
  We prove: after both stages complete, the pipelined PE
  produces the same accumulator value as the abstract PE.
-/

import Mathlib.Tactic

/-! ## Abstract (spec) PE — single-cycle MAC -/

structure AbsPE where
  weight : Int
  accum  : Int := 0

def AbsPE.step (s : AbsPE) (a : Int) : AbsPE :=
  { s with accum := s.accum + a * s.weight }

/-! ## Pipelined PE — 2-stage MAC -/

structure PipePE where
  weight : Int
  accum      : Int := 0
  product    : Int := 0       -- pipeline register between stages
  product_valid : Bool := false  -- is there a value in the pipeline?

/-- Stage 1: compute product, latch into pipeline register -/
def PipePE.stage1 (s : PipePE) (a : Int) : PipePE :=
  { s with product := a * s.weight, product_valid := true }

/-- Stage 2: accumulate the latched product -/
def PipePE.stage2 (s : PipePE) : PipePE :=
  if s.product_valid then
    { s with accum := s.accum + s.product, product_valid := false }
  else s

/-- Both stages for one activation (stage1 then stage2 on next cycle) -/
def PipePE.fullStep (s : PipePE) (a : Int) : PipePE :=
  (s.stage1 a).stage2

/-! ## Abstraction function: maps pipelined state to abstract state -/

/-- The abstract view of a pipelined PE (ignoring pipeline registers) -/
def PipePE.toAbs (s : PipePE) : AbsPE :=
  { weight := s.weight, accum := s.accum }

/-! ## Refinement theorem -/

/-- After a full step (stage1 + stage2), the pipelined PE's accumulator
    matches what the abstract PE would have computed. -/
theorem pipe_refines_abs (s : PipePE) (a : Int)
    : (PipePE.fullStep s a).toAbs = AbsPE.step (s.toAbs) a := by
  simp [PipePE.fullStep, PipePE.stage1, PipePE.stage2,
        PipePE.toAbs, AbsPE.step]

/-- After N activations, pipelined PE matches N abstract steps. -/
theorem pipe_refines_abs_multi (s : PipePE) (acts : List Int) :
    (acts.foldl PipePE.fullStep s).toAbs = acts.foldl AbsPE.step s.toAbs := by
  induction acts generalizing s with
  | nil => simp
  | cons a as ih =>
    have hstep : (PipePE.fullStep s a).toAbs = AbsPE.step (s.toAbs) a :=
      pipe_refines_abs s a
    simpa [List.foldl_cons, hstep] using ih (PipePE.fullStep s a)

/-! ## OS accumulate composition theorem -/

/--
OS-mode accumulator-hold across two dispatches.

Running K₁ activations, then K₂ more, produces the same accumulator
as running the concatenated K₁ ++ K₂ list in one pass. This is the
core correctness property of Controller.startOsAccumulate — the
drain-time `clearAll` skip must not lose psum state between tiles.
-/
theorem accumulate_compose (s : AbsPE) (xs ys : List Int) :
    (ys.foldl AbsPE.step (xs.foldl AbsPE.step s)) =
    (xs ++ ys).foldl AbsPE.step s := by
  rw [List.foldl_append]

/-! ## Zero-weight inertness -/

/--
A PE with `weight = 0` is inert: any activation sequence leaves the
accumulator unchanged. This is the correctness property for the
"cleared weight tile" idiom — after `startAccumulate` skips the
drain-time clear but a fresh weight hasn't been loaded yet, any
leftover activation stream must not corrupt the psum.
-/
theorem zero_weight_accum_unchanged (s : AbsPE) (h : s.weight = 0)
    (xs : List Int) :
    (xs.foldl AbsPE.step s).accum = s.accum := by
  induction xs generalizing s with
  | nil => simp
  | cons x xs ih =>
    have hstep_w : (AbsPE.step s x).weight = 0 := by
      simp [AbsPE.step, h]
    have hstep_a : (AbsPE.step s x).accum = s.accum := by
      simp [AbsPE.step, h]
    rw [List.foldl_cons, ih (AbsPE.step s x) hstep_w, hstep_a]

/-! ## VPU_SIGN semantics -/

/--
Integer sign function matching the BSV `VPU_SIGN` opcode: returns
-1 / 0 / +1 lane-wise.
-/
def sign (x : Int) : Int :=
  if x > 0 then 1 else if x < 0 then -1 else 0

/-- `sign(sign(x)) = sign(x)` — the image of `sign` is in `{-1, 0, 1}`
    and `sign` is the identity on that set. Matches the idempotency
    property downstream renderers can rely on for clip/relu chains. -/
theorem sign_idempotent (x : Int) : sign (sign x) = sign x := by
  unfold sign
  split_ifs <;> omega

/-- `sign` is odd: `sign(-x) = -sign(x)`. -/
theorem sign_odd (x : Int) : sign (-x) = -sign x := by
  unfold sign
  split_ifs <;> omega

/-! ## PE weight invariance -/

/-- A single `AbsPE.step` preserves the weight register — only
    `accum` changes. This is the invariant downstream Controller
    logic relies on across dispatches (weight stays stationary in WS). -/
theorem abs_step_preserves_weight (s : AbsPE) (a : Int) :
    (AbsPE.step s a).weight = s.weight := by
  simp [AbsPE.step]

/-- Folding any activation list preserves the weight. -/
theorem abs_fold_preserves_weight (s : AbsPE) (xs : List Int) :
    (xs.foldl AbsPE.step s).weight = s.weight := by
  induction xs generalizing s with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl_cons, ih, abs_step_preserves_weight]

/-- `sign x` always returns a value in `{-1, 0, 1}`. Downstream
    renderers (clip / relu chains) rely on this bound. -/
theorem sign_bounded (x : Int) : -1 ≤ sign x ∧ sign x ≤ 1 := by
  unfold sign
  split_ifs <;> constructor <;> omega

/-- `sign x * x` ≥ 0 (i.e. |x|, modulo the `sign 0 = 0` case). -/
theorem sign_times_self_nonneg (x : Int) : sign x * x ≥ 0 := by
  unfold sign
  split_ifs
  · omega
  · have : x < 0 := by assumption
    have : -x > 0 := by omega
    nlinarith
  · simp

/-! ## VPU dual-issue (SXU_DISPATCH_VPU_BG) refinement

The SXU's DISPATCH_VPU_BG path splits a single synchronous VPU op into
two stages: a dispatch that advances pc immediately while setting a
pending flag, and a background collect that retires the result into
the destination register one (or more) cycles later. These theorems
formalize the core correctness claim: dispatch-then-collect on the BG
path produces the same destination-register value as the synchronous
path, given no intervening read of the destination. -/

/-- A VPU op is modeled abstractly as a pure function on the source
    register value. Real VPU ops take a vector and a second source,
    but this scalarized view captures the dispatch→collect correctness
    property without dragging in the tile shape. -/
structure VpuState where
  reg : Int := 0        -- the destination vreg
  pending : Option Int := none  -- in-flight BG writeback (SXU-side)

/-- Synchronous dispatch: compute f(reg) and land it in reg atomically.
    Matches the `DISPATCH_VPU` opcode plus its 1-cycle collect. -/
def VpuState.syncStep (s : VpuState) (f : Int → Int) : VpuState :=
  { s with reg := f s.reg, pending := none }

/-- Background dispatch: compute f(reg), latch into `pending`. The
    destination register does not change yet. Matches `DISPATCH_VPU_BG`
    at its dispatch cycle. -/
def VpuState.bgDispatch (s : VpuState) (f : Int → Int) : VpuState :=
  { s with pending := some (f s.reg) }

/-- Background collect: if a pending writeback exists, retire it into
    the destination register and clear the flag. Matches the unified
    `do_vpu_collect` rule firing at the first cycle after dispatch
    where `vpu.isDone` holds. -/
def VpuState.bgCollect (s : VpuState) : VpuState :=
  match s.pending with
  | some v => { s with reg := v, pending := none }
  | none   => s

/-- Core equivalence: BG dispatch followed by BG collect, starting
    from a clean state (no prior pending writeback), retires the
    same destination value as a single synchronous step. This is
    the invariant SXU relies on: a BG op behaves like the sync op,
    modulo the cycle at which the register visibly updates. -/
theorem bg_dispatch_then_collect_matches_sync
    (s : VpuState) (f : Int → Int) (_h : s.pending = none) :
    ((s.bgDispatch f).bgCollect).reg = (s.syncStep f).reg := by
  simp [VpuState.bgDispatch, VpuState.bgCollect, VpuState.syncStep]

/-- BG collect is idempotent when there's nothing pending — firing it
    an extra time (e.g. the main FSM happens to schedule it in a cycle
    with no writeback to retire) is a no-op. Mirrors the guard
    `vpu_wb_pending && vpu.isDone`: when `pending = none`, no state
    change. -/
theorem bg_collect_idempotent_when_empty (s : VpuState)
    (h : s.pending = none) : s.bgCollect = s := by
  simp [VpuState.bgCollect, h]

/-- Dispatch leaves the destination register unchanged — the whole
    point of BG is that pc (and other SXU state) can proceed while the
    writeback waits on `pending`. -/
theorem bg_dispatch_preserves_reg (s : VpuState) (f : Int → Int) :
    (s.bgDispatch f).reg = s.reg := by
  simp [VpuState.bgDispatch]

