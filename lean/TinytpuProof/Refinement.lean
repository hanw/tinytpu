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

