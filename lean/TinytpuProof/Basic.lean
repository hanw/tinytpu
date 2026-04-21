/-
  TinyTPU MXU Correctness Proof — Full Skewing Model
  Parameterized by array size N (works for any N×N systolic array).
-/

import Mathlib.Tactic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

open Finset BigOperators

/-! ## 1. Skewed activation model -/

def controllerInput (acts : Fin T → Fin N → Int) (r : Fin N) (t : ℕ) : Int :=
  if h : t < T then acts ⟨t, h⟩ r else 0

def skewedInput (acts : Fin T → Fin N → Int) (r : Fin N) (c : Fin N)
    (t : ℕ) : Int :=
  if t < c.val then 0 else controllerInput acts r (t - c.val)

abbrev totalCycles (T N : ℕ) : ℕ := T + N

/-! ## 2. PE accumulator as sum over cycles -/

def peAccum (W : Fin N → Fin N → Int) (acts : Fin T → Fin N → Int)
    (r c : Fin N) : Int :=
  ∑ t ∈ Finset.range (totalCycles T N), skewedInput acts r c t * W r c

def colSum (W : Fin N → Fin N → Int) (acts : Fin T → Fin N → Int)
    (c : Fin N) : Int :=
  ∑ r : Fin N, peAccum W acts r c

/-! ## 3. GEMM specification -/

def gemmSpec (W : Fin N → Fin N → Int) (acts : Fin T → Fin N → Int)
    (c : Fin N) : Int :=
  ∑ r : Fin N, ∑ k : Fin T, acts k r * W r c

/-! ## 4. Skewing lemmas -/

lemma skewedInput_at_valid (acts : Fin T → Fin N → Int) (r : Fin N)
    (c : Fin N) (k : Fin T) :
    skewedInput acts r c (k.val + c.val) = acts k r := by
  unfold skewedInput controllerInput
  have h1 : ¬ (k.val + c.val < c.val) := by omega
  have h2 : k.val + c.val - c.val < T := by
    have := k.isLt; omega
  simp only [h1, ↓reduceIte, dif_pos h2]
  congr 1
  apply Fin.ext
  change k.val + c.val - c.val = k.val
  omega

lemma skewedInput_zero_outside (acts : Fin T → Fin N → Int) (r : Fin N)
    (c : Fin N) (t : ℕ) (ht : t < c.val ∨ t - c.val ≥ T) :
    skewedInput acts r c t = 0 := by
  unfold skewedInput controllerInput
  rcases ht with h | h
  · simp [if_pos h]
  · by_cases hc : t < c.val
    · simp [if_pos hc]
    · simp [hc]
      have : ¬ (t - c.val < T) := by omega
      simp [this]

lemma skewed_sum_eq (acts : Fin T → Fin N → Int) (r : Fin N) (c : Fin N)
    (w : Int) :
    ∑ t ∈ Finset.range (totalCycles T N), skewedInput acts r c t * w =
    (∑ k : Fin T, acts k r) * w := by
  rw [Finset.sum_mul]
  set f := fun k : Fin T => k.val + c.val
  set g := fun t => skewedInput acts r c t * w
  set img := Finset.univ.image f
  have hcN : c.val < N := c.isLt
  have h_img_sub : img ⊆ Finset.range (totalCycles T N) := by
    intro t ht
    simp only [img, f, Finset.mem_image, Finset.mem_univ, true_and] at ht
    obtain ⟨k, rfl⟩ := ht
    simp only [Finset.mem_range, totalCycles]
    have := k.isLt
    omega
  have h_zero : ∀ t ∈ Finset.range (totalCycles T N), t ∉ img → g t = 0 := by
    intro t _ ht_nimg
    suffices skewedInput acts r c t = 0 by simp [g, this]
    apply skewedInput_zero_outside
    by_contra habs
    simp only [not_or, not_lt, not_le] at habs
    obtain ⟨hge, hlt⟩ := habs
    apply ht_nimg
    simp only [img, f, Finset.mem_image, Finset.mem_univ, true_and]
    refine ⟨⟨t - c.val, by omega⟩, ?_⟩
    change t - c.val + c.val = t
    omega
  have h_inj : ∀ k₁ ∈ Finset.univ, ∀ k₂ ∈ Finset.univ,
      f k₁ = f k₂ → k₁ = k₂ := by
    intro k₁ _ k₂ _ h; ext; simp [f] at h; omega
  calc ∑ t ∈ Finset.range (totalCycles T N), g t
      = ∑ t ∈ img, g t :=
        (Finset.sum_subset h_img_sub h_zero).symm
    _ = ∑ k ∈ Finset.univ, g (f k) :=
        Finset.sum_image h_inj
    _ = ∑ k : Fin T, acts k r * w := by
        congr 1; ext k; simp [g, f, skewedInput_at_valid]

/-! ## 5. Per-PE correctness -/

lemma pe_accum_correct (W : Fin N → Fin N → Int) (acts : Fin T → Fin N → Int)
    (r c : Fin N) :
    peAccum W acts r c = (∑ k : Fin T, acts k r) * W r c := by
  exact skewed_sum_eq acts r c (W r c)

/-! ## 6. Main theorem — holds for any N×N array and any T -/

theorem mxu_correct (W : Fin N → Fin N → Int) (acts : Fin T → Fin N → Int)
    (c : Fin N) :
    colSum W acts c = gemmSpec W acts c := by
  simp only [colSum, gemmSpec]
  congr 1; ext r
  rw [pe_accum_correct]
  rw [Finset.sum_mul]

/-! ## 7. Sanity checks: specialize to concrete sizes -/

example (W : Fin 4 → Fin 4 → Int) (acts : Fin 10 → Fin 4 → Int) (c : Fin 4) :
    colSum W acts c = gemmSpec W acts c := mxu_correct W acts c

example (W : Fin 8 → Fin 8 → Int) (acts : Fin 20 → Fin 8 → Int) (c : Fin 8) :
    colSum W acts c = gemmSpec W acts c := mxu_correct W acts c

example (W : Fin 16 → Fin 16 → Int) (acts : Fin 100 → Fin 16 → Int)
    (c : Fin 16) :
    colSum W acts c = gemmSpec W acts c := mxu_correct W acts c
