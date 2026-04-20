package TranscUnit;

// Multi-cycle shared transcendental unit. One FP adder + one FP multiplier
// walk a tile one lane at a time, applying a polynomial Horner sequence
// for 2^x / log2(x) / sin(x). First iteration only wires up EXP2; LOG2
// and SIN slot into the same FSM by adding new TranscOp values and new
// coefficient rows to the step table.
//
// Cost model: per lane, N horner steps (N=4 for EXP2); each step
// consumes one cycle and one FP op. Tile of 16 lanes → ~16 * N cycles
// to process, plus a setup cycle. This avoids elaborating N parallel
// FP cones per lane inside the per-lane case branch — the critical
// trade that lets VPU keep adding transcendental opcodes without
// blowing through bsc's step budget at TensorCore elaboration.
//
// Interface mirrors FpReducer: start() with flat inputs and op,
// poll isDone, read getResult. VPU routes VPU_EXP2 through here while
// isDone==False gates the VPU collect path.

import Vector         :: *;
import FloatingPoint  :: *;

typedef enum { TR_EXP2, TR_LOG2, TR_SIN, TR_COS } TranscOp deriving (Bits, Eq, FShow);

// Integer-to-float conversion for the small signed exponent that falls
// out of LOG2's range reduction. Only ever handles values in [-126, 127],
// so we only need to look at 8 mantissa bits. Duplicated locally so
// TranscUnit stays self-contained (VPU imports TranscUnit, so pulling
// int32_to_float out of VPU would be circular).
function Float tr_i9_to_float(Int#(9) x);
   Bool     neg  = x < 0;
   Bit#(9)  xmag = neg ? pack(-x) : pack(x);
   Bit#(8)  mag_bits = truncate(xmag);
   UInt#(8) shift = 0;
   for (Integer i = 7; i >= 0; i = i - 1)
      if (mag_bits[i] == 1 && shift == 0) shift = fromInteger(i);
   Bit#(8)  exp_bits = (mag_bits == 0) ? 0 : truncate(pack(shift + 127));
   Bit#(23) mant     = (mag_bits == 0) ? 0 :
                       zeroExtend(mag_bits & ~(8'h80 >> (7 - shift))) << (23 - shift);
   Float f;
   f.sign = neg && (mag_bits != 0);
   f.exp  = (mag_bits == 0) ? 0 : exp_bits;
   f.sfd  = (mag_bits == 0) ? 0 : mant;
   return f;
endfunction

interface TranscUnit_IFC#(numeric type n);
   method Action start(TranscOp op, Vector#(n, Int#(32)) inputs);
   method Bool isDone;
   method Vector#(n, Int#(32)) getResult;
endinterface

function Float tr_bits2fp(Int#(32) x);
   return unpack(pack(x));
endfunction

function Int#(32) tr_fp2bits(Float x);
   return unpack(pack(x));
endfunction

// Truncate toward zero: zero out fractional mantissa bits of x.
// Scaffolding for EXP2 range reduction. For |x| < 1 the result is 0;
// for |x| >= 2^24 the input is assumed already integer (sfd preserved);
// for 0 <= e < 24 the low (23 - e) mantissa bits are cleared.
function Float tr_trunc(Float x);
   Bit#(8) raw_exp = x.exp;
   Int#(9) e = unpack(zeroExtend(raw_exp)) - 127;
   Bool is_small = (raw_exp < 127);     // |x| < 1
   Bool is_big   = (raw_exp >= 151);    // |x| >= 2^24: already integer
   Bit#(5)  frac_bits = truncate(pack(23 - e));            // 0..23
   Bit#(32) mask32    = ~((32'h1 << frac_bits) - 32'h1);
   Bit#(23) keep_mask = truncate(mask32);
   Bit#(23) trunc_sfd = x.sfd & keep_mask;
   Float r;
   r.sign = is_small ? False : x.sign;
   r.exp  = is_small ? 8'h00 : x.exp;
   r.sfd  = is_small ? 23'h0 : (is_big ? x.sfd : trunc_sfd);
   return r;
endfunction

// Convert a Float that represents an integer to Int#(32) (truncate
// toward zero for non-integer inputs, same semantics as VPU F2I but
// duplicated to keep TranscUnit self-contained).
function Int#(32) tr_fp_to_int(Float f);
   Bit#(8) raw_exp = f.exp;
   Int#(9) exp_val = unpack(zeroExtend(raw_exp)) - 127;
   Bool is_zero = (raw_exp == 0 && f.sfd == 0);
   Bool too_small = (exp_val < 0);
   UInt#(32) mantissa = (zeroExtend(1'b1) << 23) | unpack(zeroExtend(f.sfd));
   Bit#(5) shift_r = truncate(pack(23 - exp_val));
   Bit#(5) shift_l = truncate(pack(exp_val - 23));
   UInt#(32) mag = (exp_val >= 23) ? (mantissa << shift_l)
                                    : (mantissa >> shift_r);
   Int#(32) result = unpack(pack(mag));
   Int#(32) signed_result = f.sign ? (-result) : result;
   return (is_zero || too_small) ? 0 : signed_result;
endfunction

// Construct 2^n as Float for integer n. n outside [-126, 127] saturates
// to 0 / +inf respectively (the only n values TranscUnit will pass in
// live inside [-126, 127] by far).
function Float tr_pow2_int(Int#(32) n);
   Int#(32) biased = n + 127;
   Bool overflow  = (biased > 255);
   Bool underflow = (biased <= 0);
   Bit#(8) exp_bits = overflow ? 8'hFF
                                : (underflow ? 8'h00 : truncate(pack(biased)));
   Float r;
   r.sign = False;
   r.exp  = exp_bits;
   r.sfd  = 23'h0;
   return r;
endfunction

module mkTranscUnit(TranscUnit_IFC#(n))
   provisos(
      Add#(1, n_, n)
   );

   // FSM state
   Reg#(Bool)           busy     <- mkReg(False);
   Reg#(Bool)           done     <- mkReg(False);
   Reg#(TranscOp)       op_r     <- mkRegU;
   Reg#(Vector#(n, Int#(32))) buf_r <- mkRegU;

   // Per-lane FSM state. Each step is one FP op (FMUL or FADD); the FSM
   // dispatches the step's op by both `op_r` and `step`.
   // Use UInt#(32) for lane_idx so `== valueOf(n)` comparison fits even
   // when the flattened tile width is a power of two.
   Reg#(UInt#(32))      lane_idx <- mkReg(0);
   Reg#(UInt#(3))       step     <- mkReg(0);
   Reg#(Float)          y_r      <- mkRegU;
   Reg#(Float)          acc_r    <- mkRegU;
   Reg#(Float)          e_r      <- mkRegU;   // LOG2: exp-as-float; SIN: original x

   Float ln2_c    = unpack(32'h3F317218);  // ln(2)           ≈ 0.6931472
   Float half_c   = unpack(32'h3F000000);  // 0.5
   Float one_c    = unpack(32'h3F800000);  // 1.0
   Float neg_one  = unpack(32'hBF800000);  // -1.0
   // Remez minimax for log2(1+u) ≈ log2_a·u + log2_b·u² fit over [0, 1]
   // with p(0) = 0 enforced. Replaces Taylor (log2e, -log2e/2) for 8×
   // peak-error reduction: max |err| drops from 2.8e-1 to 3.4e-2.
   // Variable names kept historic for diffing; the constants now hold
   // the Remez values, not log(2)e / -log(2)e/2.
   Float log2e_c  = unpack(32'h3FC161E5);  //  1.5108  (Remez u coef)
   Float neg_half_log2e = unpack(32'hBF0B851F);  // -0.545  (Remez u² coef)
   // Remez minimax for sin(x) ≈ x + sin_c2·x³ + sin_c4·x⁵ fit over
   // [-π/2, π/2]. Replaces Taylor (-1/6, 1/120) for 40× peak-error
   // reduction on the fit range: max |err| drops from 4.5e-3 to 1.2e-4.
   // Outside [-π/2, π/2] both polynomials diverge; wide-angle range
   // reduction is a future iter.
   Float sin_c2   = unpack(32'hBE2A0E41);  // -0.16607  (Remez)
   Float sin_c4   = unpack(32'h3BFA0514);  //  0.00763  (Remez)
   // Remez minimax for cos(x) ≈ 1 + cos_c2·x² + cos_c4·x⁴ fit over
   // [-π/2, π/2]. Replaces Taylor (-1/2, 1/24) for 27× peak-error
   // reduction: max |err| drops from 2.0e-2 to 7.4e-4.
   Float cos_c2   = unpack(32'hBEFE42E1);  // -0.49660  (Remez)
   Float cos_c4   = unpack(32'h3D1817B9);  //  0.03713  (Remez)
   // Remez minimax quadratic for 2^x on [-1,1] with p(0) = 1 enforced.
   //   2^x ≈ 1 + exp2_p * x + exp2_q * x²     max |err| ≈ 1.6e-2 on [-1,1]
   // Fit over a symmetric range so Tensor.exp() (which scales inputs by
   // log2(e)≈1.44 and therefore drives VPU_EXP2 with both signs) does
   // not pay an asymmetric error tax like a [0,1]-only fit. Max err
   // improves 4× vs the earlier Taylor (1 + y + y²/2 where y = x·ln2,
   // max err ≈ 6.7e-2 over [-1,1]).
   Float exp2_p_c = unpack(32'h3F3C0592);  //  0.7344 (Remez)
   Float exp2_q_c = unpack(32'h3E800000);  //  0.25    (exact power of 2)

   // Per-op step schedules. Each step runs at most one FP op (FMUL or FADD)
   // so one multiplier + one adder suffice. EXP2 finishes at step 4; LOG2
   // and SIN use steps 0..5.
   //
   // EXP2 (Remez minimax quadratic, exact at x=0):
   //   0:  y_r   = x * 1        (pass-through; keeps pipeline shape)
   //   1:  acc_r = y_r * Q       (Q ≈ 0.3357)
   //   2:  acc_r = acc_r + P     (P ≈ 0.6613)
   //   3:  acc_r = acc_r * y_r   → P·x + Q·x²
   //   4:  buf[i] = acc_r + 1    → 1 + P·x + Q·x²   ≈ 2^x
   //
   // LOG2 (exact at powers of 2, range-reduced):
   //   0:  split x = m * 2^e, m in [1,2); y_r = m, e_r = float(e)
   //   1:  y_r = m - 1 → u
   //   2:  acc_r = y_r * (-0.5/ln2)
   //   3:  acc_r = acc_r + (1/ln2)
   //   4:  acc_r = acc_r * y_r
   //   5:  buf[i] = acc_r + e_r
   //
   // SIN (degree-5 Taylor, exact at x=0; ~0.5 rad error at |x|=π):
   //   0:  e_r = x (save); y_r = x * x   (x² in y_r)
   //   1:  acc_r = y_r * sin_c4           (sin_c4 = 1/120)
   //   2:  acc_r = acc_r + sin_c2         (sin_c2 = -1/6)
   //   3:  acc_r = acc_r * y_r            (back to scaled x²)
   //   4:  acc_r = acc_r + 1
   //   5:  buf[i] = acc_r * e_r           (final *x)
   rule step_rule (busy && !done);
      Float x    = tr_bits2fp(buf_r[lane_idx]);
      case (step)
         0: begin
            case (op_r)
               TR_EXP2: y_r <= tpl_1(multFP(x, one_c, Rnd_Nearest_Even));
               TR_LOG2: begin
                  Bit#(32) xb = pack(buf_r[lane_idx]);
                  Int#(9)  e_unbiased = unpack(extend(xb[30:23])) - 127;
                  Bit#(32) m_bits = {1'b0, 8'd127, xb[22:0]};
                  y_r <= unpack(m_bits);
                  e_r <= tr_i9_to_float(e_unbiased);
               end
               TR_SIN: begin
                  e_r <= x;                                     // save x
                  y_r <= tpl_1(multFP(x, x, Rnd_Nearest_Even)); // x²
               end
               TR_COS: y_r <= tpl_1(multFP(x, x, Rnd_Nearest_Even)); // x²
            endcase
            step <= 1;
         end
         1: begin
            case (op_r)
               TR_EXP2: acc_r <= tpl_1(multFP(y_r, exp2_q_c, Rnd_Nearest_Even));
               TR_LOG2: y_r   <= tpl_1(addFP(y_r, neg_one, Rnd_Nearest_Even));
               TR_SIN:  acc_r <= tpl_1(multFP(y_r, sin_c4, Rnd_Nearest_Even));
               TR_COS:  acc_r <= tpl_1(multFP(y_r, cos_c4, Rnd_Nearest_Even));
            endcase
            step <= 2;
         end
         2: begin
            case (op_r)
               TR_EXP2: acc_r <= tpl_1(addFP(acc_r, exp2_p_c, Rnd_Nearest_Even));
               TR_LOG2: acc_r <= tpl_1(multFP(y_r, neg_half_log2e, Rnd_Nearest_Even));
               TR_SIN:  acc_r <= tpl_1(addFP(acc_r, sin_c2, Rnd_Nearest_Even));
               TR_COS:  acc_r <= tpl_1(addFP(acc_r, cos_c2, Rnd_Nearest_Even));
            endcase
            step <= 3;
         end
         3: begin
            case (op_r)
               TR_EXP2: acc_r <= tpl_1(multFP(acc_r, y_r, Rnd_Nearest_Even));
               TR_LOG2: acc_r <= tpl_1(addFP(acc_r, log2e_c, Rnd_Nearest_Even));
               TR_SIN:  acc_r <= tpl_1(multFP(acc_r, y_r, Rnd_Nearest_Even));
               TR_COS:  acc_r <= tpl_1(multFP(acc_r, y_r, Rnd_Nearest_Even));
            endcase
            step <= 4;
         end
         4: begin
            case (op_r)
               TR_EXP2, TR_COS: begin
                  Float final_f = tpl_1(addFP(acc_r, one_c, Rnd_Nearest_Even));
                  Vector#(n, Int#(32)) next_buf = buf_r;
                  next_buf[lane_idx] = tr_fp2bits(final_f);
                  buf_r <= next_buf;
                  step  <= 0;
                  if (lane_idx + 1 == fromInteger(valueOf(n))) begin
                     done <= True;
                     busy <= False;
                  end else begin
                     lane_idx <= lane_idx + 1;
                  end
               end
               TR_LOG2: begin
                  acc_r <= tpl_1(multFP(acc_r, y_r, Rnd_Nearest_Even));
                  step  <= 5;
               end
               TR_SIN: begin
                  acc_r <= tpl_1(addFP(acc_r, one_c, Rnd_Nearest_Even));
                  step  <= 5;
               end
            endcase
         end
         5: begin
            Float final_f;
            case (op_r)
               TR_LOG2: final_f = tpl_1(addFP(acc_r, e_r, Rnd_Nearest_Even));
               TR_SIN:  final_f = tpl_1(multFP(acc_r, e_r, Rnd_Nearest_Even));
               default: final_f = acc_r;  // unreachable for TR_EXP2
            endcase
            Vector#(n, Int#(32)) next_buf = buf_r;
            next_buf[lane_idx] = tr_fp2bits(final_f);
            buf_r <= next_buf;
            step  <= 0;
            if (lane_idx + 1 == fromInteger(valueOf(n))) begin
               done <= True;
               busy <= False;
            end else begin
               lane_idx <= lane_idx + 1;
            end
         end
      endcase
   endrule

   method Action start(TranscOp op, Vector#(n, Int#(32)) inputs) if (!busy);
      op_r     <= op;
      buf_r    <= inputs;
      lane_idx <= 0;
      step     <= 0;
      busy     <= True;
      done     <= False;
   endmethod

   method Bool isDone = done;
   method Vector#(n, Int#(32)) getResult = buf_r;

endmodule

export TranscOp(..);
export TranscUnit_IFC(..);
export mkTranscUnit;
export tr_trunc;
export tr_fp_to_int;
export tr_pow2_int;

endpackage
