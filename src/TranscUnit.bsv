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

typedef enum { TR_EXP2, TR_LOG2 } TranscOp deriving (Bits, Eq, FShow);

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
   Reg#(Float)          e_r      <- mkRegU;   // LOG2: exponent-as-float

   Float ln2_c    = unpack(32'h3F317218);  // ln(2)           ≈ 0.6931472
   Float half_c   = unpack(32'h3F000000);  // 0.5
   Float one_c    = unpack(32'h3F800000);  // 1.0
   Float neg_one  = unpack(32'hBF800000);  // -1.0
   Float log2e_c  = unpack(32'h3FB8AA3B);  // 1/ln(2)         ≈ 1.4426950
   Float neg_half_log2e = unpack(32'hBF389A43);  // -1/(2*ln(2)) ≈ -0.7213475

   // EXP2 per-lane schedule (5 steps, exact at x=0):
   //   0:  y_r   = x * ln2
   //   1:  acc_r = y_r * 0.5
   //   2:  acc_r = acc_r + 1
   //   3:  acc_r = acc_r * y_r
   //   4:  buf[i] = acc_r + 1; advance or done
   //
   // LOG2 per-lane schedule (5 steps, exact at x=1, range-reduced):
   //   0:  split x = m * 2^e, m in [1,2); y_r = m, e_r = float(e)
   //   1:  y_r = m - 1        (u in [0, 1); FADD)
   //   2:  acc_r = y_r * c2   (c2 = -0.5 / ln2; FMUL)
   //   3:  acc_r = acc_r + c1 (c1 =  1    / ln2; FADD)
   //   4:  buf[i] = acc_r * y_r + e_r? — but we need both an FMUL and an
   //       FADD in step 4. Split into two sub-steps: 4 then 5.
   //   4:  acc_r = acc_r * y_r  (FMUL)
   //   5:  buf[i] = acc_r + e_r; advance or done
   //
   // To keep the FSM uniform at 6 steps, EXP2 ignores step 5 and finishes
   // in step 4; LOG2 uses steps 0..5.
   rule step_rule (busy && !done);
      Float x    = tr_bits2fp(buf_r[lane_idx]);
      case (step)
         0: begin
            if (op_r == TR_EXP2) begin
               y_r <= tpl_1(multFP(x, ln2_c, Rnd_Nearest_Even));
            end else begin
               // LOG2: decompose buf_r[i] into (e_unbiased, mantissa).
               // Reuse sign bit in case x ≤ 0 (LOG2(-x) is ill-defined;
               // we still produce a bit pattern so the unit never stalls).
               Bit#(32) xb = pack(buf_r[lane_idx]);
               Int#(9)  e_unbiased = unpack(extend(xb[30:23])) - 127;
               Bit#(32) m_bits = {1'b0, 8'd127, xb[22:0]};
               y_r <= unpack(m_bits);               // m in [1, 2)
               e_r <= tr_i9_to_float(e_unbiased);
            end
            step <= 1;
         end
         1: begin
            if (op_r == TR_EXP2) begin
               acc_r <= tpl_1(multFP(y_r, half_c, Rnd_Nearest_Even));
            end else begin
               y_r <= tpl_1(addFP(y_r, neg_one, Rnd_Nearest_Even));  // u = m-1
            end
            step <= 2;
         end
         2: begin
            if (op_r == TR_EXP2) begin
               acc_r <= tpl_1(addFP(acc_r, one_c, Rnd_Nearest_Even));
            end else begin
               acc_r <= tpl_1(multFP(y_r, neg_half_log2e, Rnd_Nearest_Even));
            end
            step <= 3;
         end
         3: begin
            if (op_r == TR_EXP2) begin
               acc_r <= tpl_1(multFP(acc_r, y_r, Rnd_Nearest_Even));
            end else begin
               acc_r <= tpl_1(addFP(acc_r, log2e_c, Rnd_Nearest_Even));
            end
            step <= 4;
         end
         4: begin
            if (op_r == TR_EXP2) begin
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
            end else begin
               // LOG2 step 4: FMUL acc_r * y_r (Horner middle term).
               acc_r <= tpl_1(multFP(acc_r, y_r, Rnd_Nearest_Even));
               step  <= 5;
            end
         end
         5: begin
            // LOG2 step 5: final FADD with exponent, writeback.
            Float final_f = tpl_1(addFP(acc_r, e_r, Rnd_Nearest_Even));
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

endpackage
