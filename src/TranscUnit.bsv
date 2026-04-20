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

typedef enum { TR_EXP2 } TranscOp deriving (Bits, Eq, FShow);

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

   // Per-lane Horner: y = x * ln2; then out = ((y*c2 + c1) * y) + 1 where
   //   c1 = 1, c2 = 0.5       (pure Taylor degree-2 in y = x*ln2 gives
   //   2^x ≈ 1 + y + y²/2). Each step is one FP op:
   //     step 0: y_r   = buf_r[i] * ln2        (FMUL)
   //     step 1: acc_r = y_r * half            (FMUL)
   //     step 2: acc_r = acc_r + one           (FADD)
   //     step 3: acc_r = acc_r * y_r           (FMUL)
   //     step 4: buf_r[i] = fp2bits(acc_r + one) (FADD + writeback + next lane)
   // Use UInt#(32) for lane_idx so `== valueOf(n)` comparison fits even
   // when the flattened tile width is a power of two.
   Reg#(UInt#(32))      lane_idx <- mkReg(0);
   Reg#(UInt#(3))       step     <- mkReg(0);
   Reg#(Float)          y_r      <- mkRegU;
   Reg#(Float)          acc_r    <- mkRegU;

   Float ln2_c  = unpack(32'h3F317218);  // ln(2)
   Float half_c = unpack(32'h3F000000);  // 0.5
   Float one_c  = unpack(32'h3F800000);  // 1.0

   rule step_rule (busy && !done);
      Float x    = tr_bits2fp(buf_r[lane_idx]);
      case (step)
         0: begin
            y_r <= tpl_1(multFP(x, ln2_c, Rnd_Nearest_Even));
            step <= 1;
         end
         1: begin
            acc_r <= tpl_1(multFP(y_r, half_c, Rnd_Nearest_Even));
            step  <= 2;
         end
         2: begin
            acc_r <= tpl_1(addFP(acc_r, one_c, Rnd_Nearest_Even));
            step  <= 3;
         end
         3: begin
            acc_r <= tpl_1(multFP(acc_r, y_r, Rnd_Nearest_Even));
            step  <= 4;
         end
         4: begin
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
