package FpReducer;

// Multi-cycle shared FP reducer. One addFP + one compareFP + a small FSM
// amortized across all float SUM/MAX/MIN reductions (tile, row, col).
//
// The point of this module: when the VPU wants a float reduction, every
// opcode-specific combinational tree we wrote before (FSUM_REDUCE_TILE,
// FMAX_REDUCE_TILE, FMIN_REDUCE_TILE, FSUM_REDUCE, ...) is a dedicated
// bundle of FP adders/comparators. With N opcodes × 4×4 lanes that's a
// lot of FP gates AND a lot of bsc elaboration. Here we keep one FP
// adder and one FP comparator, and walk the inputs sequentially across
// cycles. Cost per new reducer opcode is just a new `FpReducerOp` enum
// value and an FSM edge, not a new parallel tree.
//
// Interface:
//   start(op, inputs)  -- begin reduction. `inputs` is the flattened
//                         vector (caller packs tile/row/col as needed).
//   isDone             -- True once `acc` holds the final value.
//   getResult          -- returns the accumulated Float (bit-equivalent
//                         to fp2bits()).
//
// Usage pattern (from VPU side):
//   fpr.start(FPR_SUM, flat_inputs);
//   ... wait until fpr.isDone ...
//   Float r = fpr.getResult;

import Vector :: *;
import FloatingPoint :: *;

typedef enum { FPR_SUM, FPR_MAX, FPR_MIN, FPR_PROD } FpReducerOp
   deriving (Bits, Eq, FShow);

interface FpReducer_IFC#(numeric type n);
   method Action start(FpReducerOp op, Vector#(n, Int#(32)) inputs);
   method Bool isDone;
   method Float getResult;
endinterface

// Bit patterns for +0.0 / +inf / -inf kept local; callers don't supply
// identities.
function Float fpr_bits2fp(Int#(32) x);
   return unpack(pack(x));
endfunction

module mkFpReducer(FpReducer_IFC#(n))
   provisos(Add#(1, n_, n));

   // Reduction state. `busy` gates the advance rule; `idx` walks from 1
   // up to valueOf(n); when idx==valueOf(n), we stop and set done.
   Reg#(Bool)       busy  <- mkReg(False);
   Reg#(Bool)       done  <- mkReg(False);
   Reg#(FpReducerOp) op_r <- mkRegU;
   Reg#(Vector#(n, Int#(32))) buf_r <- mkRegU;
   Reg#(UInt#(32))  idx_r <- mkReg(0);
   Reg#(Float)      acc   <- mkRegU;

   // Step rule: one addFP / compareFP per cycle, reading the next
   // element out of buf_r. This is the only place FP hardware is
   // instantiated; every reducer opcode shares it.
   rule step (busy && !done);
      let i = idx_r;
      Float next = fpr_bits2fp(buf_r[i]);
      // Compute sum/prod/max/min candidates in parallel (bsc shares the FP
      // adder/multiplier/comparator), then select via the op register.
      // Keeping the final mux simple sidesteps BSV's strict definite-
      // initialization rule for the Float record.
      Float sum_acc  = tpl_1(addFP (acc, next, Rnd_Nearest_Even));
      Float prod_acc = tpl_1(multFP(acc, next, Rnd_Nearest_Even));
      let   cmp      = compareFP(acc, next);
      Float max_acc  = (cmp == GT || cmp == EQ) ? acc : next;
      Float min_acc  = (cmp == LT || cmp == EQ) ? acc : next;
      Float new_acc  = (op_r == FPR_SUM)  ? sum_acc  :
                       (op_r == FPR_PROD) ? prod_acc :
                       (op_r == FPR_MAX)  ? max_acc  : min_acc;
      acc <= new_acc;
      if (i + 1 == fromInteger(valueOf(n))) begin
         done <= True;
         busy <= False;
      end else begin
         idx_r <= i + 1;
      end
   endrule

   method Action start(FpReducerOp op, Vector#(n, Int#(32)) inputs)
      if (!busy);
      op_r  <= op;
      buf_r <= inputs;
      acc   <= fpr_bits2fp(inputs[0]);
      idx_r <= 1;
      busy  <= (valueOf(n) > 1);
      // If the input is size-1, the result is already the first element
      // and we can report done on the next cycle without stepping.
      done  <= (valueOf(n) == 1);
   endmethod

   method Bool isDone = done;
   method Float getResult = acc;

endmodule

export FpReducerOp(..);
export FpReducer_IFC(..);
export mkFpReducer;

endpackage
