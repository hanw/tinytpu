package PE;

import Vector :: *;

export PE_IFC(..);
export mkPE;

interface PE_IFC;
   // Weight-stationary path (preloaded once, never streamed).
   method Action loadWeight(Int#(8) w);
   method Action feedActivation(Int#(8) a);
   method Int#(8)  passActivation;

   // Output-stationary streaming path: both operands arrive from neighbors
   // each cycle (weight flows down, activation flows right). The PE stores
   // neither operand as state — it accumulates a*w into its psum.
   method Action  feedPair(Int#(8) w, Int#(8) a);
   method Int#(8) passWeight;

   // Shared read/clear for the accumulator (psum) in either dataflow.
   method Int#(32) getAccum;
   method Action clearAccum;
endinterface

module mkPE(PE_IFC);

   Reg#(Int#(8))  weight     <- mkReg(0);
   Reg#(Int#(32)) accum      <- mkReg(0);
   Reg#(Int#(8))  act_pass   <- mkReg(0);
   Reg#(Int#(8))  w_pass     <- mkReg(0);

   method Action loadWeight(Int#(8) w);
      weight <= w;
   endmethod

   method Action feedActivation(Int#(8) a);
      let product = signExtend(a) * signExtend(weight);
      accum    <= accum + product;
      act_pass <= a;
   endmethod

   method Int#(8) passActivation;
      return act_pass;
   endmethod

   method Action feedPair(Int#(8) w, Int#(8) a);
      let product = signExtend(a) * signExtend(w);
      accum    <= accum + product;
      act_pass <= a;
      w_pass   <= w;
   endmethod

   method Int#(8) passWeight;
      return w_pass;
   endmethod

   method Int#(32) getAccum;
      return accum;
   endmethod

   method Action clearAccum;
      accum <= 0;
   endmethod

endmodule

endpackage
