package PE;

import Vector :: *;

export PE_IFC(..);
export mkPE;

interface PE_IFC;
   method Action loadWeight(Int#(8) w);
   method Action feedActivation(Int#(8) a);
   method Int#(32) getAccum;
   method Action clearAccum;
   method Int#(8) passActivation;
endinterface

module mkPE(PE_IFC);

   Reg#(Int#(8))  weight   <- mkReg(0);
   Reg#(Int#(32)) accum    <- mkReg(0);
   Reg#(Int#(8))  act_pass <- mkReg(0);

   method Action loadWeight(Int#(8) w);
      weight <= w;
   endmethod

   method Action feedActivation(Int#(8) a);
      let product = signExtend(a) * signExtend(weight);
      accum <= accum + product;
      act_pass <= a;
   endmethod

   method Int#(32) getAccum;
      return accum;
   endmethod

   method Action clearAccum;
      accum <= 0;
   endmethod

   method Int#(8) passActivation;
      return act_pass;
   endmethod

endmodule

endpackage
