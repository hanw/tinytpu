package TbSystolicArray;

import Vector :: *;
import SystolicArray :: *;

(* synthesize *)
module mkTbSystolicArray();

   SystolicArray_IFC#(2, 2) arr <- mkSystolicArray;

   Reg#(UInt#(8)) cycle <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 30) begin
         $display("FAIL: test timed out");
         $finish(1);
      end
   endrule

   // Cycle 0: load weights
   // W = [[1, 2], [3, 4]]
   rule load_weights (cycle == 0);
      Vector#(2, Vector#(2, Int#(8))) w = newVector;
      w[0][0] = 1; w[0][1] = 2;
      w[1][0] = 3; w[1][1] = 4;
      arr.loadWeights(w);
      $display("Cycle %0d: weights loaded", cycle);
   endrule

   // Cycle 1: feed activations [5, 6]
   // Column 0 PEs get their activations directly.
   // Column 1 PEs get passActivation from column 0, which is 0 (initial value).
   rule feed1 (cycle == 1);
      Vector#(2, Int#(8)) a = newVector;
      a[0] = 5;
      a[1] = 6;
      arr.feedActivations(a);
      $display("Cycle %0d: fed [5, 6]", cycle);
   endrule

   // Cycle 2: feed zeros to let column 1 see the propagated activations
   // Now passActivation from column 0 PEs will be 5 and 6 (from cycle 1).
   rule feed2 (cycle == 2);
      Vector#(2, Int#(8)) a = replicate(0);
      arr.feedActivations(a);
      $display("Cycle %0d: fed [0, 0] to propagate", cycle);
   endrule

   // Cycle 3: check results
   // After 2 feed cycles:
   //   PE[0][0]: weight=1, fed 5 then 0 => accum = 1*5 + 1*0 = 5
   //   PE[1][0]: weight=3, fed 6 then 0 => accum = 3*6 + 3*0 = 18
   //   PE[0][1]: weight=2, fed pass(0)=0 then pass(5)=5 => accum = 2*0 + 2*5 = 10
   //   PE[1][1]: weight=4, fed pass(0)=0 then pass(6)=6 => accum = 4*0 + 4*6 = 24
   //   col0 = 5 + 18 = 23
   //   col1 = 10 + 24 = 34
   rule check_results (cycle == 3);
      Vector#(2, Int#(32)) res = arr.getResults;
      $display("Cycle %0d: results = [%0d, %0d]", cycle, res[0], res[1]);
      if (res[0] == 23 && res[1] == 34) begin
         $display("PASS: 2x2 matrix-vector multiply correct");
      end else begin
         $display("FAIL: expected [23, 34]");
      end
      $finish(0);
   endrule

endmodule

endpackage
