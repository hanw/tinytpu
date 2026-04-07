package TbAccelerator4x4;

import Vector :: *;
import TensorAccelerator :: *;

(* synthesize *)
module mkTbAccelerator4x4();

   TensorAccelerator_IFC#(4, 4, 8) accel <- mkTensorAccelerator;

   Reg#(UInt#(8)) cycle <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 100) begin
         $display("FAIL: test timed out");
         $finish(1);
      end
   endrule

   // Load identity weights: w[r][c] = (r == c) ? 1 : 0
   rule load_weights (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1;
      w[1][1] = 1;
      w[2][2] = 1;
      w[3][3] = 1;
      accel.loadWeightTile(0, w);
      $display("Cycle %0d: 4x4 identity weights loaded", cycle);
   endrule

   // Load activation vector [10, 20, 30, 40]
   rule load_acts (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 10; a[1] = 20; a[2] = 30; a[3] = 40;
      accel.loadActivationTile(0, a);
      $display("Cycle %0d: activations loaded", cycle);
   endrule

   rule start (cycle == 2);
      accel.startCompute(0, 0, 1);
      $display("Cycle %0d: compute started", cycle);
   endrule

   rule wait_done (cycle > 2 && !accel.computeDone);
      $display("Cycle %0d: computing...", cycle);
   endrule

   // With identity weights, output should equal input: [10, 20, 30, 40]
   rule check_done (cycle > 2 && accel.computeDone);
      Vector#(4, Int#(32)) out = accel.getOutput;
      $display("Cycle %0d: compute done", cycle);
      $display("  output = [%0d, %0d, %0d, %0d]", out[0], out[1], out[2], out[3]);
      if (out[0] == 10 && out[1] == 20 && out[2] == 30 && out[3] == 40) begin
         $display("PASS: 4x4 identity test");
      end else begin
         $display("FAIL: expected [10, 20, 30, 40]");
      end
      $finish(0);
   endrule

endmodule

endpackage
