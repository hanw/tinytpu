package TbXLU;

import Vector :: *;
import XLU :: *;

(* synthesize *)
module mkTbXLU();

   XLU_IFC#(4, 4) xlu <- mkXLU;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin
         $display("FAIL: test timed out at cycle %0d", cycle);
         $finish(1);
      end
   endrule

   // ---- Test 1: ROTATE ----
   // Input row0: [0, 1, 2, 3], rotate by 1 -> [1, 2, 3, 0]
   // Input row1: [10, 20, 30, 40], rotate by 1 -> [20, 30, 40, 10]
   rule dispatch_rotate (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 0;  src[0][1] = 1;  src[0][2] = 2;  src[0][3] = 3;
      src[1][0] = 10; src[1][1] = 20; src[1][2] = 30; src[1][3] = 40;
      xlu.executeRotate(src, 1);
      $display("Cycle %0d: dispatched ROTATE by 1", cycle);
   endrule

   rule check_rotate (cycle == 1);
      let res = xlu.result;
      Bool ok = (res[0][0] == 1  && res[0][1] == 2  && res[0][2] == 3  && res[0][3] == 0 &&
                 res[1][0] == 20 && res[1][1] == 30 && res[1][2] == 40 && res[1][3] == 10);
      if (ok) begin
         $display("Cycle %0d: PASS ROTATE", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL ROTATE row0=[%0d,%0d,%0d,%0d] row1=[%0d,%0d,%0d,%0d]",
            cycle,
            res[0][0], res[0][1], res[0][2], res[0][3],
            res[1][0], res[1][1], res[1][2], res[1][3]);
         failed <= failed + 1;
      end
   endrule

   // ---- Test 2: BROADCAST ----
   // Input row0: [10, 20, 30, 40], broadcast lane 2 (value 30)
   // Expected row0: [30, 30, 30, 30]
   // Input row1: [1, 2, 3, 4], broadcast lane 2 (value 3)
   // Expected row1: [3, 3, 3, 3]
   rule dispatch_broadcast (cycle == 2);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 10; src[0][1] = 20; src[0][2] = 30; src[0][3] = 40;
      src[1][0] = 1;  src[1][1] = 2;  src[1][2] = 3;  src[1][3] = 4;
      xlu.executeBroadcast(src, 2);
      $display("Cycle %0d: dispatched BROADCAST lane 2", cycle);
   endrule

   rule check_broadcast (cycle == 3);
      let res = xlu.result;
      Bool ok = (res[0][0] == 30 && res[0][1] == 30 && res[0][2] == 30 && res[0][3] == 30 &&
                 res[1][0] == 3  && res[1][1] == 3  && res[1][2] == 3  && res[1][3] == 3);
      if (ok) begin
         $display("Cycle %0d: PASS BROADCAST", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL BROADCAST row0=[%0d,%0d,%0d,%0d] row1=[%0d,%0d,%0d,%0d]",
            cycle,
            res[0][0], res[0][1], res[0][2], res[0][3],
            res[1][0], res[1][1], res[1][2], res[1][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 4);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0)
         $finish(0);
      else
         $finish(1);
   endrule

endmodule

endpackage
