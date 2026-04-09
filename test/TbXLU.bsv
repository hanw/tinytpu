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

   // ---- Test 3a: PERMUTE identity (all ctrl False -> no change) ----
   // Input row0: [5, 10, 15, 20] -> expected [5, 10, 15, 20]
   rule dispatch_permute_id (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 5; src[0][1] = 10; src[0][2] = 15; src[0][3] = 20;
      Vector#(2, Vector#(4, Bool)) ctrl = replicate(replicate(False));
      xlu.executePermute(src, ctrl);
      $display("Cycle %0d: dispatched PERMUTE identity", cycle);
   endrule

   rule check_permute_id (cycle == 5);
      let res = xlu.result;
      Bool ok = (res[0][0] == 5 && res[0][1] == 10 && res[0][2] == 15 && res[0][3] == 20);
      if (ok) begin
         $display("Cycle %0d: PASS PERMUTE identity", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL PERMUTE identity got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // ---- Test 3b: PERMUTE reversal (all ctrl True -> lane-reversed) ----
   // Input row0: [5, 10, 15, 20]
   // Stage0 (stride=1, all swap): [10, 5, 20, 15]
   // Stage1 (stride=2, all swap): [20, 15, 10, 5]
   rule dispatch_permute_rev (cycle == 6);
      Vector#(4, Vector#(4, Int#(32))) src = replicate(replicate(0));
      src[0][0] = 5; src[0][1] = 10; src[0][2] = 15; src[0][3] = 20;
      Vector#(2, Vector#(4, Bool)) ctrl = replicate(replicate(True));
      xlu.executePermute(src, ctrl);
      $display("Cycle %0d: dispatched PERMUTE reversal", cycle);
   endrule

   rule check_permute_rev (cycle == 7);
      let res = xlu.result;
      Bool ok = (res[0][0] == 20 && res[0][1] == 15 && res[0][2] == 10 && res[0][3] == 5);
      if (ok) begin
         $display("Cycle %0d: PASS PERMUTE reversal", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL PERMUTE reversal got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // ---- Test 4: TRANSPOSE ----
   // Input (4x4):  row0=[1,2,3,4] row1=[5,6,7,8] row2=[9,10,11,12] row3=[13,14,15,16]
   // Expected:     row0=[1,5,9,13] row1=[2,6,10,14] row2=[3,7,11,15] row3=[4,8,12,16]
   rule dispatch_transpose (cycle == 8);
      Vector#(4, Vector#(4, Int#(32))) src = newVector;
      src[0] = cons(1,  cons(2,  cons(3,  cons(4,  nil))));
      src[1] = cons(5,  cons(6,  cons(7,  cons(8,  nil))));
      src[2] = cons(9,  cons(10, cons(11, cons(12, nil))));
      src[3] = cons(13, cons(14, cons(15, cons(16, nil))));
      xlu.executeTranspose(src);
      $display("Cycle %0d: dispatched TRANSPOSE", cycle);
   endrule

   rule check_transpose (cycle == 9);
      let res = xlu.result;
      Bool ok = (res[0][0] == 1  && res[0][1] == 5  && res[0][2] == 9  && res[0][3] == 13 &&
                 res[1][0] == 2  && res[1][1] == 6  && res[1][2] == 10 && res[1][3] == 14 &&
                 res[2][0] == 3  && res[2][1] == 7  && res[2][2] == 11 && res[2][3] == 15 &&
                 res[3][0] == 4  && res[3][1] == 8  && res[3][2] == 12 && res[3][3] == 16);
      if (ok) begin
         $display("Cycle %0d: PASS TRANSPOSE", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL TRANSPOSE", cycle);
         $display("  row0: [%0d,%0d,%0d,%0d]", res[0][0], res[0][1], res[0][2], res[0][3]);
         $display("  row1: [%0d,%0d,%0d,%0d]", res[1][0], res[1][1], res[1][2], res[1][3]);
         $display("  row2: [%0d,%0d,%0d,%0d]", res[2][0], res[2][1], res[2][2], res[2][3]);
         $display("  row3: [%0d,%0d,%0d,%0d]", res[3][0], res[3][1], res[3][2], res[3][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 10);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0)
         $finish(0);
      else
         $finish(1);
   endrule

endmodule

endpackage
