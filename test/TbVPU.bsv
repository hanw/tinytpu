package TbVPU;

import Vector :: *;
import VPU :: *;

(* synthesize *)
module mkTbVPU();

   VPU_IFC#(4, 4) vpu <- mkVPU;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Test 1: VPU_ADD
   // src1 row0: [1, 2, 3, 4], src2 row0: [10, 20, 30, 40]
   // expected row0: [11, 22, 33, 44]
   rule dispatch_add (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 2; s1[0][2] = 3;  s1[0][3] = 4;
      s2[0][0] = 10; s2[0][1] = 20; s2[0][2] = 30; s2[0][3] = 40;
      vpu.execute(VPU_ADD, s1, s2);
      $display("Cycle %0d: dispatched VPU_ADD", cycle);
   endrule

   rule check_add (cycle == 1);
      let res = vpu.result;
      Bool ok = (res[0][0] == 11 && res[0][1] == 22 && res[0][2] == 33 && res[0][3] == 44);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_ADD", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_ADD got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 2: VPU_MUL
   // [2,3,4,5] * [3,4,5,6] = [6,12,20,30]
   rule dispatch_mul (cycle == 2);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 2; s1[0][1] = 3; s1[0][2] = 4; s1[0][3] = 5;
      s2[0][0] = 3; s2[0][1] = 4; s2[0][2] = 5; s2[0][3] = 6;
      vpu.execute(VPU_MUL, s1, s2);
      $display("Cycle %0d: dispatched VPU_MUL", cycle);
   endrule

   rule check_mul (cycle == 3);
      let res = vpu.result;
      Bool ok = (res[0][0] == 6 && res[0][1] == 12 && res[0][2] == 20 && res[0][3] == 30);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MUL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MUL got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 3: VPU_RELU
   // src1 row0: [-5, -1, 0, 7] -> [0, 0, 0, 7]
   rule dispatch_relu (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = -5; s1[0][1] = -1; s1[0][2] = 0; s1[0][3] = 7;
      vpu.execute(VPU_RELU, s1, s2);
      $display("Cycle %0d: dispatched VPU_RELU", cycle);
   endrule

   rule check_relu (cycle == 5);
      let res = vpu.result;
      Bool ok = (res[0][0] == 0 && res[0][1] == 0 && res[0][2] == 0 && res[0][3] == 7);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_RELU", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_RELU got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 4: VPU_MAX
   // max([1,5,2,6], [3,4,3,4]) = [3,5,3,6]
   rule dispatch_max (cycle == 6);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 5; s1[0][2] = 2; s1[0][3] = 6;
      s2[0][0] = 3; s2[0][1] = 4; s2[0][2] = 3; s2[0][3] = 4;
      vpu.execute(VPU_MAX, s1, s2);
      $display("Cycle %0d: dispatched VPU_MAX", cycle);
   endrule

   rule check_max (cycle == 7);
      let res = vpu.result;
      Bool ok = (res[0][0] == 3 && res[0][1] == 5 && res[0][2] == 3 && res[0][3] == 6);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MAX", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MAX got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 5: VPU_SUM_REDUCE
   // sum([10, 20, 30, 40]) = 100, broadcast -> [100, 100, 100, 100]
   rule dispatch_sum (cycle == 8);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 10; s1[0][1] = 20; s1[0][2] = 30; s1[0][3] = 40;
      vpu.execute(VPU_SUM_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_SUM_REDUCE", cycle);
   endrule

   rule check_sum (cycle == 9);
      let res = vpu.result;
      Bool ok = (res[0][0] == 100 && res[0][1] == 100 && res[0][2] == 100 && res[0][3] == 100);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SUM_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SUM_REDUCE got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 6: VPU_CMPLT
   // [1,5,2,6] < [3,4,3,4] = [1,0,1,0]
   rule dispatch_cmplt (cycle == 10);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 5; s1[0][2] = 2; s1[0][3] = 6;
      s2[0][0] = 3; s2[0][1] = 4; s2[0][2] = 3; s2[0][3] = 4;
      vpu.execute(VPU_CMPLT, s1, s2);
      $display("Cycle %0d: dispatched VPU_CMPLT", cycle);
   endrule

   rule check_cmplt (cycle == 11);
      let res = vpu.result;
      Bool ok = (res[0][0] == 1 && res[0][1] == 0 && res[0][2] == 1 && res[0][3] == 0);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_CMPLT", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_CMPLT got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 7: VPU_CMPNE
   // [1,5,2,6] != [1,4,2,4] = [0,1,0,1]
   rule dispatch_cmpne (cycle == 12);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 5; s1[0][2] = 2; s1[0][3] = 6;
      s2[0][0] = 1; s2[0][1] = 4; s2[0][2] = 2; s2[0][3] = 4;
      vpu.execute(VPU_CMPNE, s1, s2);
      $display("Cycle %0d: dispatched VPU_CMPNE", cycle);
   endrule

   rule check_cmpne (cycle == 13);
      let res = vpu.result;
      Bool ok = (res[0][0] == 0 && res[0][1] == 1 && res[0][2] == 0 && res[0][3] == 1);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_CMPNE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_CMPNE got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 8: VPU_SUB
   // [5,7,-2,0] - [2,3,4,-9] = [3,4,-6,9]
   rule dispatch_sub (cycle == 14);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 5; s1[0][1] = 7; s1[0][2] = -2; s1[0][3] = 0;
      s2[0][0] = 2; s2[0][1] = 3; s2[0][2] = 4;  s2[0][3] = -9;
      vpu.execute(VPU_SUB, s1, s2);
      $display("Cycle %0d: dispatched VPU_SUB", cycle);
   endrule

   rule check_sub (cycle == 15);
      let res = vpu.result;
      Bool ok = (res[0][0] == 3 && res[0][1] == 4 && res[0][2] == -6 && res[0][3] == 9);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SUB", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SUB got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 16);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
