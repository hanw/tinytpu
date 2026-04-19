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
      if (cycle > 100) begin $display("FAIL: timeout"); $finish(1); end
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

   // Test 9: VPU_CMPEQ
   // [1,5,2,6] == [1,4,2,4] = [1,0,1,0]
   rule dispatch_cmpeq (cycle == 16);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 5; s1[0][2] = 2; s1[0][3] = 6;
      s2[0][0] = 1; s2[0][1] = 4; s2[0][2] = 2; s2[0][3] = 4;
      vpu.execute(VPU_CMPEQ, s1, s2);
      $display("Cycle %0d: dispatched VPU_CMPEQ", cycle);
   endrule

   rule check_cmpeq (cycle == 17);
      let res = vpu.result;
      Bool ok = (res[0][0] == 1 && res[0][1] == 0 && res[0][2] == 1 && res[0][3] == 0);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_CMPEQ", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_CMPEQ got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 10: VPU_MAX_REDUCE
   // max([3, 7, 1, 5]) = 7, broadcast -> [7, 7, 7, 7]
   rule dispatch_max_reduce (cycle == 18);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 3; s1[0][1] = 7; s1[0][2] = 1; s1[0][3] = 5;
      vpu.execute(VPU_MAX_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_MAX_REDUCE", cycle);
   endrule

   rule check_max_reduce (cycle == 19);
      let res = vpu.result;
      Bool ok = (res[0][0] == 7 && res[0][1] == 7 && res[0][2] == 7 && res[0][3] == 7);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MAX_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MAX_REDUCE got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 11: VPU_SHL
   // [1,2,3,4] << [1,2,3,0] = [2,8,24,4]
   rule dispatch_shl (cycle == 20);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 2; s1[0][2] = 3; s1[0][3] = 4;
      s2[0][0] = 1; s2[0][1] = 2; s2[0][2] = 3; s2[0][3] = 0;
      vpu.execute(VPU_SHL, s1, s2);
      $display("Cycle %0d: dispatched VPU_SHL", cycle);
   endrule

   rule check_shl (cycle == 21);
      let res = vpu.result;
      Bool ok = (res[0][0] == 2 && res[0][1] == 8 && res[0][2] == 24 && res[0][3] == 4);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SHL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SHL got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 12: VPU_SHR
   // [16,32,48,64] >> [2,3,1,0] = [4,4,24,64]
   rule dispatch_shr (cycle == 22);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 16; s1[0][1] = 32; s1[0][2] = 48; s1[0][3] = 64;
      s2[0][0] = 2;  s2[0][1] = 3;  s2[0][2] = 1;  s2[0][3] = 0;
      vpu.execute(VPU_SHR, s1, s2);
      $display("Cycle %0d: dispatched VPU_SHR", cycle);
   endrule

   rule check_shr (cycle == 23);
      let res = vpu.result;
      Bool ok = (res[0][0] == 4 && res[0][1] == 4 && res[0][2] == 24 && res[0][3] == 64);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SHR", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SHR got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 13: VPU_MIN
   // min([1,5,2,6], [3,4,3,4]) = [1,4,2,4]
   rule dispatch_min (cycle == 24);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 5; s1[0][2] = 2; s1[0][3] = 6;
      s2[0][0] = 3; s2[0][1] = 4; s2[0][2] = 3; s2[0][3] = 4;
      vpu.execute(VPU_MIN, s1, s2);
      $display("Cycle %0d: dispatched VPU_MIN", cycle);
   endrule

   rule check_min (cycle == 25);
      let res = vpu.result;
      Bool ok = (res[0][0] == 1 && res[0][1] == 4 && res[0][2] == 2 && res[0][3] == 4);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MIN", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MIN got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 14: VPU_MIN_REDUCE
   // min([3, 7, 1, 5]) = 1, broadcast -> [1, 1, 1, 1]
   rule dispatch_min_reduce (cycle == 26);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 3; s1[0][1] = 7; s1[0][2] = 1; s1[0][3] = 5;
      vpu.execute(VPU_MIN_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_MIN_REDUCE", cycle);
   endrule

   rule check_min_reduce (cycle == 27);
      let res = vpu.result;
      Bool ok = (res[0][0] == 1 && res[0][1] == 1 && res[0][2] == 1 && res[0][3] == 1);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MIN_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MIN_REDUCE got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 15: VPU_DIV
   // [10,20,-9,-10] / [3,5,3,-3] = [3,4,-3,3]
   rule dispatch_div (cycle == 28);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 10; s1[0][1] = 20; s1[0][2] = -9;  s1[0][3] = -10;
      s2[0][0] = 3;  s2[0][1] = 5;  s2[0][2] = 3;   s2[0][3] = -3;
      vpu.execute(VPU_DIV, s1, s2);
      $display("Cycle %0d: dispatched VPU_DIV", cycle);
   endrule

   rule check_div (cycle == 29);
      let res = vpu.result;
      Bool ok = (res[0][0] == 3 && res[0][1] == 4 && res[0][2] == -3 && res[0][3] == 3);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_DIV", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_DIV got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 16: divide by zero returns 0 on the affected lanes.
   rule dispatch_div_zero (cycle == 30);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 7; s1[0][1] = -8; s1[0][2] = 9; s1[0][3] = 10;
      s2[0][0] = 0; s2[0][1] = 2;  s2[0][2] = 0; s2[0][3] = -5;
      vpu.execute(VPU_DIV, s1, s2);
      $display("Cycle %0d: dispatched VPU_DIV zero-divisor case", cycle);
   endrule

   rule check_div_zero (cycle == 31);
      let res = vpu.result;
      Bool ok = (res[0][0] == 0 && res[0][1] == -4 && res[0][2] == 0 && res[0][3] == -2);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_DIV zero-divisor case", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_DIV zero-divisor case got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 17: VPU_AND
   // [1,0,3,4] & [1,1,2,4] = [1,0,2,4]
   rule dispatch_and (cycle == 32);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 0; s1[0][2] = 3; s1[0][3] = 4;
      s2[0][0] = 1; s2[0][1] = 1; s2[0][2] = 2; s2[0][3] = 4;
      vpu.execute(VPU_AND, s1, s2);
      $display("Cycle %0d: dispatched VPU_AND", cycle);
   endrule

   rule check_and (cycle == 33);
      let res = vpu.result;
      Bool ok = (res[0][0] == 1 && res[0][1] == 0 && res[0][2] == 2 && res[0][3] == 4);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_AND", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_AND got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 18: VPU_OR
   // [1,0,2,4] | [0,1,1,2] = [1,1,3,6]
   rule dispatch_or (cycle == 34);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 0; s1[0][2] = 2; s1[0][3] = 4;
      s2[0][0] = 0; s2[0][1] = 1; s2[0][2] = 1; s2[0][3] = 2;
      vpu.execute(VPU_OR, s1, s2);
      $display("Cycle %0d: dispatched VPU_OR", cycle);
   endrule

   rule check_or (cycle == 35);
      let res = vpu.result;
      Bool ok = (res[0][0] == 1 && res[0][1] == 1 && res[0][2] == 3 && res[0][3] == 6);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_OR", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_OR got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 19: VPU_XOR
   // [1,0,3,7] ^ [1,1,2,3] = [0,1,1,4]
   rule dispatch_xor (cycle == 36);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 1; s1[0][1] = 0; s1[0][2] = 3; s1[0][3] = 7;
      s2[0][0] = 1; s2[0][1] = 1; s2[0][2] = 2; s2[0][3] = 3;
      vpu.execute(VPU_XOR, s1, s2);
      $display("Cycle %0d: dispatched VPU_XOR", cycle);
   endrule

   rule check_xor (cycle == 37);
      let res = vpu.result;
      Bool ok = (res[0][0] == 0 && res[0][1] == 1 && res[0][2] == 1 && res[0][3] == 4);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_XOR", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_XOR got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 20: VPU_NOT
   // ~[0, 1, -1, 255] = [-1, -2, 0, -256]
   rule dispatch_not (cycle == 38);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 0; s1[0][1] = 1; s1[0][2] = -1; s1[0][3] = 255;
      vpu.execute(VPU_NOT, s1, s2);
      $display("Cycle %0d: dispatched VPU_NOT", cycle);
   endrule

   rule check_not (cycle == 39);
      let res = vpu.result;
      Bool ok = (res[0][0] == -1 && res[0][1] == -2 && res[0][2] == 0 && res[0][3] == -256);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_NOT", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_NOT got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 21: VPU_COPY
   // copy [10, 20, 30, 40] = [10, 20, 30, 40]
   rule dispatch_copy (cycle == 40);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 10; s1[0][1] = 20; s1[0][2] = 30; s1[0][3] = 40;
      vpu.execute(VPU_COPY, s1, s2);
      $display("Cycle %0d: dispatched VPU_COPY", cycle);
   endrule

   rule check_copy (cycle == 41);
      let res = vpu.result;
      Bool ok = (res[0][0] == 10 && res[0][1] == 20 && res[0][2] == 30 && res[0][3] == 40);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_COPY", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_COPY got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 22: VPU_SELECT
   // cond=[1,0,1,0], true=[10,20,30,40], false=resultReg (from COPY above = [10,20,30,40])
   // First set resultReg to false values via COPY, then SELECT
   rule dispatch_select_setup (cycle == 42);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 100; s1[0][1] = 200; s1[0][2] = 300; s1[0][3] = 400;
      vpu.execute(VPU_COPY, s1, s2);  // set resultReg = [100,200,300,400] (false values)
      $display("Cycle %0d: dispatched VPU_COPY (SELECT setup)", cycle);
   endrule

   rule dispatch_select (cycle == 43);
      Vector#(4, Vector#(4, Int#(32))) cond = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) true_val = replicate(replicate(0));
      cond[0][0] = 1; cond[0][1] = 0; cond[0][2] = 1; cond[0][3] = 0;
      true_val[0][0] = 10; true_val[0][1] = 20; true_val[0][2] = 30; true_val[0][3] = 40;
      vpu.execute(VPU_SELECT, cond, true_val);
      $display("Cycle %0d: dispatched VPU_SELECT", cycle);
   endrule

   rule check_select (cycle == 44);
      let res = vpu.result;
      // cond[0]=1 → true[0]=10, cond[1]=0 → false[1]=200, cond[2]=1 → true[2]=30, cond[3]=0 → false[3]=400
      Bool ok = (res[0][0] == 10 && res[0][1] == 200 && res[0][2] == 30 && res[0][3] == 400);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SELECT", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SELECT got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 23: VPU_SUM_REDUCE_COL
   // tile:
   //   [1, 2, 3, 4]
   //   [5, 6, 7, 8]
   //   [9, 10, 11, 12]
   //   [13, 14, 15, 16]
   // col sums = [28, 32, 36, 40], broadcast down each column
   rule dispatch_sum_reduce_col (cycle == 46);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = fromInteger(r * 4 + c + 1);
      vpu.execute(VPU_SUM_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_SUM_REDUCE_COL", cycle);
   endrule

   rule check_sum_reduce_col (cycle == 47);
      let res = vpu.result;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1) begin
         if (res[r][0] != 28 || res[r][1] != 32 || res[r][2] != 36 || res[r][3] != 40)
            ok = False;
      end
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SUM_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SUM_REDUCE_COL got row0=[%0d,%0d,%0d,%0d] row3=[%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3],
            res[3][0], res[3][1], res[3][2], res[3][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 24: VPU_MAX_REDUCE_COL
   // Same tile, col maxes = [13, 14, 15, 16]
   rule dispatch_max_reduce_col (cycle == 48);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = fromInteger(r * 4 + c + 1);
      vpu.execute(VPU_MAX_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_MAX_REDUCE_COL", cycle);
   endrule

   rule check_max_reduce_col (cycle == 49);
      let res = vpu.result;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1) begin
         if (res[r][0] != 13 || res[r][1] != 14 || res[r][2] != 15 || res[r][3] != 16)
            ok = False;
      end
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MAX_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MAX_REDUCE_COL got row0=[%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 25: VPU_MIN_REDUCE_COL
   // Same tile, col mins = [1, 2, 3, 4]
   rule dispatch_min_reduce_col (cycle == 50);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = fromInteger(r * 4 + c + 1);
      vpu.execute(VPU_MIN_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_MIN_REDUCE_COL", cycle);
   endrule

   rule check_min_reduce_col (cycle == 51);
      let res = vpu.result;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1) begin
         if (res[r][0] != 1 || res[r][1] != 2 || res[r][2] != 3 || res[r][3] != 4)
            ok = False;
      end
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MIN_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MIN_REDUCE_COL got row0=[%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 26: VPU_SUM_REDUCE_TILE
   // tile values 1..16 sum = 136, broadcast to full tile
   rule dispatch_sum_reduce_tile (cycle == 52);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = fromInteger(r * 4 + c + 1);
      vpu.execute(VPU_SUM_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_SUM_REDUCE_TILE", cycle);
   endrule

   rule check_sum_reduce_tile (cycle == 53);
      let res = vpu.result;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (res[r][c] != 136) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SUM_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SUM_REDUCE_TILE got [0][0]=%0d", cycle, res[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Test 27: VPU_MAX_REDUCE_TILE
   // max of 1..16 = 16
   rule dispatch_max_reduce_tile (cycle == 54);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = fromInteger(r * 4 + c + 1);
      vpu.execute(VPU_MAX_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_MAX_REDUCE_TILE", cycle);
   endrule

   rule check_max_reduce_tile (cycle == 55);
      let res = vpu.result;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (res[r][c] != 16) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MAX_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MAX_REDUCE_TILE got [0][0]=%0d", cycle, res[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Test 28: VPU_MIN_REDUCE_TILE
   // min of 1..16 = 1
   rule dispatch_min_reduce_tile (cycle == 56);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = fromInteger(r * 4 + c + 1);
      vpu.execute(VPU_MIN_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_MIN_REDUCE_TILE", cycle);
   endrule

   rule check_min_reduce_tile (cycle == 57);
      let res = vpu.result;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (res[r][c] != 1) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MIN_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MIN_REDUCE_TILE got [0][0]=%0d", cycle, res[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Test 29: VPU_MUL_REDUCE (per-row)
   // row = [2, 3, 4, 5] -> product 120, broadcast across row
   rule dispatch_mul_reduce (cycle == 58);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(1));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = 2; s1[0][1] = 3; s1[0][2] = 4; s1[0][3] = 5;
      vpu.execute(VPU_MUL_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_MUL_REDUCE", cycle);
   endrule

   rule check_mul_reduce (cycle == 59);
      let res = vpu.result;
      Bool ok = (res[0][0] == 120 && res[0][1] == 120 && res[0][2] == 120 && res[0][3] == 120);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MUL_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MUL_REDUCE got [%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 30: VPU_MUL_REDUCE_COL
   // tile of 1..16; col products: [585, 960, 1365, 1792]
   rule dispatch_mul_reduce_col (cycle == 60);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = fromInteger(r * 4 + c + 1);
      vpu.execute(VPU_MUL_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_MUL_REDUCE_COL", cycle);
   endrule

   rule check_mul_reduce_col (cycle == 61);
      let res = vpu.result;
      // col products: col0=1*5*9*13=585, col1=2*6*10*14=1680, col2=3*7*11*15=3465, col3=4*8*12*16=6144
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1) begin
         if (res[r][0] != 585 || res[r][1] != 1680 || res[r][2] != 3465 || res[r][3] != 6144)
            ok = False;
      end
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MUL_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MUL_REDUCE_COL got row0=[%0d,%0d,%0d,%0d]",
            cycle, res[0][0], res[0][1], res[0][2], res[0][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 31: VPU_MUL_REDUCE_TILE with small values to avoid int32 overflow
   // tile = [1,2,1,1 ; 2,1,1,1 ; 1,1,3,1 ; 1,1,1,2] -> product 24
   rule dispatch_mul_reduce_tile (cycle == 62);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(1));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][1] = 2; s1[1][0] = 2; s1[2][2] = 3; s1[3][3] = 2;
      vpu.execute(VPU_MUL_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_MUL_REDUCE_TILE", cycle);
   endrule

   rule check_mul_reduce_tile (cycle == 63);
      let res = vpu.result;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (res[r][c] != 24) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_MUL_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_MUL_REDUCE_TILE got [0][0]=%0d", cycle, res[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Test 32: VPU_FSUM_REDUCE_TILE — float tile sum.
   // Values 1.0..16.0 (as IEEE 754 bit patterns packed into Int#(32)).
   // Expected float sum: 136.0 = 0x43080000.
   rule dispatch_fsum_reduce_tile (cycle == 64);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      Bit#(32) fbits[16] = {
         32'h3F800000, 32'h40000000, 32'h40400000, 32'h40800000,  // 1,2,3,4
         32'h40A00000, 32'h40C00000, 32'h40E00000, 32'h41000000,  // 5,6,7,8
         32'h41100000, 32'h41200000, 32'h41300000, 32'h41400000,  // 9..12
         32'h41500000, 32'h41600000, 32'h41700000, 32'h41800000}; // 13..16
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = unpack(fbits[r * 4 + c]);
      vpu.execute(VPU_FSUM_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FSUM_REDUCE_TILE", cycle);
   endrule

   rule check_fsum_reduce_tile (cycle == 65);
      let res = vpu.result;
      Bit#(32) expected = 32'h43080000;  // 136.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (pack(res[r][c]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FSUM_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FSUM_REDUCE_TILE got [0][0]=0x%08x (want 0x%08x)",
            cycle, pack(res[0][0]), expected);
         failed <= failed + 1;
      end
   endrule

   // Test 33: VPU_FMAX_REDUCE_TILE — float tile max.
   // Values 1.0..16.0 as IEEE 754 bit patterns, expected max = 16.0 = 0x41800000.
   rule dispatch_fmax_reduce_tile (cycle == 66);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      Bit#(32) fbits[16] = {
         32'h3F800000, 32'h40000000, 32'h40400000, 32'h40800000,  // 1,2,3,4
         32'h40A00000, 32'h40C00000, 32'h40E00000, 32'h41000000,  // 5,6,7,8
         32'h41100000, 32'h41200000, 32'h41300000, 32'h41400000,  // 9..12
         32'h41500000, 32'h41600000, 32'h41700000, 32'h41800000}; // 13..16
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = unpack(fbits[r * 4 + c]);
      vpu.execute(VPU_FMAX_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FMAX_REDUCE_TILE", cycle);
   endrule

   rule check_fmax_reduce_tile (cycle == 67);
      let res = vpu.result;
      Bit#(32) expected = 32'h41800000;  // 16.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (pack(res[r][c]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FMAX_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FMAX_REDUCE_TILE got [0][0]=0x%08x (want 0x%08x)",
            cycle, pack(res[0][0]), expected);
         failed <= failed + 1;
      end
   endrule

   // Test 34: VPU_FMIN_REDUCE_TILE — float tile min.
   // Values 1.0..16.0, expected min = 1.0 = 0x3F800000.
   rule dispatch_fmin_reduce_tile (cycle == 68);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      Bit#(32) fbits[16] = {
         32'h3F800000, 32'h40000000, 32'h40400000, 32'h40800000,  // 1,2,3,4
         32'h40A00000, 32'h40C00000, 32'h40E00000, 32'h41000000,  // 5,6,7,8
         32'h41100000, 32'h41200000, 32'h41300000, 32'h41400000,  // 9..12
         32'h41500000, 32'h41600000, 32'h41700000, 32'h41800000}; // 13..16
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            s1[r][c] = unpack(fbits[r * 4 + c]);
      vpu.execute(VPU_FMIN_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FMIN_REDUCE_TILE", cycle);
   endrule

   rule check_fmin_reduce_tile (cycle == 69);
      let res = vpu.result;
      Bit#(32) expected = 32'h3F800000;  // 1.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (pack(res[r][c]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FMIN_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FMIN_REDUCE_TILE got [0][0]=0x%08x (want 0x%08x)",
            cycle, pack(res[0][0]), expected);
         failed <= failed + 1;
      end
   endrule

   // Test 35: VPU_FMIN — per-lane float min.
   // row0: [3.0, 1.0, 4.0, 2.0] fmin [2.0, 2.0, 2.0, 2.0] -> [2.0, 1.0, 2.0, 2.0]
   rule dispatch_fmin (cycle == 70);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h40400000);  // 3.0
      s1[0][1] = unpack(32'h3F800000);  // 1.0
      s1[0][2] = unpack(32'h40800000);  // 4.0
      s1[0][3] = unpack(32'h40000000);  // 2.0
      for (Integer l = 0; l < 4; l = l + 1) s2[0][l] = unpack(32'h40000000);  // 2.0
      vpu.execute(VPU_FMIN, s1, s2);
      $display("Cycle %0d: dispatched VPU_FMIN", cycle);
   endrule

   rule check_fmin (cycle == 71);
      let res = vpu.result;
      // Expected row0: [2.0, 1.0, 2.0, 2.0]
      Bool ok = (pack(res[0][0]) == 32'h40000000
              && pack(res[0][1]) == 32'h3F800000
              && pack(res[0][2]) == 32'h40000000
              && pack(res[0][3]) == 32'h40000000);
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FMIN", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FMIN got [0]=[0x%08x,0x%08x,0x%08x,0x%08x]",
            cycle, pack(res[0][0]), pack(res[0][1]), pack(res[0][2]), pack(res[0][3]));
         failed <= failed + 1;
      end
   endrule

   // Test 36: VPU_FSUM_REDUCE — per-sublane (row) float sum.
   // Row0: [1.0, 2.0, 3.0, 4.0] -> 10.0 (0x41200000), broadcast across row0.
   rule dispatch_fsum_reduce (cycle == 72);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h3F800000);  // 1.0
      s1[0][1] = unpack(32'h40000000);  // 2.0
      s1[0][2] = unpack(32'h40400000);  // 3.0
      s1[0][3] = unpack(32'h40800000);  // 4.0
      vpu.execute(VPU_FSUM_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FSUM_REDUCE", cycle);
   endrule

   rule check_fsum_reduce (cycle == 73);
      let res = vpu.result;
      Bit#(32) expected = 32'h41200000;  // 10.0
      Bool ok = True;
      for (Integer l = 0; l < 4; l = l + 1)
         if (pack(res[0][l]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FSUM_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FSUM_REDUCE got [0][0]=0x%08x", cycle, pack(res[0][0]));
         failed <= failed + 1;
      end
   endrule

   // Test 37: VPU_FMAX_REDUCE — per-row float max.
   // Row0: [3.0, 1.0, 4.0, 2.0] -> 4.0 (0x40800000).
   rule dispatch_fmax_reduce (cycle == 74);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h40400000);  // 3.0
      s1[0][1] = unpack(32'h3F800000);  // 1.0
      s1[0][2] = unpack(32'h40800000);  // 4.0
      s1[0][3] = unpack(32'h40000000);  // 2.0
      vpu.execute(VPU_FMAX_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FMAX_REDUCE", cycle);
   endrule

   rule check_fmax_reduce (cycle == 75);
      let res = vpu.result;
      Bit#(32) expected = 32'h40800000;  // 4.0
      Bool ok = True;
      for (Integer l = 0; l < 4; l = l + 1)
         if (pack(res[0][l]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FMAX_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FMAX_REDUCE got [0][0]=0x%08x", cycle, pack(res[0][0]));
         failed <= failed + 1;
      end
   endrule

   // Test 38: VPU_FMIN_REDUCE — per-row float min.
   // Row0: [3.0, 1.0, 4.0, 2.0] -> 1.0 (0x3F800000).
   rule dispatch_fmin_reduce (cycle == 76);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h40400000);  // 3.0
      s1[0][1] = unpack(32'h3F800000);  // 1.0
      s1[0][2] = unpack(32'h40800000);  // 4.0
      s1[0][3] = unpack(32'h40000000);  // 2.0
      vpu.execute(VPU_FMIN_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FMIN_REDUCE", cycle);
   endrule

   rule check_fmin_reduce (cycle == 77);
      let res = vpu.result;
      Bit#(32) expected = 32'h3F800000;  // 1.0
      Bool ok = True;
      for (Integer l = 0; l < 4; l = l + 1)
         if (pack(res[0][l]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FMIN_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FMIN_REDUCE got [0][0]=0x%08x", cycle, pack(res[0][0]));
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 79);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
