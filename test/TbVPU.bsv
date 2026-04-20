package TbVPU;

import Vector :: *;
import FloatingPoint :: *;
import VPU :: *;

(* synthesize *)
module mkTbVPU();

   VPU_IFC#(4, 4) vpu <- mkVPU;

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 900) begin $display("FAIL: timeout"); $finish(1); end
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

   // --- FP tile reducers now go through the multi-cycle shared FpReducer ---
   // Each takes ~16 FSM cycles for a 16-element reduction + 1 collect cycle,
   // so we give them a 25-cycle window (dispatch..check) each and keep
   // them clustered after all single-cycle tests finish at cycle 77.

   // Test 32: VPU_FSUM_REDUCE_TILE — float tile sum.
   // Values 1.0..16.0 (as IEEE 754 bit patterns packed into Int#(32)).
   // Expected float sum: 136.0 = 0x43080000.
   rule dispatch_fsum_reduce_tile (cycle == 100);
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

   rule check_fsum_reduce_tile (cycle == 120);
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
   rule dispatch_fmax_reduce_tile (cycle == 125);
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

   rule check_fmax_reduce_tile (cycle == 145);
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
   // Test 42: VPU_FPROD_REDUCE_TILE — float tile product via shared FpReducer.
   // Values [2.0, 3.0, 4.0, 1.0*13] -> 24.0 = 0x41C00000.
   rule dispatch_fprod_reduce_tile (cycle == 200);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(unpack(32'h3F800000)));  // 1.0 fill
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h40000000);  // 2.0
      s1[0][1] = unpack(32'h40400000);  // 3.0
      s1[0][2] = unpack(32'h40800000);  // 4.0
      vpu.execute(VPU_FPROD_REDUCE_TILE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FPROD_REDUCE_TILE", cycle);
   endrule

   rule check_fprod_reduce_tile (cycle == 220);
      let res = vpu.result;
      Bit#(32) expected = 32'h41C00000;  // 24.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (pack(res[r][c]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FPROD_REDUCE_TILE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FPROD_REDUCE_TILE got [0][0]=0x%08x (want 0x%08x)",
            cycle, pack(res[0][0]), expected);
         failed <= failed + 1;
      end
   endrule

   rule dispatch_fmin_reduce_tile (cycle == 150);
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

   rule check_fmin_reduce_tile (cycle == 170);
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

   // --- Float column reductions (single-cycle, hoisted pre-compute) ---

   // Test 39: VPU_FSUM_REDUCE_COL. Fill (r,c) = 1.0; each column sum = 4.0.
   rule dispatch_fsum_reduce_col (cycle == 78);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(unpack(32'h3F800000)));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      vpu.execute(VPU_FSUM_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_FSUM_REDUCE_COL", cycle);
   endrule

   rule check_fsum_reduce_col (cycle == 79);
      let res = vpu.result;
      Bit#(32) expected = 32'h40800000;  // 4.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (pack(res[r][c]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FSUM_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FSUM_REDUCE_COL got [0][0]=0x%08x", cycle, pack(res[0][0]));
         failed <= failed + 1;
      end
   endrule

   // Test 40: VPU_FMAX_REDUCE_COL.
   // col0 = [1.0, 2.0, 3.0, 4.0] -> max 4.0; col1 = [4,3,2,1] -> 4.0;
   // col2/3 = 0 -> max 0.0.
   rule dispatch_fmax_reduce_col (cycle == 80);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      Bit#(32) fbits[4] = {32'h3F800000, 32'h40000000, 32'h40400000, 32'h40800000};
      for (Integer r = 0; r < 4; r = r + 1) begin
         s1[r][0] = unpack(fbits[r]);
         s1[r][1] = unpack(fbits[3 - r]);
      end
      vpu.execute(VPU_FMAX_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_FMAX_REDUCE_COL", cycle);
   endrule

   rule check_fmax_reduce_col (cycle == 81);
      let res = vpu.result;
      Bit#(32) e01 = 32'h40800000;      // 4.0
      Bit#(32) e23 = 32'h00000000;      // +0.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1) begin
         if (pack(res[r][0]) != e01) ok = False;
         if (pack(res[r][1]) != e01) ok = False;
         if (pack(res[r][2]) != e23) ok = False;
         if (pack(res[r][3]) != e23) ok = False;
      end
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FMAX_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FMAX_REDUCE_COL got [0]=[0x%08x,0x%08x,0x%08x,0x%08x]",
            cycle, pack(res[0][0]), pack(res[0][1]), pack(res[0][2]), pack(res[0][3]));
         failed <= failed + 1;
      end
   endrule

   // Test 41: VPU_FMIN_REDUCE_COL.
   // col0 = [3,1,4,2] -> min 1.0. Fill other columns with 4.0.
   rule dispatch_fmin_reduce_col (cycle == 82);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(unpack(32'h40800000)));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h40400000);
      s1[1][0] = unpack(32'h3F800000);
      s1[2][0] = unpack(32'h40800000);
      s1[3][0] = unpack(32'h40000000);
      vpu.execute(VPU_FMIN_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_FMIN_REDUCE_COL", cycle);
   endrule

   rule check_fmin_reduce_col (cycle == 83);
      let res = vpu.result;
      Bit#(32) e0    = 32'h3F800000;  // 1.0
      Bit#(32) erest = 32'h40800000;  // 4.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1) begin
         if (pack(res[r][0]) != e0) ok = False;
         for (Integer c = 1; c < 4; c = c + 1)
            if (pack(res[r][c]) != erest) ok = False;
      end
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FMIN_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FMIN_REDUCE_COL got [0]=[0x%08x,0x%08x,0x%08x,0x%08x]",
            cycle, pack(res[0][0]), pack(res[0][1]), pack(res[0][2]), pack(res[0][3]));
         failed <= failed + 1;
      end
   endrule

   // Test 43: VPU_FPROD_REDUCE — per-row float product.
   // Row0 = [2.0, 3.0, 4.0, 0.5] -> 12.0 = 0x41400000.
   rule dispatch_fprod_reduce (cycle == 222);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h40000000);  // 2.0
      s1[0][1] = unpack(32'h40400000);  // 3.0
      s1[0][2] = unpack(32'h40800000);  // 4.0
      s1[0][3] = unpack(32'h3F000000);  // 0.5
      vpu.execute(VPU_FPROD_REDUCE, s1, s2);
      $display("Cycle %0d: dispatched VPU_FPROD_REDUCE", cycle);
   endrule

   rule check_fprod_reduce (cycle == 223);
      let res = vpu.result;
      Bit#(32) expected = 32'h41400000;  // 12.0
      Bool ok = True;
      for (Integer l = 0; l < 4; l = l + 1)
         if (pack(res[0][l]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FPROD_REDUCE", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FPROD_REDUCE got [0][0]=0x%08x", cycle, pack(res[0][0]));
         failed <= failed + 1;
      end
   endrule

   // Test 44: VPU_FPROD_REDUCE_COL — column float product.
   // Fill (r,c) = 2.0; each column product = 16.0 = 0x41800000.
   rule dispatch_fprod_reduce_col (cycle == 224);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(unpack(32'h40000000)));  // 2.0
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      vpu.execute(VPU_FPROD_REDUCE_COL, s1, s2);
      $display("Cycle %0d: dispatched VPU_FPROD_REDUCE_COL", cycle);
   endrule

   rule check_fprod_reduce_col (cycle == 225);
      let res = vpu.result;
      Bit#(32) expected = 32'h41800000;  // 16.0
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (pack(res[r][c]) != expected) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_FPROD_REDUCE_COL", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_FPROD_REDUCE_COL got [0][0]=0x%08x", cycle, pack(res[0][0]));
         failed <= failed + 1;
      end
   endrule

   // Test 45: VPU_EXP2 — 2^x polynomial approximation (multi-cycle walker).
   // Inputs: [0.0, 1.0, 2.0, -1.0] → expected [1.0, ~2.0, ~3.35, ~0.55].
   // Degree-2 Taylor in y=x*ln2 gives big error at |x|=2 (~16% low at 4.0)
   // and is exact at 0. Check wide bands per lane. Unit walks 16 lanes × 5
   // cycles/lane → ~80 cycles, so check well after dispatch.
   rule dispatch_exp2 (cycle == 226);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h00000000);  //  0.0
      s1[0][1] = unpack(32'h3F800000);  //  1.0
      s1[0][2] = unpack(32'h40000000);  //  2.0
      s1[0][3] = unpack(32'hBF800000);  // -1.0
      vpu.execute(VPU_EXP2, s1, s2);
      $display("Cycle %0d: dispatched VPU_EXP2", cycle);
   endrule

   rule check_exp2 (cycle == 360 && vpu.isDone);
      let res = vpu.result;
      // Range-reduced Remez EXP2: splits x = n + f, runs Remez on
      // f ∈ [-1, 1], then scales by 2^n via exponent manipulation.
      // Integer inputs give exact results.
      //   x= 0: exact 1.0
      //   x= 1: exact 2.0 (trunc=1, f=0, 2^n=2)
      //   x= 2: exact 4.0 (trunc=2, f=0, 2^n=4)
      //   x=-1: exact 0.5 (trunc=-1, f=0, 2^n=0.5)
      Float got_0 = unpack(pack(res[0][0]));
      Float got_1 = unpack(pack(res[0][1]));
      Float got_2 = unpack(pack(res[0][2]));
      Float got_3 = unpack(pack(res[0][3]));
      Float lo_0  = unpack(32'h3F7D70A4);  // 0.99
      Float hi_0  = unpack(32'h3F828F5C);  // 1.02
      Float lo_1  = unpack(32'h3FFD70A4);  // 1.98
      Float hi_1  = unpack(32'h4002851E);  // 2.04
      Float lo_2  = unpack(32'h407D70A4);  // 3.96
      Float hi_2  = unpack(32'h40825C29);  // 4.08
      Float lo_3  = unpack(32'h3EF5C28F);  // 0.48
      Float hi_3  = unpack(32'h3F051EB8);  // 0.52
      Bool ok = True;
      if (compareFP(got_0, lo_0) == LT || compareFP(got_0, hi_0) == GT) ok = False;
      if (compareFP(got_1, lo_1) == LT || compareFP(got_1, hi_1) == GT) ok = False;
      if (compareFP(got_2, lo_2) == LT || compareFP(got_2, hi_2) == GT) ok = False;
      if (compareFP(got_3, lo_3) == LT || compareFP(got_3, hi_3) == GT) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_EXP2", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_EXP2 got [0x%08x,0x%08x,0x%08x,0x%08x]",
            cycle, pack(res[0][0]), pack(res[0][1]), pack(res[0][2]), pack(res[0][3]));
         failed <= failed + 1;
      end
   endrule

   // Test 46: VPU_LOG2 — log2(x) via range-reduced polynomial in TranscUnit.
   // Inputs: [1.0, 2.0, 4.0, 0.5] → true [0, 1, 2, -1].
   // Degree-2 Taylor on (m-1) gives ~28% error at u=1 (x=2.0), ~0 at
   // exact powers of 2 below 2.0 because range-reduction exactly lands
   // them at m=1 (e contributes the integer answer). At x=2 the split
   // lands m=1, e=1; polynomial on u=0 adds 0 → returns exactly 1.0.
   rule dispatch_log2 (cycle == 370);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h3F800000);  // 1.0
      s1[0][1] = unpack(32'h40000000);  // 2.0
      s1[0][2] = unpack(32'h40800000);  // 4.0
      s1[0][3] = unpack(32'h3F000000);  // 0.5
      vpu.execute(VPU_LOG2, s1, s2);
      $display("Cycle %0d: dispatched VPU_LOG2", cycle);
   endrule

   rule check_log2 (cycle == 480 && vpu.isDone);
      let res = vpu.result;
      // Each lane's expected true log2 is 0, 1, 2, -1. Range-reduction
      // puts all test inputs at m=1 exactly (since they're powers of 2),
      // so the polynomial returns 0 and the exponent-as-float dominates.
      // Accept small wobble from FP rounding.
      Float got_0 = unpack(pack(res[0][0]));
      Float got_1 = unpack(pack(res[0][1]));
      Float got_2 = unpack(pack(res[0][2]));
      Float got_3 = unpack(pack(res[0][3]));
      Float lo_0  = unpack(32'hBD800000);  // -0.0625
      Float hi_0  = unpack(32'h3D800000);  //  0.0625
      Float lo_1  = unpack(32'h3F700000);  //  0.9375
      Float hi_1  = unpack(32'h3F880000);  //  1.0625
      Float lo_2  = unpack(32'h3FF80000);  //  1.9375
      Float hi_2  = unpack(32'h40040000);  //  2.0625
      Float lo_3  = unpack(32'hBF880000);  // -1.0625
      Float hi_3  = unpack(32'hBF700000);  // -0.9375
      Bool ok = True;
      if (compareFP(got_0, lo_0) == LT || compareFP(got_0, hi_0) == GT) ok = False;
      if (compareFP(got_1, lo_1) == LT || compareFP(got_1, hi_1) == GT) ok = False;
      if (compareFP(got_2, lo_2) == LT || compareFP(got_2, hi_2) == GT) ok = False;
      if (compareFP(got_3, lo_3) == LT || compareFP(got_3, hi_3) == GT) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_LOG2", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_LOG2 got [0x%08x,0x%08x,0x%08x,0x%08x]",
            cycle, pack(res[0][0]), pack(res[0][1]), pack(res[0][2]), pack(res[0][3]));
         failed <= failed + 1;
      end
   endrule

   // Test 47: VPU_SIN — sin(x) via degree-5 Taylor.
   // Inputs: [0.0, 0.5236, 1.5708, -0.5236] (0, π/6, π/2, -π/6)
   // → expected [0.0, 0.5, 1.0, -0.5]. Taylor degree-5 is nearly exact
   //   for |x| <= π/2 (error < 0.001).
   rule dispatch_sin (cycle == 490);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h00000000);  // 0.0
      s1[0][1] = unpack(32'h3F060A92);  // π/6  ≈ 0.5235988
      s1[0][2] = unpack(32'h3FC90FDB);  // π/2  ≈ 1.5707964
      s1[0][3] = unpack(32'hBF060A92);  // -π/6
      vpu.execute(VPU_SIN, s1, s2);
      $display("Cycle %0d: dispatched VPU_SIN", cycle);
   endrule

   rule check_sin (cycle == 600 && vpu.isDone);
      let res = vpu.result;
      Float got_0 = unpack(pack(res[0][0]));
      Float got_1 = unpack(pack(res[0][1]));
      Float got_2 = unpack(pack(res[0][2]));
      Float got_3 = unpack(pack(res[0][3]));
      Float lo_0  = unpack(32'hBD800000);  // -0.0625
      Float hi_0  = unpack(32'h3D800000);  //  0.0625
      Float lo_1  = unpack(32'h3EF33333);  //  0.475
      Float hi_1  = unpack(32'h3F066666);  //  0.525
      Float lo_2  = unpack(32'h3F70A3D7);  //  0.94
      Float hi_2  = unpack(32'h3F826666);  //  1.02
      Float lo_3  = unpack(32'hBF066666);  // -0.525
      Float hi_3  = unpack(32'hBEF33333);  // -0.475
      Bool ok = True;
      if (compareFP(got_0, lo_0) == LT || compareFP(got_0, hi_0) == GT) ok = False;
      if (compareFP(got_1, lo_1) == LT || compareFP(got_1, hi_1) == GT) ok = False;
      if (compareFP(got_2, lo_2) == LT || compareFP(got_2, hi_2) == GT) ok = False;
      if (compareFP(got_3, lo_3) == LT || compareFP(got_3, hi_3) == GT) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_SIN", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_SIN got [0x%08x,0x%08x,0x%08x,0x%08x]",
            cycle, pack(res[0][0]), pack(res[0][1]), pack(res[0][2]), pack(res[0][3]));
         failed <= failed + 1;
      end
   endrule

   // Test 48: VPU_COS — cos(x) via degree-4 Taylor.
   // Inputs: [0.0, π/3, π/2, -π/3] → expected [1.0, 0.5, 0.0, 0.5].
   // Degree-4 accurate for |x| ≤ π/2 (error < 0.02).
   rule dispatch_cos (cycle == 610);
      Vector#(4, Vector#(4, Int#(32))) s1 = replicate(replicate(0));
      Vector#(4, Vector#(4, Int#(32))) s2 = replicate(replicate(0));
      s1[0][0] = unpack(32'h00000000);  // 0.0
      s1[0][1] = unpack(32'h3F860A92);  // π/3  ≈ 1.047198
      s1[0][2] = unpack(32'h3FC90FDB);  // π/2
      s1[0][3] = unpack(32'hBF860A92);  // -π/3
      vpu.execute(VPU_COS, s1, s2);
      $display("Cycle %0d: dispatched VPU_COS", cycle);
   endrule

   rule check_cos (cycle == 710 && vpu.isDone);
      let res = vpu.result;
      Float got_0 = unpack(pack(res[0][0]));
      Float got_1 = unpack(pack(res[0][1]));
      Float got_2 = unpack(pack(res[0][2]));
      Float got_3 = unpack(pack(res[0][3]));
      Float lo_0  = unpack(32'h3F733333);  //  0.95
      Float hi_0  = unpack(32'h3F866666);  //  1.05
      Float lo_1  = unpack(32'h3ECCCCCD);  //  0.40
      Float hi_1  = unpack(32'h3F19999A);  //  0.60
      Float lo_2  = unpack(32'hBE000000);  // -0.125
      Float hi_2  = unpack(32'h3E000000);  //  0.125
      Float lo_3  = unpack(32'h3ECCCCCD);  //  0.40
      Float hi_3  = unpack(32'h3F19999A);  //  0.60
      Bool ok = True;
      if (compareFP(got_0, lo_0) == LT || compareFP(got_0, hi_0) == GT) ok = False;
      if (compareFP(got_1, lo_1) == LT || compareFP(got_1, hi_1) == GT) ok = False;
      if (compareFP(got_2, lo_2) == LT || compareFP(got_2, hi_2) == GT) ok = False;
      if (compareFP(got_3, lo_3) == LT || compareFP(got_3, hi_3) == GT) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS VPU_COS", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VPU_COS got [0x%08x,0x%08x,0x%08x,0x%08x]",
            cycle, pack(res[0][0]), pack(res[0][1]), pack(res[0][2]), pack(res[0][3]));
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 780);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
