package TbVMEM;

import Vector :: *;
import VMEM :: *;

(* synthesize *)
module mkTbVMEM();

   // depth=16, sublanes=4, lanes=4
   VMEM_IFC#(16, 4, 4) mem <- mkVMEM;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin
         $display("FAIL: timeout"); $finish(1);
      end
   endrule

   // Test 1: write addr 0, read it back
   rule write_0 (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) tile = replicate(replicate(0));
      tile[0][0] = 1; tile[0][1] = 2; tile[1][0] = 5; tile[3][3] = 99;
      mem.write(0, tile);
      $display("Cycle %0d: wrote tile to addr 0", cycle);
   endrule

   rule read_req_0 (cycle == 1);
      mem.readReq(0);
      $display("Cycle %0d: issued readReq addr 0", cycle);
   endrule

   rule check_0 (cycle == 2);
      let t = mem.readResp;
      Bool ok = (t[0][0] == 1 && t[0][1] == 2 && t[1][0] == 5 && t[3][3] == 99);
      if (ok) begin
         $display("Cycle %0d: PASS write/read addr 0", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL write/read addr 0 [0][0]=%0d [3][3]=%0d",
            cycle, t[0][0], t[3][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 2: write two different addresses, verify no aliasing
   rule write_addr3 (cycle == 3);
      Vector#(4, Vector#(4, Int#(32))) tileA = replicate(replicate(0));
      tileA[0][0] = 42;
      mem.write(3, tileA);
      $display("Cycle %0d: wrote addr 3 (42)", cycle);
   endrule

   rule write_addr7 (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) tileB = replicate(replicate(0));
      tileB[0][0] = 99;
      mem.write(7, tileB);
      $display("Cycle %0d: wrote addr 7 (99)", cycle);
   endrule

   rule read_req_3 (cycle == 5);
      mem.readReq(3);
   endrule

   rule check_addr3 (cycle == 6);
      let t = mem.readResp;
      Bool ok = (t[0][0] == 42);
      if (ok) begin
         $display("Cycle %0d: PASS addr 3 isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL addr 3 got %0d", cycle, t[0][0]);
         failed <= failed + 1;
      end
   endrule

   rule read_req_7 (cycle == 7);
      mem.readReq(7);
   endrule

   rule check_addr7 (cycle == 8);
      let t = mem.readResp;
      Bool ok = (t[0][0] == 99);
      if (ok) begin
         $display("Cycle %0d: PASS addr 7 isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL addr 7 got %0d", cycle, t[0][0]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 9);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
