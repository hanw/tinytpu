package TbWeightSRAMDB;

import Vector       :: *;
import WeightSRAMDB :: *;

(* synthesize *)
module mkTbWeightSRAMDB();

   WeightSRAMDB_IFC#(8, 4, 4) ws <- mkWeightSRAMDB;

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   function Vector#(4, Vector#(4, Int#(8))) mk_tile(Int#(8) base);
      Vector#(4, Vector#(4, Int#(8))) t = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            t[r][c] = base + fromInteger(r * 4 + c);
      return t;
   endfunction

   // Step 1: write tile A to addr 0 (goes to INACTIVE bank = bank B).
   //         write tile X to addr 0 of active bank via swap later.
   // Step 2: swap; now bank B is active.
   // Step 3: read addr 0 → should return tile A (the write above).
   rule do_step1 (cycle == 0);
      ws.write(0, mk_tile(1));       // bank B[0] = [1..16]
      $display("Cycle %0d: write bank-B[0]=[1..16]", cycle);
   endrule

   rule do_step2 (cycle == 2);
      ws.swap;                        // now bank B is active
      $display("Cycle %0d: swap", cycle);
   endrule

   rule do_step3 (cycle == 4);
      ws.readReq(0);                  // read active (B) [0]
   endrule

   rule check_step3 (cycle == 6);
      let got = ws.readResp;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (got[r][c] != fromInteger(1 + r * 4 + c)) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS read-after-swap", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL read-after-swap got[0][0]=%0d", cycle, got[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Step 4: write tile X to addr 0 while B is active (goes to INACTIVE A).
   rule do_step4 (cycle == 8);
      ws.write(0, mk_tile(100));     // bank A[0] = [100..115]
      $display("Cycle %0d: write bank-A[0]=[100..115]", cycle);
   endrule

   rule do_step5 (cycle == 10);
      ws.readReq(0);                 // still reading active B[0]
   endrule

   rule check_unchanged (cycle == 12);
      let got = ws.readResp;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (got[r][c] != fromInteger(1 + r * 4 + c)) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS read not disturbed by inactive-bank write", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL read disturbed got[0][0]=%0d", cycle, got[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Step 6: swap again, read A[0] → should return [100..115].
   rule do_step6 (cycle == 14);
      ws.swap;
   endrule

   rule do_step7 (cycle == 16);
      ws.readReq(0);
   endrule

   rule check_swap2 (cycle == 18);
      let got = ws.readResp;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (got[r][c] != fromInteger(100 + r * 4 + c)) ok = False;
      if (ok) begin
         $display("Cycle %0d: PASS second swap", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL second swap got[0][0]=%0d", cycle, got[0][0]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 30);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
