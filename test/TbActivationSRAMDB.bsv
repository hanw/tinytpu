package TbActivationSRAMDB;

import Vector           :: *;
import ActivationSRAM   :: *;
import ActivationSRAMDB :: *;

(* synthesize *)
module mkTbActivationSRAMDB();

   ActivationSRAMDB_IFC#(8, 4) as <- mkActivationSRAMDB;

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 40) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   function Vector#(4, Int#(8)) mk_row(Int#(8) base);
      Vector#(4, Int#(8)) v = newVector;
      for (Integer c = 0; c < 4; c = c + 1) v[c] = base + fromInteger(c);
      return v;
   endfunction

   rule do_w1 (cycle == 0);
      as.plain.write(0, mk_row(10));      // → inactive (B)[0] = [10,11,12,13]
   endrule
   rule do_swap1 (cycle == 2); as.swap; endrule
   rule do_r1 (cycle == 4); as.plain.readReq(0); endrule
   rule check_r1 (cycle == 6);
      let got = as.plain.readResp;
      Bool ok = (got[0] == 10 && got[1] == 11 && got[2] == 12 && got[3] == 13);
      if (ok) passed <= passed + 1; else failed <= failed + 1;
      $display("Cycle %0d: %s read-after-swap got=[%0d,%0d,%0d,%0d]",
         cycle, ok ? "PASS" : "FAIL", got[0], got[1], got[2], got[3]);
   endrule

   // Write to A while B is active; verify active reads unchanged.
   rule do_w2 (cycle == 8); as.plain.write(0, mk_row(100)); endrule
   rule do_r2 (cycle == 10); as.plain.readReq(0); endrule
   rule check_r2 (cycle == 12);
      let got = as.plain.readResp;
      Bool ok = (got[0] == 10 && got[1] == 11 && got[2] == 12 && got[3] == 13);
      if (ok) passed <= passed + 1; else failed <= failed + 1;
      $display("Cycle %0d: %s inactive-bank write", cycle, ok ? "PASS" : "FAIL");
   endrule

   // Swap and verify A shows up.
   rule do_swap2 (cycle == 14); as.swap; endrule
   rule do_r3 (cycle == 16); as.plain.readReq(0); endrule
   rule check_r3 (cycle == 18);
      let got = as.plain.readResp;
      Bool ok = (got[0] == 100 && got[1] == 101 && got[2] == 102 && got[3] == 103);
      if (ok) passed <= passed + 1; else failed <= failed + 1;
      $display("Cycle %0d: %s second swap", cycle, ok ? "PASS" : "FAIL");
   endrule

   rule finish (cycle == 25);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
