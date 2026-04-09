package TbHBMModel;

import Vector :: *;
import HBMModel :: *;

(* synthesize *)
module mkTbHBMModel();

   // depth=64, sublanes=4, lanes=4, latency=3
   HBMModel_IFC#(64, 4, 4, 3) hbm <- mkHBMModel;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Write to addr 10
   rule write_data (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) t = replicate(replicate(0));
      t[0][0] = 42; t[2][3] = 77;
      hbm.write(10, t);
      $display("Cycle %0d: wrote addr 10 (42, 77)", cycle);
   endrule

   // Issue readReq at cycle 1 — response valid at cycle 1+latency = cycle 4
   rule read_req (cycle == 1);
      hbm.readReq(10);
      $display("Cycle %0d: issued readReq addr 10", cycle);
   endrule

   // Check at cycle 4 (latency=3 after req at cycle 1)
   rule check_resp (cycle == 4);
      let t = hbm.readResp;
      Bool ok = (t[0][0] == 42 && t[2][3] == 77);
      if (ok) begin
         $display("Cycle %0d: PASS HBM read with latency-3", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL HBM got [0][0]=%0d [2][3]=%0d",
            cycle, t[0][0], t[2][3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 5);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
