package TbVRegFile;

import Vector :: *;
import VRegFile :: *;

(* synthesize *)
module mkTbVRegFile();

   // 16 vregs, sublanes=4, lanes=4
   VRegFile_IFC#(16, 4, 4) vrf <- mkVRegFile;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin
         $display("FAIL: timeout"); $finish(1);
      end
   endrule

   // Test 1: write vreg 0, read back next cycle
   rule write_vreg0 (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) v = replicate(replicate(0));
      v[0][0] = 7; v[1][2] = 13; v[3][3] = 55;
      vrf.write(0, v);
      $display("Cycle %0d: wrote vreg 0", cycle);
   endrule

   rule check_vreg0 (cycle == 1);
      let v = vrf.read(0);
      Bool ok = (v[0][0] == 7 && v[1][2] == 13 && v[3][3] == 55);
      if (ok) begin
         $display("Cycle %0d: PASS write/read vreg 0", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL vreg 0: [0][0]=%0d [3][3]=%0d",
            cycle, v[0][0], v[3][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 2: write vregs 3 and 12 in separate cycles, verify no cross-contamination
   rule write_vreg3 (cycle == 2);
      Vector#(4, Vector#(4, Int#(32))) vA = replicate(replicate(0));
      vA[0][0] = 100;
      vrf.write(3, vA);
      $display("Cycle %0d: wrote vreg 3=100", cycle);
   endrule

   rule write_vreg12 (cycle == 3);
      Vector#(4, Vector#(4, Int#(32))) vB = replicate(replicate(0));
      vB[0][0] = 200;
      vrf.write(12, vB);
      $display("Cycle %0d: wrote vreg 12=200", cycle);
   endrule

   rule check_multi (cycle == 4);
      let vA = vrf.read(3);
      let vB = vrf.read(12);
      Bool ok = (vA[0][0] == 100 && vB[0][0] == 200);
      if (ok) begin
         $display("Cycle %0d: PASS multi-register isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL vreg3=%0d vreg12=%0d", cycle, vA[0][0], vB[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Test 3: overwrite vreg 3, verify vreg 12 unchanged
   rule overwrite (cycle == 5);
      Vector#(4, Vector#(4, Int#(32))) vNew = replicate(replicate(0));
      vNew[0][0] = 999;
      vrf.write(3, vNew);
      $display("Cycle %0d: overwrote vreg 3 with 999", cycle);
   endrule

   rule check_overwrite (cycle == 6);
      let vA = vrf.read(3);
      let vB = vrf.read(12);
      Bool ok = (vA[0][0] == 999 && vB[0][0] == 200);
      if (ok) begin
         $display("Cycle %0d: PASS overwrite isolation", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL overwrite: vreg3=%0d vreg12=%0d",
            cycle, vA[0][0], vB[0][0]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 7);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
