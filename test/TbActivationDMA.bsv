package TbActivationDMA;

// ActivationDMA stub test: stream 3 synthetic vectors into the
// inactive bank of an ActivationSRAMDB, swap, verify each row-vector
// matches the 0..3, 4..7, 8..11 pattern.

import Vector            :: *;
import ActivationSRAM    :: *;
import ActivationSRAMDB  :: *;
import ActivationDMA     :: *;

(* synthesize *)
module mkTbActivationDMA();

   ActivationSRAMDB_IFC#(8, 4) db  <- mkActivationSRAMDB;
   ActivationDMA_IFC#(8, 4)    dma <- mkActivationDMA(db);

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  phase  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   rule kick_dma (cycle == 1 && phase == 0);
      dma.kick(3);
      phase <= 1;
      $display("Cycle %0d: ADMA kicked for 3 vectors", cycle);
   endrule

   rule swap_banks (phase == 1 && dma.isDone);
      db.swap;
      phase <= 2;
      $display("Cycle %0d: ADMA done; swapped banks", cycle);
   endrule

   rule read_vec0 (phase == 2);
      db.plain.readReq(0);
      phase <= 3;
   endrule

   rule check_vec0 (phase == 3);
      let d = db.plain.readResp;
      Bool ok = (d[0] == 0 && d[1] == 1 && d[2] == 2 && d[3] == 3);
      if (!ok) begin
         $display("FAIL: vec0=[%0d,%0d,%0d,%0d]", d[0], d[1], d[2], d[3]);
         failed <= failed + 1;
      end else begin
         $display("Cycle %0d: vec0 ok [0,1,2,3]", cycle);
         passed <= passed + 1;
      end
      db.plain.readReq(1);
      phase <= 4;
   endrule

   rule check_vec1 (phase == 4);
      let d = db.plain.readResp;
      Bool ok = (d[0] == 4 && d[1] == 5 && d[2] == 6 && d[3] == 7);
      if (!ok) begin
         $display("FAIL: vec1=[%0d,%0d,%0d,%0d]", d[0], d[1], d[2], d[3]);
         failed <= failed + 1;
      end else begin
         $display("Cycle %0d: vec1 ok [4,5,6,7]", cycle);
         passed <= passed + 1;
      end
      db.plain.readReq(2);
      phase <= 5;
   endrule

   rule check_vec2 (phase == 5);
      let d = db.plain.readResp;
      Bool ok = (d[0] == 8 && d[1] == 9 && d[2] == 10 && d[3] == 11);
      if (!ok) begin
         $display("FAIL: vec2=[%0d,%0d,%0d,%0d]", d[0], d[1], d[2], d[3]);
         $display("Results: %0d passed, %0d failed", passed, failed + 1);
         $finish(1);
      end
      $display("Cycle %0d: vec2 ok [8,9,10,11]", cycle);
      $display("Results: %0d passed, 0 failed", passed + 1);
      $finish(0);
   endrule

endmodule

endpackage
