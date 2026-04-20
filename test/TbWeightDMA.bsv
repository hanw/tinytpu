package TbWeightDMA;

// Exercise the minimal WeightDMA stub. The DMA synthesizes three tiles
// into the inactive bank of a WeightSRAMDB; after kick completes the
// testbench swaps banks and verifies the three tiles are readable on
// the newly-active side.
//
// Pattern for tile idx i at (r, c): i*rows*cols + r*cols + c, truncated
// to Int#(8). For 4x4 tiles: tile[0][(0,0)..(3,3)] = 0..15,
// tile[1] = 16..31, tile[2] = 32..47.

import Vector       :: *;
import WeightSRAM   :: *;
import WeightSRAMDB :: *;
import WeightDMA    :: *;

(* synthesize *)
module mkTbWeightDMA();

   WeightSRAMDB_IFC#(8, 4, 4)   db  <- mkWeightSRAMDB;
   WeightDMA_IFC#(8, 4, 4)      dma <- mkWeightDMA(db);

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  phase  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Phase 0 (cycle 1): kick off DMA to write 3 tiles into inactive bank.
   rule kick_dma (cycle == 1 && phase == 0);
      dma.kick(3);
      phase <= 1;
      $display("Cycle %0d: DMA kicked for 3 tiles (writes to inactive bank)", cycle);
   endrule

   // Phase 1: wait for DMA to finish, then swap banks.
   rule swap_banks (phase == 1 && dma.isDone);
      db.swap;
      phase <= 2;
      $display("Cycle %0d: DMA done; swapped banks", cycle);
   endrule

   // Phase 2: issue readReq for addr 0.
   rule read_tile0_req (phase == 2);
      db.plain.readReq(0);
      phase <= 3;
   endrule

   // Phase 3: verify tile0 matches pattern (values 0..15), request tile1.
   rule check_tile0 (phase == 3);
      let d = db.plain.readResp;
      Bool ok = True;
      Int#(8) exp = 0;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1) begin
            if (d[r][c] != exp) ok = False;
            exp = exp + 1;
         end
      if (!ok) begin
         $display("FAIL: tile0 pattern mismatch row0=[%0d,%0d,%0d,%0d]",
                  d[0][0], d[0][1], d[0][2], d[0][3]);
         failed <= failed + 1;
      end else begin
         $display("Cycle %0d: tile0 ok [0..15]", cycle);
         passed <= passed + 1;
      end
      db.plain.readReq(1);
      phase <= 4;
   endrule

   // Phase 4: verify tile1 (values 16..31), request tile2.
   rule check_tile1 (phase == 4);
      let d = db.plain.readResp;
      Bool ok = True;
      Int#(8) exp = 16;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1) begin
            if (d[r][c] != exp) ok = False;
            exp = exp + 1;
         end
      if (!ok) begin
         $display("FAIL: tile1 pattern mismatch row0=[%0d,%0d,%0d,%0d]",
                  d[0][0], d[0][1], d[0][2], d[0][3]);
         failed <= failed + 1;
      end else begin
         $display("Cycle %0d: tile1 ok [16..31]", cycle);
         passed <= passed + 1;
      end
      db.plain.readReq(2);
      phase <= 5;
   endrule

   // Phase 5: verify tile2 (values 32..47). Finish.
   rule check_tile2 (phase == 5);
      let d = db.plain.readResp;
      Bool ok = True;
      Int#(8) exp = 32;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1) begin
            if (d[r][c] != exp) ok = False;
            exp = exp + 1;
         end
      if (!ok) begin
         $display("FAIL: tile2 pattern mismatch row0=[%0d,%0d,%0d,%0d]",
                  d[0][0], d[0][1], d[0][2], d[0][3]);
         $display("Results: %0d passed, %0d failed", passed, failed + 1);
         $finish(1);
      end else begin
         $display("Cycle %0d: tile2 ok [32..47]", cycle);
         $display("Results: %0d passed, 0 failed", passed + 1);
         $finish(0);
      end
   endrule

endmodule

endpackage
