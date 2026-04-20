package TbCtrlDBDMA;

// End-to-end ping-pong test: weight DMA preloads tile B in the
// inactive bank of a WeightSRAMDB while the Controller dispatches tile
// A on the active bank. After both finish, swap and dispatch tile B.
//
// Tile A: identity 4x4.
// Activation vector: [1, 2, 3, 4].
// Dispatch A result: identity * [1,2,3,4] = [1, 2, 3, 4].
//
// Tile B (synthesized by WeightDMA): W_B[r][c] = r*4 + c.
// Dispatch B result: sum_r a[r] * W_B[r][c] = [80, 90, 100, 110].

import Vector         :: *;
import WeightSRAM     :: *;
import WeightSRAMDB   :: *;
import WeightDMA      :: *;
import ActivationSRAM :: *;
import SystolicArray  :: *;
import Controller     :: *;
import PSUMBank       :: *;

(* synthesize *)
module mkTbCtrlDBDMA();

   SystolicArray_IFC#(4, 4)     array <- mkSystolicArray;
   WeightSRAMDB_IFC#(16, 4, 4)  wdb   <- mkWeightSRAMDB;
   ActivationSRAM_IFC#(16, 4)   asram <- mkActivationSRAM;
   PSUMBank_IFC#(8, 4, 4)       psum  <- mkPSUMBank;
   Controller_IFC#(4, 4, 16)    ctrl  <- mkController(array, wdb.plain, asram, psum);
   WeightDMA_IFC#(16, 4, 4)     dma   <- mkWeightDMA(wdb);

   Reg#(UInt#(16)) cycle <- mkReg(0);
   Reg#(UInt#(4))  phase <- mkReg(0);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Cycle 0: write tile A (identity) to inactive bank (mem_b; active=0).
   rule preload_a (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      wdb.plain.write(0, w);
   endrule

   // Cycle 1: preload activation [1,2,3,4].
   rule preload_acts (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      asram.write(0, a);
   endrule

   // Cycle 2: swap - bank B (holding tile A) becomes active for reads.
   rule swap_for_a (cycle == 2 && phase == 0);
      wdb.swap;
      phase <= 1;
      $display("Cycle %0d: swap #1 - tile A active", cycle);
   endrule

   // Cycle 3: dispatch A and kick DMA for tile B concurrently.
   rule dispatch_a (cycle == 3 && phase == 1);
      ctrl.start(0, 0, 1);
      // DMA writes 1 synthetic tile (idx 0) to the now-inactive bank
      // (bank A, the original mem_a). Kick issues over multiple cycles
      // in parallel with the MXU dispatch.
      dma.kick(1);
      phase <= 2;
      $display("Cycle %0d: dispatch A + kick DMA for tile B", cycle);
   endrule

   // When both A dispatch and DMA finish: verify A result, swap, dispatch B.
   rule check_a_start_b (phase == 2 && ctrl.isDone && dma.isDone);
      let r = ctrl.results;
      Bool ok = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      if (!ok) begin
         $display("FAIL: A result=[%0d,%0d,%0d,%0d] want [1,2,3,4]",
                  r[0], r[1], r[2], r[3]);
         $display("Results: 0 passed, 1 failed");
         $finish(1);
      end
      $display("Cycle %0d: tile A dispatch ok [1,2,3,4]", cycle);
      // Tile B is already in the inactive bank courtesy of DMA. Swap
      // banks so reads serve tile B.
      wdb.swap;
      ctrl.start(0, 0, 1);
      phase <= 3;
      $display("Cycle %0d: swap #2 + dispatch B", cycle);
   endrule

   rule check_b (phase == 3 && ctrl.isDone);
      let r = ctrl.results;
      Bool ok = (r[0] == 80 && r[1] == 90 && r[2] == 100 && r[3] == 110);
      if (!ok) begin
         $display("FAIL: B result=[%0d,%0d,%0d,%0d] want [80,90,100,110]",
                  r[0], r[1], r[2], r[3]);
         $display("Results: 1 passed, 1 failed");
         $finish(1);
      end
      $display("Cycle %0d: tile B dispatch ok [80,90,100,110]", cycle);
      $display("Results: 2 passed, 0 failed");
      $finish(0);
   endrule

endmodule

endpackage
