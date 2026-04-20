package TbCtrlDB;

// Drive mkController with WeightSRAMDB.plain / ActivationSRAMDB.plain.
// Demonstrates the preload-parallel pattern: preload tile-A into the
// inactive bank, dispatch with tile-A active, swap, preload tile-B into
// the now-inactive bank while reading tile-A (read not disturbed), swap
// again, dispatch tile-B. The double-buffered SRAMs transparently
// replace the plain SRAMs at the Controller interface.

import Vector             :: *;
import SystolicArray      :: *;
import WeightSRAM         :: *;
import ActivationSRAM     :: *;
import WeightSRAMDB       :: *;
import ActivationSRAMDB   :: *;
import Controller         :: *;
import PSUMBank           :: *;

(* synthesize *)
module mkTbCtrlDB();

   SystolicArray_IFC#(4, 4)         array  <- mkSystolicArray;
   WeightSRAMDB_IFC#(16, 4, 4)      wdb    <- mkWeightSRAMDB;
   ActivationSRAMDB_IFC#(16, 4)     adb    <- mkActivationSRAMDB;
   PSUMBank_IFC#(8, 4, 4)           psum   <- mkPSUMBank;
   Controller_IFC#(4, 4, 16)        ctrl   <- mkController(array, wdb.plain, adb.plain, psum);

   Reg#(UInt#(16)) cycle <- mkReg(0);
   Reg#(UInt#(3))  phase <- mkReg(0);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Preload 1: write identity weights + activations [1,2,3,4] into the
   // inactive bank. Active bank starts as A (reads untouched), so writes
   // go to B.
   rule preload_1 (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      wdb.plain.write(0, w);
   endrule
   rule preload_1a (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      adb.plain.write(0, a);
   endrule

   // Swap tile-A into the active banks.
   rule swap_1 (cycle == 2 && phase == 0);
      wdb.swap;
      adb.swap;
      $display("Cycle %0d: swap banks - tile A now active", cycle);
   endrule

   // Fire dispatch reading tile A.
   rule dispatch_a (cycle == 3 && phase == 0);
      ctrl.start(0, 0, 1);
      phase <= 1;
      $display("Cycle %0d: dispatch tile A", cycle);
   endrule

   // While dispatch is in flight, preload tile-B (weights scaled by 2,
   // activations [5,6,7,8]) into the now-inactive bank. Read port on
   // the active bank should be undisturbed.
   rule preload_2 (cycle == 5);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 2; w[1][1] = 2; w[2][2] = 2; w[3][3] = 2;
      wdb.plain.write(0, w);
      $display("Cycle %0d: preload tile B weights into inactive bank", cycle);
   endrule
   rule preload_2a (cycle == 6);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 5; a[1] = 6; a[2] = 7; a[3] = 8;
      adb.plain.write(0, a);
      $display("Cycle %0d: preload tile B activations", cycle);
   endrule

   // When dispatch-A completes, check result = [1,2,3,4].
   rule check_a (phase == 1 && ctrl.isDone);
      let r = ctrl.results;
      Bool ok = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      if (!ok) begin
         $display("FAIL tile A: [%0d,%0d,%0d,%0d]", r[0], r[1], r[2], r[3]);
         $finish(1);
      end
      $display("Cycle %0d: PASS tile A [%0d,%0d,%0d,%0d]",
               cycle, r[0], r[1], r[2], r[3]);
      phase <= 2;
   endrule

   // Swap to tile-B and dispatch again.
   rule swap_2 (phase == 2 && ctrl.isDone);
      wdb.swap;
      adb.swap;
      phase <= 3;
      $display("Cycle %0d: swap banks - tile B now active", cycle);
   endrule

   rule dispatch_b (phase == 3 && ctrl.isDone);
      ctrl.start(0, 0, 1);
      phase <= 4;
      $display("Cycle %0d: dispatch tile B", cycle);
   endrule

   // Tile B: weights=2*I, activations=[5..8] -> MXU output = [10,12,14,16].
   rule check_b (phase == 4 && ctrl.isDone);
      let r = ctrl.results;
      Bool ok = (r[0] == 10 && r[1] == 12 && r[2] == 14 && r[3] == 16);
      if (ok) begin
         $display("Cycle %0d: PASS tile B [%0d,%0d,%0d,%0d]",
                  cycle, r[0], r[1], r[2], r[3]);
         $display("Results: 2 passed, 0 failed");
         $finish(0);
      end else begin
         $display("FAIL tile B: [%0d,%0d,%0d,%0d]", r[0], r[1], r[2], r[3]);
         $display("Results: 1 passed, 1 failed");
         $finish(1);
      end
   endrule

endmodule

endpackage
