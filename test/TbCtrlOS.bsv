package TbCtrlOS;

// Exercise the new startOS() Controller path. The FSM still runs
// weight-stationary behavior (operand-swap + PE accumulator-hold land
// in later iters); the observable contract today is:
//   * startOS() accepts weight/act/tileLen same as start()
//   * getDataflowMode() returns DF_OUTPUT_STATIONARY while the dispatch
//     is in flight and remains latched after completion
//   * MXU output for identity-weights * [1..4] is [1,2,3,4] (WS semantics)
//   * A subsequent start() (WS) flips the mode back to DF_WEIGHT_STATIONARY

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;
import PSUMBank :: *;

(* synthesize *)
module mkTbCtrlOS();

   SystolicArray_IFC#(4, 4)   array  <- mkSystolicArray;
   WeightSRAM_IFC#(16, 4, 4)  wsram  <- mkWeightSRAM;
   ActivationSRAM_IFC#(16, 4) asram  <- mkActivationSRAM;
   PSUMBank_IFC#(8, 4, 4)     psum   <- mkPSUMBank;
   Controller_IFC#(4, 4, 16)  ctrl   <- mkController(array, wsram, asram, psum);

   Reg#(UInt#(16)) cycle <- mkReg(0);
   Reg#(UInt#(2))  phase <- mkReg(0);
   Reg#(Bool)      saw_os_mode <- mkReg(False);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Weights = identity int8 4x4.
   rule load_w (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      wsram.write(0, w);
   endrule

   // Activations [1,2,3,4].
   rule load_a (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      asram.write(0, a);
   endrule

   // Phase 0: fire startOS. Default mode must be WS beforehand.
   rule dispatch_os (cycle == 2 && phase == 0);
      if (ctrl.getDataflowMode != DF_WEIGHT_STATIONARY) begin
         $display("FAIL: initial mode != WS");
         $finish(1);
      end
      ctrl.startOS(0, 0, 1);
      phase <= 1;
      $display("Cycle %0d: dispatch OS", cycle);
   endrule

   // While OS dispatch runs, sample the mode reg. It must read OS.
   rule observe_os_mode (phase == 1 && !saw_os_mode);
      if (ctrl.getDataflowMode == DF_OUTPUT_STATIONARY) begin
         saw_os_mode <= True;
      end
   endrule

   // Phase 1 → 2: when OS dispatch completes, check result + latched mode.
   rule finish_os (phase == 1 && ctrl.isDone);
      let r = ctrl.results;
      Bool val_ok  = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      Bool mode_ok = (ctrl.getDataflowMode == DF_OUTPUT_STATIONARY);
      if (!val_ok) begin
         $display("FAIL: OS WS-equivalent output row=[%0d,%0d,%0d,%0d]",
                  r[0], r[1], r[2], r[3]);
         $finish(1);
      end
      if (!mode_ok) begin
         $display("FAIL: getDataflowMode after startOS is not OS");
         $finish(1);
      end
      if (!saw_os_mode) begin
         $display("FAIL: never observed OS mode during dispatch");
         $finish(1);
      end
      $display("Cycle %0d: OS dispatch complete result=[%0d,%0d,%0d,%0d]",
               cycle, r[0], r[1], r[2], r[3]);
      phase <= 2;
   endrule

   // Phase 2: fire a regular start() which must flip mode back to WS.
   rule dispatch_ws (phase == 2 && ctrl.isDone);
      ctrl.start(0, 0, 1);
      phase <= 3;
      $display("Cycle %0d: dispatch WS", cycle);
   endrule

   rule finish_ws (phase == 3 && ctrl.isDone);
      let r = ctrl.results;
      Bool val_ok  = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      Bool mode_ok = (ctrl.getDataflowMode == DF_WEIGHT_STATIONARY);
      if (val_ok && mode_ok) begin
         $display("Cycle %0d: PASS startOS+start toggle dataflow mode",
                  cycle);
         $display("Results: 1 passed, 0 failed");
         $finish(0);
      end else begin
         $display("FAIL: WS-after-OS val_ok=%0d mode_ok=%0d",
                  pack(val_ok), pack(mode_ok));
         $display("Results: 0 passed, 1 failed");
         $finish(1);
      end
   endrule

endmodule

endpackage
