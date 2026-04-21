package TbCtrlAccumulate;

// Exercise the Controller startAccumulate() path — WS feed with the
// drain-time PE accumulator clear skipped. The observable contract:
//   * startAccumulate() accepts (weight, act, tileLen) like start()
//   * getDataflowMode() returns DF_WEIGHT_STATIONARY_ACCUMULATE while
//     the dispatch is in flight and remains latched after completion
//   * MXU output for identity-weights * [1..4] is [1,2,3,4]
//   * A second startAccumulate() keeps the previous psum, so col-sums
//     become [2,4,6,8]
//   * clearArray() + startAccumulate() returns fresh [1,2,3,4]
//   * A subsequent start() (WS) flips the mode back to DF_WEIGHT_STATIONARY

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;
import PSUMBank :: *;

(* synthesize *)
module mkTbCtrlAccumulate();

   SystolicArray_IFC#(4, 4)   array  <- mkSystolicArray;
   WeightSRAM_IFC#(16, 4, 4)  wsram  <- mkWeightSRAM;
   ActivationSRAM_IFC#(16, 4) asram  <- mkActivationSRAM;
   PSUMBank_IFC#(8, 4, 4)     psum   <- mkPSUMBank;
   Controller_IFC#(4, 4, 16)  ctrl   <- mkController(array, wsram, asram, psum);

   Reg#(UInt#(16)) cycle <- mkReg(0);
   Reg#(UInt#(4))  phase <- mkReg(0);
   Reg#(Bool)      saw_mode <- mkReg(False);

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 400) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   rule load_w (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      wsram.write(0, w);
   endrule

   rule load_a (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      asram.write(0, a);
   endrule

   rule dispatch_1 (cycle == 2 && phase == 0);
      if (ctrl.getDataflowMode != DF_WEIGHT_STATIONARY) begin
         $display("FAIL: initial mode != WS"); $finish(1);
      end
      ctrl.startAccumulate(0, 0, 1);
      phase <= 1;
      $display("Cycle %0d: startAccumulate #1", cycle);
   endrule

   rule observe_mode (phase == 1 && !saw_mode);
      if (ctrl.getDataflowMode == DF_WEIGHT_STATIONARY_ACCUMULATE)
         saw_mode <= True;
   endrule

   rule finish_1 (phase == 1 && ctrl.isDone);
      let r = ctrl.results;
      Bool val_ok = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      Bool mode_ok = (ctrl.getDataflowMode == DF_WEIGHT_STATIONARY_ACCUMULATE);
      if (!val_ok || !mode_ok || !saw_mode) begin
         $display("FAIL #1: val=[%0d,%0d,%0d,%0d] mode_ok=%0d saw=%0d",
                  r[0], r[1], r[2], r[3], pack(mode_ok), pack(saw_mode));
         $finish(1);
      end
      $display("Cycle %0d: #1 result=[%0d,%0d,%0d,%0d]",
               cycle, r[0], r[1], r[2], r[3]);
      phase <= 2;
   endrule

   rule dispatch_2 (phase == 2 && ctrl.isDone);
      ctrl.startAccumulate(0, 0, 1);
      phase <= 3;
      $display("Cycle %0d: startAccumulate #2 (expect +prev)", cycle);
   endrule

   rule finish_2 (phase == 3 && ctrl.isDone);
      let r = ctrl.results;
      Bool val_ok = (r[0] == 2 && r[1] == 4 && r[2] == 6 && r[3] == 8);
      if (!val_ok) begin
         $display("FAIL #2 accumulate: got [%0d,%0d,%0d,%0d] want [2,4,6,8]",
                  r[0], r[1], r[2], r[3]);
         $display("Results: 0 passed, 1 failed"); $finish(1);
      end
      $display("Cycle %0d: #2 accumulated [%0d,%0d,%0d,%0d]",
               cycle, r[0], r[1], r[2], r[3]);
      phase <= 4;
   endrule

   rule clear_then_3 (phase == 4 && ctrl.isDone);
      ctrl.clearArray;
      phase <= 5;
      $display("Cycle %0d: clearArray", cycle);
   endrule

   rule dispatch_3 (phase == 5 && ctrl.isDone);
      ctrl.startAccumulate(0, 0, 1);
      phase <= 6;
      $display("Cycle %0d: startAccumulate #3 (after clear)", cycle);
   endrule

   rule finish_3 (phase == 6 && ctrl.isDone);
      let r = ctrl.results;
      Bool val_ok = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      if (!val_ok) begin
         $display("FAIL #3 after-clear: [%0d,%0d,%0d,%0d] want [1,2,3,4]",
                  r[0], r[1], r[2], r[3]);
         $display("Results: 0 passed, 1 failed"); $finish(1);
      end
      phase <= 7;
      $display("Cycle %0d: #3 fresh [%0d,%0d,%0d,%0d]",
               cycle, r[0], r[1], r[2], r[3]);
   endrule

   rule dispatch_ws (phase == 7 && ctrl.isDone);
      ctrl.start(0, 0, 1);
      phase <= 8;
      $display("Cycle %0d: dispatch WS", cycle);
   endrule

   rule finish_ws (phase == 8 && ctrl.isDone);
      let r = ctrl.results;
      Bool val_ok  = (r[0] == 1 && r[1] == 2 && r[2] == 3 && r[3] == 4);
      Bool mode_ok = (ctrl.getDataflowMode == DF_WEIGHT_STATIONARY);
      if (val_ok && mode_ok) begin
         $display("Cycle %0d: PASS accumulate + clear + WS toggle", cycle);
         $display("Results: 1 passed, 0 failed");
         $finish(0);
      end else begin
         $display("FAIL WS-after-accumulate val_ok=%0d mode_ok=%0d",
                  pack(val_ok), pack(mode_ok));
         $display("Results: 0 passed, 1 failed");
         $finish(1);
      end
   endrule

endmodule

endpackage
