package TbTensorCore;

import Vector :: *;
import TensorCore :: *;
import ScalarUnit :: *;
import VPU :: *;

(* synthesize *)
module mkTbTensorCore();

   // rows=4, cols=4, depth=16
   TensorCore_IFC#(4, 4, 16) tc <- mkTensorCore;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Load 4x4 identity weight matrix into WeightSRAM[0]
   rule load_weights (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      tc.loadWeightTile(0, w);
      $display("Cycle %0d: loaded identity weights", cycle);
   endrule

   // Load activation [1, 2, 3, 4] into ActivationSRAM[1]
   rule load_acts (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      tc.loadActivationTile(1, a);
      $display("Cycle %0d: loaded activations [1,2,3,4]", cycle);
   endrule

   // Load program one instruction per cycle
   // Program: DISPATCH_MXU wBase=0 aBase=1 tLen=1 → WAIT_MXU → HALT
   rule load_prog0 (cycle == 2);
      tc.loadProgram(0, SxuInstr { op: SXU_DISPATCH_MXU, vmemAddr:0, vregDst:0, vregSrc:0,
                                   vpuOp: VPU_ADD, vregSrc2:0,
                                   mxuWBase:0, mxuABase:1, mxuTLen:1 });
   endrule

   rule load_prog1 (cycle == 3);
      tc.loadProgram(1, SxuInstr { op: SXU_WAIT_MXU, vmemAddr:0, vregDst:0, vregSrc:0,
                                   vpuOp: VPU_ADD, vregSrc2:0,
                                   mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_prog2 (cycle == 4);
      tc.loadProgram(2, SxuInstr { op: SXU_HALT, vmemAddr:0, vregDst:0, vregSrc:0,
                                   vpuOp: VPU_ADD, vregSrc2:0,
                                   mxuWBase:0, mxuABase:0, mxuTLen:0 });
      $display("Cycle %0d: program loaded (3 instrs)", cycle);
   endrule

   rule start_tc (cycle == 5);
      tc.start(3);
      $display("Cycle %0d: TensorCore started", cycle);
   endrule

   rule wait_done (cycle > 5 && !tc.isDone);
      $display("Cycle %0d: computing...", cycle);
   endrule

   rule check_done (cycle > 5 && tc.isDone);
      Vector#(4, Int#(32)) out = tc.getMxuResult;
      // Identity × [1,2,3,4] = [1,2,3,4]
      Bool ok = (out[0] == 1 && out[1] == 2 && out[2] == 3 && out[3] == 4);
      if (ok) begin
         $display("Cycle %0d: PASS TensorCore GEMM [%0d,%0d,%0d,%0d]",
            cycle, out[0], out[1], out[2], out[3]);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL expected [1,2,3,4] got [%0d,%0d,%0d,%0d]",
            cycle, out[0], out[1], out[2], out[3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed",
         passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
