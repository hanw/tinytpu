package TbTinyTPUChip;

import Vector :: *;
import TinyTPUChip :: *;
import TensorCore :: *;
import ScalarUnit :: *;
import VPU :: *;

(* synthesize *)
module mkTbTinyTPUChip();

   TinyTPUChip_IFC chip <- mkTinyTPUChip;

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 300) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Step 1: Load TC0 identity weights into WeightSRAM[0]
   rule setup_weights (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0]=1; w[1][1]=1; w[2][2]=1; w[3][3]=1;
      chip.loadTC0Weights(0, w);
      $display("Cycle %0d: TC0 identity weights loaded", cycle);
   endrule

   // Step 2: Load TC0 activations [1,2,3,4] into ActivationSRAM[1]
   rule setup_acts (cycle == 1);
      Vector#(4, Int#(8)) a = newVector;
      a[0]=1; a[1]=2; a[2]=3; a[3]=4;
      chip.loadTC0Activations(1, a);
      $display("Cycle %0d: TC0 activations [1,2,3,4] loaded", cycle);
   endrule

   // Step 3: Load SparseCore embedding table
   // embedding[1] = [10, 20, 30, 40]
   rule setup_sc (cycle == 2);
      Vector#(4, Int#(32)) emb = replicate(0);
      emb[0]=10; emb[1]=20; emb[2]=30; emb[3]=40;
      chip.loadSCEmbedding(1, emb);
      $display("Cycle %0d: SC embedding[1]=[10,20,30,40]", cycle);
   endrule

   // Step 4: Load TC0 program (one instruction per cycle)
   // Program: DISPATCH_MXU(wBase=0, aBase=1, tLen=1) -> WAIT_MXU -> HALT
   rule load_prog0 (cycle == 3);
      chip.loadTC0Program(0, SxuInstr { op: SXU_DISPATCH_MXU, vmemAddr:0,
                                        vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0,
                                        mxuWBase:0, mxuABase:1, mxuTLen:1 });
   endrule

   rule load_prog1 (cycle == 4);
      chip.loadTC0Program(1, SxuInstr { op: SXU_WAIT_MXU, vmemAddr:0,
                                        vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0,
                                        mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_prog2 (cycle == 5);
      chip.loadTC0Program(2, SxuInstr { op: SXU_HALT, vmemAddr:0,
                                        vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0,
                                        mxuWBase:0, mxuABase:0, mxuTLen:0 });
      $display("Cycle %0d: TC0 program loaded", cycle);
   endrule

   // Step 5: Start TC0
   rule start_tc0 (cycle == 6);
      chip.startTC0(3);
      $display("Cycle %0d: TC0 started", cycle);
   endrule

   rule wait_tc0 (cycle > 6 && !chip.tc0Done);
      $display("Cycle %0d: TC0 computing...", cycle);
   endrule

   // Step 6: TC0 done — use result[0] (=1) as sparse index to look up embedding[1]
   Reg#(Bool) forwarded <- mkReg(False);

   rule forward_to_sc (chip.tc0Done && !forwarded);
      forwarded <= True;
      // result[0] = 1 (first output of identity GEMM with input [1,2,3,4])
      chip.forwardTC0ResultToSC(0);
      $display("Cycle %0d: TC0 done, forwarding result[0] as SC index", cycle);
   endrule

   rule wait_sc (forwarded && !chip.scDone);
      $display("Cycle %0d: SC computing...", cycle);
   endrule

   // Step 7: SC done — verify embedding[1] = [10, 20, 30, 40]
   rule check_sc (forwarded && chip.scDone);
      Vector#(4, Int#(32)) emb = chip.getSCResult;
      Bool ok = (emb[0] == 10 && emb[1] == 20 && emb[2] == 30 && emb[3] == 40);
      if (ok) begin
         $display("Cycle %0d: PASS chip pipeline [10,20,30,40]", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL chip result [%0d,%0d,%0d,%0d]",
            cycle, emb[0], emb[1], emb[2], emb[3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed",
         passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
