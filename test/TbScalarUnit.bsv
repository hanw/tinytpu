package TbScalarUnit;

import Vector :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import ScalarUnit :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;

(* synthesize *)
module mkTbScalarUnit();

   VMEM_IFC#(16, 4, 4)     vmem <- mkVMEM;
   VRegFile_IFC#(8, 4, 4)  vrf  <- mkVRegFile;
   VPU_IFC#(4, 4)          vpu  <- mkVPU;
   XLU_IFC#(4, 4)          xlu  <- mkXLU;

   // Stub Controller for the SXU (not used by this test, just required by interface)
   SystolicArray_IFC#(4, 4)   arr   <- mkSystolicArray;
   WeightSRAM_IFC#(16, 4, 4)  wsram <- mkWeightSRAM;
   ActivationSRAM_IFC#(16, 4) asram <- mkActivationSRAM;
   Controller_IFC#(4, 4, 16)  ctrl  <- mkController(arr, wsram, asram);

   // progDepth=16: room for both the VPU smoke test and XLU broadcast.
   SXU_IFC#(16, 16, 8, 4, 4) sxu <- mkScalarUnit(vmem, vrf, vpu, xlu, ctrl);

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 100) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Preload VMEM — one write per cycle (RegFile single-port)
   // VMEM[0]: row0 = [1, 2, 3, 4]
   rule preload_vmem0 (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) tA = replicate(replicate(0));
      tA[0][0]=1; tA[0][1]=2; tA[0][2]=3; tA[0][3]=4;
      vmem.write(0, tA);
      $display("Cycle %0d: VMEM[0] = [1,2,3,4]", cycle);
   endrule

   // VMEM[1]: row0 = [10, 20, 30, 40]
   rule preload_vmem1 (cycle == 1);
      Vector#(4, Vector#(4, Int#(32))) tB = replicate(replicate(0));
      tB[0][0]=10; tB[0][1]=20; tB[0][2]=30; tB[0][3]=40;
      vmem.write(1, tB);
      $display("Cycle %0d: VMEM[1] = [10,20,30,40]", cycle);
   endrule

   // VMEM[3]: row0 = [9, 8, 7, 6]
   rule preload_vmem3 (cycle == 2);
      Vector#(4, Vector#(4, Int#(32))) tC = replicate(replicate(0));
      tC[0][0]=9; tC[0][1]=8; tC[0][2]=7; tC[0][3]=6;
      vmem.write(3, tC);
      $display("Cycle %0d: VMEM[3] = [9,8,7,6]", cycle);
   endrule

   // Load program: one instruction per cycle
   // Program:
   // LOAD 0→v0, LOAD 1→v1, ADD v0 v1→v2, STORE v2→2,
   // LOAD 3→v3, XLU_BROADCAST lane1 of v3→v4, STORE v4→4, HALT
   rule load_instr0 (cycle == 3);
      sxu.loadInstr(0, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr1 (cycle == 4);
      sxu.loadInstr(1, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:1, vregDst:1, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr2 (cycle == 5);
      sxu.loadInstr(2, SxuInstr { op: SXU_DISPATCH_VPU, vmemAddr:0, vregDst:2, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:1, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr3 (cycle == 6);
      sxu.loadInstr(3, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:2, vregDst:0, vregSrc:2, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr4 (cycle == 7);
      sxu.loadInstr(4, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:3, vregDst:3, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr5 (cycle == 8);
      sxu.loadInstr(5, SxuInstr { op: SXU_DISPATCH_XLU_BROADCAST, vmemAddr:0, vregDst:4, vregSrc:3, vpuOp:VPU_ADD, vregSrc2:1, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr6 (cycle == 9);
      sxu.loadInstr(6, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:4, vregDst:0, vregSrc:4, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr7 (cycle == 10);
      sxu.loadInstr(7, SxuInstr { op: SXU_HALT,         vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
      $display("Cycle %0d: program loaded (8 instrs)", cycle);
   endrule

   rule start_sxu (cycle == 11);
      sxu.start(8);
      $display("Cycle %0d: SXU started", cycle);
   endrule

   rule wait_sxu (cycle > 11 && !sxu.isDone);
      $display("Cycle %0d: SXU running...", cycle);
   endrule

   // Issue VMEM reads once SXU halts
   Reg#(UInt#(2)) readPhase <- mkReg(0);

   rule issue_read_sum (sxu.isDone && readPhase == 0);
      vmem.readReq(2);
      $display("Cycle %0d: SXU done, issued readReq VMEM[2]", cycle);
      readPhase <= 1;
   endrule

   rule check_vmem2 (readPhase == 1);
      let t = vmem.readResp;
      Bool ok = (t[0][0] == 11 && t[0][1] == 22 && t[0][2] == 33 && t[0][3] == 44);
      if (ok) begin
         $display("Cycle %0d: PASS SXU program: VMEM[2]=[11,22,33,44]", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VMEM[2] row0=[%0d,%0d,%0d,%0d]",
            cycle, t[0][0], t[0][1], t[0][2], t[0][3]);
         failed <= failed + 1;
      end
      vmem.readReq(4);
      $display("Cycle %0d: issued readReq VMEM[4]", cycle);
      readPhase <= 2;
   endrule

   rule check_vmem4 (readPhase == 2);
      let t = vmem.readResp;
      Bool ok = (t[0][0] == 8 && t[0][1] == 8 && t[0][2] == 8 && t[0][3] == 8);
      if (ok) begin
         $display("Cycle %0d: PASS SXU broadcast program: VMEM[4]=[8,8,8,8]", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VMEM[4] row0=[%0d,%0d,%0d,%0d]",
            cycle, t[0][0], t[0][1], t[0][2], t[0][3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed", passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (failed == 0 && ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
