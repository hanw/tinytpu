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

   // progDepth=16: room for VPU smoke, XLU broadcast, and SELECT.
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

   // VMEM[5]: cond row0 = [1, 0, 2, 0]
   rule preload_vmem5 (cycle == 3);
      Vector#(4, Vector#(4, Int#(32))) tCond = replicate(replicate(0));
      tCond[0][0]=1; tCond[0][1]=0; tCond[0][2]=2; tCond[0][3]=0;
      vmem.write(5, tCond);
      $display("Cycle %0d: VMEM[5] = [1,0,2,0]", cycle);
   endrule

   // VMEM[6]: lhs row0 = [10, 20, 30, 40]
   rule preload_vmem6 (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) tLhs = replicate(replicate(0));
      tLhs[0][0]=10; tLhs[0][1]=20; tLhs[0][2]=30; tLhs[0][3]=40;
      vmem.write(6, tLhs);
      $display("Cycle %0d: VMEM[6] = [10,20,30,40]", cycle);
   endrule

   // VMEM[7]: rhs row0 = [5, 6, 7, 8]
   rule preload_vmem7 (cycle == 5);
      Vector#(4, Vector#(4, Int#(32))) tRhs = replicate(replicate(0));
      tRhs[0][0]=5; tRhs[0][1]=6; tRhs[0][2]=7; tRhs[0][3]=8;
      vmem.write(7, tRhs);
      $display("Cycle %0d: VMEM[7] = [5,6,7,8]", cycle);
   endrule

   // Load program: one instruction per cycle
   // Program:
   // LOAD 0→v0, LOAD 1→v1, ADD v0 v1→v2, STORE v2→2,
   // LOAD 3→v3, XLU_BROADCAST lane1 of v3→v4, STORE v4→4,
   // LOAD 5→v5, LOAD 6→v6, LOAD 7→v7, SELECT(cond=v5,lhs=v6,rhs=v7)→v0, STORE v0→8, HALT
   rule load_instr0 (cycle == 6);
      sxu.loadInstr(0, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr1 (cycle == 7);
      sxu.loadInstr(1, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:1, vregDst:1, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr2 (cycle == 8);
      sxu.loadInstr(2, SxuInstr { op: SXU_DISPATCH_VPU, vmemAddr:0, vregDst:2, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:1, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr3 (cycle == 9);
      sxu.loadInstr(3, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:2, vregDst:0, vregSrc:2, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr4 (cycle == 10);
      sxu.loadInstr(4, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:3, vregDst:3, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr5 (cycle == 11);
      sxu.loadInstr(5, SxuInstr { op: SXU_DISPATCH_XLU_BROADCAST, vmemAddr:0, vregDst:4, vregSrc:3, vpuOp:VPU_ADD, vregSrc2:1, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr6 (cycle == 12);
      sxu.loadInstr(6, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:4, vregDst:0, vregSrc:4, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr7 (cycle == 13);
      sxu.loadInstr(7, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:5, vregDst:5, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr8 (cycle == 14);
      sxu.loadInstr(8, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:6, vregDst:6, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr9 (cycle == 15);
      sxu.loadInstr(9, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:7, vregDst:7, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr10 (cycle == 16);
      sxu.loadInstr(10, SxuInstr { op: SXU_DISPATCH_SELECT, vmemAddr:0, vregDst:0, vregSrc:5, vpuOp:VPU_ADD, vregSrc2:6, mxuWBase:7, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr11 (cycle == 17);
      sxu.loadInstr(11, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:8, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule load_instr12 (cycle == 18);
      sxu.loadInstr(12, SxuInstr { op: SXU_HALT,         vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
      $display("Cycle %0d: program loaded (13 instrs)", cycle);
   endrule

   rule start_sxu (cycle == 19);
      sxu.start(13);
      $display("Cycle %0d: SXU started", cycle);
   endrule

   rule wait_sxu (cycle > 19 && !sxu.isDone);
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
      vmem.readReq(8);
      $display("Cycle %0d: issued readReq VMEM[8]", cycle);
      readPhase <= 3;
   endrule

   rule check_vmem8 (readPhase == 3);
      let t = vmem.readResp;
      Bool ok = (t[0][0] == 10 && t[0][1] == 6 && t[0][2] == 30 && t[0][3] == 8);
      if (ok) begin
         $display("Cycle %0d: PASS SXU select program: VMEM[8]=[10,6,30,8]", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL VMEM[8] row0=[%0d,%0d,%0d,%0d]",
            cycle, t[0][0], t[0][1], t[0][2], t[0][3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed", passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (failed == 0 && ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
