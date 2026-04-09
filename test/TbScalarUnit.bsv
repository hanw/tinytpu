package TbScalarUnit;

import Vector :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import ScalarUnit :: *;

(* synthesize *)
module mkTbScalarUnit();

   VMEM_IFC#(16, 4, 4)     vmem <- mkVMEM;
   VRegFile_IFC#(8, 4, 4)  vrf  <- mkVRegFile;
   VPU_IFC#(4, 4)          vpu  <- mkVPU;
   XLU_IFC#(4, 4)          xlu  <- mkXLU;

   // progDepth=8: up to 8 instructions
   SXU_IFC#(8, 16, 8, 4, 4) sxu <- mkScalarUnit(vmem, vrf, vpu, xlu);

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

   // Load program: one instruction per cycle
   // Program: LOAD 0→v0, LOAD 1→v1, DISPATCH_VPU ADD v0 v1→v2, STORE v2→2, HALT
   rule load_instr0 (cycle == 2);
      sxu.loadInstr(0, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
   endrule

   rule load_instr1 (cycle == 3);
      sxu.loadInstr(1, SxuInstr { op: SXU_LOAD_VREG,    vmemAddr:1, vregDst:1, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
   endrule

   rule load_instr2 (cycle == 4);
      sxu.loadInstr(2, SxuInstr { op: SXU_DISPATCH_VPU, vmemAddr:0, vregDst:2, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:1 });
   endrule

   rule load_instr3 (cycle == 5);
      sxu.loadInstr(3, SxuInstr { op: SXU_STORE_VREG,   vmemAddr:2, vregDst:0, vregSrc:2, vpuOp:VPU_ADD, vregSrc2:0 });
   endrule

   rule load_instr4 (cycle == 6);
      sxu.loadInstr(4, SxuInstr { op: SXU_HALT,         vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0 });
      $display("Cycle %0d: program loaded (5 instrs)", cycle);
   endrule

   rule start_sxu (cycle == 7);
      sxu.start(5);
      $display("Cycle %0d: SXU started", cycle);
   endrule

   rule wait_sxu (cycle > 7 && !sxu.isDone);
      $display("Cycle %0d: SXU running...", cycle);
   endrule

   // Issue VMEM read once SXU halts
   Reg#(Bool) readIssued <- mkReg(False);

   rule issue_read (sxu.isDone && !readIssued);
      vmem.readReq(2);
      readIssued <= True;
      $display("Cycle %0d: SXU done, issued readReq VMEM[2]", cycle);
   endrule

   rule check_vmem2 (readIssued);
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
      $display("Results: %0d passed, %0d failed", passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (ok) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
