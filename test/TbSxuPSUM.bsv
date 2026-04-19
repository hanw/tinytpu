package TbSxuPSUM;

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
import PSUMBank :: *;

// Drive the PSUM opcodes through the SXU front-end:
//   VMEM[0] = 1..16; VMEM[1] = ones
//   LOAD  VMEM[0] -> v0
//   LOAD  VMEM[1] -> v1
//   PSUM_WRITE   psum[0] := v0
//   PSUM_ACC     psum[0] += v1
//   PSUM_READ    v2 := psum[0]
//   STORE v2 -> VMEM[2]
//   HALT
// Expected VMEM[2] = 2..17.

(* synthesize *)
module mkTbSxuPSUM();

   VMEM_IFC#(16, 4, 4)     vmem <- mkVMEM;
   VRegFile_IFC#(8, 4, 4)  vrf  <- mkVRegFile;
   VPU_IFC#(4, 4)          vpu  <- mkVPU;
   XLU_IFC#(4, 4)          xlu  <- mkXLU;

   SystolicArray_IFC#(4, 4)   arr   <- mkSystolicArray;
   WeightSRAM_IFC#(16, 4, 4)  wsram <- mkWeightSRAM;
   ActivationSRAM_IFC#(16, 4) asram <- mkActivationSRAM;
   Controller_IFC#(4, 4, 16)  ctrl  <- mkController(arr, wsram, asram);

   PSUMBank_IFC#(8, 4, 4)     psum  <- mkPSUMBank;

   SXU_IFC#(16, 16, 8, 4, 4)  sxu <-
      mkScalarUnit(vmem, vrf, vpu, xlu, ctrl, psum);

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 120) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // VMEM[0] = 1..16 (full 4x4 tile)
   rule preload0 (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) t = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            t[r][c] = fromInteger(r*4 + c + 1);
      vmem.write(0, t);
   endrule

   // VMEM[1] = all ones
   rule preload1 (cycle == 1);
      vmem.write(1, replicate(replicate(1)));
   endrule

   // LOAD VMEM[0] -> v0
   rule i0 (cycle == 2);
      sxu.loadInstr(0, SxuInstr { op: SXU_LOAD_VREG, vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   // LOAD VMEM[1] -> v1
   rule i1 (cycle == 3);
      sxu.loadInstr(1, SxuInstr { op: SXU_LOAD_VREG, vmemAddr:1, vregDst:1, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   // PSUM_WRITE psum[0] := v0
   rule i2 (cycle == 4);
      sxu.loadInstr(2, SxuInstr { op: SXU_PSUM_WRITE, vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   // PSUM_ACCUMULATE psum[0] += v1
   rule i3 (cycle == 5);
      sxu.loadInstr(3, SxuInstr { op: SXU_PSUM_ACCUMULATE, vmemAddr:0, vregDst:0, vregSrc:1, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   // PSUM_READ v2 := psum[0]
   rule i4 (cycle == 6);
      sxu.loadInstr(4, SxuInstr { op: SXU_PSUM_READ, vmemAddr:0, vregDst:2, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   // STORE v2 -> VMEM[2]
   rule i5 (cycle == 7);
      sxu.loadInstr(5, SxuInstr { op: SXU_STORE_VREG, vmemAddr:2, vregDst:0, vregSrc:2, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   // HALT
   rule i6 (cycle == 8);
      sxu.loadInstr(6, SxuInstr { op: SXU_HALT, vmemAddr:0, vregDst:0, vregSrc:0, vpuOp:VPU_ADD, vregSrc2:0, mxuWBase:0, mxuABase:0, mxuTLen:0 });
   endrule

   rule start_sxu (cycle == 9);
      sxu.start(7);
      $display("Cycle %0d: SXU started", cycle);
   endrule

   Reg#(UInt#(3)) phase <- mkReg(0);

   rule issue_read (sxu.isDone && phase == 0);
      vmem.readReq(2);
      $display("Cycle %0d: SXU done, readReq VMEM[2]", cycle);
      phase <= 1;
   endrule

   rule check (phase == 1);
      let t = vmem.readResp;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (t[r][c] != fromInteger(r*4 + c + 2)) ok = False;
      if (ok) begin
         $display("PASS PSUM write+acc+read: VMEM[2] row0=[%0d,%0d,%0d,%0d] row3=[%0d,%0d,%0d,%0d]",
            t[0][0], t[0][1], t[0][2], t[0][3], t[3][0], t[3][1], t[3][2], t[3][3]);
         passed <= passed + 1;
      end else begin
         $display("FAIL VMEM[2] row0=[%0d,%0d,%0d,%0d] row3=[%0d,%0d,%0d,%0d]",
            t[0][0], t[0][1], t[0][2], t[0][3], t[3][0], t[3][1], t[3][2], t[3][3]);
         failed <= failed + 1;
      end
      $display("Results: %0d passed, %0d failed", passed + (ok ? 1 : 0), failed + (ok ? 0 : 1));
      if (failed == 0 && ok) $finish(0); else $finish(1);
   endrule

endmodule

endpackage
