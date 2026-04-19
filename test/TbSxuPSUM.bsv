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

// Drive the PSUM opcodes through the SXU front-end.
// Part 1 — VPU-side PSUM (opcodes 15/16/17):
//   VMEM[0] = 1..16; VMEM[1] = ones
//   LOAD  VMEM[0] -> v0
//   LOAD  VMEM[1] -> v1
//   PSUM_WRITE   psum[0] := v0
//   PSUM_ACC     psum[0] += v1
//   PSUM_READ    v2 := psum[0]
//   STORE v2 -> VMEM[2]          ; expect row0=[2,3,4,5], row3=[14,15,16,17]
// Part 2 — MXU-side PSUM (DISPATCH_MXU fields carry psum target):
//   weights[0] = identity int8; activations[0] = [1,2,3,4]
//   DISPATCH_MXU (psum addr=1 row=2 mode=WRITE)
//   WAIT_MXU
//   DISPATCH_MXU (psum addr=1 row=2 mode=ACCUMULATE)
//   WAIT_MXU
//   PSUM_READ    v3 := psum[1]
//   STORE v3 -> VMEM[3]          ; expect row2=[2,4,6,8]
//   HALT

(* synthesize *)
module mkTbSxuPSUM();

   VMEM_IFC#(16, 4, 4)     vmem <- mkVMEM;
   VRegFile_IFC#(8, 4, 4)  vrf  <- mkVRegFile;
   VPU_IFC#(4, 4)          vpu  <- mkVPU;
   XLU_IFC#(4, 4)          xlu  <- mkXLU;

   PSUMBank_IFC#(8, 4, 4)     psum  <- mkPSUMBank;

   SystolicArray_IFC#(4, 4)   arr   <- mkSystolicArray;
   WeightSRAM_IFC#(16, 4, 4)  wsram <- mkWeightSRAM;
   ActivationSRAM_IFC#(16, 4) asram <- mkActivationSRAM;
   Controller_IFC#(4, 4, 16)  ctrl  <- mkController(arr, wsram, asram, psum);

   SXU_IFC#(16, 16, 8, 4, 4)  sxu <-
      mkScalarUnit(vmem, vrf, vpu, xlu, ctrl, psum);

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 200) begin $display("FAIL: timeout at cycle %0d", cycle); $finish(1); end
   endrule

   // VMEM[0] = 1..16
   rule preload_vmem0 (cycle == 0);
      Vector#(4, Vector#(4, Int#(32))) t = replicate(replicate(0));
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            t[r][c] = fromInteger(r*4 + c + 1);
      vmem.write(0, t);
   endrule
   // VMEM[1] = ones
   rule preload_vmem1 (cycle == 1);
      vmem.write(1, replicate(replicate(1)));
   endrule

   // Weights[0] = identity int8 4x4
   rule preload_w (cycle == 0);
      Vector#(4, Vector#(4, Int#(8))) w = replicate(replicate(0));
      w[0][0] = 1; w[1][1] = 1; w[2][2] = 1; w[3][3] = 1;
      wsram.write(0, w);
   endrule
   // Activations[0] = [1,2,3,4]
   rule preload_a (cycle == 0);
      Vector#(4, Int#(8)) a = newVector;
      a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4;
      asram.write(0, a);
   endrule

   // Program (13 instrs). Load one per cycle starting at cycle 2.
   // Mode encoding on DISPATCH_MXU: vregSrc2[1:0] = 01 WRITE, 10 ACCUMULATE.
   function SxuInstr mk(SxuOpCode op, UInt#(8) va, UInt#(4) vd, UInt#(4) vs, VpuOp vop,
                        UInt#(4) vs2, UInt#(8) wb, UInt#(8) ab, UInt#(8) tl);
      return SxuInstr { op: op, vmemAddr: va, vregDst: vd, vregSrc: vs, vpuOp: vop,
                        vregSrc2: vs2, mxuWBase: wb, mxuABase: ab, mxuTLen: tl };
   endfunction

   rule p0 (cycle == 2);  sxu.loadInstr(0,  mk(SXU_LOAD_VREG,        0, 0, 0, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p1 (cycle == 3);  sxu.loadInstr(1,  mk(SXU_LOAD_VREG,        1, 1, 0, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p2 (cycle == 4);  sxu.loadInstr(2,  mk(SXU_PSUM_WRITE,       0, 0, 0, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p3 (cycle == 5);  sxu.loadInstr(3,  mk(SXU_PSUM_ACCUMULATE,  0, 0, 1, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p4 (cycle == 6);  sxu.loadInstr(4,  mk(SXU_PSUM_READ,        0, 2, 0, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p5 (cycle == 7);  sxu.loadInstr(5,  mk(SXU_STORE_VREG,       2, 0, 2, VPU_ADD, 0, 0, 0, 0)); endrule
   // DISPATCH_MXU with psum target = (addr=1, row=2, mode=WRITE). Fields
   // repurposed per ScalarUnit.do_mxu: vregDst=addr, vregSrc[1:0]=row, vregSrc2[1:0]=mode.
   rule p6 (cycle == 8);  sxu.loadInstr(6,  mk(SXU_DISPATCH_MXU,     0, 1, 2, VPU_ADD, 1, 0, 0, 1)); endrule
   rule p7 (cycle == 9);  sxu.loadInstr(7,  mk(SXU_WAIT_MXU,         0, 0, 0, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p8 (cycle == 10); sxu.loadInstr(8,  mk(SXU_DISPATCH_MXU,     0, 1, 2, VPU_ADD, 2, 0, 0, 1)); endrule
   rule p9 (cycle == 11); sxu.loadInstr(9,  mk(SXU_WAIT_MXU,         0, 0, 0, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p10 (cycle == 12); sxu.loadInstr(10, mk(SXU_PSUM_READ,        1, 3, 0, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p11 (cycle == 13); sxu.loadInstr(11, mk(SXU_STORE_VREG,       3, 0, 3, VPU_ADD, 0, 0, 0, 0)); endrule
   rule p12 (cycle == 14); sxu.loadInstr(12, mk(SXU_HALT,             0, 0, 0, VPU_ADD, 0, 0, 0, 0)); endrule

   rule start_sxu (cycle == 15);
      sxu.start(13);
      $display("Cycle %0d: SXU started", cycle);
   endrule

   Reg#(UInt#(3)) phase <- mkReg(0);

   rule issue_read_part1 (sxu.isDone && phase == 0);
      vmem.readReq(2);
      $display("Cycle %0d: SXU done, readReq VMEM[2]", cycle);
      phase <= 1;
   endrule

   rule check_part1 (phase == 1);
      let t = vmem.readResp;
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (t[r][c] != fromInteger(r*4 + c + 2)) ok = False;
      if (ok) begin
         $display("PASS Part 1 (VPU PSUM): VMEM[2] row0=[%0d,%0d,%0d,%0d] row3=[%0d,%0d,%0d,%0d]",
            t[0][0], t[0][1], t[0][2], t[0][3], t[3][0], t[3][1], t[3][2], t[3][3]);
         passed <= passed + 1;
      end else begin
         $display("FAIL Part 1: VMEM[2] row0=[%0d,%0d,%0d,%0d] row3=[%0d,%0d,%0d,%0d]",
            t[0][0], t[0][1], t[0][2], t[0][3], t[3][0], t[3][1], t[3][2], t[3][3]);
         failed <= failed + 1;
      end
      vmem.readReq(3);
      phase <= 2;
   endrule

   rule check_part2 (phase == 2);
      let t = vmem.readResp;
      Bool ok = (t[2][0] == 2 && t[2][1] == 4 && t[2][2] == 6 && t[2][3] == 8);
      if (ok) begin
         $display("PASS Part 2 (MXU PSUM): VMEM[3] row2=[%0d,%0d,%0d,%0d]",
            t[2][0], t[2][1], t[2][2], t[2][3]);
         passed <= passed + 1;
      end else begin
         $display("FAIL Part 2: VMEM[3] row2=[%0d,%0d,%0d,%0d]",
            t[2][0], t[2][1], t[2][2], t[2][3]);
         failed <= failed + 1;
      end
      UInt#(8) totalPassed = passed + (ok ? 1 : 0);
      UInt#(8) totalFailed = failed + (ok ? 0 : 1);
      $display("Results: %0d passed, %0d failed", totalPassed, totalFailed);
      $finish(totalFailed == 0 ? 0 : 1);
   endrule

endmodule

endpackage
