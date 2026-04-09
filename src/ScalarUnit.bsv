package ScalarUnit;

import Vector :: *;
import RegFile :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import Controller :: *;

typedef enum {
   SXU_LOAD_VREG,
   SXU_STORE_VREG,
   SXU_DISPATCH_VPU,
   SXU_DISPATCH_MXU,
   SXU_WAIT_MXU,
   SXU_HALT
} SxuOpCode deriving (Bits, Eq, FShow);

typedef struct {
   SxuOpCode op;
   UInt#(8)  vmemAddr;   // VMEM address (for LOAD/STORE)
   UInt#(4)  vregDst;    // Destination vreg index
   UInt#(4)  vregSrc;    // Source vreg index (STORE: source to write; VPU: src1)
   VpuOp     vpuOp;      // VPU operation (for DISPATCH_VPU)
   UInt#(4)  vregSrc2;   // Second source vreg (for VPU src2)
   UInt#(8)  mxuWBase;   // MXU weight base addr
   UInt#(8)  mxuABase;   // MXU activation base addr
   UInt#(8)  mxuTLen;    // MXU tile length
} SxuInstr deriving (Bits, Eq);

interface SXU_IFC#(numeric type progDepth,
                   numeric type vmDepth,
                   numeric type numRegs,
                   numeric type sublanes,
                   numeric type lanes);
   method Action loadInstr(UInt#(TLog#(progDepth)) pc, SxuInstr instr);
   method Action start(UInt#(TLog#(progDepth)) len);
   method Bool isDone;
endinterface

typedef enum { SXU_IDLE, SXU_FETCH, SXU_EXEC_LOAD_REQ, SXU_EXEC_LOAD_RESP,
               SXU_EXEC_STORE, SXU_EXEC_VPU, SXU_EXEC_VPU_COLLECT,
               SXU_EXEC_MXU, SXU_WAIT_MXU_STATE, SXU_HALTED }
   SxuState deriving (Bits, Eq, FShow);

module mkScalarUnit#(
   VMEM_IFC#(vmDepth, sublanes, lanes)             vmem,
   VRegFile_IFC#(numRegs, sublanes, lanes)          vrf,
   VPU_IFC#(sublanes, lanes)                        vpu,
   XLU_IFC#(sublanes, lanes)                        xlu,
   Controller_IFC#(sublanes, lanes, vmDepth)        ctrl
)(SXU_IFC#(progDepth, vmDepth, numRegs, sublanes, lanes))
   provisos(
      Add#(1, p_, progDepth),
      Add#(1, v_, vmDepth),
      Add#(1, r_, numRegs),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz),
      Bits#(SxuInstr, isz),
      Add#(0, sublanes, lanes),          // square vregs (XLU requirement)
      Add#(a__, TLog#(vmDepth), 8),      // vmemAddr/mxu*:UInt#(8) truncates to TLog#(vmDepth)
      Add#(b__, TLog#(numRegs), 4),      // vregDst/vregSrc:UInt#(4) truncates to TLog#(numRegs)
      Add#(logd_, TLog#(vmDepth), 32)    // needed by Controller
   );

   RegFile#(UInt#(TLog#(progDepth)), SxuInstr) prog <- mkRegFileFull;

   Reg#(SxuState)                pc_state <- mkReg(SXU_IDLE);
   Reg#(UInt#(TLog#(progDepth))) pc       <- mkReg(0);
   Reg#(SxuInstr)                curInstr <- mkRegU;

   // FETCH: read instruction at current pc, decode
   rule do_fetch (pc_state == SXU_FETCH);
      let instr = prog.sub(pc);
      curInstr <= instr;
      case (instr.op)
         SXU_LOAD_VREG:    pc_state <= SXU_EXEC_LOAD_REQ;
         SXU_STORE_VREG:   pc_state <= SXU_EXEC_STORE;
         SXU_DISPATCH_VPU: pc_state <= SXU_EXEC_VPU;
         SXU_DISPATCH_MXU: pc_state <= SXU_EXEC_MXU;
         SXU_WAIT_MXU:     pc_state <= SXU_WAIT_MXU_STATE;
         SXU_HALT:         pc_state <= SXU_HALTED;
      endcase
   endrule

   // LOAD step 1: issue VMEM readReq
   rule do_load_req (pc_state == SXU_EXEC_LOAD_REQ);
      vmem.readReq(truncate(curInstr.vmemAddr));
      pc_state <= SXU_EXEC_LOAD_RESP;
   endrule

   // LOAD step 2: collect readResp, write to VRegFile, advance pc
   rule do_load_resp (pc_state == SXU_EXEC_LOAD_RESP);
      vrf.write(truncate(curInstr.vregDst), vmem.readResp);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // STORE: read VRegFile, write to VMEM, advance pc
   rule do_store (pc_state == SXU_EXEC_STORE);
      let data = vrf.read(truncate(curInstr.vregSrc));
      vmem.write(truncate(curInstr.vmemAddr), data);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_VPU: read two source vregs, dispatch VPU
   rule do_vpu (pc_state == SXU_EXEC_VPU);
      let s1 = vrf.read(truncate(curInstr.vregSrc));
      let s2 = vrf.read(truncate(curInstr.vregSrc2));
      vpu.execute(curInstr.vpuOp, s1, s2);
      pc_state <= SXU_EXEC_VPU_COLLECT;
   endrule

   // Collect VPU result (1-cycle latency), write to vregDst, advance pc
   rule do_vpu_collect (pc_state == SXU_EXEC_VPU_COLLECT);
      vrf.write(truncate(curInstr.vregDst), vpu.result);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_MXU: trigger Controller, advance pc
   rule do_mxu (pc_state == SXU_EXEC_MXU);
      ctrl.start(truncate(curInstr.mxuWBase),
                 truncate(curInstr.mxuABase),
                 truncate(curInstr.mxuTLen));
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // WAIT_MXU: stall until Controller isDone, then advance pc
   rule do_wait_mxu (pc_state == SXU_WAIT_MXU_STATE && ctrl.isDone);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   method Action loadInstr(UInt#(TLog#(progDepth)) addr, SxuInstr instr);
      prog.upd(addr, instr);
   endmethod

   method Action start(UInt#(TLog#(progDepth)) len) if (pc_state == SXU_IDLE);
      pc       <= 0;
      pc_state <= SXU_FETCH;
   endmethod

   method Bool isDone;
      return pc_state == SXU_HALTED;
   endmethod

endmodule

export SxuOpCode(..);
export SxuInstr(..);
export SXU_IFC(..);
export mkScalarUnit;

endpackage
