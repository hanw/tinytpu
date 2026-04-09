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
`ifdef TRACE
   Reg#(UInt#(32))               cycle    <- mkReg(0);

   rule count_trace_cycles;
      cycle <= cycle + 1;
   endrule
`endif

   // FETCH: read instruction at current pc, decode
   rule do_fetch (pc_state == SXU_FETCH);
      let instr = prog.sub(pc);
      curInstr <= instr;
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=FETCH pc=%0d", cycle, pc);
`endif
      case (instr.op)
         SXU_LOAD_VREG:    pc_state <= SXU_EXEC_LOAD_REQ;
         SXU_STORE_VREG:   pc_state <= SXU_EXEC_STORE;
         SXU_DISPATCH_VPU: pc_state <= SXU_EXEC_VPU;
         SXU_DISPATCH_MXU: pc_state <= SXU_EXEC_MXU;
         SXU_WAIT_MXU:     pc_state <= SXU_WAIT_MXU_STATE;
         SXU_HALT: begin
`ifdef TRACE
            $display("TRACE cycle=%0d unit=SXU ev=HALT pc=%0d", cycle, pc);
`endif
            pc_state <= SXU_HALTED;
         end
      endcase
   endrule

   // LOAD step 1: issue VMEM readReq
   rule do_load_req (pc_state == SXU_EXEC_LOAD_REQ);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=LOAD_REQ pc=%0d addr=%0d", cycle, pc, curInstr.vmemAddr);
      $display("TRACE cycle=%0d unit=VMEM ev=READ_REQ addr=%0d", cycle, curInstr.vmemAddr);
`endif
      vmem.readReq(truncate(curInstr.vmemAddr));
      pc_state <= SXU_EXEC_LOAD_RESP;
   endrule

   // LOAD step 2: collect readResp, write to VRegFile, advance pc
   rule do_load_resp (pc_state == SXU_EXEC_LOAD_RESP);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=LOAD_RESP pc=%0d", cycle, pc);
      $display("TRACE cycle=%0d unit=VMEM ev=READ_RESP", cycle);
`endif
      vrf.write(truncate(curInstr.vregDst), vmem.readResp);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // STORE: read VRegFile, write to VMEM, advance pc
   rule do_store (pc_state == SXU_EXEC_STORE);
      let data = vrf.read(truncate(curInstr.vregSrc));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=STORE pc=%0d addr=%0d", cycle, pc, curInstr.vmemAddr);
      $display("TRACE cycle=%0d unit=VMEM ev=WRITE addr=%0d", cycle, curInstr.vmemAddr);
`endif
      vmem.write(truncate(curInstr.vmemAddr), data);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_VPU: read two source vregs, dispatch VPU
   rule do_vpu (pc_state == SXU_EXEC_VPU);
      let s1 = vrf.read(truncate(curInstr.vregSrc));
      let s2 = vrf.read(truncate(curInstr.vregSrc2));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_VPU pc=%0d op=%0d", cycle, pc, pack(curInstr.vpuOp));
      $display("TRACE cycle=%0d unit=VPU ev=EXEC op=%0d", cycle, pack(curInstr.vpuOp));
`endif
      vpu.execute(curInstr.vpuOp, s1, s2);
      pc_state <= SXU_EXEC_VPU_COLLECT;
   endrule

   // Collect VPU result (1-cycle latency), write to vregDst, advance pc
   rule do_vpu_collect (pc_state == SXU_EXEC_VPU_COLLECT);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=VPU_COLLECT pc=%0d", cycle, pc);
      $display("TRACE cycle=%0d unit=VPU ev=RESULT", cycle);
`endif
      vrf.write(truncate(curInstr.vregDst), vpu.result);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_MXU: trigger Controller, advance pc
   rule do_mxu (pc_state == SXU_EXEC_MXU);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_MXU pc=%0d", cycle, pc);
`endif
      ctrl.start(truncate(curInstr.mxuWBase),
                 truncate(curInstr.mxuABase),
                 truncate(curInstr.mxuTLen));
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // WAIT_MXU: stall until Controller isDone, then advance pc
   rule do_wait_mxu (pc_state == SXU_WAIT_MXU_STATE);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=WAIT_MXU pc=%0d", cycle, pc);
`endif
      if (ctrl.isDone) begin
         pc <= pc + 1;
         pc_state <= SXU_FETCH;
      end
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
