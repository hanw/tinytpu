package ScalarUnit;

import Vector :: *;
import RegFile :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import Controller :: *;
import PSUMBank :: *;

typedef enum {
   SXU_LOAD_VREG,
   SXU_STORE_VREG,
   SXU_DISPATCH_VPU,
   SXU_DISPATCH_XLU_BROADCAST,
   SXU_DISPATCH_MXU,
   SXU_WAIT_MXU,
   SXU_LOAD_MXU_RESULT,
   SXU_HALT,
   SXU_DISPATCH_SELECT,
   SXU_BROADCAST_SCALAR,
   SXU_BROADCAST_ROW,
   SXU_BROADCAST_COL,
   SXU_DISPATCH_XLU_TRANSPOSE,
   // Generalized engine-to-engine reads. Mirror of SXU_LOAD_MXU_RESULT
   // for each accumulator column block, so kernels can chain ops
   // without round-tripping through VRegFile between every step.
   SXU_LOAD_VPU_RESULT,
   SXU_LOAD_XLU_RESULT,
   // PSUM (partial-sum) bank access. Reserved in advance; exec rules
   // are added in later iterations once the bank is wired into
   // TensorCore. Layout:
   //   SXU_PSUM_WRITE(psum_addr, vregSrc)       — tile := vreg
   //   SXU_PSUM_ACCUMULATE(psum_addr, vregSrc)  — tile += vreg
   //   SXU_PSUM_READ(psum_addr, vregDst)        — vreg := tile (1-cycle)
   SXU_PSUM_WRITE,
   SXU_PSUM_ACCUMULATE,
   SXU_PSUM_READ
} SxuOpCode deriving (Bits, Eq, FShow);

typedef struct {
   SxuOpCode op;
   UInt#(8)  vmemAddr;   // VMEM address (for LOAD/STORE)
   UInt#(4)  vregDst;    // Destination vreg index
   UInt#(4)  vregSrc;    // Source vreg index (STORE: source to write; VPU: src1)
   VpuOp     vpuOp;      // VPU operation (for DISPATCH_VPU)
   UInt#(4)  vregSrc2;   // Second source vreg (for VPU src2)
   UInt#(8)  mxuWBase;   // MXU weight base addr; SELECT uses low bits as rhs vreg
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
               SXU_EXEC_XLU_BROADCAST, SXU_EXEC_XLU_COLLECT,
               SXU_EXEC_XLU_BROADCAST_SCALAR, SXU_EXEC_XLU_BROADCAST_ROW,
               SXU_EXEC_XLU_BROADCAST_COL,
               SXU_EXEC_XLU_TRANSPOSE,
               SXU_EXEC_SELECT_COPY, SXU_EXEC_SELECT,
               SXU_EXEC_MXU, SXU_WAIT_MXU_STATE, SXU_EXEC_LOAD_MXU_RESULT,
               SXU_EXEC_LOAD_VPU_RESULT, SXU_EXEC_LOAD_XLU_RESULT,
               SXU_EXEC_PSUM_WRITE, SXU_EXEC_PSUM_ACCUMULATE,
               SXU_EXEC_PSUM_READ_REQ, SXU_EXEC_PSUM_READ_RESP,
               SXU_HALTED }
   SxuState deriving (Bits, Eq, FShow);

module mkScalarUnit#(
   VMEM_IFC#(vmDepth, sublanes, lanes)             vmem,
   VRegFile_IFC#(numRegs, sublanes, lanes)          vrf,
   VPU_IFC#(sublanes, lanes)                        vpu,
   XLU_IFC#(sublanes, lanes)                        xlu,
   Controller_IFC#(sublanes, lanes, vmDepth)        ctrl,
   PSUMBank_IFC#(psumDepth, sublanes, lanes)         psum
)(SXU_IFC#(progDepth, vmDepth, numRegs, sublanes, lanes))
   provisos(
      Add#(1, p_, progDepth),
      Add#(1, v_, vmDepth),
      Add#(1, r_, numRegs),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Add#(1, pd_, psumDepth),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz),
      Bits#(SxuInstr, isz),
      Add#(0, sublanes, lanes),          // square vregs (XLU requirement)
      Add#(a__, TLog#(vmDepth), 8),      // vmemAddr/mxu*:UInt#(8) truncates to TLog#(vmDepth)
      Add#(b__, TLog#(numRegs), 4),      // vregDst/vregSrc:UInt#(4) truncates to TLog#(numRegs)
      Add#(d__, TLog#(numRegs), 8),      // mxuWBase can be reused as a register source for SELECT
      Add#(c__, TLog#(lanes), 4),        // vregSrc2 carries XLU broadcast lane selector
      Add#(e__, TLog#(psumDepth), 8),    // vmemAddr field doubles as PSUM bucket addr
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
         SXU_DISPATCH_XLU_BROADCAST: pc_state <= SXU_EXEC_XLU_BROADCAST;
         SXU_DISPATCH_SELECT: pc_state <= SXU_EXEC_SELECT_COPY;
         SXU_BROADCAST_SCALAR: pc_state <= SXU_EXEC_XLU_BROADCAST_SCALAR;
         SXU_BROADCAST_ROW: pc_state <= SXU_EXEC_XLU_BROADCAST_ROW;
         SXU_BROADCAST_COL: pc_state <= SXU_EXEC_XLU_BROADCAST_COL;
         SXU_DISPATCH_XLU_TRANSPOSE: pc_state <= SXU_EXEC_XLU_TRANSPOSE;
         SXU_DISPATCH_MXU: pc_state <= SXU_EXEC_MXU;
         SXU_WAIT_MXU:     pc_state <= SXU_WAIT_MXU_STATE;
         SXU_LOAD_MXU_RESULT: pc_state <= SXU_EXEC_LOAD_MXU_RESULT;
         SXU_LOAD_VPU_RESULT: pc_state <= SXU_EXEC_LOAD_VPU_RESULT;
         SXU_LOAD_XLU_RESULT: pc_state <= SXU_EXEC_LOAD_XLU_RESULT;
         SXU_PSUM_WRITE:      pc_state <= SXU_EXEC_PSUM_WRITE;
         SXU_PSUM_ACCUMULATE: pc_state <= SXU_EXEC_PSUM_ACCUMULATE;
         SXU_PSUM_READ:       pc_state <= SXU_EXEC_PSUM_READ_REQ;
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

   // Collect VPU result. Single-cycle ops report isDone immediately; the
   // multi-cycle float reducer keeps isDone=False while its FSM walks, so
   // this rule stalls until the reducer finishes and resultReg holds the
   // final broadcast value.
   rule do_vpu_collect (pc_state == SXU_EXEC_VPU_COLLECT && vpu.isDone);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=VPU_COLLECT pc=%0d", cycle, pc);
      $display("TRACE cycle=%0d unit=VPU ev=RESULT", cycle);
`endif
      vrf.write(truncate(curInstr.vregDst), vpu.result);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_XLU_BROADCAST: read source vreg, broadcast selected lane.
   rule do_xlu_broadcast (pc_state == SXU_EXEC_XLU_BROADCAST);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(TLog#(lanes)) srcLane = truncate(curInstr.vregSrc2);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_XLU_BROADCAST pc=%0d src_lane=%0d", cycle, pc, srcLane);
      $display("TRACE cycle=%0d unit=XLU ev=BROADCAST src_lane=%0d", cycle, srcLane);
`endif
      xlu.executeBroadcast(src, srcLane);
      pc_state <= SXU_EXEC_XLU_COLLECT;
   endrule

   rule do_xlu_broadcast_scalar (pc_state == SXU_EXEC_XLU_BROADCAST_SCALAR);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(4) sel = curInstr.vregSrc2;
      UInt#(TLog#(sublanes)) srcRow = truncate(sel >> valueOf(TLog#(lanes)));
      UInt#(TLog#(lanes)) srcCol = truncate(sel);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=BROADCAST_SCALAR pc=%0d src=v%0d row=%0d col=%0d", cycle, pc, curInstr.vregSrc, srcRow, srcCol);
`endif
      xlu.executeBroadcastScalar(src, srcRow, srcCol);
      pc_state <= SXU_EXEC_XLU_COLLECT;
   endrule

   rule do_xlu_broadcast_row (pc_state == SXU_EXEC_XLU_BROADCAST_ROW);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(TLog#(sublanes)) srcRow = truncate(curInstr.vregSrc2);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=BROADCAST_ROW pc=%0d src=v%0d row=%0d", cycle, pc, curInstr.vregSrc, srcRow);
`endif
      xlu.executeBroadcastRow(src, srcRow);
      pc_state <= SXU_EXEC_XLU_COLLECT;
   endrule

   rule do_xlu_broadcast_col (pc_state == SXU_EXEC_XLU_BROADCAST_COL);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(TLog#(lanes)) srcCol = truncate(curInstr.vregSrc2);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=BROADCAST_COL pc=%0d src=v%0d col=%0d", cycle, pc, curInstr.vregSrc, srcCol);
`endif
      xlu.executeBroadcastCol(src, srcCol);
      pc_state <= SXU_EXEC_XLU_COLLECT;
   endrule

   rule do_xlu_transpose (pc_state == SXU_EXEC_XLU_TRANSPOSE);
      let src = vrf.read(truncate(curInstr.vregSrc));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=XLU_TRANSPOSE pc=%0d src=v%0d", cycle, pc, curInstr.vregSrc);
`endif
      xlu.executeTranspose(src);
      pc_state <= SXU_EXEC_XLU_COLLECT;
   endrule

   // Collect XLU result (1-cycle latency), write to vregDst, advance pc
   rule do_xlu_collect (pc_state == SXU_EXEC_XLU_COLLECT);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=XLU_COLLECT pc=%0d", cycle, pc);
      $display("TRACE cycle=%0d unit=XLU ev=RESULT", cycle);
`endif
      vrf.write(truncate(curInstr.vregDst), xlu.result);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_SELECT step 1: copy rhs into the VPU result register.
   // Uses:
   //   vregSrc  = condition tile register
   //   vregSrc2 = lhs/true tile register
   //   mxuWBase low bits = rhs/false tile register
   rule do_select_copy (pc_state == SXU_EXEC_SELECT_COPY);
      let rhs = vrf.read(truncate(curInstr.mxuWBase));
      let zeros = replicate(replicate(0));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_SELECT_COPY pc=%0d rhs=v%0d", cycle, pc, truncate(curInstr.mxuWBase));
      $display("TRACE cycle=%0d unit=VPU ev=EXEC op=%0d", cycle, pack(VPU_COPY));
`endif
      vpu.execute(VPU_COPY, rhs, zeros);
      pc_state <= SXU_EXEC_SELECT;
   endrule

   // DISPATCH_SELECT step 2: run native VPU_SELECT using copied rhs as the
   // implicit false path held in VPU resultReg.
   rule do_select (pc_state == SXU_EXEC_SELECT);
      let cond = vrf.read(truncate(curInstr.vregSrc));
      let lhs  = vrf.read(truncate(curInstr.vregSrc2));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_SELECT pc=%0d cond=v%0d lhs=v%0d", cycle, pc, curInstr.vregSrc, curInstr.vregSrc2);
      $display("TRACE cycle=%0d unit=VPU ev=EXEC op=%0d", cycle, pack(VPU_SELECT));
`endif
      vpu.execute(VPU_SELECT, cond, lhs);
      pc_state <= SXU_EXEC_VPU_COLLECT;
   endrule

   // LOAD_VPU_RESULT: copy vpu.result directly into vregDst.
   // Mirror of LOAD_MXU_RESULT for the VPU accumulator. Lets kernels
   // chain VPU ops into a downstream engine without round-tripping
   // through VRegFile for writeback+read.
   rule do_load_vpu_result (pc_state == SXU_EXEC_LOAD_VPU_RESULT && vpu.isDone);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=LOAD_VPU_RESULT pc=%0d dst=v%0d", cycle, pc, curInstr.vregDst);
`endif
      vrf.write(truncate(curInstr.vregDst), vpu.result);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // LOAD_XLU_RESULT: copy xlu.result directly into vregDst.
   rule do_load_xlu_result (pc_state == SXU_EXEC_LOAD_XLU_RESULT);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=LOAD_XLU_RESULT pc=%0d dst=v%0d", cycle, pc, curInstr.vregDst);
`endif
      vrf.write(truncate(curInstr.vregDst), xlu.result);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // PSUM_WRITE: read vregSrc, deposit into psum bucket. Bucket index is
   // carried in vmemAddr (same 8-bit field as VMEM addr, truncated).
   rule do_psum_write (pc_state == SXU_EXEC_PSUM_WRITE);
      let data = vrf.read(truncate(curInstr.vregSrc));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=PSUM_WRITE pc=%0d addr=%0d src=v%0d",
               cycle, pc, curInstr.vmemAddr, curInstr.vregSrc);
`endif
      psum.write(truncate(curInstr.vmemAddr), data);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // PSUM_ACCUMULATE: psum[addr] += vregSrc.
   rule do_psum_accumulate (pc_state == SXU_EXEC_PSUM_ACCUMULATE);
      let data = vrf.read(truncate(curInstr.vregSrc));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=PSUM_ACCUMULATE pc=%0d addr=%0d src=v%0d",
               cycle, pc, curInstr.vmemAddr, curInstr.vregSrc);
`endif
      psum.accumulate(truncate(curInstr.vmemAddr), data);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // PSUM_READ step 1: issue psum readReq (1-cycle latency matches VMEM).
   rule do_psum_read_req (pc_state == SXU_EXEC_PSUM_READ_REQ);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=PSUM_READ_REQ pc=%0d addr=%0d dst=v%0d",
               cycle, pc, curInstr.vmemAddr, curInstr.vregDst);
`endif
      psum.readReq(truncate(curInstr.vmemAddr));
      pc_state <= SXU_EXEC_PSUM_READ_RESP;
   endrule

   // PSUM_READ step 2: collect readResp into vregDst.
   rule do_psum_read_resp (pc_state == SXU_EXEC_PSUM_READ_RESP);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=PSUM_READ_RESP pc=%0d dst=v%0d",
               cycle, pc, curInstr.vregDst);
`endif
      vrf.write(truncate(curInstr.vregDst), psum.readResp);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // LOAD_MXU_RESULT: copy ctrl.results (1 row of cols Int#(32)) into row 0 of vregDst
   rule do_load_mxu_result (pc_state == SXU_EXEC_LOAD_MXU_RESULT);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=LOAD_MXU_RESULT pc=%0d dst=v%0d", cycle, pc, curInstr.vregDst);
`endif
      Vector#(sublanes, Vector#(lanes, Int#(32))) v = replicate(replicate(0));
      v[0] = ctrl.results;
      vrf.write(truncate(curInstr.vregDst), v);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_MXU: trigger Controller, advance pc.
   //
   // For multi-K-tile accumulation the instruction optionally names a
   // PSUM deposit target. The fields are re-purposed from otherwise
   // unused vreg slots so SxuInstr stays the same width:
   //   vregDst       -> psumAddr (truncated to TLog#(psumDepth))
   //   vregSrc[1:0]  -> psumRow (bucket row, TLog#(sublanes))
   //   vregSrc2[1:0] -> psumMode (PSUM_OFF / WRITE / ACCUMULATE)
   // Existing MXU dispatches pass these as 0, which decodes to
   // PSUM_OFF and leaves behavior unchanged.
   rule do_mxu (pc_state == SXU_EXEC_MXU);
      PsumMode mode = unpack(truncate(pack(curInstr.vregSrc2)));
      UInt#(TLog#(sublanes)) psumRow = truncate(curInstr.vregSrc);
      UInt#(8) psumAddr = extend(curInstr.vregDst);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_MXU pc=%0d psum_mode=%0d psum_addr=%0d psum_row=%0d",
               cycle, pc, pack(mode), psumAddr, psumRow);
`endif
      ctrl.startPsum(truncate(curInstr.mxuWBase),
                     truncate(curInstr.mxuABase),
                     truncate(curInstr.mxuTLen),
                     psumAddr, psumRow, mode);
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
