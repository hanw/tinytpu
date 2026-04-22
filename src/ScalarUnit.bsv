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
   SXU_PSUM_READ,
   // PSUM_READ_ROW: read psum[addr][row] (one sublane) into row 0 of
   // vregDst and zero the other rows. Mirrors LOAD_MXU_RESULT so the
   // multi-K-tile GEMM epilogue (bias/relu/store) can consume exactly
   // the accumulated row without the uninitialized-other-rows issue.
   // Layout: vmemAddr = bucket, vregSrc[1:0] = row, vregDst = target.
   SXU_PSUM_READ_ROW,
   // PSUM_CLEAR: zero a whole bucket in one cycle, without sourcing a
   // zero vreg. Lets multi-K-tile GEMM skip the "preload zero tile in
   // VMEM + LOAD into v15 + PSUM_WRITE" dance before each chain.
   // Layout: vmemAddr = bucket. vreg* fields unused.
   SXU_PSUM_CLEAR,
   // Predicate scaffolding (Architectural Refactor Item #7). A single
   // Bool pred register plus two opcodes:
   //   SET_PRED_IF_ZERO vregSrc: pred := (vreg[0][0] == 0)
   //   SKIP_IF_PRED: if pred, pc += 2 (skip next instr); auto-reset pred
   // Baby step toward BARRIER/IF/ENDIF without adding real branches.
   SXU_SET_PRED_IF_ZERO,
   SXU_SKIP_IF_PRED,
   // PSUM_ACCUMULATE_ROW: accumulate row 0 of vregSrc into
   // psum[vmemAddr][vregDst[1:0]]. VPU-side analog of the row-
   // granular MXU path — lets a VPU sequence deposit partial results
   // into a PSUM bucket row without overwriting the other rows.
   SXU_PSUM_ACCUMULATE_ROW,
   // WS with cross-dispatch accumulator-hold. Same operand layout as
   // SXU_DISPATCH_MXU but routes through Controller.startAccumulate
   // so the drain-time clear is skipped; back-to-back dispatches sum
   // their col-sums into the same PE accumulator (multi-K-tile GEMM).
   // Not a distinct dataflow — the PE still holds a preloaded weight.
   SXU_DISPATCH_MXU_ACCUMULATE,
   // Explicitly zero the systolic-array PE accumulators. Needed
   // between accumulate epochs and before re-entering WS from OS.
   // Single-cycle, no operand fields used.
   SXU_MXU_CLEAR,
   // Real output-stationary dispatch. Routes through
   // Controller.startOS: both weights and activations stream as a
   // staircase, each PE holds its own psum, drain returns the full
   // (rows x cols) matrix via resultsMatrix() (read through
   // LOAD_MXU_RESULT-row opcodes later). Operand layout reuses the
   // MXU triple (wBase/aBase/tileLen where tileLen = kLen).
   SXU_DISPATCH_MXU_OS
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

// Classify each opcode by which execution engine it consumes. The MXU
// dispatches are already overlapped with the main FSM (DISPATCH_MXU
// advances past WAIT_MXU), but XLU dispatches still stall the main
// FSM even though the XLU is physically independent. Exposing this
// classification is the first step toward a dual-issue SXU front-end
// that can overlap XLU work with the rest of the program.
function Bool sxu_is_xlu_slot(SxuOpCode op);
   case (op)
      SXU_DISPATCH_XLU_BROADCAST,
      SXU_BROADCAST_SCALAR,
      SXU_BROADCAST_ROW,
      SXU_BROADCAST_COL,
      SXU_DISPATCH_XLU_TRANSPOSE,
      SXU_LOAD_XLU_RESULT:
         return True;
      default:
         return False;
   endcase
endfunction

typedef enum { SXU_IDLE, SXU_FETCH, SXU_EXEC_LOAD_REQ, SXU_EXEC_LOAD_RESP,
               SXU_EXEC_STORE, SXU_EXEC_VPU, SXU_EXEC_VPU_COLLECT,
               SXU_EXEC_XLU_BROADCAST, SXU_EXEC_XLU_COLLECT,
               SXU_EXEC_XLU_BROADCAST_SCALAR, SXU_EXEC_XLU_BROADCAST_ROW,
               SXU_EXEC_XLU_BROADCAST_COL,
               SXU_EXEC_XLU_TRANSPOSE,
               SXU_EXEC_SELECT_COPY, SXU_EXEC_SELECT,
               SXU_EXEC_MXU, SXU_EXEC_MXU_ACCUMULATE, SXU_EXEC_MXU_OS, SXU_EXEC_MXU_CLEAR, SXU_WAIT_MXU_STATE, SXU_EXEC_LOAD_MXU_RESULT,
               SXU_EXEC_LOAD_VPU_RESULT, SXU_EXEC_LOAD_XLU_RESULT,
               SXU_EXEC_PSUM_WRITE, SXU_EXEC_PSUM_ACCUMULATE,
               SXU_EXEC_PSUM_READ_REQ, SXU_EXEC_PSUM_READ_RESP,
               SXU_EXEC_PSUM_READ_ROW, SXU_EXEC_PSUM_CLEAR,
               SXU_EXEC_PSUM_ACCUMULATE_ROW,
               SXU_EXEC_SET_PRED, SXU_EXEC_SKIP_IF_PRED,
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
   // Single-bit predicate for baby conditional execution
   // (SET_PRED_IF_ZERO / SKIP_IF_PRED).
   Reg#(Bool)                    pred     <- mkReg(False);

   // Dual-issue scoreboard (Architectural Refactor Item #4).
   // xlu_busy tracks whether an XLU dispatch is in flight (set on
   // EXEC_XLU_* dispatch, cleared on EXEC_XLU_COLLECT). xlu_dst holds
   // the target vreg so a future arbiter rule can stall on RAW against
   // a subsequent reader. The main FSM is still single-issue — the
   // scoreboard is observable today; a parallel issue slot lands in a
   // follow-up iter.
   Reg#(Bool)                    xlu_busy <- mkReg(False);
   Reg#(UInt#(4))                xlu_dst  <- mkReg(0);
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
         SXU_DISPATCH_MXU_ACCUMULATE: pc_state <= SXU_EXEC_MXU_ACCUMULATE;
         SXU_DISPATCH_MXU_OS: pc_state <= SXU_EXEC_MXU_OS;
         SXU_MXU_CLEAR:    pc_state <= SXU_EXEC_MXU_CLEAR;
         SXU_WAIT_MXU:     pc_state <= SXU_WAIT_MXU_STATE;
         SXU_LOAD_MXU_RESULT: pc_state <= SXU_EXEC_LOAD_MXU_RESULT;
         SXU_LOAD_VPU_RESULT: pc_state <= SXU_EXEC_LOAD_VPU_RESULT;
         SXU_LOAD_XLU_RESULT: pc_state <= SXU_EXEC_LOAD_XLU_RESULT;
         SXU_PSUM_WRITE:      pc_state <= SXU_EXEC_PSUM_WRITE;
         SXU_PSUM_ACCUMULATE: pc_state <= SXU_EXEC_PSUM_ACCUMULATE;
         SXU_PSUM_READ:       pc_state <= SXU_EXEC_PSUM_READ_REQ;
         SXU_PSUM_READ_ROW:   pc_state <= SXU_EXEC_PSUM_READ_ROW;
         SXU_PSUM_CLEAR:      pc_state <= SXU_EXEC_PSUM_CLEAR;
         SXU_PSUM_ACCUMULATE_ROW: pc_state <= SXU_EXEC_PSUM_ACCUMULATE_ROW;
         SXU_SET_PRED_IF_ZERO: pc_state <= SXU_EXEC_SET_PRED;
         SXU_SKIP_IF_PRED:    pc_state <= SXU_EXEC_SKIP_IF_PRED;
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

   // XLU dispatches are now dual-issue: they fire on the dispatch cycle,
   // advance pc + return to FETCH immediately, and a background collect
   // rule writes the XLU result to vregDst one cycle later. Issue is
   // guarded by !xlu_busy so a new XLU op can't be dispatched while
   // another is still outstanding (structural hazard). A RAW stall on
   // readers of xlu_dst lands in the next iter.
   rule do_xlu_broadcast (pc_state == SXU_EXEC_XLU_BROADCAST && !xlu_busy);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(TLog#(lanes)) srcLane = truncate(curInstr.vregSrc2);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_XLU_BROADCAST pc=%0d src_lane=%0d", cycle, pc, srcLane);
      $display("TRACE cycle=%0d unit=XLU ev=BROADCAST src_lane=%0d", cycle, srcLane);
`endif
      xlu.executeBroadcast(src, srcLane);
      xlu_busy <= True;
      xlu_dst  <= curInstr.vregDst;
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   rule do_xlu_broadcast_scalar (pc_state == SXU_EXEC_XLU_BROADCAST_SCALAR && !xlu_busy);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(4) sel = curInstr.vregSrc2;
      UInt#(TLog#(sublanes)) srcRow = truncate(sel >> valueOf(TLog#(lanes)));
      UInt#(TLog#(lanes)) srcCol = truncate(sel);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=BROADCAST_SCALAR pc=%0d src=v%0d row=%0d col=%0d", cycle, pc, curInstr.vregSrc, srcRow, srcCol);
`endif
      xlu.executeBroadcastScalar(src, srcRow, srcCol);
      xlu_busy <= True;
      xlu_dst  <= curInstr.vregDst;
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   rule do_xlu_broadcast_row (pc_state == SXU_EXEC_XLU_BROADCAST_ROW && !xlu_busy);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(TLog#(sublanes)) srcRow = truncate(curInstr.vregSrc2);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=BROADCAST_ROW pc=%0d src=v%0d row=%0d", cycle, pc, curInstr.vregSrc, srcRow);
`endif
      xlu.executeBroadcastRow(src, srcRow);
      xlu_busy <= True;
      xlu_dst  <= curInstr.vregDst;
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   rule do_xlu_broadcast_col (pc_state == SXU_EXEC_XLU_BROADCAST_COL && !xlu_busy);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(TLog#(lanes)) srcCol = truncate(curInstr.vregSrc2);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=BROADCAST_COL pc=%0d src=v%0d col=%0d", cycle, pc, curInstr.vregSrc, srcCol);
`endif
      xlu.executeBroadcastCol(src, srcCol);
      xlu_busy <= True;
      xlu_dst  <= curInstr.vregDst;
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   rule do_xlu_transpose (pc_state == SXU_EXEC_XLU_TRANSPOSE && !xlu_busy);
      let src = vrf.read(truncate(curInstr.vregSrc));
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=XLU_TRANSPOSE pc=%0d src=v%0d", cycle, pc, curInstr.vregSrc);
`endif
      xlu.executeTranspose(src);
      xlu_busy <= True;
      xlu_dst  <= curInstr.vregDst;
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // Background XLU collect. Fires on the cycle after an XLU dispatch
   // (xlu.result carries 1-cycle latency), writes the tile to xlu_dst,
   // and clears xlu_busy so subsequent XLU dispatches can proceed.
   // Runs independently of the main fetch/dispatch FSM, giving
   // single-slot dual-issue between XLU and non-XLU ops.
   rule do_xlu_collect_bg (xlu_busy);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=XLU_COLLECT_BG dst=v%0d", cycle, xlu_dst);
`endif
      vrf.write(truncate(xlu_dst), xlu.result);
      xlu_busy <= False;
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

   // PSUM_ACCUMULATE_ROW: accumulate row 0 of vregSrc into
   // psum[addr][row]. Mirror of Controller.psum.accumulateRow but
   // driven from a vreg (VPU-side path).
   rule do_psum_accumulate_row (pc_state == SXU_EXEC_PSUM_ACCUMULATE_ROW);
      let src = vrf.read(truncate(curInstr.vregSrc));
      UInt#(TLog#(sublanes)) row = truncate(curInstr.vregDst);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=PSUM_ACCUMULATE_ROW pc=%0d addr=%0d row=%0d src=v%0d",
               cycle, pc, curInstr.vmemAddr, row, curInstr.vregSrc);
`endif
      psum.accumulateRow(truncate(curInstr.vmemAddr), row, src[0]);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // SET_PRED_IF_ZERO: pred := (vreg[0][0] == 0). Scalar predicate
   // source is the top-left lane of vregSrc's tile.
   rule do_set_pred (pc_state == SXU_EXEC_SET_PRED);
      let src = vrf.read(truncate(curInstr.vregSrc));
      let scalar = src[0][0];
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=SET_PRED_IF_ZERO pc=%0d src=v%0d scalar=%0d",
               cycle, pc, curInstr.vregSrc, scalar);
`endif
      pred <= (scalar == 0);
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // SKIP_IF_PRED: if pred, advance pc by 2 (skipping the next
   // instruction); otherwise advance by 1. Auto-reset pred.
   rule do_skip_if_pred (pc_state == SXU_EXEC_SKIP_IF_PRED);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=SKIP_IF_PRED pc=%0d pred=%0d",
               cycle, pc, pred);
`endif
      pc <= pred ? (pc + 2) : (pc + 1);
      pred <= False;
      pc_state <= SXU_FETCH;
   endrule

   // PSUM_CLEAR: zero bucket in one cycle (all rows).
   rule do_psum_clear (pc_state == SXU_EXEC_PSUM_CLEAR);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=PSUM_CLEAR pc=%0d addr=%0d",
               cycle, pc, curInstr.vmemAddr);
`endif
      psum.clear(truncate(curInstr.vmemAddr));
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // PSUM_READ_ROW: combinational peek of psum[addr][row], deposited
   // in row 0 of vregDst with the other rows zeroed — the same shape
   // LOAD_MXU_RESULT produces, so downstream bias/relu/STORE steps
   // can consume one accumulated GEMM row without knowing PSUM exists.
   rule do_psum_read_row (pc_state == SXU_EXEC_PSUM_READ_ROW);
      UInt#(TLog#(sublanes)) row = truncate(curInstr.vregSrc);
      Vector#(lanes, Int#(32)) r = psum.peekRow(truncate(curInstr.vmemAddr), row);
      Vector#(sublanes, Vector#(lanes, Int#(32))) v = replicate(replicate(0));
      v[0] = r;
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=PSUM_READ_ROW pc=%0d addr=%0d row=%0d dst=v%0d",
               cycle, pc, curInstr.vmemAddr, row, curInstr.vregDst);
`endif
      vrf.write(truncate(curInstr.vregDst), v);
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

   // DISPATCH_MXU_ACCUMULATE: WS feed path with drain-time clear
   // skipped, so consecutive dispatches sum their col-sums into the
   // same PE accumulator. Same operand layout as SXU_DISPATCH_MXU.
   // PSUM routing is not available on this path.
   rule do_mxu_accumulate (pc_state == SXU_EXEC_MXU_ACCUMULATE);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_MXU_ACCUMULATE pc=%0d", cycle, pc);
`endif
      ctrl.startAccumulate(truncate(curInstr.mxuWBase),
                           truncate(curInstr.mxuABase),
                           truncate(curInstr.mxuTLen));
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // DISPATCH_MXU_OS: real output-stationary dispatch. Routes through
   // Controller.startOS with (wBase, aBase, kLen=tileLen).
   rule do_mxu_os (pc_state == SXU_EXEC_MXU_OS);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=DISPATCH_MXU_OS pc=%0d", cycle, pc);
`endif
      ctrl.startOS(truncate(curInstr.mxuWBase),
                   truncate(curInstr.mxuABase),
                   truncate(curInstr.mxuTLen));
      pc <= pc + 1;
      pc_state <= SXU_FETCH;
   endrule

   // MXU_CLEAR: zero the systolic-array PE accumulators (starts a fresh
   // OS accumulation epoch). Single-cycle, no operand fields used.
   rule do_mxu_clear (pc_state == SXU_EXEC_MXU_CLEAR);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=SXU ev=MXU_CLEAR pc=%0d", cycle, pc);
`endif
      ctrl.clearArray;
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

   method Action start(UInt#(TLog#(progDepth)) len)
         if (pc_state == SXU_IDLE || pc_state == SXU_HALTED);
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
