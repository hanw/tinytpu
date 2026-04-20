package Controller;

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import PSUMBank :: *;

export ControlState(..);
export DataflowMode(..);
export Controller_IFC(..);
export mkController;

typedef enum {
   Idle,
   LoadWeights,
   LoadWeightsResp,
   StreamActivations,
   Drain,
   Done
} ControlState deriving (Bits, Eq, FShow);

// Dataflow mode. WS is the existing weight-stationary behavior: load
// the tile of weights once into the systolic array, stream activations
// through, drain results. OS (output-stationary) will later be used
// for depthwise conv and attention kernels where weights vary per step
// but partial sums accumulate in-place. For now only WS is implemented;
// startOS() accepts OS but executes WS until the PE accumulator mode
// lands. The enum + start path being in place is scaffolding for the
// Architectural Refactor Item #8.
typedef enum {
   DF_WEIGHT_STATIONARY,
   DF_OUTPUT_STATIONARY
} DataflowMode deriving (Bits, Eq, FShow);

interface Controller_IFC#(numeric type rows, numeric type cols, numeric type depth);
   method Action start(UInt#(TLog#(depth)) weightBase,
                       UInt#(TLog#(depth)) actBase,
                       UInt#(TLog#(depth)) tileLen);
   // Variant that additionally deposits the dispatch result into a PSUM
   // bucket row. When `mode == PSUM_OFF` this is equivalent to `start`.
   // psumAddr is an 8-bit field truncated to the instantiated psum depth.
   method Action startPsum(UInt#(TLog#(depth)) weightBase,
                           UInt#(TLog#(depth)) actBase,
                           UInt#(TLog#(depth)) tileLen,
                           UInt#(8) psumAddr,
                           UInt#(TLog#(rows)) psumRow,
                           PsumMode psumMode);
   // Output-stationary variant. Flips dfModeReg to DF_OUTPUT_STATIONARY for
   // the duration of the dispatch and runs the existing WS FSM (operand
   // swap + PE accumulator-hold land in later iterations). Exposes the
   // mode transition to consumers so SXU can dispatch via an OS opcode.
   method Action startOS(UInt#(TLog#(depth)) weightBase,
                         UInt#(TLog#(depth)) actBase,
                         UInt#(TLog#(depth)) tileLen);
   method Bool isDone;
   method Vector#(cols, Int#(32)) results;
   method ControlState getState;
   // Currently selected dataflow mode (set via startOS in future iters;
   // exposed now so SXU/tests can observe Controller state without
   // changing the start/startPsum signatures).
   method DataflowMode getDataflowMode;
endinterface

module mkController#(
      SystolicArray_IFC#(rows, cols) array,
      WeightSRAM_IFC#(depth, rows, cols) wSRAM,
      ActivationSRAM_IFC#(depth, rows) aSRAM,
      PSUMBank_IFC#(psumDepth, rows, cols) psum
   )(Controller_IFC#(rows, cols, depth))
   provisos(Add#(1, r_, rows),
            Add#(1, c_, cols),
            Add#(1, d_, depth),
            Add#(1, pd_, psumDepth),
            Log#(depth, logd),
            Add#(logd_, TLog#(depth), 32),
            Add#(psumTrunc__, TLog#(psumDepth), 8));

   Reg#(ControlState) cstate <- mkReg(Idle);

   Reg#(UInt#(TLog#(depth))) wBase   <- mkReg(0);
   Reg#(UInt#(TLog#(depth))) aBase   <- mkReg(0);
   Reg#(UInt#(TLog#(depth))) tLen    <- mkReg(0);

   // Counts activation reads issued
   Reg#(UInt#(TLog#(depth))) actIdx  <- mkReg(0);

   // Total stream cycles (tileLen + rows - 1 for systolic skew drain)
   Reg#(UInt#(32)) streamCycle <- mkReg(0);

   // Track whether we've issued first activation read
   Reg#(Bool) firstActRead <- mkReg(False);

   Reg#(Vector#(cols, Int#(32))) outputBuf <- mkRegU;

   // PSUM deposit target for the currently running dispatch. Set at
   // start(Psum)() time; consumed in do_drain.
   Reg#(UInt#(8))                 psumAddrReg <- mkReg(0);
   Reg#(UInt#(TLog#(rows)))       psumRowReg  <- mkReg(0);
   Reg#(PsumMode)                 psumModeReg <- mkReg(PSUM_OFF);
   // Dataflow mode register. Always DF_WEIGHT_STATIONARY today; once
   // an OS start path lands this flips in startOS() for the duration
   // of a dispatch.
   Reg#(DataflowMode)             dfModeReg   <- mkReg(DF_WEIGHT_STATIONARY);
`ifdef TRACE
   Reg#(UInt#(32)) cycle <- mkReg(0);

   rule count_trace_cycles;
      cycle <= cycle + 1;
   endrule
`endif

   // Phase 1: Issue weight read request
   rule do_load_weights (cstate == LoadWeights);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=LOAD_W addr=%0d", cycle, wBase);
`endif
      wSRAM.readReq(wBase);
      cstate <= LoadWeightsResp;
   endrule

   // Phase 2: Receive weight data and load into array, issue first activation read
   rule do_load_weights_resp (cstate == LoadWeightsResp);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=LOAD_W_RESP", cycle);
`endif
      array.loadWeights(wSRAM.readResp);
      // Issue first activation read
      aSRAM.readReq(aBase);
      actIdx <= 1;
      streamCycle <= 0;
      firstActRead <= True;
      cstate <= StreamActivations;
   endrule

   // Phase 3: Stream activations through the array
   rule do_stream (cstate == StreamActivations);
      let totalCycles = extend(tLen) + fromInteger(valueOf(rows)) - 1;

      if (streamCycle < totalCycles) begin
`ifdef TRACE
         $display("TRACE cycle=%0d unit=MXU ev=STREAM_A cyc=%0d", cycle, streamCycle);
`endif
         // Feed activation data from SRAM response
         // During skew drain cycles (streamCycle >= tileLen), we feed zeros
         // but still need to call feedActivations for systolic propagation
         if (streamCycle < extend(tLen)) begin
            array.feedActivations(aSRAM.readResp);
         end else begin
            array.feedActivations(replicate(0));
         end

         // Pre-fetch next activation if available
         if (actIdx < tLen) begin
            aSRAM.readReq(aBase + actIdx);
            actIdx <= actIdx + 1;
         end

         streamCycle <= streamCycle + 1;
      end else begin
         cstate <= Drain;
      end
   endrule

   // Phase 4: Collect results. Also optionally deposit into a PSUM
   // bucket row so multi-K-tile GEMM can accumulate in hardware
   // without round-tripping through VRegFile.
   rule do_drain (cstate == Drain);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=DRAIN", cycle);
`endif
      let r = array.getResults;
      outputBuf <= r;
      array.clearAll;
      case (psumModeReg)
         PSUM_WRITE: begin
`ifdef TRACE
            $display("TRACE cycle=%0d unit=MXU ev=PSUM_WRITE addr=%0d row=%0d",
                     cycle, psumAddrReg, psumRowReg);
`endif
            psum.writeRow(truncate(psumAddrReg), psumRowReg, r);
         end
         PSUM_ACCUMULATE: begin
`ifdef TRACE
            $display("TRACE cycle=%0d unit=MXU ev=PSUM_ACCUMULATE addr=%0d row=%0d",
                     cycle, psumAddrReg, psumRowReg);
`endif
            psum.accumulateRow(truncate(psumAddrReg), psumRowReg, r);
         end
      endcase
      cstate <= Done;
   endrule

   method Action start(UInt#(TLog#(depth)) weightBase,
                       UInt#(TLog#(depth)) actBase,
                       UInt#(TLog#(depth)) tileLen) if (cstate == Idle || cstate == Done);
      wBase  <= weightBase;
      aBase  <= actBase;
      tLen   <= tileLen;
      actIdx <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_WEIGHT_STATIONARY;
      cstate <= LoadWeights;
   endmethod

   method Action startPsum(UInt#(TLog#(depth)) weightBase,
                           UInt#(TLog#(depth)) actBase,
                           UInt#(TLog#(depth)) tileLen,
                           UInt#(8) psumAddr,
                           UInt#(TLog#(rows)) psumRow,
                           PsumMode psumMode) if (cstate == Idle || cstate == Done);
      wBase       <= weightBase;
      aBase       <= actBase;
      tLen        <= tileLen;
      actIdx      <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      psumAddrReg <= psumAddr;
      psumRowReg  <= psumRow;
      psumModeReg <= psumMode;
      dfModeReg   <= DF_WEIGHT_STATIONARY;
      cstate <= LoadWeights;
   endmethod

   method Action startOS(UInt#(TLog#(depth)) weightBase,
                         UInt#(TLog#(depth)) actBase,
                         UInt#(TLog#(depth)) tileLen) if (cstate == Idle || cstate == Done);
      wBase  <= weightBase;
      aBase  <= actBase;
      tLen   <= tileLen;
      actIdx <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_OUTPUT_STATIONARY;
      cstate <= LoadWeights;
   endmethod

   method Bool isDone;
      return cstate == Done;
   endmethod

   method Vector#(cols, Int#(32)) results if (cstate == Done);
      return outputBuf;
   endmethod

   method ControlState getState;
      return cstate;
   endmethod

   method DataflowMode getDataflowMode;
      return dfModeReg;
   endmethod

endmodule

endpackage
