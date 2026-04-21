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
   LoadWeightsRespOS,   // OS-real: buffer tile, don't preload into PEs
   LoadActsOS,          // OS-real: prefetch K activation vectors into buffer
   StreamActivations,   // WS + old "OS" (accumulator-hold) feed path
   StreamOS,            // OS-real: feedPair staircase, K + skew cycles
   Drain,
   Done
} ControlState deriving (Bits, Eq, FShow);

// Dataflow mode.
//  - DF_WEIGHT_STATIONARY: the default. loadWeights once, stream
//    activations through, drain column sums. Clears the accumulator
//    at drain time so consecutive starts each begin from zero.
//  - DF_OUTPUT_STATIONARY: variant that runs the WS feed path but
//    skips the drain-time clear, so back-to-back startOS calls
//    accumulate across K-tiles. Useful for multi-tile GEMM; it is
//    NOT a distinct dataflow (the PE still holds a preloaded weight).
//  - DF_OUTPUT_STATIONARY_REAL: real OS dataflow. No preload — each
//    cycle feeds one weight row and one activation column as a
//    staircase; psums accumulate per-PE; drain returns the full
//    (rows x cols) matrix via resultsMatrix().
typedef enum {
   DF_WEIGHT_STATIONARY,
   DF_OUTPUT_STATIONARY,
   DF_OUTPUT_STATIONARY_REAL
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
   // Real output-stationary dispatch. Runs the feedPair staircase:
   // each cycle reads one activation vector (length rows) from
   // ActivationSRAM and one weight row (length cols) from the tile
   // stored at weightBase, feeding both as the systolic wavefront.
   // kLen is the inner reduction depth (must be <= rows with the
   // current single-tile weight SRAM read).
   method Action startOsReal(UInt#(TLog#(depth)) weightBase,
                             UInt#(TLog#(depth)) actBase,
                             UInt#(TLog#(depth)) kLen);
   // Explicitly reset the systolic-array PE accumulators. Needed between
   // OS-mode epochs: OS dispatches intentionally preserve accumulator
   // state across consecutive starts so the caller can accumulate over
   // streamed K-tiles; clearArray() starts a fresh epoch.
   method Action clearArray;
   method Bool isDone;
   method Vector#(cols, Int#(32)) results;
   // Full matrix drain for OS-real dispatches. In WS / old-OS mode
   // this returns the per-PE accumulator state which collapses to
   // `results` when summed down columns.
   method Vector#(rows, Vector#(cols, Int#(32))) resultsMatrix;
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
            Add#(osr_, TLog#(rows), 32),
            Add#(osrd_, TLog#(rows), TLog#(depth)),
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

   // OS-real state. The feedPair staircase runs for kLen + (rows-1) +
   // (cols-1) cycles; column c of w_top carries W[t-c][c] (else 0) and
   // row r of a_left carries A[r][t-r] (else 0). Both buffers are
   // populated before the stream starts so the feed rule can read
   // them combinationally.
   Reg#(Vector#(rows, Vector#(cols, Int#(8)))) osWeightBuf <- mkRegU;
   Reg#(Vector#(rows, Vector#(rows, Int#(8)))) osActBuf    <- mkRegU;
   Reg#(UInt#(32)) osFeedCycle <- mkReg(0);
   Reg#(UInt#(TLog#(depth))) osActLoadIdx <- mkReg(0);
   Reg#(UInt#(TLog#(depth))) osKLen <- mkReg(0);
   Reg#(Vector#(rows, Vector#(cols, Int#(32)))) matrixBuf <- mkRegU;

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

   // Phase 1: Issue weight read request. The next-state depends on
   // the dataflow mode: WS / old-OS preload into the array, OS-real
   // buffers the tile and streams rows per cycle.
   rule do_load_weights (cstate == LoadWeights);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=LOAD_W addr=%0d", cycle, wBase);
`endif
      wSRAM.readReq(wBase);
      if (dfModeReg == DF_OUTPUT_STATIONARY_REAL)
         cstate <= LoadWeightsRespOS;
      else
         cstate <= LoadWeightsResp;
   endrule

   // Phase 2 (WS / old-OS): Receive weight data, preload into array,
   // and issue first activation read.
   rule do_load_weights_resp (cstate == LoadWeightsResp);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=LOAD_W_RESP", cycle);
`endif
      array.loadWeights(wSRAM.readResp);
      aSRAM.readReq(aBase);
      actIdx <= 1;
      streamCycle <= 0;
      firstActRead <= True;
      cstate <= StreamActivations;
   endrule

   // Phase 2 (OS-real): Buffer the weight tile, issue the first
   // activation read, and go into the activation-prefetch state so
   // we can assemble the K x rows activation matrix up front.
   rule do_load_weights_resp_os (cstate == LoadWeightsRespOS);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=LOAD_W_RESP_OS", cycle);
`endif
      osWeightBuf <= wSRAM.readResp;
      aSRAM.readReq(aBase);
      osActLoadIdx <= 1;
      cstate <= LoadActsOS;
   endrule

   // Phase 2.5 (OS-real): prefetch K activation vectors into osActBuf
   // so the feedPair staircase can read them combinationally with the
   // appropriate diagonal skew.
   rule do_load_acts_os (cstate == LoadActsOS);
      let k = osActLoadIdx;
      // Activation read issued `k` cycles ago lands at aSRAM.readResp
      // on this cycle. Place it into osActBuf slot k-1.
      UInt#(TLog#(rows)) slot = unpack(truncate(pack(k - 1)));
      Vector#(rows, Vector#(rows, Int#(8))) nextBuf = osActBuf;
      nextBuf[slot] = aSRAM.readResp;
      osActBuf <= nextBuf;

      if (k < osKLen) begin
         aSRAM.readReq(aBase + k);
         osActLoadIdx <= k + 1;
      end else begin
         osFeedCycle <= 0;
         cstate <= StreamOS;
      end
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=LOAD_A_OS slot=%0d", cycle, slot);
`endif
   endrule

   // Phase 3 (OS-real): feedPair staircase. At global feed cycle t:
   //   w_top[c]  = W[t-c][c] if 0 <= t-c < kLen else 0
   //   a_left[r] = A[r][t-r] if 0 <= t-r < kLen else 0
   // osActBuf[k] is column k of A, so osActBuf[k][r] = A[r][k].
   // Total cycles = kLen + (rows-1) + (cols-1). Within kLen all
   // diagonals line up; the remaining cycles flush the in-flight
   // wavefront through the array.
   rule do_stream_os (cstate == StreamOS);
      let kU = extend(osKLen);
      let totalCycles = kU +
                        fromInteger(valueOf(rows) - 1) +
                        fromInteger(valueOf(cols) - 1);
      let t = osFeedCycle;

      if (t < totalCycles) begin
         Vector#(cols, Int#(8)) w_top  = replicate(0);
         Vector#(rows, Int#(8)) a_left = replicate(0);
         // Staircase weights: column c taps osWeightBuf[t-c][c].
         for (Integer c = 0; c < valueOf(cols); c = c + 1) begin
            if (t >= fromInteger(c)) begin
               let k = t - fromInteger(c);
               if (k < kU) begin
                  UInt#(TLog#(rows)) rIdx = unpack(truncate(pack(k)));
                  w_top[c] = osWeightBuf[rIdx][c];
               end
            end
         end
         // Staircase activations: row r taps osActBuf[t-r][r].
         for (Integer r = 0; r < valueOf(rows); r = r + 1) begin
            if (t >= fromInteger(r)) begin
               let k = t - fromInteger(r);
               if (k < kU) begin
                  UInt#(TLog#(rows)) kIdx = unpack(truncate(pack(k)));
                  a_left[r] = osActBuf[kIdx][r];
               end
            end
         end
`ifdef TRACE
         $display("TRACE cycle=%0d unit=MXU ev=FEED_OS cyc=%0d", cycle, t);
`endif
         array.feedPair(w_top, a_left);
         osFeedCycle <= t + 1;
      end else begin
         cstate <= Drain;
      end
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
      matrixBuf <= array.getMatrix;
      // In WS mode, clear the PE accumulators so the next dispatch starts
      // from zero. OS modes preserve them: the old-OS mode accumulates
      // col-sums across K-tiles; OS-real leaves a full matrix of psums
      // that the consumer reads via resultsMatrix() before clearing.
      if (dfModeReg == DF_WEIGHT_STATIONARY)
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
      // WS always starts from a zeroed accumulator. Prior dispatches may
      // have been OS mode (which intentionally does not drain-clear).
      array.clearAll;
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
      array.clearAll;
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

   method Action startOsReal(UInt#(TLog#(depth)) weightBase,
                             UInt#(TLog#(depth)) actBase,
                             UInt#(TLog#(depth)) kLen)
         if (cstate == Idle || cstate == Done);
      wBase  <= weightBase;
      aBase  <= actBase;
      osKLen <= kLen;
      // The WS-side counters are reset here too so a follow-up WS
      // dispatch starts from a clean slate even if the caller skips
      // the explicit clearArray between modes.
      tLen   <= 0;
      actIdx <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      osFeedCycle  <= 0;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_OUTPUT_STATIONARY_REAL;
      // OS-real consumers are expected to start from a cleared array
      // so each dispatch's psum matrix reflects only its own inputs.
      array.clearAll;
      cstate <= LoadWeights;
   endmethod

   method Action clearArray if (cstate == Idle || cstate == Done);
      array.clearAll;
   endmethod

   method Bool isDone;
      return cstate == Done;
   endmethod

   method Vector#(cols, Int#(32)) results if (cstate == Done);
      return outputBuf;
   endmethod

   method Vector#(rows, Vector#(cols, Int#(32))) resultsMatrix if (cstate == Done);
      return matrixBuf;
   endmethod

   method ControlState getState;
      return cstate;
   endmethod

   method DataflowMode getDataflowMode;
      return dfModeReg;
   endmethod

endmodule

endpackage
