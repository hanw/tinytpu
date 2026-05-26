package Controller;

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import PSUMBank :: *;
import VPU :: *;

export ControlState(..);
export DataflowMode(..);
export EpilogueConfig(..);
export Controller_IFC(..);
export mkController;

typedef enum {
   Idle,
   LoadWeights,
   LoadWeightsResp,
   LoadWeightsRespOS,   // OS: buffer tile, don't preload into PEs
   LoadActsOS,          // OS: prefetch K activation vectors into buffer
   StreamActivations,   // WS + WS-accumulate feed path
   StreamOS,            // OS: feedPair staircase, K + skew cycles
   Drain,
   Done
} ControlState deriving (Bits, Eq, FShow);

// Dataflow mode.
//  - DF_WEIGHT_STATIONARY: the default. loadWeights once, stream
//    activations through, drain column sums. Clears the accumulator
//    at drain time so consecutive starts each begin from zero.
//  - DF_WEIGHT_STATIONARY_ACCUMULATE: runs the WS feed path but
//    skips the drain-time clear, so back-to-back startAccumulate calls
//    add partial results into the same PE accumulator (multi-K-tile
//    GEMM). NOT a distinct dataflow — the PE still holds a preloaded
//    weight; only the clear gating differs from WS.
//  - DF_OUTPUT_STATIONARY: real OS dataflow. No preload — each cycle
//    feeds one weight row and one activation column as a staircase;
//    psums accumulate per-PE; drain returns the full (rows x cols)
//    matrix via resultsMatrix().
typedef enum {
   DF_WEIGHT_STATIONARY,
   DF_WEIGHT_STATIONARY_ACCUMULATE,
   DF_OUTPUT_STATIONARY
} DataflowMode deriving (Bits, Eq, FShow);

typedef struct {
   Bool biasEnable;
   Bool reluEnable;
   Bool reduceEnable;
   Bool reduceSumsq;
} EpilogueConfig deriving (Bits, Eq, FShow);

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
   // Fused GEMM + epilogue (bias / relu at drain time). Runs the same WS
   // feed path as startPsum; at drain the full rows x cols result matrix
   // is post-processed according to epiCfg, then stored in epilogueBuf.
   // The SXU reads the result back via epilogueResult once isDone.
   method Action startEpilogue(UInt#(TLog#(depth)) weightBase,
                               UInt#(TLog#(depth)) actBase,
                               UInt#(TLog#(depth)) tileLen,
                               Vector#(cols, Int#(32)) biasVec,
                               EpilogueConfig epiCfg);
   // Generic-VPU MXU epilogue: WS dispatch that, at drain, applies an
   // arbitrary VpuOp lane-wise between drainMatrix and a tile-shape
   // src2, depositing the result in epilogueBuf. The curated op subset
   // currently lowered combinationally is {VPU_ADD, VPU_SUB, VPU_MUL,
   // VPU_MAX, VPU_MIN}; any other op is treated as pass-through of
   // drainMatrix so that the SXU can still complete the dispatch.
   // wbVmem is captured but unused inside the Controller — the SXU
   // reads it back to choose the writeback target.
   method Action startGenericEpilogue(UInt#(TLog#(depth)) weightBase,
                                      UInt#(TLog#(depth)) actBase,
                                      UInt#(TLog#(depth)) tileLen,
                                      Vector#(rows, Vector#(cols, Int#(32))) src2Tile,
                                      VpuOp vpuOp,
                                      Bool wbVmem);
   // WS with cross-dispatch accumulator-hold. Runs the same feed path
   // as start() but skips the drain-time clearAll, so consecutive
   // startAccumulate calls sum partial psums into the same PE. The
   // caller uses clearArray() to begin a fresh accumulation epoch.
   // Useful for multi-tile GEMM K-reduction in hardware. Not a
   // distinct dataflow from WS.
   method Action startAccumulate(UInt#(TLog#(depth)) weightBase,
                                 UInt#(TLog#(depth)) actBase,
                                 UInt#(TLog#(depth)) tileLen);
   // Real output-stationary dispatch. Runs the feedPair staircase:
   // each cycle reads one activation vector (length rows) from
   // ActivationSRAM and one weight row (length cols) from the tile
   // stored at weightBase, feeding both as the systolic wavefront.
   // kLen is the inner reduction depth (must be <= rows with the
   // current single-tile weight SRAM read).
   method Action startOS(UInt#(TLog#(depth)) weightBase,
                         UInt#(TLog#(depth)) actBase,
                         UInt#(TLog#(depth)) kLen);
   // Real-OS dispatch that preserves the per-PE accumulator state,
   // so consecutive startOsAccumulate calls add another K-tile worth
   // of psums into the same matrix. Lets OS scale past K == rows.
   // Caller is expected to call clearArray() before the first tile
   // of a fresh epoch.
   method Action startOsAccumulate(UInt#(TLog#(depth)) weightBase,
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
   // Post-epilogue result matrix (valid when isDone and epilogueActive).
   method Vector#(rows, Vector#(cols, Int#(32))) epilogueResult;
   // Per-row INT64 reduction statistic. Callable only when reduceEnable was
   // set in the preceding epilogue dispatch (in addition to isDone and
   // epilogueActive) — calling it otherwise deadlocks the caller rather
   // than returning a stale/undefined value.
   method Vector#(rows, Int#(64)) epilogueStat;
   // True when the most recent dispatch was startGenericEpilogue, so the
   // SXU can read back the captured writeback-mode bit and choose between
   // VREG and VMEM writeback. Valid once isDone.
   method Bool genericEpilogueWbVmem;
   method ControlState getState;
   // Currently selected dataflow mode (set via startOS in future iters;
   // exposed now so SXU/tests can observe Controller state without
   // changing the start/startPsum signatures).
   method DataflowMode getDataflowMode;
   // SP3: drain-side requantization.
   // setRequantConfig stores the INT32 scale multiplier and shift;
   // startRequant launches a WS dispatch that, at drain time, applies
   // the requant formula per column and writes INT8 results to aSRAM.
   // Precondition: shift >= 1. With shift == 0, `scaleShift - 1` wraps in
   // UInt#(5) arithmetic to 31, producing a large wrong rounding offset
   // instead of zero. All v1 quantization flows use shift >= 1.
   method Action setRequantConfig(Int#(32) mul, UInt#(5) shift);
   method Action startRequant(UInt#(TLog#(depth)) weightBase,
                              UInt#(TLog#(depth)) actBase,
                              UInt#(TLog#(depth)) tileLen,
                              UInt#(8) asramTargetBase);
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
            Add#(psumTrunc__, TLog#(psumDepth), 8),
            Add#(logd8_, TLog#(depth), 8),
            Add#(0, cols, rows));

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

   // Requantization state: scale factor, shift, active flag, and ASRAM target.
   Reg#(Int#(32))  scaleMul             <- mkReg(0);
   Reg#(UInt#(5))  scaleShift           <- mkReg(0);
   Reg#(Bool)      requantActive        <- mkReg(False);
   Reg#(UInt#(8))  requantTargetBase    <- mkReg(0);
   Reg#(UInt#(8))  requantTargetOffset  <- mkReg(0);

   // Epilogue state: bias vector, config, active flag, and result buffer.
   Reg#(Vector#(cols, Int#(32)))             biasReg       <- mkRegU;
   Reg#(EpilogueConfig)                      epiCfgReg     <- mkRegU;
   Reg#(Bool)                                epilogueActive <- mkReg(False);
   Reg#(Vector#(rows, Vector#(cols, Int#(32)))) epilogueBuf <- mkRegU;
   // Per-row INT64 reduction accumulator (SUM or SUM-OF-SQUARES).
   // Populated at drain time when epiCfgReg.reduceEnable is set.
   Reg#(Vector#(rows, Int#(64)))             epilogueStatBuf <- mkRegU;

   // Generic-VPU epilogue state. src2Reg is the tile-shape second
   // operand captured at start time. vpuOpReg picks the lane-wise op.
   // wbVmemReg is opaque to the Controller — the SXU reads it back via
   // genericEpilogueWbVmem to choose its writeback path. The drain
   // result lands in the shared epilogueBuf so epilogueResult covers
   // both legacy and generic dispatches.
   Reg#(Vector#(rows, Vector#(cols, Int#(32)))) src2Reg          <- mkRegU;
   Reg#(VpuOp)                                  vpuOpReg         <- mkRegU;
   Reg#(Bool)                                   wbVmemReg        <- mkReg(False);
   Reg#(Bool)                                   genericEpilogueActive <- mkReg(False);
`ifdef TRACE
   Reg#(UInt#(32)) cycle <- mkReg(0);

   rule count_trace_cycles;
      cycle <= cycle + 1;
   endrule
`endif

   // Phase 1: Issue weight read request. The next-state depends on
   // the dataflow mode: WS / WS-accumulate preload into the array,
   // OS buffers the tile and streams rows per cycle.
   rule do_load_weights (cstate == LoadWeights);
`ifdef TRACE
      $display("TRACE cycle=%0d unit=MXU ev=LOAD_W addr=%0d", cycle, wBase);
`endif
      wSRAM.readReq(wBase);
      if (dfModeReg == DF_OUTPUT_STATIONARY)
         cstate <= LoadWeightsRespOS;
      else
         cstate <= LoadWeightsResp;
   endrule

   // Phase 2 (WS / WS-accumulate): Receive weight data, preload into
   // the array, and issue first activation read.
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

   // Phase 2 (OS): Buffer the weight tile, issue the first activation
   // read, and go into the activation-prefetch state so we can
   // assemble the K x rows activation matrix up front.
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
      let drainMatrix = array.getMatrix;
      outputBuf <= r;
      matrixBuf <= drainMatrix;
      // DF_WEIGHT_STATIONARY clears the array so each dispatch starts
      // from zero. The other two modes preserve the accumulator:
      // DF_WEIGHT_STATIONARY_ACCUMULATE lets consecutive dispatches
      // sum their col-sums; DF_OUTPUT_STATIONARY leaves a full matrix
      // of per-PE psums that the consumer reads via resultsMatrix()
      // and clears explicitly with clearArray().
      if (dfModeReg == DF_WEIGHT_STATIONARY)
         array.clearAll;
      if (requantActive) begin
`ifdef TRACE
         $display("TRACE cycle=%0d unit=MXU ev=REQUANT base=%0d off=%0d",
                  cycle, requantTargetBase, requantTargetOffset);
`endif
         Vector#(cols, Int#(8)) reqRow = newVector;
         for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
            Int#(32) acc     = r[ci];
            Int#(64) wide    = signExtend(acc) * signExtend(scaleMul);
            Int#(64) one     = 1;
            Int#(64) rounded = wide + (one << (scaleShift - 1));
            Int#(64) shifted = rounded >> scaleShift;
            Int#(64) clamped = (shifted >  127) ?  127 :
                               (shifted < -128) ? -128 : shifted;
            reqRow[ci] = truncate(clamped);
         end
         // requantTargetBase + requantTargetOffset is UInt#(8) arithmetic; callers must
         // keep base + tileLen <= 255 (current 4x4/depth=32 configs trivially satisfy this).
         aSRAM.writeBack(truncate(requantTargetBase + requantTargetOffset), reqRow);
         requantTargetOffset <= requantTargetOffset + 1;
      end else if (epilogueActive) begin
         Vector#(rows, Vector#(cols, Int#(32))) m = drainMatrix;
         Vector#(rows, Vector#(cols, Int#(32))) outm = m;
         for (Integer ri = 0; ri < valueOf(rows); ri = ri + 1)
            for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
               Int#(32) v = m[ri][ci];
               if (epiCfgReg.biasEnable) v = v + biasReg[ci];
               if (epiCfgReg.reluEnable && v < 0) v = 0;
               outm[ri][ci] = v;
            end
         epilogueBuf <= outm;
         if (epiCfgReg.reduceEnable) begin
            Vector#(rows, Int#(64)) stat = replicate(0);
            for (Integer ri = 0; ri < valueOf(rows); ri = ri + 1) begin
               Int#(64) acc = 0;
               for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
                  Int#(64) e = signExtend(outm[ri][ci]);
                  acc = acc + (epiCfgReg.reduceSumsq ? (e * e) : e);
               end
               stat[ri] = acc;
            end
            epilogueStatBuf <= stat;
         end
      end else if (genericEpilogueActive) begin
         // Lane-wise apply vpuOpReg(drainMatrix, src2Reg) into epilogueBuf.
         // Curated subset is inlined as combinational arithmetic; ops
         // outside the subset fall through as pass-through of drainMatrix
         // so the dispatch still completes deterministically.
         Vector#(rows, Vector#(cols, Int#(32))) m    = drainMatrix;
         Vector#(rows, Vector#(cols, Int#(32))) src2 = src2Reg;
         Vector#(rows, Vector#(cols, Int#(32))) outm = m;
         for (Integer ri = 0; ri < valueOf(rows); ri = ri + 1)
            for (Integer ci = 0; ci < valueOf(cols); ci = ci + 1) begin
               Int#(32) a = m[ri][ci];
               Int#(32) b = src2[ri][ci];
               Int#(32) v = a;
               case (vpuOpReg)
                  VPU_ADD: v = a + b;
                  VPU_SUB: v = a - b;
                  VPU_MUL: v = a * b;
                  VPU_MAX: v = (a > b) ? a : b;
                  VPU_MIN: v = (a < b) ? a : b;
               endcase
               outm[ri][ci] = v;
            end
         // Lane-pair rotation. Separate inner-loop pass because the op
         // works on adjacent lane pairs rather than a single (r,c) cell:
         //   out[r][2p]     = drain[r][2p]   * src2[r][2p]
         //                  - drain[r][2p+1] * src2[r][2p+1]
         //   out[r][2p + 1] = drain[r][2p]   * src2[r][2p+1]
         //                  + drain[r][2p+1] * src2[r][2p]
         // Matches VPU.bsv VPU_IPAIR_ROTATE semantics so a future caller
         // can lower to either the dedicated VPU dispatch or the fused
         // MXU epilogue interchangeably.
         if (vpuOpReg == VPU_IPAIR_ROTATE)
            for (Integer ri = 0; ri < valueOf(rows); ri = ri + 1)
               for (Integer p = 0; p < valueOf(cols) / 2; p = p + 1) begin
                  Int#(32) de = m[ri][2 * p];
                  Int#(32) doo = m[ri][2 * p + 1];
                  Int#(32) c  = src2[ri][2 * p];
                  Int#(32) sn = src2[ri][2 * p + 1];
                  outm[ri][2 * p]     = de * c  - doo * sn;
                  outm[ri][2 * p + 1] = de * sn + doo * c;
            end
         epilogueBuf <= outm;
      end else begin
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
      end
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
      epilogueActive <= False;
      genericEpilogueActive <= False;
      requantActive  <= False;
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
      epilogueActive <= False;
      genericEpilogueActive <= False;
      requantActive  <= False;
      array.clearAll;
      cstate <= LoadWeights;
   endmethod

   method Action startEpilogue(UInt#(TLog#(depth)) weightBase,
                               UInt#(TLog#(depth)) actBase,
                               UInt#(TLog#(depth)) tileLen,
                               Vector#(cols, Int#(32)) biasVec,
                               EpilogueConfig epiCfg) if (cstate == Idle || cstate == Done);
      wBase       <= weightBase;
      aBase       <= actBase;
      tLen        <= tileLen;
      actIdx      <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_WEIGHT_STATIONARY;
      biasReg     <= biasVec;
      epiCfgReg   <= epiCfg;
      epilogueActive <= True;
      genericEpilogueActive <= False;
      requantActive  <= False;
      array.clearAll;
      cstate <= LoadWeights;
   endmethod

   method Action startGenericEpilogue(UInt#(TLog#(depth)) weightBase,
                                      UInt#(TLog#(depth)) actBase,
                                      UInt#(TLog#(depth)) tileLen,
                                      Vector#(rows, Vector#(cols, Int#(32))) src2Tile,
                                      VpuOp vpuOp,
                                      Bool wbVmem) if (cstate == Idle || cstate == Done);
      wBase       <= weightBase;
      aBase       <= actBase;
      tLen        <= tileLen;
      actIdx      <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_WEIGHT_STATIONARY;
      src2Reg     <= src2Tile;
      vpuOpReg    <= vpuOp;
      wbVmemReg   <= wbVmem;
      epilogueActive <= False;
      genericEpilogueActive <= True;
      requantActive  <= False;
      array.clearAll;
      cstate <= LoadWeights;
   endmethod

   method Action startAccumulate(UInt#(TLog#(depth)) weightBase,
                                 UInt#(TLog#(depth)) actBase,
                                 UInt#(TLog#(depth)) tileLen) if (cstate == Idle || cstate == Done);
      wBase  <= weightBase;
      aBase  <= actBase;
      tLen   <= tileLen;
      actIdx <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_WEIGHT_STATIONARY_ACCUMULATE;
      epilogueActive <= False;
      genericEpilogueActive <= False;
      requantActive  <= False;
      cstate <= LoadWeights;
   endmethod

   method Action startOS(UInt#(TLog#(depth)) weightBase,
                         UInt#(TLog#(depth)) actBase,
                         UInt#(TLog#(depth)) kLen)
         if (cstate == Idle || cstate == Done);
      wBase  <= weightBase;
      aBase  <= actBase;
      osKLen <= kLen;
      // WS-side counters reset so a follow-up WS dispatch starts from
      // a clean slate even if the caller skips the explicit clearArray
      // between modes.
      tLen   <= 0;
      actIdx <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      osFeedCycle  <= 0;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_OUTPUT_STATIONARY;
      epilogueActive <= False;
      genericEpilogueActive <= False;
      requantActive  <= False;
      // OS consumers start from a cleared array so each dispatch's
      // psum matrix reflects only its own inputs.
      array.clearAll;
      cstate <= LoadWeights;
   endmethod

   method Action startOsAccumulate(UInt#(TLog#(depth)) weightBase,
                                   UInt#(TLog#(depth)) actBase,
                                   UInt#(TLog#(depth)) kLen)
         if (cstate == Idle || cstate == Done);
      wBase  <= weightBase;
      aBase  <= actBase;
      osKLen <= kLen;
      tLen   <= 0;
      actIdx <= 0;
      streamCycle <= 0;
      firstActRead <= False;
      osFeedCycle  <= 0;
      psumModeReg <= PSUM_OFF;
      dfModeReg   <= DF_OUTPUT_STATIONARY;
      epilogueActive <= False;
      genericEpilogueActive <= False;
      requantActive  <= False;
      // Skip the clearAll — psums carry over from the previous OS
      // dispatch, letting multi-K-tile OS scale past K == rows.
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

   method Vector#(rows, Vector#(cols, Int#(32))) epilogueResult
         if (cstate == Done && (epilogueActive || genericEpilogueActive));
      return epilogueBuf;
   endmethod

   method Vector#(rows, Int#(64)) epilogueStat if (cstate == Done && epilogueActive && epiCfgReg.reduceEnable);
      return epilogueStatBuf;
   endmethod

   method Bool genericEpilogueWbVmem if (cstate == Done && genericEpilogueActive);
      return wbVmemReg;
   endmethod

   method ControlState getState;
      return cstate;
   endmethod

   method DataflowMode getDataflowMode;
      return dfModeReg;
   endmethod

   // Precondition: shift >= 1. With shift == 0, `scaleShift - 1` wraps in
   // UInt#(5) arithmetic to 31, producing a large wrong rounding offset
   // instead of zero. All v1 quantization flows use shift >= 1.
   method Action setRequantConfig(Int#(32) mul, UInt#(5) shift) if (cstate == Idle || cstate == Done);
      scaleMul   <= mul;
      scaleShift <= shift;
   endmethod

   method Action startRequant(UInt#(TLog#(depth)) weightBase,
                              UInt#(TLog#(depth)) actBase,
                              UInt#(TLog#(depth)) tileLen,
                              UInt#(8) asramTargetBase) if (cstate == Idle || cstate == Done);
      wBase        <= weightBase;
      aBase        <= actBase;
      tLen         <= tileLen;
      actIdx       <= 0;
      streamCycle  <= 0;
      firstActRead <= False;
      psumModeReg  <= PSUM_OFF;
      dfModeReg    <= DF_WEIGHT_STATIONARY;
      epilogueActive      <= False;
      genericEpilogueActive <= False;
      requantActive       <= True;
      requantTargetBase   <= asramTargetBase;
      requantTargetOffset <= 0;
      array.clearAll;
      cstate <= LoadWeights;
   endmethod

endmodule

endpackage
