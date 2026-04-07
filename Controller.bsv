package Controller;

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;

export ControlState(..);
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

interface Controller_IFC#(numeric type rows, numeric type cols, numeric type depth);
   method Action start(UInt#(TLog#(depth)) weightBase,
                       UInt#(TLog#(depth)) actBase,
                       UInt#(TLog#(depth)) tileLen);
   method Bool isDone;
   method Vector#(cols, Int#(32)) results;
   method ControlState getState;
endinterface

module mkController#(
      SystolicArray_IFC#(rows, cols) array,
      WeightSRAM_IFC#(depth, rows, cols) wSRAM,
      ActivationSRAM_IFC#(depth, rows) aSRAM
   )(Controller_IFC#(rows, cols, depth))
   provisos(Add#(1, r_, rows),
            Add#(1, c_, cols),
            Add#(1, d_, depth),
            Log#(depth, logd),
            Add#(logd_, TLog#(depth), 32));

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

   // Phase 1: Issue weight read request
   rule do_load_weights (cstate == LoadWeights);
      wSRAM.readReq(wBase);
      cstate <= LoadWeightsResp;
   endrule

   // Phase 2: Receive weight data and load into array, issue first activation read
   rule do_load_weights_resp (cstate == LoadWeightsResp);
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

   // Phase 4: Collect results
   rule do_drain (cstate == Drain);
      outputBuf <= array.getResults;
      array.clearAll;
      cstate <= Done;
   endrule

   method Action start(UInt#(TLog#(depth)) weightBase,
                       UInt#(TLog#(depth)) actBase,
                       UInt#(TLog#(depth)) tileLen) if (cstate == Idle);
      wBase  <= weightBase;
      aBase  <= actBase;
      tLen   <= tileLen;
      actIdx <= 0;
      streamCycle <= 0;
      firstActRead <= False;
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

endmodule

endpackage
