package HBMModel;

import Vector :: *;
import RegFile :: *;

interface HBMModel_IFC#(numeric type depth,
                         numeric type sublanes,
                         numeric type lanes,
                         numeric type latency);
   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
endinterface

module mkHBMModel(HBMModel_IFC#(depth, sublanes, lanes, latency))
   provisos(
      Add#(1, d_, depth),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Add#(1, lat_, latency),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   RegFile#(UInt#(TLog#(depth)), Vector#(sublanes, Vector#(lanes, Int#(32))))
      mem <- mkRegFileFull;

   // Shift register pipeline: pipeline[0] is newest, pipeline[latency-1] is output.
   // Each cycle shift_pipeline advances data toward the output.
   Reg#(Vector#(latency, Vector#(sublanes, Vector#(lanes, Int#(32))))) pipeline
      <- mkReg(replicate(replicate(replicate(0))));
   Reg#(Vector#(latency, Bool)) pValid <- mkReg(replicate(False));

   // Wire used to communicate a new readReq to shift_pipeline in the same cycle
   RWire#(Vector#(sublanes, Vector#(lanes, Int#(32)))) reqWire <- mkRWire;

   // Single rule handles all pipeline shifts; also inserts from readReq if fired
   rule shift_pipeline;
      Vector#(latency, Vector#(sublanes, Vector#(lanes, Int#(32)))) newPipe = pipeline;
      Vector#(latency, Bool) newValid = pValid;
      // Shift existing data toward output (high index)
      for (Integer i = valueOf(latency) - 1; i > 0; i = i - 1) begin
         newPipe[i]  = pipeline[i-1];
         newValid[i] = pValid[i-1];
      end
      // Insert new request at stage 0 if readReq fired this cycle
      let mdata = reqWire.wget;
      newValid[0] = isValid(mdata);
      if (isValid(mdata)) newPipe[0] = fromMaybe(?, mdata);
      pipeline <= newPipe;
      pValid   <= newValid;
   endrule

   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      mem.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      reqWire.wset(mem.sub(addr));
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
      return pipeline[valueOf(latency) - 1];
   endmethod

endmodule

export HBMModel_IFC(..);
export mkHBMModel;

endpackage
