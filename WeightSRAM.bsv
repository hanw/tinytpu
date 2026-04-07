package WeightSRAM;

import Vector :: *;
import RegFile :: *;

export WeightSRAM_IFC(..);
export mkWeightSRAM;

interface WeightSRAM_IFC#(numeric type depth, numeric type rows, numeric type cols);
   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Vector#(cols, Int#(8))) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(rows, Vector#(cols, Int#(8))) readResp;
endinterface

module mkWeightSRAM(WeightSRAM_IFC#(depth, rows, cols))
   provisos(Add#(1, d_, depth),
            Add#(1, r_, rows),
            Add#(1, c_, cols),
            Log#(depth, logd),
            Bits#(Vector#(rows, Vector#(cols, Int#(8))), sz));

   RegFile#(UInt#(logd), Vector#(rows, Vector#(cols, Int#(8)))) mem
      <- mkRegFileFull;

   Reg#(Vector#(rows, Vector#(cols, Int#(8)))) resp <- mkRegU;

   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Vector#(cols, Int#(8))) data);
      mem.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      resp <= mem.sub(addr);
   endmethod

   method Vector#(rows, Vector#(cols, Int#(8))) readResp;
      return resp;
   endmethod

endmodule

endpackage
