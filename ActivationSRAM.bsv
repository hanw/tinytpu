package ActivationSRAM;

import Vector :: *;
import RegFile :: *;

export ActivationSRAM_IFC(..);
export mkActivationSRAM;

interface ActivationSRAM_IFC#(numeric type depth, numeric type rows);
   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(rows, Int#(8)) readResp;
endinterface

module mkActivationSRAM(ActivationSRAM_IFC#(depth, rows))
   provisos(Add#(1, _, depth),
            Add#(1, _, rows),
            Log#(depth, logd),
            Bits#(Vector#(rows, Int#(8)), sz));

   RegFile#(UInt#(logd), Vector#(rows, Int#(8))) mem
      <- mkRegFileFull;

   Reg#(Vector#(rows, Int#(8))) resp <- mkRegU;

   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
      mem.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      resp <= mem.sub(addr);
   endmethod

   method Vector#(rows, Int#(8)) readResp;
      return resp;
   endmethod

endmodule

endpackage
