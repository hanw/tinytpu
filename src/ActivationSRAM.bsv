package ActivationSRAM;

import Vector :: *;
import RegFile :: *;

export ActivationSRAM_IFC(..);
export mkActivationSRAM;

interface ActivationSRAM_IFC#(numeric type depth, numeric type rows);
   // Two write ports kept distinct so the DB wrapper can route each to
   // its own bank (write -> inactive / DMA preload, writeBack -> active
   // / drain-side writeback). In this single-bank leaf they are
   // functionally identical.
   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
   // Used by the DB wrapper's drain-side writeback path; in this single-bank
   // leaf the two write methods are functionally identical.
   method Action writeBack(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(rows, Int#(8)) readResp;
   method Vector#(rows, Int#(8)) peek(UInt#(TLog#(depth)) addr);
endinterface

module mkActivationSRAM(ActivationSRAM_IFC#(depth, rows))
   provisos(Add#(1, d_, depth),
            Add#(1, r_, rows),
            Log#(depth, logd),
            Bits#(Vector#(rows, Int#(8)), sz));

   RegFile#(UInt#(logd), Vector#(rows, Int#(8))) mem
      <- mkRegFileFull;

   Reg#(Vector#(rows, Int#(8))) resp <- mkRegU;

   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
      mem.upd(addr, data);
   endmethod

   method Action writeBack(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
      mem.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      resp <= mem.sub(addr);
   endmethod

   method Vector#(rows, Int#(8)) readResp;
      return resp;
   endmethod

   method Vector#(rows, Int#(8)) peek(UInt#(TLog#(depth)) addr);
      return mem.sub(addr);
   endmethod

endmodule

endpackage
