package VMEM;

import Vector :: *;
import RegFile :: *;

interface VMEM_IFC#(numeric type depth, numeric type sublanes, numeric type lanes);
   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
   method Vector#(sublanes, Vector#(lanes, Int#(32))) peek(UInt#(TLog#(depth)) addr);
endinterface

module mkVMEM(VMEM_IFC#(depth, sublanes, lanes))
   provisos(
      Add#(1, d_, depth),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   RegFile#(UInt#(TLog#(depth)), Vector#(sublanes, Vector#(lanes, Int#(32))))
      mem <- mkRegFileFull;

   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) resp <- mkRegU;

   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      mem.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      resp <= mem.sub(addr);
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
      return resp;
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) peek(UInt#(TLog#(depth)) addr);
      return mem.sub(addr);
   endmethod

endmodule

export VMEM_IFC(..);
export mkVMEM;

endpackage
