package VRegFile;

import Vector :: *;
import RegFile :: *;

interface VRegFile_IFC#(numeric type numRegs, numeric type sublanes, numeric type lanes);
   method Action write(UInt#(TLog#(numRegs)) idx,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Vector#(sublanes, Vector#(lanes, Int#(32))) read(UInt#(TLog#(numRegs)) idx);
endinterface

module mkVRegFile(VRegFile_IFC#(numRegs, sublanes, lanes))
   provisos(
      Add#(1, r_, numRegs),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   RegFile#(UInt#(TLog#(numRegs)), Vector#(sublanes, Vector#(lanes, Int#(32))))
      rf <- mkRegFileFull;

   method Action write(UInt#(TLog#(numRegs)) idx,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      rf.upd(idx, data);
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) read(UInt#(TLog#(numRegs)) idx);
      return rf.sub(idx);
   endmethod

endmodule

export VRegFile_IFC(..);
export mkVRegFile;

endpackage
