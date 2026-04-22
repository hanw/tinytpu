package VRegFile;

// VRegFile — vector register file backed by a Vector of independent
// Regs rather than a single-port RegFile. Each register has its own
// write port, so concurrent writes to different indices do not
// serialize.
//
// Why this matters: the dual-issue SXU has a background XLU collect
// rule that writes one vreg while the main FSM also writes a vreg.
// With the single-port RegFile, those two rules shared one write
// port and bsc forced them mutually exclusive — defeating the
// dual-issue intent. Splitting into N independent Regs lets them
// fire the same cycle (the writes target different indices).
//
// The `write` method is a single action port that picks the target
// Reg; conflicts on the SAME register index still serialize, but
// that's the expected semantics of the architectural register file.

import Vector :: *;

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

   Vector#(numRegs, Reg#(Vector#(sublanes, Vector#(lanes, Int#(32))))) regs
      <- replicateM(mkReg(replicate(replicate(0))));

   method Action write(UInt#(TLog#(numRegs)) idx,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      regs[idx] <= data;
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) read(UInt#(TLog#(numRegs)) idx);
      return regs[idx];
   endmethod

endmodule

export VRegFile_IFC(..);
export mkVRegFile;

endpackage
