package ActivationSRAMDB;

// Double-buffered ActivationSRAM. Parallel of WeightSRAMDB — same
// ping-pong semantics (writes to inactive, reads from active, swap
// flips the pointer) but for the row-oriented activation operand.

import Vector           :: *;
import RegFile          :: *;
import ActivationSRAM   :: *;

export ActivationSRAMDB_IFC(..);
export mkActivationSRAMDB;

// Embed ActivationSRAM_IFC as a sub-interface so the Controller can
// take `asramDB.plain` and the outer caller drives `swap` directly.
interface ActivationSRAMDB_IFC#(numeric type depth, numeric type rows);
   interface ActivationSRAM_IFC#(depth, rows) plain;
   method Action swap;
   method Bit#(1) activeBank;
endinterface

module mkActivationSRAMDB(ActivationSRAMDB_IFC#(depth, rows))
   provisos(Add#(1, d_, depth),
            Add#(1, r_, rows),
            Log#(depth, logd),
            Bits#(Vector#(rows, Int#(8)), sz));

   RegFile#(UInt#(logd), Vector#(rows, Int#(8))) mem_a <- mkRegFileFull;
   RegFile#(UInt#(logd), Vector#(rows, Int#(8))) mem_b <- mkRegFileFull;

   Reg#(Bit#(1)) active <- mkReg(0);
   Reg#(Vector#(rows, Int#(8))) resp <- mkRegU;

   interface ActivationSRAM_IFC plain;
      method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Int#(8)) data);
         if (active == 0) mem_b.upd(addr, data);
         else             mem_a.upd(addr, data);
      endmethod

      method Action readReq(UInt#(TLog#(depth)) addr);
         if (active == 0) resp <= mem_a.sub(addr);
         else             resp <= mem_b.sub(addr);
      endmethod

      method Vector#(rows, Int#(8)) readResp;
         return resp;
      endmethod
   endinterface

   method Action swap;
      active <= ~active;
   endmethod

   method Bit#(1) activeBank = active;

endmodule

endpackage
