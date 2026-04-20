package WeightSRAMDB;

// Double-buffered WeightSRAM (Architectural Refactor Item #5). Holds
// two independent banks (A / B) and a current-read-pointer. Writes go
// to the non-active bank (background preload); reads serve from the
// active bank. `swap` flips which bank is active — this is the ping-
// pong primitive that lets the next MXU tile preload from HBM in
// parallel with the current MXU dispatch draining.
//
// Interface extends the plain WeightSRAM_IFC with `swap`. Today the
// Controller still uses the plain SRAM (no swap calls); this module
// is scaffolding so a Controller revision can swap banks without
// touching the rest of the data path. The API is intentionally a
// superset of WeightSRAM_IFC so callers can switch with a single
// `swap` invocation when needed.
//
// Writes:  always hit the non-active bank. Use this for HBM preload
//          while the array still reads the previous tile.
// Reads:   always read from the active bank.
// swap:    toggle active-bank pointer.

import Vector   :: *;
import RegFile  :: *;

export WeightSRAMDB_IFC(..);
export mkWeightSRAMDB;

interface WeightSRAMDB_IFC#(numeric type depth, numeric type rows, numeric type cols);
   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Vector#(cols, Int#(8))) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(rows, Vector#(cols, Int#(8))) readResp;
   method Action swap;
   method Bit#(1) activeBank;
endinterface

module mkWeightSRAMDB(WeightSRAMDB_IFC#(depth, rows, cols))
   provisos(Add#(1, d_, depth),
            Add#(1, r_, rows),
            Add#(1, c_, cols),
            Log#(depth, logd),
            Bits#(Vector#(rows, Vector#(cols, Int#(8))), sz));

   RegFile#(UInt#(logd), Vector#(rows, Vector#(cols, Int#(8)))) mem_a <- mkRegFileFull;
   RegFile#(UInt#(logd), Vector#(rows, Vector#(cols, Int#(8)))) mem_b <- mkRegFileFull;

   Reg#(Bit#(1)) active <- mkReg(0);   // 0 = mem_a is active for reads
   Reg#(Vector#(rows, Vector#(cols, Int#(8)))) resp <- mkRegU;

   method Action write(UInt#(TLog#(depth)) addr, Vector#(rows, Vector#(cols, Int#(8))) data);
      // Write to the INACTIVE bank so reads continue serving the current
      // tile uninterrupted.
      if (active == 0) mem_b.upd(addr, data);
      else             mem_a.upd(addr, data);
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      // Read from the active bank.
      if (active == 0) resp <= mem_a.sub(addr);
      else             resp <= mem_b.sub(addr);
   endmethod

   method Vector#(rows, Vector#(cols, Int#(8))) readResp;
      return resp;
   endmethod

   method Action swap;
      active <= ~active;
   endmethod

   method Bit#(1) activeBank = active;

endmodule

endpackage
