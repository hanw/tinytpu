package WeightDMA;

// WeightDMA (Architectural Refactor Item #5). A minimal DMA engine
// stub that issues background writes to a double-buffered weight SRAM.
// The real target for this block is HBM -> WeightSRAMDB preload in
// parallel with an MXU dispatch draining the active bank.
//
// Today the engine has no HBM source attached — on `kick(count)` it
// synthesizes deterministic tile patterns (lane (r,c) at tile index i
// carries value i*16 + r*4 + c, truncated to Int#(8)) and writes them
// to addresses 0..count-1 of the inactive bank of the supplied
// WeightSRAMDB. The testbench then `swap`s banks and verifies the
// data is readable on the active side. Replacing the synthetic pattern
// with an HBM readReq/readResp pipeline is a follow-up iter.

import Vector       :: *;
import WeightSRAM   :: *;
import WeightSRAMDB :: *;

export DMAEngineState(..);
export WeightDMA_IFC(..);
export mkWeightDMA;

typedef enum {
   DMA_IDLE,
   DMA_STREAM,
   DMA_DONE
} DMAEngineState deriving (Bits, Eq, FShow);

interface WeightDMA_IFC#(numeric type depth, numeric type rows, numeric type cols);
   method Action kick(UInt#(TLog#(depth)) count);
   method Bool   isBusy;
   method Bool   isDone;
endinterface

// Synthetic tile generator: value[r][c] = tileIdx*rows*cols + r*cols + c.
function Vector#(rows, Vector#(cols, Int#(8))) synth_tile(UInt#(32) idx)
   provisos(Add#(1, r_, rows), Add#(1, c_, cols));
   Vector#(rows, Vector#(cols, Int#(8))) tile = newVector;
   Integer rN = valueOf(rows);
   Integer cN = valueOf(cols);
   UInt#(32) stride = fromInteger(rN * cN);
   for (Integer r = 0; r < rN; r = r + 1) begin
      tile[r] = newVector;
      for (Integer c = 0; c < cN; c = c + 1) begin
         UInt#(32) v = idx * stride
                       + fromInteger(r) * fromInteger(cN)
                       + fromInteger(c);
         tile[r][c] = unpack(truncate(pack(v)));
      end
   end
   return tile;
endfunction

module mkWeightDMA#(
      WeightSRAMDB_IFC#(depth, rows, cols) dbsram
   )(WeightDMA_IFC#(depth, rows, cols))
   provisos(Add#(1, d_, depth),
            Add#(1, r_, rows),
            Add#(1, c_, cols));

   Reg#(DMAEngineState)      state  <- mkReg(DMA_IDLE);
   Reg#(UInt#(TLog#(depth))) addr   <- mkReg(0);
   Reg#(UInt#(TLog#(depth))) count  <- mkReg(0);
   Reg#(UInt#(32))           tileCt <- mkReg(0);

   rule do_stream (state == DMA_STREAM);
      let tile = synth_tile(tileCt);
      dbsram.plain.write(addr, tile);
      if (addr + 1 == count) begin
         state <= DMA_DONE;
      end else begin
         addr   <= addr + 1;
         tileCt <= tileCt + 1;
      end
   endrule

   method Action kick(UInt#(TLog#(depth)) n) if (state == DMA_IDLE || state == DMA_DONE);
      addr   <= 0;
      tileCt <= 0;
      count  <= n;
      state  <= n == 0 ? DMA_DONE : DMA_STREAM;
   endmethod

   method Bool isBusy = state == DMA_STREAM;
   method Bool isDone = state == DMA_DONE;

endmodule

endpackage
