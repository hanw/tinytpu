package ActivationDMA;

// ActivationDMA — minimal DMA stub for the ActivationSRAMDB inactive
// bank. Sibling of WeightDMA for the activation operand. Kick starts
// streaming N synthetic activation vectors into the inactive bank of
// the DB SRAM; the testbench then swaps banks and verifies the
// content. Replacing the synthetic pattern with an HBM readReq/resp
// source is a follow-up iter.
//
// Pattern for vector idx i at lane r: i*rows + r, truncated to Int#(8).

import Vector             :: *;
import ActivationSRAM     :: *;
import ActivationSRAMDB   :: *;

export DMAActivState(..);
export ActivationDMA_IFC(..);
export mkActivationDMA;

typedef enum {
   ADMA_IDLE,
   ADMA_STREAM,
   ADMA_DONE
} DMAActivState deriving (Bits, Eq, FShow);

interface ActivationDMA_IFC#(numeric type depth, numeric type rows);
   method Action kick(UInt#(TLog#(depth)) count);
   method Bool   isBusy;
   method Bool   isDone;
endinterface

function Vector#(rows, Int#(8)) synth_act(UInt#(32) idx)
   provisos(Add#(1, r_, rows));
   Vector#(rows, Int#(8)) v = newVector;
   Integer rN = valueOf(rows);
   UInt#(32) stride = fromInteger(rN);
   for (Integer r = 0; r < rN; r = r + 1) begin
      UInt#(32) val = idx * stride + fromInteger(r);
      v[r] = unpack(truncate(pack(val)));
   end
   return v;
endfunction

module mkActivationDMA#(
      ActivationSRAMDB_IFC#(depth, rows) dbsram
   )(ActivationDMA_IFC#(depth, rows))
   provisos(Add#(1, d_, depth),
            Add#(1, r_, rows));

   Reg#(DMAActivState)       state  <- mkReg(ADMA_IDLE);
   Reg#(UInt#(TLog#(depth))) addr   <- mkReg(0);
   Reg#(UInt#(TLog#(depth))) count  <- mkReg(0);
   Reg#(UInt#(32))           idxCt  <- mkReg(0);

   rule do_stream (state == ADMA_STREAM);
      let v = synth_act(idxCt);
      dbsram.plain.write(addr, v);
      if (addr + 1 == count) begin
         state <= ADMA_DONE;
      end else begin
         addr  <= addr + 1;
         idxCt <= idxCt + 1;
      end
   endrule

   method Action kick(UInt#(TLog#(depth)) n) if (state == ADMA_IDLE || state == ADMA_DONE);
      addr  <= 0;
      idxCt <= 0;
      count <= n;
      state <= n == 0 ? ADMA_DONE : ADMA_STREAM;
   endmethod

   method Bool isBusy = state == ADMA_STREAM;
   method Bool isDone = state == ADMA_DONE;

endmodule

endpackage
