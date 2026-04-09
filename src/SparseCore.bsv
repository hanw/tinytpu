package SparseCore;

import Vector :: *;
import RegFile :: *;

interface SparseCore_IFC#(numeric type embTableDepth, numeric type embWidth, numeric type bagSize);
   // Load one embedding vector into the embTable
   method Action loadEmbedding(UInt#(TLog#(embTableDepth)) idx,
                                Vector#(embWidth, Int#(32)) emb);
   // Submit a bag of up to bagSize indices; count = number of valid entries
   method Action submitBag(Vector#(bagSize, UInt#(TLog#(embTableDepth))) indices,
                           UInt#(8) count);
   method Action reset;
   method Vector#(embWidth, Int#(32)) result;
   method Bool isDone;
endinterface

typedef enum { SC_IDLE, SC_LOOKUP, SC_DONE } SCState deriving (Bits, Eq, FShow);

module mkSparseCore(SparseCore_IFC#(embTableDepth, embWidth, bagSize))
   provisos(
      Add#(1, t_, embTableDepth),
      Add#(1, e_, embWidth),
      Add#(1, b_, bagSize),
      Bits#(Vector#(embWidth, Int#(32)), esz)
   );

   RegFile#(UInt#(TLog#(embTableDepth)), Vector#(embWidth, Int#(32)))
      embTable <- mkRegFileFull;

   Reg#(SCState)                                    state    <- mkReg(SC_IDLE);
   Reg#(Vector#(bagSize, UInt#(TLog#(embTableDepth)))) bagReg   <- mkRegU;
   Reg#(UInt#(8))                                   countReg <- mkReg(0);
   Reg#(UInt#(8))                                   cursor   <- mkReg(0);
   Reg#(Vector#(embWidth, Int#(32)))                accum    <- mkReg(replicate(0));

   rule do_lookup (state == SC_LOOKUP);
      if (cursor < countReg) begin
         let idx = bagReg[cursor];
         let emb = embTable.sub(idx);
         Vector#(embWidth, Int#(32)) newAccum = newVector;
         for (Integer i = 0; i < valueOf(embWidth); i = i + 1)
            newAccum[i] = accum[i] + emb[i];
         accum  <= newAccum;
         cursor <= cursor + 1;
      end else begin
         state <= SC_DONE;
      end
   endrule

   method Action loadEmbedding(UInt#(TLog#(embTableDepth)) idx,
                                Vector#(embWidth, Int#(32)) emb);
      embTable.upd(idx, emb);
   endmethod

   method Action submitBag(Vector#(bagSize, UInt#(TLog#(embTableDepth))) indices,
                           UInt#(8) count) if (state == SC_IDLE);
      bagReg   <= indices;
      countReg <= count;
      cursor   <= 0;
      accum    <= replicate(0);
      state    <= SC_LOOKUP;
   endmethod

   method Action reset if (state == SC_DONE);
      state <= SC_IDLE;
   endmethod

   method Vector#(embWidth, Int#(32)) result if (state == SC_DONE);
      return accum;
   endmethod

   method Bool isDone;
      return state == SC_DONE;
   endmethod

endmodule

export SparseCore_IFC(..);
export mkSparseCore;

endpackage
