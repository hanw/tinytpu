package SystolicArray;

import Vector :: *;
import PE :: *;

export SystolicArray_IFC(..);
export mkSystolicArray;

interface SystolicArray_IFC#(numeric type rows, numeric type cols);
   method Action loadWeights(Vector#(rows, Vector#(cols, Int#(8))) w);
   method Action feedActivations(Vector#(rows, Int#(8)) a);
   method Vector#(cols, Int#(32)) getResults;
   method Action clearAll;
endinterface

module mkSystolicArray(SystolicArray_IFC#(rows, cols))
   provisos(Add#(1, r_, rows), Add#(1, c_, cols));

   // 2D grid of PEs
   Vector#(rows, Vector#(cols, PE_IFC)) pes <- replicateM(replicateM(mkPE));

   method Action loadWeights(Vector#(rows, Vector#(cols, Int#(8))) w);
      for (Integer r = 0; r < valueOf(rows); r = r + 1)
         for (Integer c = 0; c < valueOf(cols); c = c + 1)
            pes[r][c].loadWeight(w[r][c]);
   endmethod

   // Feed activations to column 0 of each row.
   // Within a row, activations propagate via passActivation (systolic).
   // The controller handles inter-row skew by inserting zeros.
   method Action feedActivations(Vector#(rows, Int#(8)) a);
      for (Integer r = 0; r < valueOf(rows); r = r + 1) begin
         // Feed activation to first PE in each row
         pes[r][0].feedActivation(a[r]);
         // Propagate through rest of the row using passActivation from prior PE
         for (Integer c = 1; c < valueOf(cols); c = c + 1)
            pes[r][c].feedActivation(pes[r][c-1].passActivation);
      end
   endmethod

   // Sum accumulators down each column for matrix-vector result
   method Vector#(cols, Int#(32)) getResults;
      Vector#(cols, Int#(32)) out = replicate(0);
      for (Integer c = 0; c < valueOf(cols); c = c + 1) begin
         Int#(32) col_sum = 0;
         for (Integer r = 0; r < valueOf(rows); r = r + 1)
            col_sum = col_sum + pes[r][c].getAccum;
         out[c] = col_sum;
      end
      return out;
   endmethod

   method Action clearAll;
      for (Integer r = 0; r < valueOf(rows); r = r + 1)
         for (Integer c = 0; c < valueOf(cols); c = c + 1)
            pes[r][c].clearAccum;
   endmethod

endmodule

endpackage
