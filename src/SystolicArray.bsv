package SystolicArray;

import Vector :: *;
import PE :: *;

export SystolicArray_IFC(..);
export mkSystolicArray;

interface SystolicArray_IFC#(numeric type rows, numeric type cols);
   // Weight-stationary path: preload all weights at once, then stream
   // activations left-to-right. getResults returns column sums (the
   // existing WS dataflow used by the Controller today).
   method Action loadWeights(Vector#(rows, Vector#(cols, Int#(8))) w);
   method Action feedActivations(Vector#(rows, Int#(8)) a);
   method Vector#(cols, Int#(32)) getResults;

   // Output-stationary streaming path: each cycle, the top edge of the
   // grid (row 0) receives one weight per column and the left edge
   // (col 0) receives one activation per row. feedPair moves one cycle
   // of the systolic wavefront — weights propagate down, activations
   // propagate right, psums stay in place. Caller is responsible for
   // the staircase skew on the input vectors.
   method Action feedPair(Vector#(cols, Int#(8)) w_top,
                          Vector#(rows, Int#(8)) a_left);

   // Full-tile drain: returns the per-PE psum as a (rows x cols) matrix.
   // Only meaningful after an OS dispatch — WS dispatches leave col-sums
   // stranded in the same registers, which is what getResults collapses.
   method Vector#(rows, Vector#(cols, Int#(32))) getMatrix;

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

   // OS streaming: row 0 receives fresh weights from the top edge, all
   // lower rows take their weight from the PE directly above (via
   // passWeight). Similarly col 0 receives fresh activations from the
   // left edge; inner columns read the PE to their left.
   method Action feedPair(Vector#(cols, Int#(8)) w_top,
                          Vector#(rows, Int#(8)) a_left);
      for (Integer r = 0; r < valueOf(rows); r = r + 1) begin
         for (Integer c = 0; c < valueOf(cols); c = c + 1) begin
            Int#(8) w_in = (r == 0) ? w_top[c] : pes[r-1][c].passWeight;
            Int#(8) a_in = (c == 0) ? a_left[r] : pes[r][c-1].passActivation;
            pes[r][c].feedPair(w_in, a_in);
         end
      end
   endmethod

   method Vector#(rows, Vector#(cols, Int#(32))) getMatrix;
      Vector#(rows, Vector#(cols, Int#(32))) out = replicate(replicate(0));
      for (Integer r = 0; r < valueOf(rows); r = r + 1)
         for (Integer c = 0; c < valueOf(cols); c = c + 1)
            out[r][c] = pes[r][c].getAccum;
      return out;
   endmethod

   method Action clearAll;
      for (Integer r = 0; r < valueOf(rows); r = r + 1)
         for (Integer c = 0; c < valueOf(cols); c = c + 1)
            pes[r][c].clearAccum;
   endmethod

endmodule

endpackage
