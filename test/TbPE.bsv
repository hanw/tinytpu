package TbPE;

import PE :: *;

(* synthesize *)
module mkTbPE();

   PE_IFC pe <- mkPE;

   Reg#(UInt#(8)) cycle <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 30) begin
         $display("FAIL: test timed out");
         $finish(1);
      end
   endrule

   // Cycle 0: load weight = 3
   rule load_weight (cycle == 0);
      pe.loadWeight(3);
      $display("Cycle %0d: loadWeight(3)", cycle);
   endrule

   // Cycle 1: feed activation = 5, expect accum starts at 0
   rule feed1 (cycle == 1);
      pe.feedActivation(5);
      $display("Cycle %0d: feedActivation(5), accum before = %0d", cycle, pe.getAccum);
   endrule

   // Cycle 2: feed activation = -2, check accum = 3*5 = 15
   rule feed2 (cycle == 2);
      pe.feedActivation(-2);
      $display("Cycle %0d: feedActivation(-2), accum = %0d (expect 15)", cycle, pe.getAccum);
      $display("Cycle %0d: passActivation = %0d (expect 5)", cycle, pe.passActivation);
   endrule

   // Cycle 3: check accum = 15 + 3*(-2) = 9
   rule check3 (cycle == 3);
      $display("Cycle %0d: accum = %0d (expect 9)", cycle, pe.getAccum);
      $display("Cycle %0d: passActivation = %0d (expect -2)", cycle, pe.passActivation);
   endrule

   // Cycle 4: clear and verify
   rule clear (cycle == 4);
      pe.clearAccum;
      $display("Cycle %0d: clearAccum", cycle);
   endrule

   rule check_clear (cycle == 5);
      $display("Cycle %0d: accum after clear = %0d (expect 0)", cycle, pe.getAccum);
   endrule

   // Cycle 6: test with negative weight
   rule load_neg_weight (cycle == 6);
      pe.loadWeight(-4);
      $display("Cycle %0d: loadWeight(-4)", cycle);
   endrule

   rule feed_neg (cycle == 7);
      pe.feedActivation(10);
      $display("Cycle %0d: feedActivation(10)", cycle);
   endrule

   rule check_neg (cycle == 8);
      $display("Cycle %0d: accum = %0d (expect -40)", cycle, pe.getAccum);
   endrule

   // OS dataflow: feedPair streams both operands each cycle.
   rule os_clear (cycle == 9);
      pe.clearAccum;
   endrule

   rule os_feed1 (cycle == 10);
      pe.feedPair(6, 7);
      $display("Cycle %0d: feedPair(6,7), accum before = %0d", cycle, pe.getAccum);
   endrule

   rule os_feed2 (cycle == 11);
      pe.feedPair(-3, 4);
      $display("Cycle %0d: feedPair(-3,4), accum = %0d (expect 42)", cycle, pe.getAccum);
      $display("Cycle %0d: passWeight = %0d (expect 6), passActivation = %0d (expect 7)",
               cycle, pe.passWeight, pe.passActivation);
   endrule

   rule os_check (cycle == 12);
      // accum = 42 + (-3)*4 = 30
      $display("Cycle %0d: accum = %0d (expect 30)", cycle, pe.getAccum);
      $display("Cycle %0d: passWeight = %0d (expect -3), passActivation = %0d (expect 4)",
               cycle, pe.passWeight, pe.passActivation);
      $display("PASS");
      $finish(0);
   endrule

endmodule

endpackage
