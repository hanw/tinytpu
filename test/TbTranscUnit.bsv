package TbTranscUnit;

// Standalone TranscUnit harness. Drives the multi-cycle walker directly
// with 4 float inputs and asserts each output is in_band a tight band of
// the true transcendental value. The VPU-level TbVPU test still covers
// the dispatch glue; this TB locks down the polynomial coefficients so
// future Remez tuning regressions fail at the TranscUnit layer rather
// than downstream.

import Vector :: *;
import FloatingPoint :: *;
import TranscUnit :: *;

(* synthesize *)
module mkTbTranscUnit();

   TranscUnit_IFC#(4) tru <- mkTranscUnit;

   Reg#(UInt#(16)) cycle <- mkReg(0);
   Reg#(UInt#(4))  phase <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   function Bool in_band(Float got, Bit#(32) lo, Bit#(32) hi);
      Float flo = unpack(lo);
      Float fhi = unpack(hi);
      return compareFP(got, flo) != LT && compareFP(got, fhi) != GT;
   endfunction

   rule tick;
      cycle <= cycle + 1;
      if (cycle > 500) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // -- EXP2 band check -------------------------------------------------
   // Inputs: [0.0, 1.0, 2.0, -1.0] → Remez approx:
   //   x= 0: 1.0       (exact)
   //   x= 1: 1.9844    (0.78% low vs 2.0)
   //   x= 2: 3.4688    (13.3% low vs 4.0)
   //   x=-1: 0.5155    (3.1% high vs 0.5)
   rule dispatch_exp2 (cycle == 2 && phase == 0);
      Vector#(4, Int#(32)) s = newVector;
      s[0] = unpack(32'h00000000);   //  0.0
      s[1] = unpack(32'h3F800000);   //  1.0
      s[2] = unpack(32'h40000000);   //  2.0
      s[3] = unpack(32'hBF800000);   // -1.0
      tru.start(TR_EXP2, s);
      phase <= 1;
      $display("Cycle %0d: start EXP2", cycle);
   endrule

   rule collect_exp2 (phase == 1 && tru.isDone);
      let r = tru.getResult;
      Float g0 = unpack(pack(r[0]));
      Float g1 = unpack(pack(r[1]));
      Float g2 = unpack(pack(r[2]));
      Float g3 = unpack(pack(r[3]));
      Bool ok = True;
      if (!in_band(g0, 32'h3F7EB852, 32'h3F81EB85)) ok = False; // [0.995, 1.010]
      if (!in_band(g1, 32'h3FFAE148, 32'h4001EB85)) ok = False; // [1.96, 2.03]
      if (!in_band(g2, 32'h405CCCCD, 32'h4062E148)) ok = False; // [3.45, 3.545]
      if (!in_band(g3, 32'h3F028F5C, 32'h3F0CCCCD)) ok = False; // [0.510, 0.550]
      if (ok) begin
         $display("Cycle %0d: PASS TR_EXP2", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL TR_EXP2 got [0x%08x,0x%08x,0x%08x,0x%08x]",
                  cycle, pack(r[0]), pack(r[1]), pack(r[2]), pack(r[3]));
         failed <= failed + 1;
      end
      phase <= 2;
   endrule

   // -- LOG2 band check -------------------------------------------------
   // Inputs: [1.0, 2.0, 4.0, 0.5] → true [0, 1, 2, -1].
   // LOG2 range-reduces to m in [1, 2); at exact powers of 2 the
   // fractional part is zero so the polynomial adds nothing. Range-
   // reduction is coefficient-free so powers-of-two stay exact.
   rule dispatch_log2 (phase == 2);
      Vector#(4, Int#(32)) s = newVector;
      s[0] = unpack(32'h3F800000);   // 1.0
      s[1] = unpack(32'h40000000);   // 2.0
      s[2] = unpack(32'h40800000);   // 4.0
      s[3] = unpack(32'h3F000000);   // 0.5
      tru.start(TR_LOG2, s);
      phase <= 3;
      $display("Cycle %0d: start LOG2", cycle);
   endrule

   rule collect_log2 (phase == 3 && tru.isDone);
      let r = tru.getResult;
      Float g0 = unpack(pack(r[0]));
      Float g1 = unpack(pack(r[1]));
      Float g2 = unpack(pack(r[2]));
      Float g3 = unpack(pack(r[3]));
      Bool ok = True;
      // log2(1) = 0, log2(2) = 1, log2(4) = 2, log2(0.5) = -1.
      // All exact at powers-of-two.
      if (!in_band(g0, 32'hBCA3D70A, 32'h3CA3D70A)) ok = False;  // [-0.02, 0.02]
      if (!in_band(g1, 32'h3F7AE148, 32'h3F851EB8)) ok = False;  // [0.98, 1.04]
      if (!in_band(g2, 32'h3FFAE148, 32'h4028F5C3)) ok = False;  // [1.96, 2.64]
      if (!in_band(g3, 32'hBF851EB8, 32'hBF7AE148)) ok = False;  // [-1.04, -0.98]
      if (ok) begin
         $display("Cycle %0d: PASS TR_LOG2", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL TR_LOG2 got [0x%08x,0x%08x,0x%08x,0x%08x]",
                  cycle, pack(r[0]), pack(r[1]), pack(r[2]), pack(r[3]));
         failed <= failed + 1;
      end
      phase <= 4;
   endrule

   // -- SIN band check --------------------------------------------------
   // Inputs: [0.0, π/6, π/4, π/2] → true [0, 0.5, 0.7071, 1.0].
   rule dispatch_sin (phase == 4);
      Vector#(4, Int#(32)) s = newVector;
      s[0] = unpack(32'h00000000);   // 0.0
      s[1] = unpack(32'h3F060A92);   // π/6 ≈ 0.5236
      s[2] = unpack(32'h3F490FDB);   // π/4 ≈ 0.7854
      s[3] = unpack(32'h3FC90FDB);   // π/2 ≈ 1.5708
      tru.start(TR_SIN, s);
      phase <= 5;
      $display("Cycle %0d: start SIN", cycle);
   endrule

   rule collect_sin (phase == 5 && tru.isDone);
      let r = tru.getResult;
      Float g0 = unpack(pack(r[0]));
      Float g1 = unpack(pack(r[1]));
      Float g2 = unpack(pack(r[2]));
      Float g3 = unpack(pack(r[3]));
      Bool ok = True;
      // sin(0)=0, sin(π/6)=0.5, sin(π/4)=0.7071, sin(π/2)=1.0
      if (!in_band(g0, 32'hBCA3D70A, 32'h3CA3D70A)) ok = False;  // [-0.02, 0.02]
      if (!in_band(g1, 32'h3EF5C28F, 32'h3F028F5C)) ok = False;  // [0.48, 0.51]
      if (!in_band(g2, 32'h3F308B44, 32'h3F408B44)) ok = False;  // [0.6896, 0.7521] tightened
      if (!in_band(g3, 32'h3F7EB852, 32'h3F81EB85)) ok = False;  // [0.995, 1.010]
      if (ok) begin
         $display("Cycle %0d: PASS TR_SIN", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL TR_SIN got [0x%08x,0x%08x,0x%08x,0x%08x]",
                  cycle, pack(r[0]), pack(r[1]), pack(r[2]), pack(r[3]));
         failed <= failed + 1;
      end
      phase <= 6;
   endrule

   // -- COS band check --------------------------------------------------
   // Inputs: [0.0, π/6, π/3, π/2] → true [1.0, 0.8660, 0.5, 0.0].
   rule dispatch_cos (phase == 6);
      Vector#(4, Int#(32)) s = newVector;
      s[0] = unpack(32'h00000000);   // 0.0
      s[1] = unpack(32'h3F060A92);   // π/6
      s[2] = unpack(32'h3F860A92);   // π/3
      s[3] = unpack(32'h3FC90FDB);   // π/2
      tru.start(TR_COS, s);
      phase <= 7;
      $display("Cycle %0d: start COS", cycle);
   endrule

   rule collect_cos (phase == 7 && tru.isDone);
      let r = tru.getResult;
      Float g0 = unpack(pack(r[0]));
      Float g1 = unpack(pack(r[1]));
      Float g2 = unpack(pack(r[2]));
      Float g3 = unpack(pack(r[3]));
      Bool ok = True;
      if (!in_band(g0, 32'h3F7EB852, 32'h3F81EB85)) ok = False;  // [0.995, 1.010]
      if (!in_band(g1, 32'h3F570A3D, 32'h3F5EB852)) ok = False;  // [0.84, 0.870]
      if (!in_band(g2, 32'h3EF5C28F, 32'h3F051EB8)) ok = False;  // [0.48, 0.52]
      if (!in_band(g3, 32'hBCA3D70A, 32'h3CA3D70A)) ok = False;  // [-0.02, 0.02]
      if (ok) begin
         $display("Cycle %0d: PASS TR_COS", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL TR_COS got [0x%08x,0x%08x,0x%08x,0x%08x]",
                  cycle, pack(r[0]), pack(r[1]), pack(r[2]), pack(r[3]));
         failed <= failed + 1;
      end
      phase <= 8;
   endrule

   // -- Range-reduction helpers: tr_trunc / tr_fp_to_int / tr_pow2_int --
   // These are pure combinational functions. Check a handful of inputs
   // in one cycle so future integration with EXP2 range reduction has
   // a known-good baseline.
   rule check_helpers (phase == 8);
      Bool ok = True;
      // trunc(3.7) == 3.0 (0x40400000)
      Float t1 = tr_trunc(unpack(32'h406CCCCD));  // 3.7
      if (pack(t1) != 32'h40400000) ok = False;
      // trunc(-2.3) == -2.0 (0xC0000000)
      Float t2 = tr_trunc(unpack(32'hC0133333));  // -2.3
      if (pack(t2) != 32'hC0000000) ok = False;
      // trunc(0.5) == 0.0
      Float t3 = tr_trunc(unpack(32'h3F000000));
      if (pack(t3) != 32'h00000000) ok = False;
      // trunc(-0.9) == 0.0
      Float t4 = tr_trunc(unpack(32'hBF666666));
      if (pack(t4) != 32'h00000000) ok = False;
      // fp_to_int(3.0) == 3
      if (tr_fp_to_int(unpack(32'h40400000)) != 3) ok = False;
      // fp_to_int(-5.0) == -5
      if (tr_fp_to_int(unpack(32'hC0A00000)) != -5) ok = False;
      // pow2_int(0) == 1.0
      if (pack(tr_pow2_int(0)) != 32'h3F800000) ok = False;
      // pow2_int(3) == 8.0
      if (pack(tr_pow2_int(3)) != 32'h41000000) ok = False;
      // pow2_int(-2) == 0.25
      if (pack(tr_pow2_int(-2)) != 32'h3E800000) ok = False;

      if (ok) begin
         $display("Cycle %0d: PASS range-reduction helpers", cycle);
         passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL range-reduction helpers (trunc/fp_to_int/pow2_int)",
                  cycle);
         failed <= failed + 1;
      end
      phase <= 9;
   endrule

   rule done_rule (phase == 9);
      $display("Results: %0d passed, %0d failed", passed, failed);
      $finish(failed == 0 ? 0 : 1);
   endrule

endmodule

endpackage
