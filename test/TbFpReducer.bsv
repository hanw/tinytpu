package TbFpReducer;

import Vector :: *;
import FpReducer :: *;

// Sequences a handful of (op, inputs, expected) tests through the shared
// FP reducer. Each test launches when the previous is reported done.

(* synthesize *)
module mkTbFpReducer();

   // Size 16 covers the flattened 4x4 tile; smaller row/col cases pack
   // into the same buffer with identity pads supplied by the launcher.
   FpReducer_IFC#(16) fpr <- mkFpReducer;

   Reg#(UInt#(16)) cycle   <- mkReg(0);
   Reg#(UInt#(8))  passed  <- mkReg(0);
   Reg#(UInt#(8))  failed  <- mkReg(0);
   Reg#(UInt#(8))  test_id <- mkReg(0);
   Reg#(Bool)      armed   <- mkReg(False);
   Reg#(Bit#(32))  expected <- mkReg(0);

   Int#(32) neg_inf = unpack(32'hFF800000);
   Int#(32) pos_inf = unpack(32'h7F800000);

   // Helper: build a size-16 vector from a leading 4-element list,
   // padded with `pad` afterwards.
   function Vector#(16, Int#(32))
      v4_pad(Int#(32) a, Int#(32) b, Int#(32) c, Int#(32) d, Int#(32) pad);
      Vector#(16, Int#(32)) v = replicate(pad);
      v[0] = a; v[1] = b; v[2] = c; v[3] = d;
      return v;
   endfunction

   // Helper: values 1.0..16.0 packed into a 16-element float vector.
   function Vector#(16, Int#(32)) v_1_to_16();
      Vector#(16, Int#(32)) v = newVector;
      // 1.0, 2.0, ..., 16.0 bit patterns.
      v[0]  = unpack(32'h3F800000); v[1]  = unpack(32'h40000000);
      v[2]  = unpack(32'h40400000); v[3]  = unpack(32'h40800000);
      v[4]  = unpack(32'h40A00000); v[5]  = unpack(32'h40C00000);
      v[6]  = unpack(32'h40E00000); v[7]  = unpack(32'h41000000);
      v[8]  = unpack(32'h41100000); v[9]  = unpack(32'h41200000);
      v[10] = unpack(32'h41300000); v[11] = unpack(32'h41400000);
      v[12] = unpack(32'h41500000); v[13] = unpack(32'h41600000);
      v[14] = unpack(32'h41700000); v[15] = unpack(32'h41800000);
      return v;
   endfunction

   rule timeout (cycle > 400);
      $display("FAIL: timeout at cycle %0d", cycle);
      $finish(1);
   endrule

   rule count; cycle <= cycle + 1; endrule

   // Each completed reduction reports and advances test_id.
   rule next (armed && fpr.isDone);
      Bit#(32) got = pack(fpr.getResult);
      if (got == expected) begin
         $display("PASS test %0d (got 0x%08x)", test_id, got);
         passed <= passed + 1;
      end else begin
         $display("FAIL test %0d: got 0x%08x want 0x%08x", test_id, got, expected);
         failed <= failed + 1;
      end
      test_id <= test_id + 1;
      armed <= False;
   endrule

   // Test 0: SUM of 1..16 -> 136.0
   rule launch_0 (!armed && test_id == 0 && cycle > 0);
      fpr.start(FPR_SUM, v_1_to_16);
      expected <= 32'h43080000;
      armed <= True;
   endrule

   // Test 1: SUM of [1,2,3,4] + 12 zero pads -> 10.0
   rule launch_1 (!armed && test_id == 1);
      let a = unpack(32'h3F800000); let b = unpack(32'h40000000);
      let c = unpack(32'h40400000); let d = unpack(32'h40800000);
      fpr.start(FPR_SUM, v4_pad(a, b, c, d, unpack(32'h00000000)));
      expected <= 32'h41200000;  // 10.0
      armed <= True;
   endrule

   // Test 2: MAX of [3,1,4,2] + 12 -inf pads -> 4.0
   rule launch_2 (!armed && test_id == 2);
      let a = unpack(32'h40400000); // 3.0
      let b = unpack(32'h3F800000); // 1.0
      let c = unpack(32'h40800000); // 4.0
      let d = unpack(32'h40000000); // 2.0
      fpr.start(FPR_MAX, v4_pad(a, b, c, d, neg_inf));
      expected <= 32'h40800000;  // 4.0
      armed <= True;
   endrule

   // Test 3: MIN of [3,1,4,2] + 12 +inf pads -> 1.0
   rule launch_3 (!armed && test_id == 3);
      let a = unpack(32'h40400000);
      let b = unpack(32'h3F800000);
      let c = unpack(32'h40800000);
      let d = unpack(32'h40000000);
      fpr.start(FPR_MIN, v4_pad(a, b, c, d, pos_inf));
      expected <= 32'h3F800000;  // 1.0
      armed <= True;
   endrule

   // Test 4: MAX full tile 1..16 -> 16.0
   rule launch_4 (!armed && test_id == 4);
      fpr.start(FPR_MAX, v_1_to_16);
      expected <= 32'h41800000;
      armed <= True;
   endrule

   // Test 5: MIN full tile 1..16 -> 1.0
   rule launch_5 (!armed && test_id == 5);
      fpr.start(FPR_MIN, v_1_to_16);
      expected <= 32'h3F800000;
      armed <= True;
   endrule

   rule finish_ok (!armed && test_id == 6);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule

endpackage
