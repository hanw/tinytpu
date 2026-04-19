package TbPSUMBank;

import Vector   :: *;
import PSUMBank :: *;

// Tiny sequential driver for the PSUMBank. Covers:
//   - write then peek   (basic storage)
//   - accumulate then peek (add-into semantics)
//   - readReq / readResp (1-cycle latency, matches VMEM pattern)
//   - clear
// 8 buckets × 4×4 Int32 matches the shape we'd use for K-tile GEMM.

(* synthesize *)
module mkTbPSUMBank();

   PSUMBank_IFC#(8, 4, 4) psum <- mkPSUMBank;

   Reg#(UInt#(16)) cycle  <- mkReg(0);
   Reg#(UInt#(8))  passed <- mkReg(0);
   Reg#(UInt#(8))  failed <- mkReg(0);

   rule count; cycle <= cycle + 1; endrule
   rule timeout (cycle > 300);
      $display("FAIL: timeout"); $finish(1);
   endrule

   // Helpers: build small tiles.
   function Vector#(4, Vector#(4, Int#(32))) tile_const(Int#(32) v);
      return replicate(replicate(v));
   endfunction

   function Vector#(4, Vector#(4, Int#(32))) tile_seq();
      Vector#(4, Vector#(4, Int#(32))) t = newVector;
      for (Integer r = 0; r < 4; r = r + 1) begin
         Vector#(4, Int#(32)) row = newVector;
         for (Integer c = 0; c < 4; c = c + 1)
            row[c] = fromInteger(r * 4 + c + 1);
         t[r] = row;
      end
      return t;
   endfunction

   function Bool tile_eq(Vector#(4, Vector#(4, Int#(32))) a,
                         Vector#(4, Vector#(4, Int#(32))) b);
      Bool ok = True;
      for (Integer r = 0; r < 4; r = r + 1)
         for (Integer c = 0; c < 4; c = c + 1)
            if (a[r][c] != b[r][c]) ok = False;
      return ok;
   endfunction

   // Test 0: write bucket 1 with tile_seq (1..16), peek -> same tile.
   rule t0_write (cycle == 1);
      psum.write(1, tile_seq);
      $display("Cycle %0d: write bucket 1 with 1..16", cycle);
   endrule
   rule t0_check (cycle == 2);
      if (tile_eq(psum.peek(1), tile_seq)) begin
         $display("PASS test 0 (write/peek)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 0"); failed <= failed + 1;
      end
   endrule

   // Test 1: accumulate bucket 1 with tile_const(10) -> 11..26.
   rule t1_acc (cycle == 3); psum.accumulate(1, tile_const(10)); endrule
   rule t1_check (cycle == 4);
      Vector#(4, Vector#(4, Int#(32))) expected = newVector;
      for (Integer r = 0; r < 4; r = r + 1) begin
         Vector#(4, Int#(32)) row = newVector;
         for (Integer c = 0; c < 4; c = c + 1) row[c] = fromInteger(r*4 + c + 1 + 10);
         expected[r] = row;
      end
      if (tile_eq(psum.peek(1), expected)) begin
         $display("PASS test 1 (accumulate)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 1 got [0][0]=%0d want 11", psum.peek(1)[0][0]);
         failed <= failed + 1;
      end
   endrule

   // Test 2: readReq bucket 1 → readResp on next cycle matches peek.
   rule t2_req (cycle == 5); psum.readReq(1); endrule
   rule t2_check (cycle == 6);
      Vector#(4, Vector#(4, Int#(32))) expected = newVector;
      for (Integer r = 0; r < 4; r = r + 1) begin
         Vector#(4, Int#(32)) row = newVector;
         for (Integer c = 0; c < 4; c = c + 1) row[c] = fromInteger(r*4 + c + 1 + 10);
         expected[r] = row;
      end
      if (tile_eq(psum.readResp, expected)) begin
         $display("PASS test 2 (readReq/readResp)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 2"); failed <= failed + 1;
      end
   endrule

   // Test 3: clear bucket 1 → peek all zeros.
   rule t3_clear (cycle == 7); psum.clear(1); endrule
   rule t3_check (cycle == 8);
      if (tile_eq(psum.peek(1), tile_const(0))) begin
         $display("PASS test 3 (clear)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 3"); failed <= failed + 1;
      end
   endrule

   // Test 4: two buckets independent.
   rule t4_w0 (cycle == 9);  psum.write(0, tile_const(7)); endrule
   rule t4_w2 (cycle == 10); psum.write(2, tile_const(-3)); endrule
   rule t4_check (cycle == 11);
      if (tile_eq(psum.peek(0), tile_const(7)) &&
          tile_eq(psum.peek(2), tile_const(-3))) begin
         $display("PASS test 4 (bucket isolation)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 4"); failed <= failed + 1;
      end
   endrule

   // Test 5: accumulate into a freshly-cleared bucket.
   rule t5_acc (cycle == 12); psum.accumulate(1, tile_seq); endrule
   rule t5_check (cycle == 13);
      if (tile_eq(psum.peek(1), tile_seq)) begin
         $display("PASS test 5 (accumulate-into-zero)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 5"); failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 20);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule

endpackage
