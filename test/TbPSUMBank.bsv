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
   rule timeout (cycle > 50);
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

   // Test 6: writeRow touches only the targeted row.
   // Bucket 3 starts at tile_const(5). writeRow(3, 2, [100,101,102,103])
   // should leave rows 0,1,3 at 5 and row 2 at [100..103].
   rule t6_init (cycle == 14); psum.write(3, tile_const(5)); endrule
   rule t6_row (cycle == 15);
      Vector#(4, Int#(32)) r = newVector;
      for (Integer c = 0; c < 4; c = c + 1) r[c] = fromInteger(100 + c);
      psum.writeRow(3, 2, r);
   endrule
   rule t6_check (cycle == 16);
      let t = psum.peek(3);
      Bool ok = (t[0][0] == 5 && t[1][0] == 5 && t[3][0] == 5 &&
                 t[2][0] == 100 && t[2][1] == 101 && t[2][2] == 102 && t[2][3] == 103);
      if (ok) begin
         $display("PASS test 6 (writeRow isolation)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 6: row0[0]=%0d row2=[%0d,%0d,%0d,%0d]",
            t[0][0], t[2][0], t[2][1], t[2][2], t[2][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 7: accumulateRow adds into the targeted row only.
   // Bucket 3 row 2 is now [100..103]; accumulateRow adds [1,2,3,4] ->
   // [101,103,105,107]. Other rows stay at 5.
   rule t7_acc (cycle == 17);
      Vector#(4, Int#(32)) d = newVector;
      for (Integer c = 0; c < 4; c = c + 1) d[c] = fromInteger(c + 1);
      psum.accumulateRow(3, 2, d);
   endrule
   rule t7_check (cycle == 18);
      let t = psum.peek(3);
      Bool ok = (t[0][0] == 5 && t[1][0] == 5 && t[3][0] == 5 &&
                 t[2][0] == 101 && t[2][1] == 103 && t[2][2] == 105 && t[2][3] == 107);
      if (ok) begin
         $display("PASS test 7 (accumulateRow isolation)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 7: row2=[%0d,%0d,%0d,%0d]",
            t[2][0], t[2][1], t[2][2], t[2][3]);
         failed <= failed + 1;
      end
   endrule

   // Test 8: peekRow returns only the requested row.
   rule t8_check (cycle == 19);
      let r = psum.peekRow(3, 0);
      Bool ok = (r[0] == 5 && r[1] == 5 && r[2] == 5 && r[3] == 5);
      if (ok) begin
         $display("PASS test 8 (peekRow)"); passed <= passed + 1;
      end else begin
         $display("FAIL test 8: row0=[%0d,%0d,%0d,%0d]", r[0], r[1], r[2], r[3]);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 25);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule

endpackage
