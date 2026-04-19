package PSUMBank;

// PSUMBank — small partial-sum SRAM. Addresses hold one 4x4 tile each
// of Int#(32) accumulators. The consumer (usually MXU) writes per
// K-tile with `accumulate()`; other consumers (VPU for the epilogue)
// can `read()` the accumulated value before the final epilogue stage.
//
// Point of the module: give multi-K-tile GEMM a real hardware
// accumulator path instead of round-tripping through VRegFile between
// each K-tile. This is the PSUM-analog that the NeuronCore reference
// has; TinyTPU today approximates it with Controller.results for
// single-K-tile only.
//
// Interface is deliberately small:
//   write(addr, tile)       — zero the bucket, then deposit tile
//   accumulate(addr, tile)  — tile[r][c] += stored[r][c]
//   readReq(addr)           — schedule a read
//   readResp                — the tile returned one cycle later
//   peek(addr)              — combinational peek (testbench convenience)
// Multi-cycle semantics match VMEM so SXU can drive it with the same
// LOAD/STORE FSM shape.

import Vector   :: *;
import RegFile  :: *;

interface PSUMBank_IFC#(numeric type depth,
                        numeric type sublanes,
                        numeric type lanes);
   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Action accumulate(UInt#(TLog#(depth)) addr,
                            Vector#(sublanes, Vector#(lanes, Int#(32))) data);
   method Action readReq(UInt#(TLog#(depth)) addr);
   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
   method Vector#(sublanes, Vector#(lanes, Int#(32))) peek(UInt#(TLog#(depth)) addr);
   method Action clear(UInt#(TLog#(depth)) addr);
   // Row-granular access. One MXU dispatch produces a single lanes-wide
   // row (`Vector#(lanes, Int#(32))`), so writing / accumulating a whole
   // tile in one shot would force Controller to buffer every row first.
   // These let MXU touch one row at a time without disturbing the other
   // rows of the same bucket.
   method Action writeRow(UInt#(TLog#(depth)) addr,
                          UInt#(TLog#(sublanes)) row,
                          Vector#(lanes, Int#(32)) data);
   method Action accumulateRow(UInt#(TLog#(depth)) addr,
                               UInt#(TLog#(sublanes)) row,
                               Vector#(lanes, Int#(32)) data);
   method Vector#(lanes, Int#(32)) peekRow(UInt#(TLog#(depth)) addr,
                                           UInt#(TLog#(sublanes)) row);
endinterface

module mkPSUMBank(PSUMBank_IFC#(depth, sublanes, lanes))
   provisos(
      Add#(1, d_, depth),
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   RegFile#(UInt#(TLog#(depth)),
            Vector#(sublanes, Vector#(lanes, Int#(32)))) mem <- mkRegFileFull;
   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) respReg <- mkRegU;

   function Vector#(sublanes, Vector#(lanes, Int#(32)))
      tile_add(Vector#(sublanes, Vector#(lanes, Int#(32))) a,
               Vector#(sublanes, Vector#(lanes, Int#(32))) b);
      Vector#(sublanes, Vector#(lanes, Int#(32))) r = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1) begin
         Vector#(lanes, Int#(32)) row = newVector;
         for (Integer l = 0; l < valueOf(lanes); l = l + 1)
            row[l] = a[s][l] + b[s][l];
         r[s] = row;
      end
      return r;
   endfunction

   method Action write(UInt#(TLog#(depth)) addr,
                       Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      mem.upd(addr, data);
   endmethod

   method Action accumulate(UInt#(TLog#(depth)) addr,
                            Vector#(sublanes, Vector#(lanes, Int#(32))) data);
      mem.upd(addr, tile_add(mem.sub(addr), data));
   endmethod

   method Action readReq(UInt#(TLog#(depth)) addr);
      respReg <= mem.sub(addr);
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) readResp;
      return respReg;
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) peek(UInt#(TLog#(depth)) addr);
      return mem.sub(addr);
   endmethod

   method Action clear(UInt#(TLog#(depth)) addr);
      Vector#(sublanes, Vector#(lanes, Int#(32))) zeros = replicate(replicate(0));
      mem.upd(addr, zeros);
   endmethod

   method Action writeRow(UInt#(TLog#(depth)) addr,
                          UInt#(TLog#(sublanes)) row,
                          Vector#(lanes, Int#(32)) data);
      let t = mem.sub(addr);
      t[row] = data;
      mem.upd(addr, t);
   endmethod

   method Action accumulateRow(UInt#(TLog#(depth)) addr,
                               UInt#(TLog#(sublanes)) row,
                               Vector#(lanes, Int#(32)) data);
      let t = mem.sub(addr);
      Vector#(lanes, Int#(32)) sum = newVector;
      for (Integer l = 0; l < valueOf(lanes); l = l + 1)
         sum[l] = t[row][l] + data[l];
      t[row] = sum;
      mem.upd(addr, t);
   endmethod

   method Vector#(lanes, Int#(32)) peekRow(UInt#(TLog#(depth)) addr,
                                           UInt#(TLog#(sublanes)) row);
      return mem.sub(addr)[row];
   endmethod

endmodule

export PSUMBank_IFC(..);
export mkPSUMBank;

endpackage
