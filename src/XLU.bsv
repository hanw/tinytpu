package XLU;

import Vector :: *;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

typedef enum { PERMUTE, ROTATE, BROADCAST, TRANSPOSE }
   XluOp deriving (Bits, Eq, FShow);

// ---------------------------------------------------------------------------
// Interface
// ---------------------------------------------------------------------------

interface XLU_IFC#(numeric type sublanes, numeric type lanes);
   // Cyclic rotate: output[s][i] = input[s][(i + amount) mod lanes]
   method Action executeRotate(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) amount
   );

   // Broadcast: output[s][i] = input[s][srcLane] for all i
   method Action executeBroadcast(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) srcLane
   );

   // Scalar broadcast: output[r][c] = src[srcRow][srcCol]
   method Action executeBroadcastScalar(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(sublanes)) srcRow,
      UInt#(TLog#(lanes)) srcCol
   );

   // Row broadcast: output[r][c] = src[srcRow][c]
   method Action executeBroadcastRow(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(sublanes)) srcRow
   );

   // Col broadcast: output[r][c] = src[r][srcCol]
   method Action executeBroadcastCol(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) srcCol
   );

   // XOR-swap butterfly permutation.
   // ctrl[k][i] = True means lane i swaps with lane (i XOR 2^k) at stage k.
   method Action executePermute(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      Vector#(TLog#(lanes), Vector#(lanes, Bool)) ctrl
   );

   // Transpose: output[r][c] = input[c][r]  (requires sublanes == lanes)
   method Action executeTranspose(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src
   );

   // Result is valid the cycle after any execute* call
   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;

endinterface

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Compile-time 2^k (BSV Integer has no bitwise ops, so use multiplication)
function Integer pow2(Integer k) = (k == 0) ? 1 : 2 * pow2(k - 1);

// Barrel-shifter rotation for one lane row.
// output[i] = input[(i + amount) mod lanes]
// Stage k shifts by 2^k if bit k of 'amount' is set.
function Vector#(lanes, t) lane_rotate(
   Vector#(lanes, t) v,
   UInt#(TLog#(lanes)) amount
) provisos(Log#(lanes, logLanes));
   Vector#(lanes, t) cur = v;
   for (Integer k = 0; k < valueOf(logLanes); k = k + 1) begin
      Integer stride = pow2(k);
      // Extract bit k of amount at compile-time index k
      Bool do_shift = unpack(pack(amount)[k]);
      Vector#(lanes, t) nxt = newVector;
      for (Integer i = 0; i < valueOf(lanes); i = i + 1)
         nxt[i] = do_shift ? cur[(i + stride) % valueOf(lanes)] : cur[i];
      cur = nxt;
   end
   return cur;
endfunction

// Broadcast one lane value to all lanes
function Vector#(lanes, t) lane_broadcast(
   Vector#(lanes, t) v,
   UInt#(TLog#(lanes)) srcLane
) provisos(Log#(lanes, logLanes));
   t val = v[srcLane];
   return replicate(val);
endfunction

function Vector#(sublanes, Vector#(lanes, t)) tile_broadcast_scalar(
   Vector#(sublanes, Vector#(lanes, t)) src,
   UInt#(TLog#(sublanes)) srcRow,
   UInt#(TLog#(lanes)) srcCol
);
   t val = src[srcRow][srcCol];
   return replicate(replicate(val));
endfunction

function Vector#(sublanes, Vector#(lanes, t)) tile_broadcast_row(
   Vector#(sublanes, Vector#(lanes, t)) src,
   UInt#(TLog#(sublanes)) srcRow
);
   Vector#(lanes, t) row = src[srcRow];
   return replicate(row);
endfunction

function Vector#(sublanes, Vector#(lanes, t)) tile_broadcast_col(
   Vector#(sublanes, Vector#(lanes, t)) src,
   UInt#(TLog#(lanes)) srcCol
);
   Vector#(sublanes, Vector#(lanes, t)) res = newVector;
   for (Integer r = 0; r < valueOf(sublanes); r = r + 1)
      res[r] = replicate(src[r][srcCol]);
   return res;
endfunction

// XOR-swap butterfly permutation for one lane row.
// ctrl[k][i] = True: swap lane i with lane (i XOR 2^k) at stage k.
// Since stride = 2^k is a power of 2, (i XOR stride) == i+stride when bit k
// of i is 0, and i-stride when bit k of i is 1.  Using integer arithmetic
// avoids BSV's restriction on bitwise ops on Integer.
function Vector#(lanes, t) lane_butterfly(
   Vector#(TLog#(lanes), Vector#(lanes, Bool)) ctrl,
   Vector#(lanes, t) v
) provisos(Log#(lanes, logLanes));
   Vector#(lanes, t) cur = v;
   for (Integer k = 0; k < valueOf(logLanes); k = k + 1) begin
      Integer stride = pow2(k);
      Vector#(lanes, t) nxt = newVector;
      for (Integer i = 0; i < valueOf(lanes); i = i + 1) begin
         // i XOR stride: flip bit k of i
         Integer partner = ((i / stride) % 2 == 0) ? i + stride : i - stride;
         nxt[i] = ctrl[k][i] ? cur[partner] : cur[i];
      end
      cur = nxt;
   end
   return cur;
endfunction

// 2D transpose for square vreg: output[r][c] = input[c][r]
function Vector#(n, Vector#(n, t)) vreg_transpose(
   Vector#(n, Vector#(n, t)) v
);
   Vector#(n, Vector#(n, t)) res = newVector;
   for (Integer r = 0; r < valueOf(n); r = r + 1) begin
      res[r] = newVector;
      for (Integer c = 0; c < valueOf(n); c = c + 1)
         res[r][c] = v[c][r];
   end
   return res;
endfunction

// ---------------------------------------------------------------------------
// Module: mkXLU
// Add#(0, sublanes, lanes) enforces sublanes == lanes (required for TRANSPOSE)
// ---------------------------------------------------------------------------

module mkXLU(XLU_IFC#(sublanes, lanes))
   provisos(
      Log#(lanes, logLanes),
      Add#(1, l_, lanes),
      Add#(1, s_, sublanes),
      Add#(0, sublanes, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) resultReg <- mkRegU;

   method Action executeRotate(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) amount
   );
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1)
         res[s] = lane_rotate(src[s], amount);
      resultReg <= res;
   endmethod

   method Action executeBroadcast(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) srcLane
   );
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1)
         res[s] = lane_broadcast(src[s], srcLane);
      resultReg <= res;
   endmethod

   method Action executeBroadcastScalar(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(sublanes)) srcRow,
      UInt#(TLog#(lanes)) srcCol
   );
      resultReg <= tile_broadcast_scalar(src, srcRow, srcCol);
   endmethod

   method Action executeBroadcastRow(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(sublanes)) srcRow
   );
      resultReg <= tile_broadcast_row(src, srcRow);
   endmethod

   method Action executeBroadcastCol(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      UInt#(TLog#(lanes)) srcCol
   );
      resultReg <= tile_broadcast_col(src, srcCol);
   endmethod

   method Action executePermute(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src,
      Vector#(TLog#(lanes), Vector#(lanes, Bool)) ctrl
   );
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1)
         res[s] = lane_butterfly(ctrl, src[s]);
      resultReg <= res;
   endmethod

   method Action executeTranspose(
      Vector#(sublanes, Vector#(lanes, Int#(32))) src
   );
      resultReg <= vreg_transpose(src);
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;
      return resultReg;
   endmethod

endmodule

export XluOp(..);
export XLU_IFC(..);
export mkXLU;

endpackage
