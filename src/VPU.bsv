package VPU;

import Vector :: *;
import FloatingPoint :: *;

typedef enum { VPU_ADD, VPU_MUL, VPU_RELU, VPU_MAX, VPU_SUM_REDUCE, VPU_CMPLT, VPU_CMPNE, VPU_SUB, VPU_CMPEQ, VPU_MAX_REDUCE, VPU_SHL, VPU_SHR, VPU_MIN, VPU_MIN_REDUCE, VPU_DIV, VPU_AND, VPU_OR, VPU_XOR,
               VPU_FADD, VPU_FMUL, VPU_FSUB, VPU_FMAX, VPU_FCMPLT, VPU_FRECIP, VPU_I2F, VPU_F2I,
               VPU_NOT, VPU_SELECT, VPU_COPY,
               VPU_SUM_REDUCE_COL, VPU_MAX_REDUCE_COL, VPU_MIN_REDUCE_COL,
               VPU_SUM_REDUCE_TILE, VPU_MAX_REDUCE_TILE, VPU_MIN_REDUCE_TILE }
   VpuOp deriving (Bits, Eq, FShow);

// Reinterpret Int#(32) bits as IEEE 754 Float (bitcast, not conversion)
function Float bits2fp(Int#(32) x);
   return unpack(pack(x));
endfunction

// Reinterpret Float bits as Int#(32) (bitcast, not conversion)
function Int#(32) fp2bits(Float x);
   return unpack(pack(x));
endfunction

// Convert signed Int#(32) to Float (value conversion)
function Float int32_to_float(Int#(32) x);
   Bool neg = x < 0;
   UInt#(32) mag = neg ? unpack(pack(-x)) : unpack(pack(x));
   // Find position of leading 1 (highest set bit)
   Bit#(32) mag_bits = pack(mag);
   UInt#(8) shift = 0;
   for (Integer i = 31; i >= 0; i = i - 1)
      if (mag_bits[i] == 1 && shift == 0) shift = fromInteger(i);
   // Exponent: bias 127 + position of leading 1
   Bit#(8) exp_bits = (mag == 0) ? 0 : truncate(pack(shift + 127));
   // Mantissa: bits below the leading 1, shifted to fill 23 bits
   Bit#(23) mant = (shift > 23) ? truncate(pack(mag >> (shift - 23)))
                                 : truncate(pack(mag << (23 - shift)));
   Float f;
   f.sign = neg && (mag != 0);
   f.exp = (mag == 0) ? 0 : exp_bits;
   f.sfd = (mag == 0) ? 0 : mant;
   return f;
endfunction

// Convert Float to signed Int#(32) (truncate toward zero)
function Int#(32) float_to_int32(Float f);
   Bit#(8) raw_exp = f.exp;
   Int#(9) exp_val = unpack(zeroExtend(raw_exp)) - 127;
   Bool is_zero = (raw_exp == 0 && f.sfd == 0);
   Bool is_special = (raw_exp == 8'hFF);
   Bool too_small = (exp_val < 0);
   // Reconstruct magnitude: 1.mantissa × 2^exp
   UInt#(32) mantissa = (zeroExtend(1'b1) << 23) | unpack(zeroExtend(f.sfd));
   Bit#(5) shift_r = truncate(pack(23 - exp_val));
   Bit#(5) shift_l = truncate(pack(exp_val - 23));
   UInt#(32) mag = (exp_val >= 23) ? (mantissa << shift_l)
                                    : (mantissa >> shift_r);
   Int#(32) result = unpack(pack(mag));
   Int#(32) signed_result = f.sign ? (-result) : result;
   return (is_zero || is_special || too_small) ? 0 : signed_result;
endfunction

interface VPU_IFC#(numeric type sublanes, numeric type lanes);
   method Action execute(
      VpuOp op,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src1,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src2
   );
   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;
endinterface

// Sum all lanes in one row: unrolled adder
function Int#(32) lane_sum(Vector#(lanes, Int#(32)) row)
   provisos(Add#(1, l_, lanes));
   Int#(32) acc = 0;
   for (Integer i = 0; i < valueOf(lanes); i = i + 1)
      acc = acc + row[i];
   return acc;
endfunction

// Max of all lanes in one row: unrolled comparator
function Int#(32) lane_max(Vector#(lanes, Int#(32)) row)
   provisos(Add#(1, l_, lanes));
   Int#(32) acc = row[0];
   for (Integer i = 1; i < valueOf(lanes); i = i + 1)
      acc = (row[i] > acc) ? row[i] : acc;
   return acc;
endfunction

// Min of all lanes in one row: unrolled comparator
function Int#(32) lane_min(Vector#(lanes, Int#(32)) row)
   provisos(Add#(1, l_, lanes));
   Int#(32) acc = row[0];
   for (Integer i = 1; i < valueOf(lanes); i = i + 1)
      acc = (row[i] < acc) ? row[i] : acc;
   return acc;
endfunction

module mkVPU(VPU_IFC#(sublanes, lanes))
   provisos(
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) resultReg <- mkRegU;

   method Action execute(
      VpuOp op,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src1,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src2
   );
      // Per-column reductions across sublanes (rows): one value per column.
      Vector#(lanes, Int#(32)) col_sum = newVector;
      Vector#(lanes, Int#(32)) col_max = newVector;
      Vector#(lanes, Int#(32)) col_min = newVector;
      for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
         Int#(32) csum = 0;
         Int#(32) cmax = src1[0][l];
         Int#(32) cmin = src1[0][l];
         for (Integer s = 0; s < valueOf(sublanes); s = s + 1) begin
            csum = csum + src1[s][l];
            cmax = (src1[s][l] > cmax) ? src1[s][l] : cmax;
            cmin = (src1[s][l] < cmin) ? src1[s][l] : cmin;
         end
         col_sum[l] = csum;
         col_max[l] = cmax;
         col_min[l] = cmin;
      end
      // Full-tile reductions: reduce the per-column reductions across lanes.
      Int#(32) tile_sum = 0;
      Int#(32) tile_max = col_max[0];
      Int#(32) tile_min = col_min[0];
      for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
         tile_sum = tile_sum + col_sum[l];
         tile_max = (col_max[l] > tile_max) ? col_max[l] : tile_max;
         tile_min = (col_min[l] < tile_min) ? col_min[l] : tile_min;
      end

      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1) begin
         Vector#(lanes, Int#(32)) row = newVector;
         case (op)
            VPU_ADD: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = src1[s][l] + src2[s][l];
            end
            VPU_MUL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = src1[s][l] * src2[s][l];
            end
            VPU_RELU: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] > 0) ? src1[s][l] : 0;
            end
            VPU_MAX: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] > src2[s][l]) ? src1[s][l] : src2[s][l];
            end
            VPU_SUM_REDUCE: begin
               Int#(32) s_val = lane_sum(src1[s]);
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = s_val;
            end
            VPU_CMPLT: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] < src2[s][l]) ? 1 : 0;
            end
            VPU_CMPNE: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] != src2[s][l]) ? 1 : 0;
            end
            VPU_SUB: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = src1[s][l] - src2[s][l];
            end
            VPU_CMPEQ: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] == src2[s][l]) ? 1 : 0;
            end
            VPU_MAX_REDUCE: begin
               Int#(32) m_val = lane_max(src1[s]);
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = m_val;
            end
            VPU_MIN: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] < src2[s][l]) ? src1[s][l] : src2[s][l];
            end
            VPU_MIN_REDUCE: begin
               Int#(32) mn_val = lane_min(src1[s]);
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = mn_val;
            end
            VPU_SHL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(5) amt = truncate(pack(src2[s][l]));
                  row[l] = unpack(pack(src1[s][l]) << amt);
               end
            end
            VPU_SHR: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(5) amt = truncate(pack(src2[s][l]));
                  row[l] = unpack(pack(src1[s][l]) >> amt);
               end
            end
            VPU_DIV: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src2[s][l] == 0) ? 0 : (src1[s][l] / src2[s][l]);
            end
            VPU_AND: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = unpack(pack(src1[s][l]) & pack(src2[s][l]));
            end
            VPU_OR: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = unpack(pack(src1[s][l]) | pack(src2[s][l]));
            end
            VPU_XOR: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = unpack(pack(src1[s][l]) ^ pack(src2[s][l]));
            end
            // ── Float32 ops (bits in Int#(32) reinterpreted as IEEE 754) ──
            VPU_FADD: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  let r = addFP(bits2fp(src1[s][l]), bits2fp(src2[s][l]), Rnd_Nearest_Even);
                  row[l] = fp2bits(tpl_1(r));
               end
            end
            VPU_FMUL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  let r = multFP(bits2fp(src1[s][l]), bits2fp(src2[s][l]), Rnd_Nearest_Even);
                  row[l] = fp2bits(tpl_1(r));
               end
            end
            VPU_FSUB: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Float b_neg = bits2fp(src2[s][l]);
                  b_neg.sign = !b_neg.sign;
                  let r = addFP(bits2fp(src1[s][l]), b_neg, Rnd_Nearest_Even);
                  row[l] = fp2bits(tpl_1(r));
               end
            end
            VPU_FMAX: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  let cmp = compareFP(bits2fp(src1[s][l]), bits2fp(src2[s][l]));
                  row[l] = (cmp == GT || cmp == EQ) ? src1[s][l] : src2[s][l];
               end
            end
            VPU_FCMPLT: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  let cmp = compareFP(bits2fp(src1[s][l]), bits2fp(src2[s][l]));
                  row[l] = (cmp == LT) ? 1 : 0;
               end
            end
            VPU_FRECIP: begin
               // Reciprocal via magic-number initial estimate + Newton-Raphson.
               // Initial: reinterpret bits as int, compute 0x7EF0_0000 - x, reinterpret as float.
               // Then 3 NR iterations: x' = x * (2 - a * x)
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Float a = bits2fp(src1[s][l]);
                  // Magic number reciprocal estimate (sign-magnitude aware)
                  Bit#(32) a_bits = pack(src1[s][l]);
                  Bit#(31) mag = a_bits[30:0];
                  Bit#(31) est_mag = 31'h3F800000 + (31'h3F800000 - mag);  // ~2*bias - mag
                  Bit#(32) est_bits = {a_bits[31], est_mag};
                  Float est = unpack(est_bits);
                  Float two = unpack(32'h40000000);
                  // NR step 1
                  let a_est1 = tpl_1(multFP(a, est, Rnd_Nearest_Even));
                  Float neg1 = a_est1; neg1.sign = !neg1.sign;
                  let ref1 = tpl_1(multFP(est, tpl_1(addFP(two, neg1, Rnd_Nearest_Even)), Rnd_Nearest_Even));
                  // NR step 2
                  let a_est2 = tpl_1(multFP(a, ref1, Rnd_Nearest_Even));
                  Float neg2 = a_est2; neg2.sign = !neg2.sign;
                  let ref2 = tpl_1(multFP(ref1, tpl_1(addFP(two, neg2, Rnd_Nearest_Even)), Rnd_Nearest_Even));
                  // NR step 3
                  let a_est3 = tpl_1(multFP(a, ref2, Rnd_Nearest_Even));
                  Float neg3 = a_est3; neg3.sign = !neg3.sign;
                  let ref3 = tpl_1(multFP(ref2, tpl_1(addFP(two, neg3, Rnd_Nearest_Even)), Rnd_Nearest_Even));
                  row[l] = fp2bits(ref3);
               end
            end
            VPU_I2F: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = fp2bits(int32_to_float(src1[s][l]));
            end
            VPU_F2I: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = float_to_int32(bits2fp(src1[s][l]));
            end
            VPU_NOT: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = unpack(~pack(src1[s][l]));
            end
            VPU_SELECT: begin
               // SELECT(cond=src1, true_val=src2, false_val=resultReg)
               // result[l] = (src1[l] != 0) ? src2[l] : resultReg[s][l]
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] != 0) ? src2[s][l] : resultReg[s][l];
            end
            VPU_COPY: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = src1[s][l];
            end
            VPU_SUM_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_sum[l];
            end
            VPU_MAX_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_max[l];
            end
            VPU_MIN_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_min[l];
            end
            VPU_SUM_REDUCE_TILE: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = tile_sum;
            end
            VPU_MAX_REDUCE_TILE: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = tile_max;
            end
            VPU_MIN_REDUCE_TILE: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = tile_min;
            end
         endcase
         res[s] = row;
      end
      resultReg <= res;
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;
      return resultReg;
   endmethod

endmodule

export VpuOp(..);
export VPU_IFC(..);
export mkVPU;

endpackage
