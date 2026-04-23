package VPU;

import Vector :: *;
import FloatingPoint :: *;
import FpReducer :: *;
import TranscUnit :: *;

typedef enum { VPU_ADD, VPU_MUL, VPU_RELU, VPU_MAX, VPU_SUM_REDUCE, VPU_CMPLT, VPU_CMPNE, VPU_SUB, VPU_CMPEQ, VPU_MAX_REDUCE, VPU_SHL, VPU_SHR, VPU_MIN, VPU_MIN_REDUCE, VPU_DIV, VPU_AND, VPU_OR, VPU_XOR,
               VPU_FADD, VPU_FMUL, VPU_FSUB, VPU_FMAX, VPU_FCMPLT, VPU_FRECIP, VPU_I2F, VPU_F2I,
               VPU_NOT, VPU_SELECT, VPU_COPY,
               VPU_SUM_REDUCE_COL, VPU_MAX_REDUCE_COL, VPU_MIN_REDUCE_COL,
               VPU_SUM_REDUCE_TILE, VPU_MAX_REDUCE_TILE, VPU_MIN_REDUCE_TILE,
               VPU_MUL_REDUCE, VPU_MUL_REDUCE_COL, VPU_MUL_REDUCE_TILE,
               VPU_FSUM_REDUCE_TILE, VPU_FMAX_REDUCE_TILE, VPU_FMIN_REDUCE_TILE,
               VPU_FMIN,
               VPU_FSUM_REDUCE, VPU_FMAX_REDUCE, VPU_FMIN_REDUCE,
               VPU_FSUM_REDUCE_COL, VPU_FMAX_REDUCE_COL, VPU_FMIN_REDUCE_COL,
               VPU_FPROD_REDUCE_TILE, VPU_FPROD_REDUCE, VPU_FPROD_REDUCE_COL,
               VPU_EXP2, VPU_LOG2, VPU_SIN, VPU_COS,
               // Packed-int8 arithmetic: each Int#(32) lane holds 4 Int#(8)
               // values in two's-complement (byte 0 = bits[7:0], byte 3 =
               // bits[31:24]). Saturating add is the first quantized-
               // inference primitive — four 8-bit adders per lane, each
               // clamped to [-128, 127] independently.
               VPU_PACKED_I8_ADD, VPU_PACKED_I8_SUB,
               VPU_PACKED_I8_MAX, VPU_PACKED_I8_MIN,
               VPU_PACKED_I8_NEG, VPU_PACKED_I8_RELU,
               VPU_PACKED_I8_CMPLT, VPU_PACKED_I8_CMPEQ,
               VPU_PACKED_I8_MUL_LOW, VPU_PACKED_I8_MUL_HIGH,
               VPU_PACKED_I8_ABS,
               // Lane-wise sign: -1 if x<0, 0 if x==0, 1 if x>0.
               VPU_SIGN,
               // Byte-wise sign on packed int8 (per-byte -1/0/+1).
               VPU_PACKED_I8_SIGN,
               // Float lane-wise sign: -1.0 if x<0, +1.0 if x>0, 0.0 if x==0.
               VPU_FSIGN,
               // Integer argmin/argmax per row: output[s][*] = index
               // of the min/max value in src1[s][*] (broadcast).
               VPU_ARGMIN, VPU_ARGMAX,
               // Per-lane count-leading-zeros and popcount (lane-wise
               // unary ops on the raw 32-bit value).
               VPU_CLZ, VPU_POPCOUNT,
               // Count trailing zeros (32 for all-zero); byte-swap
               // within each 32-bit lane.
               VPU_CTZ, VPU_BYTE_REVERSE,
               // 32-bit signed saturating add / sub (clamp to Int#(32)
               // range on overflow, matching C intrinsics /
               // DSP extensions).
               VPU_SAT_ADD_I32, VPU_SAT_SUB_I32,
               // |a - b| — saturating for int32 (|Int32_MIN - 0| clamps
               // to Int32_MAX). Useful as an L1-distance primitive.
               VPU_ABS_DIFF_I32, VPU_PACKED_I8_ABS_DIFF,
               // Float absolute value (clear sign bit) — single-cycle
               // unary, replaces the FSUB + FMAX pair in softsign.
               VPU_FABS,
               // 32-bit lane-wise rotate-left / rotate-right; shift
               // amount taken from low 5 bits of src2 (mod 32).
               VPU_ROTL, VPU_ROTR,
               // Unsigned-viewed 32-bit min/max for sort keys and
               // hashing. Semantically independent of signed MIN/MAX.
               VPU_MIN_U32, VPU_MAX_U32 }
   VpuOp deriving (Bits, Eq, FShow);

// Add two signed 8-bit values with saturation to [-128, 127].
function Int#(8) sat_add_i8(Int#(8) a, Int#(8) b);
   Int#(9) wide = extend(a) + extend(b);
   Int#(8) clamped = (wide >  127) ?  127 :
                     (wide < -128) ? -128 :
                     truncate(wide);
   return clamped;
endfunction

// Subtract two signed 8-bit values with saturation to [-128, 127].
function Int#(8) sat_sub_i8(Int#(8) a, Int#(8) b);
   Int#(9) wide = extend(a) - extend(b);
   Int#(8) clamped = (wide >  127) ?  127 :
                     (wide < -128) ? -128 :
                     truncate(wide);
   return clamped;
endfunction

// Pack four Int#(8) back into one Int#(32) (little-endian byte order).
function Int#(32) pack_i8x4(Int#(8) b0, Int#(8) b1, Int#(8) b2, Int#(8) b3);
   Bit#(32) bits = {pack(b3), pack(b2), pack(b1), pack(b0)};
   return unpack(bits);
endfunction

// Unpack one Int#(32) into four Int#(8) (little-endian byte order).
function Tuple4#(Int#(8), Int#(8), Int#(8), Int#(8)) unpack_i8x4(Int#(32) x);
   Bit#(32) bits = pack(x);
   return tuple4(unpack(bits[ 7: 0]),
                 unpack(bits[15: 8]),
                 unpack(bits[23:16]),
                 unpack(bits[31:24]));
endfunction

// Packed-int8 lane-wise saturating add. Treats each 32-bit value as 4x
// Int#(8) in little-endian byte order; does four independent saturating
// adds and repacks.
function Int#(32) packed_i8_add(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4(sat_add_i8(a0, b0), sat_add_i8(a1, b1),
                    sat_add_i8(a2, b2), sat_add_i8(a3, b3));
endfunction

// Packed-int8 lane-wise saturating sub (a - b per byte, clamped).
function Int#(32) packed_i8_sub(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4(sat_sub_i8(a0, b0), sat_sub_i8(a1, b1),
                    sat_sub_i8(a2, b2), sat_sub_i8(a3, b3));
endfunction

// Byte-wise signed max.
function Int#(32) packed_i8_max(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4(max(a0, b0), max(a1, b1), max(a2, b2), max(a3, b3));
endfunction

// Byte-wise signed min.
function Int#(32) packed_i8_min(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4(min(a0, b0), min(a1, b1), min(a2, b2), min(a3, b3));
endfunction

// Byte-wise signed negate with saturation: -(-128) would overflow, so
// clamp it to 127 (standard packed-SIMD convention).
function Int#(8) sat_neg_i8(Int#(8) a);
   Int#(8) r = (a == -128) ? 127 : -a;
   return r;
endfunction

function Int#(32) packed_i8_neg(Int#(32) a);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   return pack_i8x4(sat_neg_i8(a0), sat_neg_i8(a1), sat_neg_i8(a2), sat_neg_i8(a3));
endfunction

// Byte-wise RELU (max(x, 0) per byte).
function Int#(32) packed_i8_relu(Int#(32) a);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   return pack_i8x4((a0 > 0) ? a0 : 0, (a1 > 0) ? a1 : 0,
                    (a2 > 0) ? a2 : 0, (a3 > 0) ? a3 : 0);
endfunction

// Byte-wise signed less-than: 0xFF (-1) if true, 0x00 if false per byte.
function Int#(32) packed_i8_cmplt(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4((a0 < b0) ? -1 : 0, (a1 < b1) ? -1 : 0,
                    (a2 < b2) ? -1 : 0, (a3 < b3) ? -1 : 0);
endfunction

// Byte-wise equal: 0xFF (-1) if equal, 0x00 otherwise.
function Int#(32) packed_i8_cmpeq(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4((a0 == b0) ? -1 : 0, (a1 == b1) ? -1 : 0,
                    (a2 == b2) ? -1 : 0, (a3 == b3) ? -1 : 0);
endfunction

// Byte-wise signed multiply keeping the low 8 bits (two's-complement
// wrap). Matches the semantics of ISO-C `(int8_t)((int16_t)a * b)`,
// which is what quantized-inference frameworks usually want for the
// element-wise stage before a wider accumulation.
function Int#(8) mul_low_i8(Int#(8) a, Int#(8) b);
   Int#(16) wide = extend(a) * extend(b);
   return truncate(wide);
endfunction

function Int#(32) packed_i8_mul_low(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4(mul_low_i8(a0, b0), mul_low_i8(a1, b1),
                    mul_low_i8(a2, b2), mul_low_i8(a3, b3));
endfunction

// Byte-wise signed multiply, HIGH 8 bits of the 16-bit product.
// Useful for Q1.7 × Q1.7 → Q1.7 (fixed-point scaling): lo gives the
// wrap-around result, high gives the scaled result.
function Int#(8) mul_high_i8(Int#(8) a, Int#(8) b);
   Int#(16) wide = extend(a) * extend(b);
   Bit#(16) bits = pack(wide);
   return unpack(bits[15:8]);
endfunction

function Int#(32) packed_i8_mul_high(Int#(32) a, Int#(32) b);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   match { .b0, .b1, .b2, .b3 } = unpack_i8x4(b);
   return pack_i8x4(mul_high_i8(a0, b0), mul_high_i8(a1, b1),
                    mul_high_i8(a2, b2), mul_high_i8(a3, b3));
endfunction

// Byte-wise absolute value with saturation (|-128| wraps to 127).
function Int#(8) sat_abs_i8(Int#(8) a);
   Int#(8) r = (a == -128) ? 127 : ((a < 0) ? -a : a);
   return r;
endfunction

function Int#(32) packed_i8_abs(Int#(32) a);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   return pack_i8x4(sat_abs_i8(a0), sat_abs_i8(a1),
                    sat_abs_i8(a2), sat_abs_i8(a3));
endfunction

// Byte-wise sign: -1 / 0 / +1 per byte.
function Int#(8) sign_i8(Int#(8) a);
   Int#(8) r = (a > 0) ? 1 : ((a < 0) ? -1 : 0);
   return r;
endfunction

function Int#(32) packed_i8_sign(Int#(32) a);
   match { .a0, .a1, .a2, .a3 } = unpack_i8x4(a);
   return pack_i8x4(sign_i8(a0), sign_i8(a1), sign_i8(a2), sign_i8(a3));
endfunction

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
   // True when the most recent dispatch has completed and `result` holds
   // the final value. Single-cycle ops report done immediately; multi-
   // cycle float reducers go False during the walk and True on completion.
   method Bool isDone;
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

// Product of all lanes in one row: unrolled multiplier
function Int#(32) lane_prod(Vector#(lanes, Int#(32)) row)
   provisos(Add#(1, l_, lanes));
   Int#(32) acc = 1;
   for (Integer i = 0; i < valueOf(lanes); i = i + 1)
      acc = acc * row[i];
   return acc;
endfunction

// Float sequential reducers over `n` Int#(32) bit patterns reinterpreted
// as IEEE 754 floats. Using helper functions keeps the FP adders and
// comparators in one place so bsc elaborates them once and they are
// shared between the row-reducer (per-sublane) and col-reducer
// (per-column) paths.
function Float lane_fsum(Vector#(n, Int#(32)) vec)
   provisos(Add#(1, n_, n));
   Float acc = unpack(32'h00000000);  // +0.0
   for (Integer i = 0; i < valueOf(n); i = i + 1)
      acc = tpl_1(addFP(acc, bits2fp(vec[i]), Rnd_Nearest_Even));
   return acc;
endfunction

function Float lane_fmax(Vector#(n, Int#(32)) vec)
   provisos(Add#(1, n_, n));
   Float acc = bits2fp(vec[0]);
   for (Integer i = 1; i < valueOf(n); i = i + 1) begin
      Float b = bits2fp(vec[i]);
      let cmp = compareFP(acc, b);
      acc = (cmp == GT || cmp == EQ) ? acc : b;
   end
   return acc;
endfunction

function Float lane_fmin(Vector#(n, Int#(32)) vec)
   provisos(Add#(1, n_, n));
   Float acc = bits2fp(vec[0]);
   for (Integer i = 1; i < valueOf(n); i = i + 1) begin
      Float b = bits2fp(vec[i]);
      let cmp = compareFP(acc, b);
      acc = (cmp == LT || cmp == EQ) ? acc : b;
   end
   return acc;
endfunction

function Float lane_fprod(Vector#(n, Int#(32)) vec)
   provisos(Add#(1, n_, n));
   Float acc = unpack(32'h3F800000);  // 1.0
   for (Integer i = 0; i < valueOf(n); i = i + 1)
      acc = tpl_1(multFP(acc, bits2fp(vec[i]), Rnd_Nearest_Even));
   return acc;
endfunction


module mkVPU(VPU_IFC#(sublanes, lanes))
   provisos(
      Add#(1, s_, sublanes),
      Add#(1, l_, lanes),
      Add#(1, sl_, TMul#(sublanes, lanes)),
      Bits#(Vector#(sublanes, Vector#(lanes, Int#(32))), vsz)
   );

   Reg#(Vector#(sublanes, Vector#(lanes, Int#(32)))) resultReg <- mkRegU;

   // Shared multi-cycle FP reducer for SUM/MAX/MIN across the flattened
   // tile (row/col variants will feed their vectors through the same unit
   // in later iterations with caller-supplied identity padding).
   FpReducer_IFC#(TMul#(sublanes, lanes)) fpr <- mkFpReducer;
   Reg#(Bool) fp_busy <- mkReg(False);

   // Shared multi-cycle transcendental unit for per-lane EXP2/LOG2/SIN.
   // Walks the flattened tile one lane at a time to keep elaboration cost
   // bounded (1 FMUL + 1 FADD total, not N_horner * lanes parallel cones).
   TranscUnit_IFC#(TMul#(sublanes, lanes)) tru <- mkTranscUnit;
   Reg#(Bool) transc_busy <- mkReg(False);

   // When the reducer signals done, broadcast its scalar across the full
   // tile and clear the busy flag. SXU waits on isDone before reading.
   rule fp_collect (fp_busy && fpr.isDone);
      Int#(32) bits = unpack(pack(fpr.getResult));
      Vector#(sublanes, Vector#(lanes, Int#(32))) res =
         replicate(replicate(bits));
      resultReg <= res;
      fp_busy <= False;
   endrule

   // When the transcendental unit signals done, unflatten its per-lane
   // result back into the tile shape and release the busy flag.
   rule transc_collect (transc_busy && tru.isDone);
      let flat = tru.getResult;
      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1) begin
         Vector#(lanes, Int#(32)) row = newVector;
         for (Integer l = 0; l < valueOf(lanes); l = l + 1)
            row[l] = flat[s * valueOf(lanes) + l];
         res[s] = row;
      end
      resultReg <= res;
      transc_busy <= False;
   endrule

   method Action execute(
      VpuOp op,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src1,
      Vector#(sublanes, Vector#(lanes, Int#(32))) src2
   ) if (!fp_busy && !transc_busy);
      // Multi-cycle float tile reducers dispatch through the shared
      // FpReducer and set fp_busy; the fp_collect rule later writes
      // resultReg and clears fp_busy. The combinational case block below
      // runs for everything else.
      Bool use_fp_tile_reducer =
             (op == VPU_FSUM_REDUCE_TILE)
          || (op == VPU_FMAX_REDUCE_TILE)
          || (op == VPU_FMIN_REDUCE_TILE)
          || (op == VPU_FPROD_REDUCE_TILE);
      Bool use_transc = (op == VPU_EXP2) || (op == VPU_LOG2) || (op == VPU_SIN) || (op == VPU_COS);
      if (use_fp_tile_reducer) begin
         Vector#(TMul#(sublanes, lanes), Int#(32)) flat = newVector;
         for (Integer sf = 0; sf < valueOf(sublanes); sf = sf + 1)
            for (Integer lf = 0; lf < valueOf(lanes); lf = lf + 1)
               flat[sf * valueOf(lanes) + lf] = src1[sf][lf];
         FpReducerOp frop = (op == VPU_FSUM_REDUCE_TILE)  ? FPR_SUM  :
                             (op == VPU_FMAX_REDUCE_TILE)  ? FPR_MAX  :
                             (op == VPU_FMIN_REDUCE_TILE)  ? FPR_MIN  : FPR_PROD;
         fpr.start(frop, flat);
         fp_busy <= True;
      end else if (use_transc) begin
         Vector#(TMul#(sublanes, lanes), Int#(32)) flat = newVector;
         for (Integer sf = 0; sf < valueOf(sublanes); sf = sf + 1)
            for (Integer lf = 0; lf < valueOf(lanes); lf = lf + 1)
               flat[sf * valueOf(lanes) + lf] = src1[sf][lf];
         TranscOp trop = (op == VPU_LOG2) ? TR_LOG2 :
                         (op == VPU_SIN)  ? TR_SIN  :
                         (op == VPU_COS)  ? TR_COS  : TR_EXP2;
         tru.start(trop, flat);
         transc_busy <= True;
      end else begin
      // Per-column reductions across sublanes (rows): one value per column.
      Vector#(lanes, Int#(32)) col_sum = newVector;
      Vector#(lanes, Int#(32)) col_max = newVector;
      Vector#(lanes, Int#(32)) col_min = newVector;
      Vector#(lanes, Int#(32)) col_prod = newVector;
      for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
         Int#(32) csum = 0;
         Int#(32) cmax = src1[0][l];
         Int#(32) cmin = src1[0][l];
         Int#(32) cprod = 1;
         for (Integer s = 0; s < valueOf(sublanes); s = s + 1) begin
            csum = csum + src1[s][l];
            cmax = (src1[s][l] > cmax) ? src1[s][l] : cmax;
            cmin = (src1[s][l] < cmin) ? src1[s][l] : cmin;
            cprod = cprod * src1[s][l];
         end
         col_sum[l] = csum;
         col_max[l] = cmax;
         col_min[l] = cmin;
         col_prod[l] = cprod;
      end

      // Float per-column reductions. Computed unconditionally (like the
      // integer variants above) so each FP adder/comparator cone is
      // elaborated once, not re-elaborated per case branch. One col_f*
      // helper call per column; the helpers themselves (lane_fsum etc.)
      // are shared between row and col paths.
      Vector#(lanes, Int#(32)) col_fsum  = newVector;
      Vector#(lanes, Int#(32)) col_fmax  = newVector;
      Vector#(lanes, Int#(32)) col_fmin  = newVector;
      Vector#(lanes, Int#(32)) col_fprod = newVector;
      for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
         Vector#(sublanes, Int#(32)) colv = newVector;
         for (Integer sc = 0; sc < valueOf(sublanes); sc = sc + 1)
            colv[sc] = src1[sc][l];
         col_fsum[l]  = unpack(pack(lane_fsum(colv)));
         col_fmax[l]  = unpack(pack(lane_fmax(colv)));
         col_fmin[l]  = unpack(pack(lane_fmin(colv)));
         col_fprod[l] = unpack(pack(lane_fprod(colv)));
      end
      // Full-tile reductions: reduce the per-column reductions across lanes.
      Int#(32) tile_sum = 0;
      Int#(32) tile_max = col_max[0];
      Int#(32) tile_min = col_min[0];
      Int#(32) tile_prod = 1;
      for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
         tile_sum = tile_sum + col_sum[l];
         tile_max = (col_max[l] > tile_max) ? col_max[l] : tile_max;
         tile_min = (col_min[l] < tile_min) ? col_min[l] : tile_min;
         tile_prod = tile_prod * col_prod[l];
      end


      Vector#(sublanes, Vector#(lanes, Int#(32))) res = newVector;
      for (Integer s = 0; s < valueOf(sublanes); s = s + 1) begin
         Vector#(lanes, Int#(32)) row = newVector;
         case (op)
            VPU_ADD: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = src1[s][l] + src2[s][l];
            end
            VPU_PACKED_I8_ADD: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_add(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_SUB: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_sub(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_MAX: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_max(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_MIN: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_min(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_NEG: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_neg(src1[s][l]);
            end
            VPU_PACKED_I8_RELU: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_relu(src1[s][l]);
            end
            VPU_PACKED_I8_CMPLT: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_cmplt(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_CMPEQ: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_cmpeq(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_MUL_LOW: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_mul_low(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_MUL_HIGH: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_mul_high(src1[s][l], src2[s][l]);
            end
            VPU_PACKED_I8_ABS: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_abs(src1[s][l]);
            end
            VPU_SIGN: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = (src1[s][l] > 0) ? 1 :
                            ((src1[s][l] < 0) ? -1 : 0);
            end
            VPU_PACKED_I8_SIGN: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = packed_i8_sign(src1[s][l]);
            end
            VPU_FSIGN: begin
               // -1.0 (0xBF800000), 0.0 (0x00000000), +1.0 (0x3F800000).
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Float x = bits2fp(src1[s][l]);
                  Bool is_zero = (x.exp == 0 && x.sfd == 0);
                  Int#(32) sign_bits = is_zero ? 32'h00000000 :
                                        (x.sign ? 32'hBF800000 :
                                                  32'h3F800000);
                  row[l] = sign_bits;
               end
            end
            VPU_ARGMIN: begin
               // Per-row argmin of src1. First-lane-wins tiebreak.
               Int#(32) m_val = src1[s][0];
               UInt#(32) m_idx = 0;
               for (Integer l = 1; l < valueOf(lanes); l = l + 1)
                  if (src1[s][l] < m_val) begin
                     m_val = src1[s][l];
                     m_idx = fromInteger(l);
                  end
               Int#(32) bcast = unpack(pack(m_idx));
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = bcast;
            end
            VPU_ARGMAX: begin
               Int#(32) m_val = src1[s][0];
               UInt#(32) m_idx = 0;
               for (Integer l = 1; l < valueOf(lanes); l = l + 1)
                  if (src1[s][l] > m_val) begin
                     m_val = src1[s][l];
                     m_idx = fromInteger(l);
                  end
               Int#(32) bcast = unpack(pack(m_idx));
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = bcast;
            end
            VPU_CLZ: begin
               // Count leading zeros of the raw 32-bit value. All-zero
               // input returns 32.
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(32) b = pack(src1[s][l]);
                  UInt#(32) cnt = 32;
                  for (Integer bi = 31; bi >= 0; bi = bi - 1)
                     if (b[bi] == 1 && cnt == 32)
                        cnt = fromInteger(31 - bi);
                  row[l] = unpack(pack(cnt));
               end
            end
            VPU_POPCOUNT: begin
               // Population count (number of 1 bits) of the raw value.
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(32) b = pack(src1[s][l]);
                  UInt#(32) cnt = 0;
                  for (Integer bi = 0; bi < 32; bi = bi + 1)
                     if (b[bi] == 1)
                        cnt = cnt + 1;
                  row[l] = unpack(pack(cnt));
               end
            end
            VPU_CTZ: begin
               // Count trailing zeros; 32 for all-zero input.
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(32) b = pack(src1[s][l]);
                  UInt#(32) cnt = 32;
                  for (Integer bi = 0; bi < 32; bi = bi + 1)
                     if (b[bi] == 1 && cnt == 32)
                        cnt = fromInteger(bi);
                  row[l] = unpack(pack(cnt));
               end
            end
            VPU_BYTE_REVERSE: begin
               // Byte-swap each 32-bit lane (little-endian ↔ big-endian).
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(32) b = pack(src1[s][l]);
                  Bit#(32) r = { b[ 7: 0], b[15: 8], b[23:16], b[31:24] };
                  row[l] = unpack(r);
               end
            end
            VPU_SAT_ADD_I32: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Int#(33) wide = extend(src1[s][l]) + extend(src2[s][l]);
                  Int#(32) clamped =
                     (wide >  2147483647)  ?  2147483647 :
                     (wide < -2147483648)  ? -2147483648 :
                     truncate(wide);
                  row[l] = clamped;
               end
            end
            VPU_SAT_SUB_I32: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Int#(33) wide = extend(src1[s][l]) - extend(src2[s][l]);
                  Int#(32) clamped =
                     (wide >  2147483647)  ?  2147483647 :
                     (wide < -2147483648)  ? -2147483648 :
                     truncate(wide);
                  row[l] = clamped;
               end
            end
            VPU_ABS_DIFF_I32: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Int#(33) diff = extend(src1[s][l]) - extend(src2[s][l]);
                  Int#(33) mag  = (diff < 0) ? -diff : diff;
                  Int#(32) clamped = (mag > 2147483647) ? 2147483647
                                                        : truncate(mag);
                  row[l] = clamped;
               end
            end
            VPU_PACKED_I8_ABS_DIFF: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  match { .a0, .a1, .a2, .a3 } = unpack_i8x4(src1[s][l]);
                  match { .b0, .b1, .b2, .b3 } = unpack_i8x4(src2[s][l]);
                  row[l] = pack_i8x4(sat_abs_i8(sat_sub_i8(a0, b0)),
                                     sat_abs_i8(sat_sub_i8(a1, b1)),
                                     sat_abs_i8(sat_sub_i8(a2, b2)),
                                     sat_abs_i8(sat_sub_i8(a3, b3)));
               end
            end
            VPU_FABS: begin
               // Clear the IEEE-754 sign bit of each lane.
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(32) b = pack(src1[s][l]);
                  Bit#(32) r = { 1'b0, b[30:0] };
                  row[l] = unpack(r);
               end
            end
            VPU_ROTL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(32) b = pack(src1[s][l]);
                  Bit#(5)  amt = truncate(pack(src2[s][l]));
                  Bit#(5)  comp = 5'd0 - amt;
                  Bit#(32) r = (b << amt) | (b >> comp);
                  row[l] = unpack(r);
               end
            end
            VPU_ROTR: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  Bit#(32) b = pack(src1[s][l]);
                  Bit#(5)  amt = truncate(pack(src2[s][l]));
                  Bit#(5)  comp = 5'd0 - amt;
                  Bit#(32) r = (b >> amt) | (b << comp);
                  row[l] = unpack(r);
               end
            end
            VPU_MIN_U32: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  UInt#(32) a = unpack(pack(src1[s][l]));
                  UInt#(32) b = unpack(pack(src2[s][l]));
                  row[l] = unpack(pack((a < b) ? a : b));
               end
            end
            VPU_MAX_U32: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  UInt#(32) a = unpack(pack(src1[s][l]));
                  UInt#(32) b = unpack(pack(src2[s][l]));
                  row[l] = unpack(pack((a > b) ? a : b));
               end
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
            VPU_FMIN: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) begin
                  let cmp = compareFP(bits2fp(src1[s][l]), bits2fp(src2[s][l]));
                  row[l] = (cmp == LT || cmp == EQ) ? src1[s][l] : src2[s][l];
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
            VPU_MUL_REDUCE: begin
               Int#(32) p_val = lane_prod(src1[s]);
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = p_val;
            end
            VPU_MUL_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_prod[l];
            end
            VPU_MUL_REDUCE_TILE: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = tile_prod;
            end
            // FSUM/FMAX/FMIN_REDUCE_TILE are handled by the multi-cycle
            // FpReducer at the top of execute(). They never enter this
            // combinational case block.
            VPU_FSUM_REDUCE: begin
               // Row/lane float sum: reduce this sublane's `lanes` values to
               // one scalar via a binary addFP tree, broadcast across lanes.
               Integer n = valueOf(lanes);
               Vector#(lanes, Float) level = newVector;
               for (Integer c = 0; c < n; c = c + 1)
                  level[c] = bits2fp(src1[s][c]);
               for (Integer stride = 1; stride < n; stride = stride * 2)
                  for (Integer i = 0; i + stride < n; i = i + 2 * stride)
                     level[i] = tpl_1(addFP(level[i], level[i + stride], Rnd_Nearest_Even));
               Int#(32) bits = fp2bits(level[0]);
               for (Integer l = 0; l < n; l = l + 1) row[l] = bits;
            end
            VPU_FMAX_REDUCE: begin
               // Row/lane float max: compareFP binary tree, keep GT/EQ.
               Integer n = valueOf(lanes);
               Vector#(lanes, Float) level = newVector;
               for (Integer c = 0; c < n; c = c + 1)
                  level[c] = bits2fp(src1[s][c]);
               for (Integer stride = 1; stride < n; stride = stride * 2)
                  for (Integer i = 0; i + stride < n; i = i + 2 * stride) begin
                     let cmp = compareFP(level[i], level[i + stride]);
                     level[i] = (cmp == GT || cmp == EQ) ? level[i] : level[i + stride];
                  end
               Int#(32) bits = fp2bits(level[0]);
               for (Integer l = 0; l < n; l = l + 1) row[l] = bits;
            end
            VPU_FMIN_REDUCE: begin
               // Row/lane float min: compareFP binary tree, keep LT/EQ.
               Integer n = valueOf(lanes);
               Vector#(lanes, Float) level = newVector;
               for (Integer c = 0; c < n; c = c + 1)
                  level[c] = bits2fp(src1[s][c]);
               for (Integer stride = 1; stride < n; stride = stride * 2)
                  for (Integer i = 0; i + stride < n; i = i + 2 * stride) begin
                     let cmp = compareFP(level[i], level[i + stride]);
                     level[i] = (cmp == LT || cmp == EQ) ? level[i] : level[i + stride];
                  end
               Int#(32) bits = fp2bits(level[0]);
               for (Integer l = 0; l < n; l = l + 1) row[l] = bits;
            end
            // Float column reductions: just index the precomputed col_f*
            // vectors. The FP arithmetic happens above, unconditionally.
            VPU_FSUM_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_fsum[l];
            end
            VPU_FMAX_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_fmax[l];
            end
            VPU_FMIN_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_fmin[l];
            end
            VPU_FPROD_REDUCE: begin
               Int#(32) bits = unpack(pack(lane_fprod(src1[s])));
               for (Integer l = 0; l < valueOf(lanes); l = l + 1) row[l] = bits;
            end
            VPU_FPROD_REDUCE_COL: begin
               for (Integer l = 0; l < valueOf(lanes); l = l + 1)
                  row[l] = col_fprod[l];
            end
            // VPU_EXP2 is dispatched at the top of execute() through the
            // multi-cycle TranscUnit; this case block never sees it.
            default: noAction;
         endcase
         res[s] = row;
      end
      resultReg <= res;
      end
   endmethod

   method Vector#(sublanes, Vector#(lanes, Int#(32))) result;
      return resultReg;
   endmethod

   // True when no multi-cycle reducer or transcendental walk is in flight.
   // Single-cycle ops always observe True here on the next cycle after
   // dispatch.
   method Bool isDone = !fp_busy && !transc_busy;

endmodule

export VpuOp(..);
export VPU_IFC(..);
export mkVPU;

endpackage
