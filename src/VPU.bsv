package VPU;

import Vector :: *;

typedef enum { VPU_ADD, VPU_MUL, VPU_RELU, VPU_MAX, VPU_SUM_REDUCE, VPU_CMPLT, VPU_CMPNE, VPU_SUB, VPU_CMPEQ }
   VpuOp deriving (Bits, Eq, FShow);

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
