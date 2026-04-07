package TensorAccelerator;

import Vector :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;

export TensorAccelerator_IFC(..);
export mkTensorAccelerator;

interface TensorAccelerator_IFC#(numeric type rows, numeric type cols, numeric type depth);
   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) wData);
   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) aData);
   method Action startCompute(UInt#(TLog#(depth)) weightBase,
                              UInt#(TLog#(depth)) actBase,
                              UInt#(TLog#(depth)) tileLen);
   method Bool computeDone;
   method Vector#(cols, Int#(32)) getOutput;
endinterface

module mkTensorAccelerator(TensorAccelerator_IFC#(rows, cols, depth))
   provisos(Add#(1, _, rows),
            Add#(1, _, cols),
            Add#(1, _, depth),
            Log#(depth, logd),
            Add#(a__, TLog#(depth), 32),
            Bits#(Vector#(rows, Vector#(cols, Int#(8))), wsz),
            Bits#(Vector#(rows, Int#(8)), asz));

   SystolicArray_IFC#(rows, cols) array <- mkSystolicArray;
   WeightSRAM_IFC#(depth, rows, cols) wSRAM <- mkWeightSRAM;
   ActivationSRAM_IFC#(depth, rows) aSRAM <- mkActivationSRAM;
   Controller_IFC#(rows, cols, depth) ctrl <- mkController(array, wSRAM, aSRAM);

   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) wData);
      wSRAM.write(addr, wData);
   endmethod

   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) aData);
      aSRAM.write(addr, aData);
   endmethod

   method Action startCompute(UInt#(TLog#(depth)) weightBase,
                              UInt#(TLog#(depth)) actBase,
                              UInt#(TLog#(depth)) tileLen);
      ctrl.start(weightBase, actBase, tileLen);
   endmethod

   method Bool computeDone;
      return ctrl.isDone;
   endmethod

   method Vector#(cols, Int#(32)) getOutput;
      return ctrl.results;
   endmethod

endmodule

endpackage
