package TensorCore;

import Vector :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import ScalarUnit :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import ActivationSRAM :: *;
import Controller :: *;

// TCInstr is SxuInstr; export it so testbench only needs to import TensorCore
typedef SxuInstr TCInstr;

interface TensorCore_IFC#(numeric type rows, numeric type cols, numeric type depth);
   // Pre-load weight tile into WeightSRAM at addr
   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) data);
   // Pre-load activation vector into ActivationSRAM at addr
   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) data);
   // Load one SXU microprogram instruction
   method Action loadProgram(UInt#(8) pc, TCInstr instr);
   // Pre-load/read unified VMEM tiles for VPU/XLU programs.
   method Action loadVmemTile(UInt#(TLog#(depth)) addr,
                              Vector#(rows, Vector#(cols, Int#(32))) data);
   method Action readVmemTile(UInt#(TLog#(depth)) addr);
   method Vector#(rows, Vector#(cols, Int#(32))) getVmemResult;
   method Vector#(rows, Vector#(cols, Int#(32))) peekVmemTile(UInt#(TLog#(depth)) addr);
   // Start SXU execution
   method Action start(UInt#(8) len);
   method Bool isDone;
   // MXU result vector (valid after isDone)
   method Vector#(cols, Int#(32)) getMxuResult;
endinterface

module mkTensorCore(TensorCore_IFC#(rows, cols, depth))
   provisos(
      Add#(1, r_, rows),
      Add#(1, c_, cols),
      Add#(1, d_, depth),
      Add#(0, rows, cols),               // square (rows == cols, needed for XLU)
      Log#(depth, logDepth),
      Add#(logd_, TLog#(depth), 32),
      Bits#(Vector#(rows, Vector#(cols, Int#(8))), wsz),
      Bits#(Vector#(rows, Int#(8)), asz),
      Bits#(Vector#(rows, Vector#(rows, Int#(32))), vrsz),
      Bits#(SxuInstr, isz),
      Add#(a__, TLog#(depth), 8),        // mxu*:UInt#(8) truncates to TLog#(depth)
      Add#(b__, TLog#(rows), 4)          // XLU broadcast lane selector truncates from UInt#(4)
   );

   // MXU sub-system
   SystolicArray_IFC#(rows, cols)     array <- mkSystolicArray;
   WeightSRAM_IFC#(depth, rows, cols) wsram <- mkWeightSRAM;
   ActivationSRAM_IFC#(depth, rows)   asram <- mkActivationSRAM;
   Controller_IFC#(rows, cols, depth) ctrl  <- mkController(array, wsram, asram);

   // VPU/XLU sub-system — rows is both sublanes and lanes (square)
   VMEM_IFC#(depth, rows, rows)    vmem <- mkVMEM;
   VRegFile_IFC#(16, rows, rows)   vrf  <- mkVRegFile;
   VPU_IFC#(rows, rows)            vpu  <- mkVPU;
   XLU_IFC#(rows, rows)            xlu  <- mkXLU;

   // Scalar Unit drives everything
   SXU_IFC#(256, depth, 16, rows, rows) sxu <-
      mkScalarUnit(vmem, vrf, vpu, xlu, ctrl);

   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) data);
      wsram.write(addr, data);
   endmethod

   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) data);
      asram.write(addr, data);
   endmethod

   method Action loadProgram(UInt#(8) pc, TCInstr instr);
      sxu.loadInstr(truncate(pc), instr);
   endmethod

   method Action loadVmemTile(UInt#(TLog#(depth)) addr,
                              Vector#(rows, Vector#(cols, Int#(32))) data);
      vmem.write(addr, data);
   endmethod

   method Action readVmemTile(UInt#(TLog#(depth)) addr);
      vmem.readReq(addr);
   endmethod

   method Vector#(rows, Vector#(cols, Int#(32))) getVmemResult;
      return vmem.readResp;
   endmethod

   method Vector#(rows, Vector#(cols, Int#(32))) peekVmemTile(UInt#(TLog#(depth)) addr);
      return vmem.peek(addr);
   endmethod

   method Action start(UInt#(8) len);
      sxu.start(truncate(len));
   endmethod

   method Bool isDone = sxu.isDone;

   method Vector#(cols, Int#(32)) getMxuResult;
      return ctrl.results;
   endmethod

endmodule

export TCInstr(..);
export TensorCore_IFC(..);
export mkTensorCore;

endpackage
