package TensorCore;

import Vector :: *;
import VMEM :: *;
import VRegFile :: *;
import VPU :: *;
import XLU :: *;
import ScalarUnit :: *;
import SystolicArray :: *;
import WeightSRAM :: *;
import WeightSRAMDB :: *;
import ActivationSRAM :: *;
import ActivationSRAMDB :: *;
import Controller :: *;
import PSUMBank :: *;

// TCInstr is SxuInstr; export it so testbench only needs to import TensorCore
typedef SxuInstr TCInstr;

interface TensorCore_IFC#(numeric type rows, numeric type cols, numeric type depth);
   // Pre-load weight tile into the ACTIVE bank of WeightSRAM — the
   // "preload then dispatch immediately" path.
   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) data);
   // Pre-load activation vector into the ACTIVE bank of ActivationSRAM.
   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) data);
   // Preload into the INACTIVE bank for DMA-overlap flows. The data is
   // only visible to the array after a swapWeightBanks() / swapActivationBanks()
   // call — use this when you want to preload tile N+1 while tile N is
   // draining through the array.
   method Action preloadWeightTile(UInt#(TLog#(depth)) addr,
                                   Vector#(rows, Vector#(cols, Int#(8))) data);
   method Action preloadActivationTile(UInt#(TLog#(depth)) addr,
                                       Vector#(rows, Int#(8)) data);
   method Action swapWeightBanks;
   method Action swapActivationBanks;
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
      Add#(b__, TLog#(rows), 4),         // XLU broadcast lane selector truncates from UInt#(4)
      Add#(c__, TLog#(rows), 32),        // OS feed cycle counter (Controller)
      Add#(d__, TLog#(rows), TLog#(depth)), // OS act-load idx fits a tile offset
      Add#(1, sl_, TMul#(rows, rows))    // FpReducer inside VPU reduces rows*rows elems
   );

   // PSUM bucket bank — 8 tile-shaped Int32 accumulators. Shared
   // between MXU (per-K-tile row accumulate via Controller) and SXU
   // (via the SXU_PSUM_{WRITE,ACCUMULATE,READ} opcodes). Declared
   // before Controller so it can be passed as a module argument.
   PSUMBank_IFC#(8, rows, rows)    psum <- mkPSUMBank;

   // MXU sub-system. Weight and Activation SRAMs are double-buffered so
   // a DMA engine (or test host) can preload tile N+1 into the inactive
   // bank while tile N drains through the array. Controller sees the
   // ACTIVE bank via the DB module's `plain` sub-interface.
   SystolicArray_IFC#(rows, cols)          array    <- mkSystolicArray;
   WeightSRAMDB_IFC#(depth, rows, cols)    wsramDB  <- mkWeightSRAMDB;
   ActivationSRAMDB_IFC#(depth, rows)      asramDB  <- mkActivationSRAMDB;
   Controller_IFC#(rows, cols, depth)      ctrl     <- mkController(array, wsramDB.plain, asramDB.plain, psum);

   // VPU/XLU sub-system — rows is both sublanes and lanes (square)
   VMEM_IFC#(depth, rows, rows)    vmem <- mkVMEM;
   VRegFile_IFC#(16, rows, rows)   vrf  <- mkVRegFile;
   VPU_IFC#(rows, rows)            vpu  <- mkVPU;
   XLU_IFC#(rows, rows)            xlu  <- mkXLU;

   // Scalar Unit drives everything
   SXU_IFC#(256, depth, 16, rows, rows) sxu <-
      mkScalarUnit(vmem, vrf, vpu, xlu, ctrl, psum);

   method Action loadWeightTile(UInt#(TLog#(depth)) addr,
                                Vector#(rows, Vector#(cols, Int#(8))) data);
      wsramDB.writeActive(addr, data);
   endmethod

   method Action loadActivationTile(UInt#(TLog#(depth)) addr,
                                    Vector#(rows, Int#(8)) data);
      asramDB.writeActive(addr, data);
   endmethod

   method Action preloadWeightTile(UInt#(TLog#(depth)) addr,
                                   Vector#(rows, Vector#(cols, Int#(8))) data);
      wsramDB.plain.write(addr, data);
   endmethod

   method Action preloadActivationTile(UInt#(TLog#(depth)) addr,
                                       Vector#(rows, Int#(8)) data);
      asramDB.plain.write(addr, data);
   endmethod

   method Action swapWeightBanks;
      wsramDB.swap;
   endmethod

   method Action swapActivationBanks;
      asramDB.swap;
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
