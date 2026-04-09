package TinyTPUChip;

import Vector :: *;
import TensorCore :: *;
import SparseCore :: *;
import ChipNoC :: *;
import ScalarUnit :: *;
import VPU :: *;

interface TinyTPUChip_IFC;
   // TC0 setup
   method Action loadTC0Weights(UInt#(4) addr,
                                Vector#(4, Vector#(4, Int#(8))) data);
   method Action loadTC0Activations(UInt#(4) addr,
                                    Vector#(4, Int#(8)) data);
   method Action loadTC0Program(UInt#(4) pc, TCInstr instr);
   method Action startTC0(UInt#(4) len);
   method Bool tc0Done;
   method Vector#(4, Int#(32)) getTC0Result;

   // Forward TC0 result lane 'laneIdx' (0..3) as sparse index to SparseCore
   method Action forwardTC0ResultToSC(UInt#(2) laneIdx);

   // SparseCore setup
   method Action loadSCEmbedding(UInt#(5) idx, Vector#(4, Int#(32)) emb);
   method Bool scDone;
   method Vector#(4, Int#(32)) getSCResult;
endinterface

module mkTinyTPUChip(TinyTPUChip_IFC);

   // TC0: 4x4 matrix multiply, depth=16 VMEM/SRAM
   TensorCore_IFC#(4, 4, 16) tc0 <- mkTensorCore;

   // SparseCore: 32-entry table, embWidth=4, bagSize=1
   SparseCore_IFC#(32, 4, 1)  sc <- mkSparseCore;

   // NOC (instantiated but used implicitly via forwardTC0ResultToSC)
   ChipNoC_IFC#(2)            noc <- mkChipNoC;

   Reg#(Bool) scSubmitted <- mkReg(False);

   method Action loadTC0Weights(UInt#(4) addr,
                                Vector#(4, Vector#(4, Int#(8))) data);
      tc0.loadWeightTile(extend(addr), data);
   endmethod

   method Action loadTC0Activations(UInt#(4) addr,
                                    Vector#(4, Int#(8)) data);
      tc0.loadActivationTile(extend(addr), data);
   endmethod

   method Action loadTC0Program(UInt#(4) pc, TCInstr instr);
      tc0.loadProgram(pc, instr);
   endmethod

   method Action startTC0(UInt#(4) len);
      tc0.start(len);
   endmethod

   method Bool tc0Done = tc0.isDone;

   method Vector#(4, Int#(32)) getTC0Result;
      return tc0.getMxuResult;
   endmethod

   method Action forwardTC0ResultToSC(UInt#(2) laneIdx) if (!scSubmitted);
      Vector#(4, Int#(32)) res = tc0.getMxuResult;
      UInt#(5) idx = unpack(truncate(pack(res[laneIdx])));
      Vector#(1, UInt#(5)) bag = replicate(idx);
      sc.submitBag(bag, 1);
      scSubmitted <= True;
   endmethod

   method Action loadSCEmbedding(UInt#(5) idx, Vector#(4, Int#(32)) emb);
      sc.loadEmbedding(idx, emb);
   endmethod

   method Bool scDone = sc.isDone;

   method Vector#(4, Int#(32)) getSCResult;
      return sc.result;
   endmethod

endmodule

export TinyTPUChip_IFC(..);
export mkTinyTPUChip;

endpackage
