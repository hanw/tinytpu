// Concrete instantiation of mkVPU for bscwave testing.
//
// mkVPU is polymorphic in (sublanes, lanes); bscwave-gen-ports needs a
// non-polymorphic top so it can emit fixed-width Inputs/Outputs records.
// 1x4 is the smallest config that still satisfies the even-lane proviso
// required by VPU_PAIR_ROTATE / VPU_IPAIR_ROTATE.
package VPUWrapper;

import VPU :: *;

(* synthesize *)
module mkVPU_S1L4(VPU_IFC#(1, 4));
   let _vpu <- mkVPU;
   return _vpu;
endmodule

endpackage
