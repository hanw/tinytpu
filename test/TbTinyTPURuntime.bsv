// Generic TinyTPU co-simulation testbench.
// Reads a numeric text bundle from the file pointed to by the $TINYTPU_BUNDLE
// environment variable, loads it into a TensorCore#(4,4,16), runs it, and
// prints results to stdout for the Python driver to parse.
//
// Bundle format — one record per line; comment lines start with '#':
//   0 addr w00 w01 w02 w03 w10 w11 w12 w13 w20 w21 w22 w23 w30 w31 w32 w33
//       WEIGHT_TILE: SRAM addr + 4x4 Int8 weights (row-major)
//   1 addr a0 a1 a2 a3
//       ACT_TILE: SRAM addr + 4 Int8 activations
//   2 opcode vmemAddr vregDst vregSrc vpuOp vregSrc2 mxuWBase mxuABase mxuTLen
//       INSTR: SxuOpCode (packed int) + all SxuInstr fields
//   3 flag
//       OUTPUT_MXU: 1 = emit getMxuResult line after HALT
//   4
//       END: stop loading, start execution
//   5 addr v00 v01 v02 v03 v10 ... v33
//       VMEM_TILE: SRAM addr + 4x4 Int32 values (row-major)
//   6 addr
//       OUTPUT_VMEM: emit VMEM tile at addr after HALT
//
// Output lines:
//   mxu_result v0 v1 v2 v3   (if OUTPUT_MXU 1)
//   vmem_result v00 v01 v02 v03 v10 ... v33   (if OUTPUT_VMEM is present)
//   cycles N
//   status ok

package TbTinyTPURuntime;

import Vector :: *;
import TensorCore :: *;
import ScalarUnit :: *;
import VPU :: *;

// BDPI imports — implemented in bdpi/tinytpu_io.c
import "BDPI" function ActionValue#(Int#(32)) tinytpu_bundle_open();
import "BDPI" function ActionValue#(Int#(32)) tinytpu_bundle_read_int();

typedef enum {
   TSIM_OPEN,
   TSIM_READ_TYPE,
   TSIM_WTILE,
   TSIM_ATILE,
   TSIM_INSTR,
   TSIM_OUTPUT_MXU,
   TSIM_VMEM_TILE,
   TSIM_OUTPUT_VMEM,
   TSIM_START,
   TSIM_RUNNING,
   TSIM_OUTPUT_REQ,
   TSIM_OUTPUT
} TsimState deriving(Bits, Eq, FShow);

(* synthesize *)
module mkTbTinyTPURuntime();

   TensorCore_IFC#(4, 4, 16) tc <- mkTensorCore;

   Reg#(TsimState) state  <- mkReg(TSIM_OPEN);
   Reg#(UInt#(4))  pc     <- mkReg(0);      // instruction counter
   Reg#(Bool)      outMxu <- mkReg(False);
   Reg#(Bool)      outVmem <- mkReg(False);
   Reg#(UInt#(4))  outVmemAddr <- mkReg(0);
   Reg#(UInt#(16)) cycle  <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 5000) begin
         $display("FAIL: timeout");
         $finish(1);
      end
   endrule

   // Open bundle file via BDPI
   rule do_open (state == TSIM_OPEN);
      Int#(32) rc <- tinytpu_bundle_open();
      if (rc != 0) begin
         $display("ERROR: could not open bundle (TINYTPU_BUNDLE not set or file missing)");
         $finish(1);
      end
      state <= TSIM_READ_TYPE;
   endrule

   // Read next record type (0-4) and dispatch
   rule do_read_type (state == TSIM_READ_TYPE);
      Int#(32) typ <- tinytpu_bundle_read_int();
      if      (typ == 0) state <= TSIM_WTILE;
      else if (typ == 1) state <= TSIM_ATILE;
      else if (typ == 2) state <= TSIM_INSTR;
      else if (typ == 3) state <= TSIM_OUTPUT_MXU;
      else if (typ == 4) state <= TSIM_START;
      else if (typ == 5) state <= TSIM_VMEM_TILE;
      else if (typ == 6) state <= TSIM_OUTPUT_VMEM;
      else begin
         $display("ERROR: unknown bundle record type %0d", typ);
         $finish(1);
      end
   endrule

   // Record 0: WEIGHT_TILE — addr + 16 Int8 values (4 rows × 4 cols, row-major)
   rule do_wtile (state == TSIM_WTILE);
      Int#(32) addr    <- tinytpu_bundle_read_int();
      Int#(32) w00 <- tinytpu_bundle_read_int();
      Int#(32) w01 <- tinytpu_bundle_read_int();
      Int#(32) w02 <- tinytpu_bundle_read_int();
      Int#(32) w03 <- tinytpu_bundle_read_int();
      Int#(32) w10 <- tinytpu_bundle_read_int();
      Int#(32) w11 <- tinytpu_bundle_read_int();
      Int#(32) w12 <- tinytpu_bundle_read_int();
      Int#(32) w13 <- tinytpu_bundle_read_int();
      Int#(32) w20 <- tinytpu_bundle_read_int();
      Int#(32) w21 <- tinytpu_bundle_read_int();
      Int#(32) w22 <- tinytpu_bundle_read_int();
      Int#(32) w23 <- tinytpu_bundle_read_int();
      Int#(32) w30 <- tinytpu_bundle_read_int();
      Int#(32) w31 <- tinytpu_bundle_read_int();
      Int#(32) w32 <- tinytpu_bundle_read_int();
      Int#(32) w33 <- tinytpu_bundle_read_int();
      Vector#(4, Vector#(4, Int#(8))) wtile = replicate(replicate(0));
      wtile[0][0] = unpack(truncate(pack(w00)));
      wtile[0][1] = unpack(truncate(pack(w01)));
      wtile[0][2] = unpack(truncate(pack(w02)));
      wtile[0][3] = unpack(truncate(pack(w03)));
      wtile[1][0] = unpack(truncate(pack(w10)));
      wtile[1][1] = unpack(truncate(pack(w11)));
      wtile[1][2] = unpack(truncate(pack(w12)));
      wtile[1][3] = unpack(truncate(pack(w13)));
      wtile[2][0] = unpack(truncate(pack(w20)));
      wtile[2][1] = unpack(truncate(pack(w21)));
      wtile[2][2] = unpack(truncate(pack(w22)));
      wtile[2][3] = unpack(truncate(pack(w23)));
      wtile[3][0] = unpack(truncate(pack(w30)));
      wtile[3][1] = unpack(truncate(pack(w31)));
      wtile[3][2] = unpack(truncate(pack(w32)));
      wtile[3][3] = unpack(truncate(pack(w33)));
      tc.loadWeightTile(unpack(truncate(pack(addr))), wtile);
      state <= TSIM_READ_TYPE;
   endrule

   // Record 1: ACT_TILE — addr + 4 Int8 activations
   rule do_atile (state == TSIM_ATILE);
      Int#(32) addr <- tinytpu_bundle_read_int();
      Int#(32) a0   <- tinytpu_bundle_read_int();
      Int#(32) a1   <- tinytpu_bundle_read_int();
      Int#(32) a2   <- tinytpu_bundle_read_int();
      Int#(32) a3   <- tinytpu_bundle_read_int();
      Vector#(4, Int#(8)) atile = replicate(0);
      atile[0] = unpack(truncate(pack(a0)));
      atile[1] = unpack(truncate(pack(a1)));
      atile[2] = unpack(truncate(pack(a2)));
      atile[3] = unpack(truncate(pack(a3)));
      tc.loadActivationTile(unpack(truncate(pack(addr))), atile);
      state <= TSIM_READ_TYPE;
   endrule

   // Record 2: INSTR — SxuOpCode (int) + 8 SxuInstr field values
   // Fields in order: opcode vmemAddr vregDst vregSrc vpuOp vregSrc2 mxuWBase mxuABase mxuTLen
   rule do_instr (state == TSIM_INSTR);
      Int#(32) opc      <- tinytpu_bundle_read_int();
      Int#(32) vmemAddr <- tinytpu_bundle_read_int();
      Int#(32) vregDst  <- tinytpu_bundle_read_int();
      Int#(32) vregSrc  <- tinytpu_bundle_read_int();
      Int#(32) vpuOp    <- tinytpu_bundle_read_int();
      Int#(32) vregSrc2 <- tinytpu_bundle_read_int();
      Int#(32) mxuWBase <- tinytpu_bundle_read_int();
      Int#(32) mxuABase <- tinytpu_bundle_read_int();
      Int#(32) mxuTLen  <- tinytpu_bundle_read_int();
      TCInstr instr = TCInstr {
         op:       unpack(truncate(pack(opc))),
         vmemAddr: unpack(truncate(pack(vmemAddr))),
         vregDst:  unpack(truncate(pack(vregDst))),
         vregSrc:  unpack(truncate(pack(vregSrc))),
         vpuOp:    unpack(truncate(pack(vpuOp))),
         vregSrc2: unpack(truncate(pack(vregSrc2))),
         mxuWBase: unpack(truncate(pack(mxuWBase))),
         mxuABase: unpack(truncate(pack(mxuABase))),
         mxuTLen:  unpack(truncate(pack(mxuTLen)))
      };
      tc.loadProgram(pc, instr);
      pc    <= pc + 1;
      state <= TSIM_READ_TYPE;
   endrule

   // Record 3: OUTPUT_MXU flag (1 = emit mxu_result line)
   rule do_output_mxu (state == TSIM_OUTPUT_MXU);
      Int#(32) flag <- tinytpu_bundle_read_int();
      outMxu <= (flag != 0);
      state  <= TSIM_READ_TYPE;
   endrule

   // Record 5: VMEM_TILE — addr + 16 Int32 values (4 rows × 4 cols, row-major)
   rule do_vmem_tile (state == TSIM_VMEM_TILE);
      Int#(32) addr <- tinytpu_bundle_read_int();
      Int#(32) v00 <- tinytpu_bundle_read_int();
      Int#(32) v01 <- tinytpu_bundle_read_int();
      Int#(32) v02 <- tinytpu_bundle_read_int();
      Int#(32) v03 <- tinytpu_bundle_read_int();
      Int#(32) v10 <- tinytpu_bundle_read_int();
      Int#(32) v11 <- tinytpu_bundle_read_int();
      Int#(32) v12 <- tinytpu_bundle_read_int();
      Int#(32) v13 <- tinytpu_bundle_read_int();
      Int#(32) v20 <- tinytpu_bundle_read_int();
      Int#(32) v21 <- tinytpu_bundle_read_int();
      Int#(32) v22 <- tinytpu_bundle_read_int();
      Int#(32) v23 <- tinytpu_bundle_read_int();
      Int#(32) v30 <- tinytpu_bundle_read_int();
      Int#(32) v31 <- tinytpu_bundle_read_int();
      Int#(32) v32 <- tinytpu_bundle_read_int();
      Int#(32) v33 <- tinytpu_bundle_read_int();
      Vector#(4, Vector#(4, Int#(32))) tile = replicate(replicate(0));
      tile[0][0] = v00; tile[0][1] = v01; tile[0][2] = v02; tile[0][3] = v03;
      tile[1][0] = v10; tile[1][1] = v11; tile[1][2] = v12; tile[1][3] = v13;
      tile[2][0] = v20; tile[2][1] = v21; tile[2][2] = v22; tile[2][3] = v23;
      tile[3][0] = v30; tile[3][1] = v31; tile[3][2] = v32; tile[3][3] = v33;
      tc.loadVmemTile(unpack(truncate(pack(addr))), tile);
      state <= TSIM_READ_TYPE;
   endrule

   // Record 6: OUTPUT_VMEM — addr to emit after HALT
   rule do_output_vmem (state == TSIM_OUTPUT_VMEM);
      Int#(32) addr <- tinytpu_bundle_read_int();
      outVmem <= True;
      outVmemAddr <= unpack(truncate(pack(addr)));
      state <= TSIM_READ_TYPE;
   endrule

   // Record 4: END — start TensorCore execution
   rule do_start (state == TSIM_START);
      tc.start(pc);
      state <= TSIM_RUNNING;
   endrule

   // Wait for TensorCore to finish and issue optional VMEM read before printing.
   rule do_output_req (state == TSIM_RUNNING && tc.isDone);
      if (outVmem) tc.readVmemTile(outVmemAddr);
      state <= TSIM_OUTPUT;
   endrule

   // Print results, finish simulation
   rule do_output (state == TSIM_OUTPUT);
      if (outMxu) begin
         Vector#(4, Int#(32)) res = tc.getMxuResult;
         $display("mxu_result %0d %0d %0d %0d",
                  res[0], res[1], res[2], res[3]);
      end
      if (outVmem) begin
         Vector#(4, Vector#(4, Int#(32))) tile = tc.getVmemResult;
         $display("vmem_result %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d %0d",
                  tile[0][0], tile[0][1], tile[0][2], tile[0][3],
                  tile[1][0], tile[1][1], tile[1][2], tile[1][3],
                  tile[2][0], tile[2][1], tile[2][2], tile[2][3],
                  tile[3][0], tile[3][1], tile[3][2], tile[3][3]);
      end
      $display("cycles %0d", cycle);
      $display("status ok");
      $finish(0);
   endrule

endmodule
endpackage
