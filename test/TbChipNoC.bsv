package TbChipNoC;

import Vector :: *;
import ChipNoC :: *;

(* synthesize *)
module mkTbChipNoC();

   // 3-node ring: node 0, node 1, node 2
   ChipNoC_IFC#(3) noc <- mkChipNoC;

   Reg#(UInt#(8)) cycle  <- mkReg(0);
   Reg#(UInt#(8)) passed <- mkReg(0);
   Reg#(UInt#(8)) failed <- mkReg(0);

   rule count_cycles;
      cycle <= cycle + 1;
      if (cycle > 50) begin $display("FAIL: timeout"); $finish(1); end
   endrule

   // Test 1: node 0 sends to node 1 (single hop)
   rule send_pkt (cycle == 0);
      NocPacket pkt = NocPacket { dst: 1, src: 0, payload: 32'hDEAD };
      noc.send(0, pkt);
      $display("Cycle %0d: node 0 -> node 1 payload=0xDEAD", cycle);
   endrule

   rule check_node1 (cycle == 2 && noc.canRecv(1));
      let pkt <- noc.recv(1);
      Bool ok = (pkt.src == 0 && pkt.payload == 32'hDEAD);
      if (ok) begin
         $display("Cycle %0d: PASS node 1 received packet", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL node 1 got src=%0d payload=0x%0x",
            cycle, pkt.src, pkt.payload);
         failed <= failed + 1;
      end
   endrule

   // Test 2: node 0 sends to node 2 (two hops in ring)
   rule send_two_hop (cycle == 5);
      NocPacket pkt = NocPacket { dst: 2, src: 0, payload: 32'hBEEF };
      noc.send(0, pkt);
      $display("Cycle %0d: node 0 -> node 2 payload=0xBEEF", cycle);
   endrule

   rule check_node2 (cycle >= 7 && noc.canRecv(2));
      let pkt <- noc.recv(2);
      Bool ok = (pkt.src == 0 && pkt.payload == 32'hBEEF);
      if (ok) begin
         $display("Cycle %0d: PASS node 2 two-hop delivery", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL node 2 src=%0d payload=0x%0x",
            cycle, pkt.src, pkt.payload);
         failed <= failed + 1;
      end
   endrule

   // Test 3: loopback — node 1 sends to itself
   rule send_loopback (cycle == 8);
      NocPacket pkt = NocPacket { dst: 1, src: 1, payload: 32'hCAFE };
      noc.send(1, pkt);
      $display("Cycle %0d: node 1 -> node 1 (loopback)", cycle);
   endrule

   rule check_loopback (cycle == 9 && noc.canRecv(1));
      let pkt <- noc.recv(1);
      Bool ok = (pkt.src == 1 && pkt.payload == 32'hCAFE);
      if (ok) begin
         $display("Cycle %0d: PASS loopback", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL loopback payload=0x%0x", cycle, pkt.payload);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 15);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
