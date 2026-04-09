package ChipNoC;

import Vector :: *;
import FIFOF :: *;

typedef struct {
   UInt#(4) dst;
   UInt#(4) src;
   Bit#(32) payload;
} NocPacket deriving (Bits, Eq, FShow);

interface ChipNoC_IFC#(numeric type numNodes);
   method Action send(UInt#(TLog#(numNodes)) srcNode, NocPacket pkt);
   method Bool canRecv(UInt#(TLog#(numNodes)) dstNode);
   method ActionValue#(NocPacket) recv(UInt#(TLog#(numNodes)) dstNode);
endinterface

module mkChipNoC(ChipNoC_IFC#(numNodes))
   provisos(
      Add#(1, n_, numNodes),
      Add#(a__, TLog#(numNodes), 4)   // pkt.dst:UInt#(4) truncates to TLog#(numNodes)
   );

   // Each node has a local inbox FIFO for received packets
   Vector#(numNodes, FIFOF#(NocPacket)) inbox <- replicateM(mkFIFOF);

   // Ring FIFO: packets in transit between nodes
   FIFOF#(NocPacket) ring <- mkFIFOF;

   // Route ring head: deliver to inbox if destination matches
   rule do_route (ring.notEmpty);
      let pkt = ring.first;
      ring.deq;
      UInt#(TLog#(numNodes)) dst = truncate(pkt.dst);
      if (inbox[dst].notFull)
         inbox[dst].enq(pkt);
      // Packet dropped if inbox full (back-pressure not modeled)
   endrule

   method Action send(UInt#(TLog#(numNodes)) srcNode, NocPacket pkt);
      UInt#(TLog#(numNodes)) dst = truncate(pkt.dst);
      if (dst == srcNode) begin
         // Loopback: deliver directly to inbox
         if (inbox[srcNode].notFull)
            inbox[srcNode].enq(pkt);
      end else begin
         ring.enq(pkt);
      end
   endmethod

   method Bool canRecv(UInt#(TLog#(numNodes)) dstNode);
      return inbox[dstNode].notEmpty;
   endmethod

   method ActionValue#(NocPacket) recv(UInt#(TLog#(numNodes)) dstNode);
      let pkt = inbox[dstNode].first;
      inbox[dstNode].deq;
      return pkt;
   endmethod

endmodule

export NocPacket(..);
export ChipNoC_IFC(..);
export mkChipNoC;

endpackage
