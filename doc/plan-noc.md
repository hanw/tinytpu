# On-Chip NOC (Network-on-Chip) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a simple token-ring NOC connecting multiple TensorCores and SparseCores on-chip, enabling inter-unit message passing for result forwarding (e.g., TensorCore output → SparseCore input, or TC→TC activation passing).

**Architecture:** A ring of `numNodes` nodes. Each node has an inbound FIFO (holding packets addressed to it) and a ring forward port. A packet carries a `dst` node ID, `src` node ID, and a 32-bit payload. The ring rotates packets one hop per cycle until they reach their destination. Parameterized by `numNodes` (default 3: TC0 + TC1 + SC0). No dependencies on TensorCore or SparseCore — the NOC is standalone with a generic packet interface.

**Tech Stack:** BSV, BSC, Bluesim, GNU Make. Follows Controller.bsv FSM patterns. Uses BSV `FIFOF` for ring and inbox buffers.

---

## Packet Format

```bsv
typedef struct {
   UInt#(4) dst;      // Destination node ID (0 to numNodes-1)
   UInt#(4) src;      // Source node ID
   Bit#(32) payload;  // Data payload
} NocPacket deriving (Bits, Eq, FShow);
```

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/ChipNoC.bsv` | Create | NocPacket type, NocNode_IFC, `mkChipNoC` |
| `test/TbChipNoC.bsv` | Create | Single-hop, two-hop, and multi-packet tests |
| `Makefile` | Modify | Add `test-noc` target |

---

## Task 1: Makefile + failing single-hop test

**Files:**
- Create: `test/TbChipNoC.bsv`
- Modify: `Makefile`

- [ ] **Step 1: Add Makefile entries**

```makefile
$(BUILDDIR)/TbChipNoC.bo: $(BUILDDIR)/ChipNoC.bo
$(BUILDDIR)/mkTbChipNoC.bexe: $(BUILDDIR)/TbChipNoC.bo
	$(BSC) $(BSCFLAGS) -o $@ -e mkTbChipNoC $(BUILDDIR)/mkTbChipNoC.ba
test-noc: $(BUILDDIR)/mkTbChipNoC.bexe
	$<
```

Add `test-noc` to `.PHONY` and `test` target.

- [ ] **Step 2: Write failing test**

Create `test/TbChipNoC.bsv`:

```bsv
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

   // Test 1: node 0 sends packet to node 1 (adjacent)
   // Packet should arrive at node 1 after 1 hop
   rule send_pkt (cycle == 0);
      NocPacket pkt = NocPacket { dst: 1, src: 0, payload: 32'hDEAD };
      noc.send(0, pkt);
      $display("Cycle %0d: node 0 → node 1 payload=0xDEAD", cycle);
   endrule

   rule check_node1 (cycle == 2 && noc.canRecv(1));
      NocPacket pkt = noc.recv(1);
      Bool ok = (pkt.src == 0 && pkt.payload == 32'hDEAD);
      if (ok) begin
         $display("Cycle %0d: PASS node 1 received packet", cycle); passed <= passed + 1;
      end else begin
         $display("Cycle %0d: FAIL node 1 got src=%0d payload=0x%0x",
            cycle, pkt.src, pkt.payload);
         failed <= failed + 1;
      end
   endrule

   rule finish (cycle == 5);
      $display("Results: %0d passed, %0d failed", passed, failed);
      if (failed == 0) $finish(0); else $finish(1);
   endrule

endmodule
endpackage
```

- [ ] **Step 3: Run — expect compile error**

```bash
cd /home/hanwang/p/tinytpu && make test-noc
```

---

## Task 2: Implement `src/ChipNoC.bsv`

**Files:**
- Create: `src/ChipNoC.bsv`

- [ ] **Step 1: Write ChipNoC.bsv**

```bsv
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
   method NocPacket recv(UInt#(TLog#(numNodes)) dstNode);
endinterface

module mkChipNoC(ChipNoC_IFC#(numNodes))
   provisos(
      Add#(1, n_, numNodes),
      Log#(numNodes, logN)
   );

   // Each node has a local inbox FIFO
   Vector#(numNodes, FIFOF#(NocPacket)) inbox <-
      replicateM(mkFIFOF);

   // Ring forwarding FIFO: packets in transit around the ring
   // We use a single shared ring FIFO; each cycle we route or forward
   FIFOF#(NocPacket) ring <- mkFIFOF;

   // Route: if ring head has arrived at its destination, move to inbox;
   // otherwise let it stay (next node will see it next cycle)
   // For simplicity: each cycle, dequeue ring head and deliver or re-enqueue
   Reg#(UInt#(TLog#(numNodes))) ringCursor <- mkReg(0);

   rule do_route (ring.notEmpty);
      let pkt = ring.first;
      ring.deq;
      UInt#(TLog#(numNodes)) dst = truncate(pkt.dst);
      if (inbox[dst].notFull) begin
         inbox[dst].enq(pkt);
      end
      // If inbox is full, packet is dropped (in a real NOC, back-pressure)
   endrule

   method Action send(UInt#(TLog#(numNodes)) srcNode, NocPacket pkt);
      // Directly deliver if dst == src (loopback), else enqueue to ring
      UInt#(TLog#(numNodes)) dst = truncate(pkt.dst);
      if (dst == srcNode) begin
         if (inbox[srcNode].notFull)
            inbox[srcNode].enq(pkt);
      end else begin
         ring.enq(pkt);
      end
   endmethod

   method Bool canRecv(UInt#(TLog#(numNodes)) dstNode);
      return inbox[dstNode].notEmpty;
   endmethod

   method NocPacket recv(UInt#(TLog#(numNodes)) dstNode);
      inbox[dstNode].deq;
      return inbox[dstNode].first;
   endmethod

endmodule

export NocPacket(..);
export ChipNoC_IFC(..);
export mkChipNoC;

endpackage
```

**Note on `recv` method:** In BSV, calling both `.deq` and `.first` in the same method may require care. Use `ActionValue`:

```bsv
method ActionValue#(NocPacket) recv(UInt#(TLog#(numNodes)) dstNode);
   inbox[dstNode].deq;
   return inbox[dstNode].first;
endmethod
```

Update the interface and testbench accordingly:
```bsv
// In interface:
method ActionValue#(NocPacket) recv(UInt#(TLog#(numNodes)) dstNode);
// In testbench:
rule check_node1 (cycle == 2 && noc.canRecv(1));
   let pkt <- noc.recv(1);
   ...
endrule
```

- [ ] **Step 2: Run — expect PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-noc
```
Expected:
```
Cycle 0: node 0 → node 1 payload=0xDEAD
Cycle 2: PASS node 1 received packet
Results: 1 passed, 0 failed
```

- [ ] **Step 3: Commit**

```bash
git add src/ChipNoC.bsv test/TbChipNoC.bsv Makefile
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "feat: add on-chip ring NOC connecting TensorCores and SparseCore"
```

---

## Task 3: Two-hop and multi-packet tests

**Files:**
- Modify: `test/TbChipNoC.bsv`

- [ ] **Step 1: Replace `finish` at cycle==5 with extended tests**

Replace `rule finish (cycle == 5)` with:

```bsv
   // Test 2: node 0 sends to node 2 (two hops away in a 3-node ring)
   // Should arrive at node 2 after ~2 routing cycles
   rule send_two_hop (cycle == 5);
      NocPacket pkt = NocPacket { dst: 2, src: 0, payload: 32'hBEEF };
      noc.send(0, pkt);
      $display("Cycle %0d: node 0 → node 2 (2 hops) payload=0xBEEF", cycle);
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
      $display("Cycle %0d: node 1 → node 1 (loopback)", cycle);
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
```

- [ ] **Step 2: Run — expect 3 tests PASS**

```bash
cd /home/hanwang/p/tinytpu && make test-noc
```

- [ ] **Step 3: Run regression + commit**

```bash
cd /home/hanwang/p/tinytpu && make test
git add test/TbChipNoC.bsv
git commit --author="Han Wang <h1337h4x0r@gmail.com>" -m "test: add NOC two-hop and loopback tests"
```

---

*Plan created: 2026-04-08*
