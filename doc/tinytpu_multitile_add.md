# TinyTPU Multi-Tile ADD Plan

Current VPU elementwise lowering handles one 4x4 VMEM tile, so `numel > 16`
is rejected. A first multi-tile `ADD` implementation should keep the same
single-op semantics and only loop over tile chunks.

## Proposed Steps

1. Split flat int32 buffers into 16-element chunks.
2. Build one VMEM/VPU instruction sequence per chunk.
3. Assign distinct VMEM input/output addresses per chunk or run chunks as
   separate runtime bundles.
4. Copy partial output chunks back into the tinygrad output buffer in order.
5. Add a tail policy for the final partial tile.

The separate-runtime-bundle approach is simpler and matches the current
host-driven GEMM loop. The single-bundle approach is closer to hardware intent
but needs multiple VMEM output records or a richer output protocol.
