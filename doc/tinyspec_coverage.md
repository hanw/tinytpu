# TinyTPU Tinyspec Coverage

This table summarizes current tinyspec-facing coverage. The detailed estimate
and next-iteration checklist live in `TODO.md`.

| Area | Status | Notes |
| --- | --- | --- |
| GEMM / matmul | Partial | 4x4 tiled int8 hardware operands with int32 host buffers, including multi-row, deep-K, wide-N, and batched coverage. |
| VPU binary int32 | Partial | Single VMEM tile for `ADD`, `MUL`, `MAX`, `SUB`, `CMPLT`, `CMPNE`, and `CMPEQ`. |
| VPU unary int32 | Partial | ReLU and 4-element sum reduction. |
| Scalar constants | Partial | `x+c`, `x-c`, `c-x`, `x*c`, `maximum(x,c)`, `x<c`, `x!=c`, and `x==c` on one tile. |
| Bool outputs | Partial | Comparison outputs are written as tinygrad bool buffers. |
| Movement ops | Open | General reshape/permute/transpose/indexing lowering is not implemented. |
| Multi-tile elementwise | Open | Current VPU lowering rejects `numel > 16`. |
| Multi-kernel planning | Open | No intermediate VMEM allocation or chained VPU bundle scheduling. |
| Profiler | Partial | Bundle parsing, VMEM records, bundle dump, trace parsing, and Perfetto emission exist. |
