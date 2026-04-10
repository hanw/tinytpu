# TinyTPU Reduction Notes

Current reduction support is intentionally narrow: a 4-element int32 vector can
be reduced to one scalar with `VPU_SUM_REDUCE`.

## Row-Wise Sum

The natural next hardware-backed step is one row sum per VMEM row. The current
VPU already computes a lane sum for each row and broadcasts that sum across the
row. The missing lowering work is selecting the first lane from each row or
storing a compact four-element result.

## Full-Tile Sum

A full 4x4 sum needs either a second reduction pass over the four row sums or a
new VPU op that reduces all sublanes into one scalar. The two-pass path is
closer to the current hardware but requires multi-instruction VPU bundles and
intermediate VMEM/VReg allocation.

## Open Semantics

- `keepdim` result layout
- non-contiguous reduction axes
- multi-tile accumulation order
- overflow behavior for int32 reductions
