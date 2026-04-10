# TinyTPU Unsupported Tinyspec Areas

This manifest records known unsupported tinyspec areas so gaps are explicit
instead of rediscovered during each iteration batch.

## Elementwise

- `WHERE`: no general select lowering yet. ReLU is handled as a special case.
- Bitwise ops: `AND`, `OR`, `XOR`, shifts, and integer division/modulo are not
  implemented in the VPU.
- Chained elementwise graphs still require host round trips between lowered
  kernels.

## Movement

- `RESHAPE` / `PERMUTE` / `TRANSPOSE`: only shape preservation for already
  contiguous small elementwise outputs is covered. General movement kernels are
  unsupported.
- `EXPAND`, `PAD`, `SHRINK`, `FLIP`, `CAT`, and gather-like indexing are
  unsupported.

## Reductions

- Only 4-element int32 sum to scalar is lowered.
- Row-wise, column-wise, full-tile, keepdim, max, and multi-tile reductions are
  unsupported.

## Dtypes

- VPU paths execute int32 values and bool comparison outputs.
- MXU paths accept int32 host buffers that must fit int8 hardware operands.
- int8/uint8/int16/uint16/uint32 elementwise and float32 policy are open.

## Runtime

- VPU lowering is single VMEM tile only.
- Multiple VMEM output tiles and multiple VPU instructions in one tinygrad
  lowered program are not supported.
