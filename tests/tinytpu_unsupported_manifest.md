# TinyTPU Unsupported Tinyspec Areas

This manifest records known unsupported tinyspec areas so gaps are explicit
instead of rediscovered during each iteration batch.

## Elementwise

- `WHERE`: general compare/select fusion is still incomplete, but direct
  `WHERE`, `clip`, `abs`, and fused `ADD+RELU` now lower through multi-step
  TinyTPU programs.
- Bitwise ops: `AND`, `OR`, `XOR`, unary `NOT`, and shifts now have direct VPU
  lowering for supported elementwise shapes.
- `IDIV` lowers to `VPU_DIV`; `MOD` lowers through a hardware-backed
  `DIV -> MUL -> SUB` TinyTPU program.
- Scalar broadcasting for size-1 elementwise inputs is supported for lowered
  binary kernels and lowered VPU programs via XLU broadcast.

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
- `bool -> int32` cast is lowered.
- `TRUNC` / `RECIPROCAL` on float32 use host software fallback.
- int8/uint8/int16/uint16/uint32 elementwise and broader float32 policy are
  still open.

## Runtime

- VPU lowering handles multiple VMEM tiles by chunking and supports
  multi-instruction TinyTPU programs.
- General movement kernels and arbitrary mixed MXU+VPU programs are still not
  supported.
