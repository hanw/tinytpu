# Agent Workflow

## ONNX -> tinygrad -> TinyTPU

For model bring-up and backend expansion, use this process:

1. Start from an ONNX model.
2. Compile it through tinygrad targeting `TINYTPU`.
3. Inspect what fails in the TinyTPU renderer/lowering path.
4. Classify the gap:
   - If it is a software/compiler problem, fix it in the tinygrad TinyTPU renderer/runtime path.
   - If it is a hardware/instruction-set problem, create an issue describing the missing TinyTPU capability and then fix it in TinyTPU.

## Triage Rule

The default assumption is:

- First treat ONNX bring-up failures as a renderer/lowering problem.
- Only escalate to a TinyTPU hardware issue when the required behavior cannot be expressed with the current instruction set or data paths.

## Expected Output

For each investigated ONNX model or kernel:

- Record what tinygrad produced.
- Record what the TinyTPU renderer could and could not lower.
- Name the missing instruction sequence or hardware capability explicitly.
- Route the fix to either tinygrad software or TinyTPU hardware based on that analysis.
