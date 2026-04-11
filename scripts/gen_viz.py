#!/usr/bin/env python3
"""
gen_viz.py — Regenerate viz_pipeline.html with a new embedded test case.

Runs the traced BSV simulator on a bundle, captures cycle-accurate events,
and splices them into the HTML template as new RAW_EVENTS / PROGRAM data.

Usage
─────
  # Built-in sample bundle (identity-weight GEMM + RELU):
  python3 scripts/gen_viz.py --sample

  # Any TASM source file:
  python3 scripts/gen_viz.py --tasm path/to/program.tasm

  # Pre-assembled numeric bundle (from dump_tinytpu_bundle.py, etc.):
  python3 scripts/gen_viz.py path/to/bundle.txt

  # Explicit output path (default: viz_out.html in current directory):
  python3 scripts/gen_viz.py --sample -o docs/trace_relu.html

Requires
────────
  build/mkTbTinyTPURuntimeTrace.bexe  — run `make runtime-tb-trace` once
  scripts/viz_pipeline.html           — the HTML template (shipped with the repo)

Known limitation
────────────────
  The trace binary must match the current ScalarUnit.bsv opcode enum.  If the
  binary is stale (e.g. built before SXU_DISPATCH_XLU_BROADCAST was added),
  the HALT opcode (wire value 6) maps to an undefined enum entry and the
  simulator loops until --timeout.  Fix: `make runtime-tb-trace`, then retry.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT  = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from profiler.bundle import Bundle, BundleInstr, parse_bundle_file, parse_bundle_text, write_bundle_file
from profiler.sample_program import make_sample_bundle
from profiler.trace_parser import Event, parse_trace_output

# Lazy — only imported when --tasm is used
def _tasm_assemble(src: str) -> str:
    import tasm
    return tasm.assemble(src)

# ─── opcode / field helpers ───────────────────────────────────────────────────

VPU_OP_NAMES: dict[int, str] = {
    0:"ADD",  1:"MUL",  2:"RELU", 3:"MAX",  4:"SUM_REDUCE",
    5:"CMPLT",6:"CMPNE",7:"SUB",  8:"CMPEQ",9:"MAX_REDUCE",
    10:"SHL", 11:"SHR", 12:"MIN", 13:"MIN_REDUCE",14:"DIV",
    15:"AND", 16:"OR",  17:"XOR",
}
# VPU ops that take a single source (no vregSrc2 meaningful)
_UNARY_VPU = {2, 4, 9, 13}


def _instr_asm(instr: BundleInstr) -> str:
    """Reconstruct a TASM-style assembly string from a BundleInstr."""
    op = instr.opcode
    if op == 0:
        return f"LOAD  v{instr.vreg_dst}, VMEM[{instr.vmem_addr}]"
    if op == 1:
        return f"STORE VMEM[{instr.vmem_addr}], v{instr.vreg_src}"
    if op == 2:
        vop = VPU_OP_NAMES.get(instr.vpu_op, f"OP{instr.vpu_op}")
        if instr.vpu_op in _UNARY_VPU:
            return f"VPU   v{instr.vreg_dst} = {vop}(v{instr.vreg_src})"
        return f"VPU   v{instr.vreg_dst} = {vop}(v{instr.vreg_src}, v{instr.vreg_src2})"
    if op == 3:
        lane = instr.vreg_src2
        return f"BROADCAST v{instr.vreg_dst}" + (f", lane={lane}" if lane else "")
    if op == 4:
        return f"MXU   WMEM[{instr.mxu_w_base}], AMEM[{instr.mxu_a_base}], tiles={instr.mxu_t_len}"
    if op == 5:
        return "WAIT_MXU"
    if op == 6:
        return "HALT"
    return f"OP{op}"


def _instr_units(instr: BundleInstr) -> str:
    return {
        0: "SXU·VMEM", 1: "SXU·VMEM",
        2: "SXU·VPU",  3: "SXU·XLU",
        4: "SXU→MXU",  5: "SXU·stall",
        6: "SXU",
    }.get(instr.opcode, "SXU")


# ─── JS generation ────────────────────────────────────────────────────────────

def _field_val(v: object) -> str:
    """Render a field value as JS: numeric strings become numbers, rest quoted."""
    s = str(v)
    if re.fullmatch(r"-?\d+", s):
        return s
    return f'"{s}"'


def _events_to_js(events: list[Event], cycle_offset: int) -> str:
    """Render a list of Event objects as a JS array literal (RAW_EVENTS)."""
    lines = ["["]
    for ev in events:
        cycle = ev.cycle - cycle_offset
        fields_pairs = ", ".join(f"{k}:{_field_val(v)}" for k, v in ev.fields.items())
        fields_str = "{" + fields_pairs + "}"
        lines.append(f'  {{ cycle:{cycle:4d}, unit:"{ev.unit}", ev:"{ev.ev}", fields:{fields_str} }},')
    lines.append("]")
    return "\n".join(lines)


def _program_to_js(instrs: list[BundleInstr], events: list[Event], cycle_offset: int) -> str:
    """Render the PROGRAM listing JS array from bundle instructions + trace events."""
    # Count SXU cycles per PC to produce the cyc annotation
    pc_sxu_cycles: dict[int, int] = {}
    for ev in events:
        if ev.unit == "SXU" and "pc" in ev.fields:
            try:
                pc = int(ev.fields["pc"])
                pc_sxu_cycles[pc] = pc_sxu_cycles.get(pc, 0) + 1
            except (ValueError, TypeError):
                pass

    lines = ["["]
    for i, instr in enumerate(instrs):
        asm   = _instr_asm(instr).replace('"', '\\"')
        units = _instr_units(instr)
        cyc   = str(pc_sxu_cycles.get(i, "?"))
        lines.append(f'  {{ pc:{i}, asm:"{asm}", units:"{units}", cyc:"{cyc}" }},')
    lines.append("]")
    return "\n".join(lines)


# ─── HTML template patching ───────────────────────────────────────────────────

def _find_matching_bracket(text: str, open_pos: int) -> int:
    """Return the index of the `]` that closes the `[` at open_pos."""
    depth = 0
    for i in range(open_pos, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    raise ValueError(f"Unmatched '[' at position {open_pos}")


def _replace_js_array(html: str, varname: str, new_body: str) -> str:
    """
    Replace the body of `const VARNAME = [...];` in the HTML script block.
    Uses bracket counting so inner `[` in string literals don't confuse it.
    """
    prefix = f"const {varname} = "
    idx = html.find(prefix)
    if idx < 0:
        raise ValueError(f"'const {varname} = ' not found in template HTML")
    open_bracket = html.index("[", idx + len(prefix))
    close_bracket = _find_matching_bracket(html, open_bracket)
    return html[:open_bracket] + new_body + html[close_bracket + 1:]


# ─── simulator runner ─────────────────────────────────────────────────────────

def _run_trace_sim(bundle: Bundle, timeout: int = 30) -> tuple[list[Event], int]:
    """
    Write bundle to a temp file, run the traced simulator, parse TRACE lines.
    Returns (events, cycle_offset) where cycle_offset is the first event cycle
    (subtracted before embedding so cycles start at 0 in the visualizer).
    """
    sim_path = REPO_ROOT / "build" / "mkTbTinyTPURuntimeTrace.bexe"
    if not sim_path.exists():
        sys.exit(
            f"error: traced simulator not found: {sim_path}\n"
            "Run `make runtime-tb-trace` from the repo root first."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        write_bundle_file(f.name, bundle)
        bundle_path = f.name

    try:
        proc = subprocess.run(
            [str(sim_path)],
            env={**os.environ, "TINYTPU_BUNDLE": bundle_path},
            capture_output=True, text=True, check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        sys.exit(
            f"error: trace simulator timed out after {timeout}s.\n"
            "Common cause: HALT opcode mismatch — the trace binary may be stale.\n"
            "Fix: run `make runtime-tb-trace` from the repo root, then retry."
        )
    finally:
        try: os.unlink(bundle_path)
        except FileNotFoundError: pass

    if proc.returncode not in (0, 1):   # BSV $finish(1) is a test-failure, still parse
        sys.exit(f"error: simulator exited {proc.returncode}\nstderr:\n{proc.stderr}")

    if "FAIL: timeout" in proc.stdout:
        sys.exit(
            "error: BSV testbench hit its 5000-cycle internal timeout.\n"
            "The program did not terminate.  If it uses HALT, the trace binary\n"
            "may be stale — run `make runtime-tb-trace` and retry."
        )

    events, other_lines = parse_trace_output(proc.stdout)
    if not events:
        sys.exit("error: simulator produced no TRACE events — is the binary built with -D TRACE?")

    cycle_offset = min(ev.cycle for ev in events)
    return events, cycle_offset


# ─── main entry point ─────────────────────────────────────────────────────────

def generate(
    bundle: Bundle,
    output: Path,
    template: Path = SCRIPT_DIR / "viz_pipeline.html",
    verbose: bool = True,
    timeout: int = 30,
) -> None:
    """Run simulation and splice results into the HTML template."""
    vprint = (lambda msg: print(msg, file=sys.stderr)) if verbose else (lambda _: None)

    vprint("Running traced simulator …")
    events, cycle_offset = _run_trace_sim(bundle, timeout=timeout)
    total_cycles = max(ev.cycle for ev in events) - cycle_offset + 1
    vprint(f"  {len(events)} events across {total_cycles} cycles")

    raw_js  = _events_to_js(events, cycle_offset)
    prog_js = _program_to_js(bundle.instructions, events, cycle_offset)

    vprint(f"Patching template: {template}")
    html = template.read_text(encoding="utf-8")
    html = _replace_js_array(html, "RAW_EVENTS", raw_js)
    html = _replace_js_array(html, "PROGRAM",    prog_js)

    output.write_text(html, encoding="utf-8")
    vprint(f"Written: {output}  ({output.stat().st_size // 1024} KB)")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "bundle", nargs="?",
        help="Numeric TinyTPU bundle file (wire format from dump_tinytpu_bundle.py etc.)",
    )
    src.add_argument(
        "--sample", action="store_true",
        help="Use the built-in sample bundle (identity GEMM + RELU)",
    )
    src.add_argument(
        "--tasm", metavar="FILE",
        help="Assemble a TASM source file and use it as the bundle",
    )
    p.add_argument(
        "-o", "--output", default="viz_out.html",
        help="Output HTML file path (default: viz_out.html)",
    )
    p.add_argument(
        "--template", default=str(SCRIPT_DIR / "viz_pipeline.html"),
        help="HTML template to patch (default: scripts/viz_pipeline.html)",
    )
    p.add_argument(
        "--timeout", type=int, default=30, metavar="SEC",
        help="Seconds before killing the traced simulator (default: 30)",
    )
    args = p.parse_args(argv[1:])

    if args.sample:
        bundle = make_sample_bundle()
    elif args.tasm:
        tasm_path = Path(args.tasm)
        if not tasm_path.exists():
            p.error(f"TASM file not found: {tasm_path}")
        wire_text = _tasm_assemble(tasm_path.read_text(encoding="utf-8"))
        bundle = parse_bundle_text(wire_text)
    else:
        bundle_path = Path(args.bundle)
        if not bundle_path.exists():
            p.error(f"Bundle file not found: {bundle_path}")
        bundle = parse_bundle_file(bundle_path)

    generate(bundle, Path(args.output), template=Path(args.template), timeout=args.timeout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
