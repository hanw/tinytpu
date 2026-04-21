#!/usr/bin/env python3
"""Generate doc/architecture.html with draw.io-rendered diagrams.

Produces 4 mxGraphModel XML blocks (compilation flow, SXU FSM, timeline,
datapath), embeds them in data-mxgraph JSON, and writes a self-contained
HTML page that renders them via viewer.diagrams.net's viewer-static.min.js.
"""

import html as htmllib
import json
from pathlib import Path

# ----- cell builders -----------------------------------------------------
BASE = "rounded=1;whiteSpace=wrap;html=1;fontColor=#e6edf3;fontSize=11;"
STYLE = {
    "mxu": BASE + "fillColor=#1f2d3a;strokeColor=#58a6ff;",
    "vpu": BASE + "fillColor=#24293a;strokeColor=#d2a8ff;",
    "xlu": BASE + "fillColor=#1e2a1f;strokeColor=#7ee787;",
    "sxu": BASE + "fillColor=#2a241a;strokeColor=#f2cc60;",
    "mem": BASE + "fillColor=#2a1e22;strokeColor=#ff7b72;",
    "box": BASE + "fillColor=#1f2630;strokeColor=#30363d;",
    "ell": BASE + "fillColor=#2a241a;strokeColor=#f2cc60;ellipse=1;",
    "pill": BASE + "fillColor=#1f2630;strokeColor=#30363d;rounded=1;arcSize=40;",
}
ORTH = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;jettySize=auto;orthogonalLoop=1;"
EDGE = {
    "solid": ORTH + "endArrow=classic;strokeColor=#8b949e;strokeWidth=1.5;",
    "hot":   ORTH + "endArrow=classic;strokeColor=#7ee787;strokeWidth=2;",
    "fb":    ORTH + "endArrow=classic;strokeColor=#f2cc60;dashed=1;",
    "bg":    ORTH + "endArrow=classic;strokeColor=#7ee787;dashed=1;dashPattern=2 3;",
    "guard": ORTH + "endArrow=classic;strokeColor=#f2cc60;dashed=1;dashPattern=5 3;",
    "epi":   ORTH + "endArrow=classic;strokeColor=#d2a8ff;strokeWidth=1.2;",
    "back":  ORTH + "endArrow=classic;strokeColor=#8b949e;dashed=1;dashPattern=3 4;opacity=65;",
}

def box(cid, value, x, y, w, h, kind="box"):
    return (f'<mxCell id="{cid}" value="{htmllib.escape(value)}" style="{STYLE[kind]}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def edge(cid, src, tgt, kind="solid", label="", exit=None, entry=None):
    """Orthogonal edge with optional explicit source/target anchors.
    exit/entry are (x, y) in 0..1 (fraction of the box's bbox)."""
    lbl = f' value="{htmllib.escape(label)}"' if label else ""
    style = EDGE[kind]
    if exit is not None:
        style += f"exitX={exit[0]};exitY={exit[1]};exitDx=0;exitDy=0;"
    if entry is not None:
        style += f"entryX={entry[0]};entryY={entry[1]};entryDx=0;entryDy=0;"
    return (f'<mxCell id="{cid}"{lbl} style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{tgt}"><mxGeometry relative="1" as="geometry"/></mxCell>')

def free_edge(cid, pts, kind="solid", label=""):
    """Edge with explicit waypoints (no source/target attachment)."""
    lbl = f' value="{htmllib.escape(label)}"' if label else ""
    src = f'<mxPoint x="{pts[0][0]}" y="{pts[0][1]}" as="sourcePoint"/>'
    tgt = f'<mxPoint x="{pts[-1][0]}" y="{pts[-1][1]}" as="targetPoint"/>'
    wps = "".join(f'<mxPoint x="{x}" y="{y}"/>' for x, y in pts[1:-1])
    arr = f'<Array as="points">{wps}</Array>' if wps else ""
    return (f'<mxCell id="{cid}"{lbl} style="{EDGE[kind]}" edge="1" parent="1">'
            f'<mxGeometry relative="1" as="geometry">{src}{tgt}{arr}</mxGeometry></mxCell>')

def text(cid, value, x, y, w=120, h=18, color="#8b949e", size=10, bold=False):
    fw = "fontStyle=1;" if bold else ""
    style = (f"text;html=1;strokeColor=none;fillColor=none;align=center;"
             f"verticalAlign=middle;fontColor={color};fontSize={size};{fw}")
    return (f'<mxCell id="{cid}" value="{htmllib.escape(value)}" style="{style}" '
            f'vertex="1" parent="1"><mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def wrap(cells):
    inner = "\n    ".join(cells)
    return ('<mxGraphModel dx="1200" dy="800" grid="0" gridSize="10" guides="0" tooltips="1" connect="1" '
            'arrows="1" fold="1" page="1" pageScale="1" pageWidth="1200" pageHeight="800" math="0" shadow="0">'
            '<root><mxCell id="0"/><mxCell id="1" parent="0"/>'
            f'    {inner}'
            '</root></mxGraphModel>')

# ----- diagram 1: compilation flow --------------------------------------
def compilation_flow():
    c = []
    # 6 stage boxes
    stages = [
        ("cf-1", "Tensor program\n\nPython user code via\ntinygrad API", 20, 60, 170, 80, "mem"),
        ("cf-2", "tinygrad UOp graph\n\nADD/MUL/MAX/WHERE/EXP2/LOG2...\nmovement / reduce / load", 210, 60, 170, 80, "mem"),
        ("cf-3", "TINYTPU Renderer\n\nops_tinytpu.py\npattern-match → SXU_PROGRAM", 400, 60, 200, 80, "sxu"),
        ("cf-4", "Bundle text\n\nTASM format\ndata_plan + instrs + out", 620, 60, 180, 80, "sxu"),
        ("cf-5", "mkTbTinyTPURuntime\n\nBSV simulator\n(bluesim, co-sim BDPI)", 820, 60, 180, 80, "mxu"),
        ("cf-6", "Result\n\nmxu_result /\nvmem_result lines", 1020, 60, 160, 80, "mem"),
    ]
    for cid, val, x, y, w, h, k in stages:
        c.append(box(cid, val, x, y, w, h, k))
    # edges
    c.append(edge("cf-e1", "cf-1", "cf-2", "solid"))
    c.append(edge("cf-e2", "cf-2", "cf-3", "solid"))
    c.append(edge("cf-e3", "cf-3", "cf-4", "hot"))
    c.append(edge("cf-e4", "cf-4", "cf-5", "hot"))
    c.append(edge("cf-e5", "cf-5", "cf-6", "hot"))
    # sub-stages
    c.append(box("cf-7", "Renderer dispatch chain (~40 patterns)\nwmma → reduction → scaled-log2/exp2/sin →\ntanh → sigmoid → swish → softsign → clip → clamp → ...",
                 360, 180, 280, 66, "box"))
    c.append(box("cf-8", "TinyTPUProgram executor\n_exec_sxu_program\nemits VMEM/WMEM/AMEM lines",
                 660, 180, 200, 66, "box"))
    c.append(edge("cf-e6", "cf-7", "cf-3", "fb"))
    c.append(edge("cf-e7", "cf-8", "cf-4", "fb"))
    return wrap(c)

# ----- diagram 2: SXU FSM ----------------------------------------------
def sxu_fsm():
    c = []
    # pill states
    c.append(box("sf-idle", "IDLE", 540, 14, 120, 34, "pill"))
    c.append(box("sf-halt", "HALTED", 540, 500, 120, 34, "pill"))
    # FETCH (ellipse)
    c.append(box("sf-fetch", "FETCH\nprog[pc] → curInstr; route", 490, 70, 220, 50, "sxu"))
    # row 1 (dispatch)
    c.append(box("sf-load-req",   "LOAD_REQ\nvmem.readReq",       40,  140, 120, 40, "sxu"))
    c.append(box("sf-store",      "STORE\nvrf → vmem",            180, 140, 120, 40, "sxu"))
    c.append(box("sf-vpu",        "VPU\nvpu.execute(op)",         320, 140, 120, 40, "vpu"))
    c.append(box("sf-xlu",        "XLU_*\nbc / transpose / perm", 460, 140, 130, 40, "xlu"))
    c.append(box("sf-sel-copy",   "SELECT_COPY\nVPU_COPY rhs→reg",610, 140, 120, 40, "vpu"))
    c.append(box("sf-mxu",        "MXU / MXU_OS\nctrl.start[OS]; pc++", 750, 140, 130, 40, "mxu"))
    c.append(box("sf-wait-mxu",   "WAIT_MXU\nstall !ctrl.isDone",  900, 140, 120, 40, "mxu"))
    c.append(box("sf-ld-mxu",     "LD_MXU_RESULT\nctrl.results → vrf", 1040, 140, 140, 40, "mxu"))
    # row 2 (step 2)
    c.append(box("sf-load-resp",  "LOAD_RESP\nvrf.write ← resp",  40,  230, 120, 40, "sxu"))
    c.append(box("sf-vpu-coll",   "VPU_COLLECT\nvrf ← vpu.result",320, 230, 120, 40, "vpu"))
    c.append(box("sf-sel",        "SELECT\nVPU_SELECT(cond,lhs)", 610, 230, 120, 40, "vpu"))
    c.append(box("sf-mxu-clr",    "MXU_CLEAR\nctrl.clearArray; pc++", 750, 230, 130, 40, "mxu"))
    c.append(box("sf-ld-vpu",     "LD_VPU_RESULT\nvpu.result → vrf (bypass)", 1040, 230, 140, 40, "vpu"))
    # row 3 (collect)
    c.append(box("sf-psum", "PSUM_{WRITE, ACC, READ, READ_ROW, CLEAR, ACC_ROW}\n8-bucket bank; 1-cycle except READ (2-step)", 700, 320, 320, 44, "mxu"))
    c.append(box("sf-pred", "SET_PRED_IF_ZERO / SKIP_IF_PRED\n1-bit pred reg; baby IF/BARRIER", 40, 320, 260, 44, "sxu"))
    c.append(box("sf-ld-xlu", "LD_XLU_RESULT\nxlu.result → vrf", 1040, 320, 140, 40, "xlu"))
    # row 4 (bg rule)
    c.append(box("sf-xlu-bg", "do_xlu_collect_bg (parallel rule)\nfires when xlu_busy=True; writes\nvrf[xlu_dst] ← xlu.result; clears xlu_busy",
                 440, 410, 320, 60, "xlu"))

    # rails labels
    c.append(text("sf-l1", "dispatch",           20, 145, 80, 16))
    c.append(text("sf-l2", "step 2",             20, 235, 80, 16))
    c.append(text("sf-l3", "collect / step 3",   20, 330, 120, 16))
    c.append(text("sf-l4", "background rules",   20, 425, 120, 16))

    # entry
    c.append(edge("sf-e-start", "sf-idle", "sf-fetch", "solid", "start()",
                  exit=(0.5, 1), entry=(0.5, 0)))
    # forward dispatches from FETCH — fan out from bottom edge at different x
    # positions so the bundles don't stack. FETCH is x=[490,710], bottom y=120.
    c.append(edge("sf-ef-load",   "sf-fetch", "sf-load-req", "solid", "LOAD_VREG",
                  exit=(0.05, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-store",  "sf-fetch", "sf-store",    "solid", "STORE_VREG",
                  exit=(0.12, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-vpu",    "sf-fetch", "sf-vpu",      "solid", "DISPATCH_VPU",
                  exit=(0.25, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-xlu",    "sf-fetch", "sf-xlu",      "solid", "XLU_*",
                  exit=(0.38, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-sel",    "sf-fetch", "sf-sel-copy", "solid", "SELECT",
                  exit=(0.55, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-mxu",    "sf-fetch", "sf-mxu",      "solid", "MXU[_OS]",
                  exit=(0.70, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-wmxu",   "sf-fetch", "sf-wait-mxu", "solid", "WAIT_MXU",
                  exit=(0.85, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-ldmxu",  "sf-fetch", "sf-ld-mxu",   "solid", "LD_MXU_R",
                  exit=(0.95, 1), entry=(0.5, 0)))
    # MXU_CLEAR / PSUM / PRED: route through step-2 rail directly since they
    # share a dispatch edge in step-1 above conceptually.
    c.append(edge("sf-ef-mxuc",   "sf-fetch", "sf-mxu-clr",  "solid", "MXU_CLEAR",
                  exit=(0.75, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-psum",   "sf-fetch", "sf-psum",     "solid", "PSUM_*",
                  exit=(0.80, 1), entry=(0.5, 0)))
    c.append(edge("sf-ef-pred",   "sf-fetch", "sf-pred",     "solid", "PRED_*",
                  exit=(0.15, 1), entry=(0.5, 0)))
    # step 2 edges (within column)
    c.append(edge("sf-s1", "sf-load-req", "sf-load-resp", "solid", "next cycle",
                  exit=(0.5, 1), entry=(0.5, 0)))
    c.append(edge("sf-s2", "sf-vpu",      "sf-vpu-coll",  "guard", "vpu.isDone",
                  exit=(0.5, 1), entry=(0.5, 0)))
    c.append(edge("sf-s3", "sf-sel-copy", "sf-sel",       "solid", "next cycle",
                  exit=(0.5, 1), entry=(0.5, 0)))
    c.append(edge("sf-s4", "sf-sel",      "sf-vpu-coll", "guard", "vpu.isDone",
                  exit=(0, 0.5), entry=(1, 0.5)))
    # bg rule edge (dashed, not a state transition)
    c.append(edge("sf-s5", "sf-xlu", "sf-xlu-bg", "bg", "1-cycle later",
                  exit=(0.5, 1), entry=(0.5, 0)))
    # back edges pc++ — route from each terminal to FETCH's left or right edge
    c.append(edge("sf-b1",  "sf-store",     "sf-fetch", "back", "pc++",
                  exit=(0.5, 0), entry=(0.2, 1)))
    c.append(edge("sf-b2",  "sf-mxu",       "sf-fetch", "back", "pc++",
                  exit=(0.5, 0), entry=(0.7, 1)))
    c.append(edge("sf-b3",  "sf-xlu",       "sf-fetch", "back", "pc++",
                  exit=(0, 0.5), entry=(0.4, 1)))
    c.append(edge("sf-b4",  "sf-mxu-clr",   "sf-fetch", "back", "",
                  exit=(0.5, 0), entry=(0.75, 1)))
    c.append(edge("sf-b5",  "sf-load-resp", "sf-fetch", "back", "",
                  exit=(1, 0.5), entry=(0.1, 1)))
    c.append(edge("sf-b6",  "sf-vpu-coll",  "sf-fetch", "back", "",
                  exit=(1, 0.5), entry=(0.3, 1)))
    c.append(edge("sf-b7",  "sf-psum",      "sf-fetch", "back", "",
                  exit=(0.5, 0), entry=(0.85, 1)))
    c.append(edge("sf-b8",  "sf-pred",      "sf-fetch", "back", "",
                  exit=(1, 0.5), entry=(0.15, 1)))
    c.append(edge("sf-b9",  "sf-ld-mxu",    "sf-fetch", "back", "",
                  exit=(0, 0.5), entry=(0.9, 1)))
    c.append(edge("sf-b10", "sf-ld-vpu",    "sf-fetch", "back", "",
                  exit=(0, 0.5), entry=(0.95, 1)))
    c.append(edge("sf-b11", "sf-ld-xlu",    "sf-fetch", "back", "",
                  exit=(0, 0.5), entry=(1, 0.8)))
    c.append(edge("sf-b12", "sf-wait-mxu",  "sf-fetch", "back", "done",
                  exit=(0.5, 0), entry=(0.9, 1)))
    # halt
    c.append(edge("sf-halte", "sf-fetch", "sf-halt", "solid", "HALT",
                  exit=(0.5, 1), entry=(0.5, 0)))
    return wrap(c)

# ----- diagram 3: parallel execution timeline ----------------------------
def timeline():
    c = []
    # row labels
    rows = [("SXU", 60), ("MXU", 120), ("VPU", 180), ("XLU", 240), ("VMEM", 300)]
    for lbl, y in rows:
        c.append(text(f"tl-lbl-{lbl}", lbl, 10, y+4, 60, 18, color="#e6edf3", bold=True))
    # SXU row cells (cycles 0-17)
    sxu = [
        (0,  "FETCH", "sxu"), (1,  "EXEC_MXU", "sxu"), (2,  "FETCH", "sxu"),
        (3,  "EXEC_VPU", "vpu"), (4,  "VPU_COLL", "vpu"), (5,  "FETCH", "sxu"),
        (6,  "XLU_DSP", "xlu"), (7,  "FETCH", "sxu"),
        (8,  "EXEC_VPU", "vpu"), (9,  "VPU_COLL", "vpu"), (10, "FETCH", "sxu"),
        (11, "WAIT", "mxu"), (12, "WAIT", "mxu"), (13, "WAIT", "mxu"), (14, "WAIT", "mxu"),
        (15, "FETCH", "sxu"), (16, "LD_MXU_R", "mxu"),
    ]
    for col, lbl, kind in sxu:
        c.append(box(f"tl-sxu-{col}", lbl, 80 + col*62, 48, 60, 28, kind))
    # MXU row
    c.append(box("tl-mxu-long", "MXU pipeline (weight-stationary: stream → systolic → drain)",
                 80 + 1*62, 108, 15*62, 28, "mxu"))
    # VPU row
    c.append(box("tl-vpu-fadd", "FADD", 80 + 3*62, 168, 60, 28, "vpu"))
    c.append(box("tl-vpu-frelu", "FRELU", 80 + 8*62, 168, 60, 28, "vpu"))
    # XLU row — dispatch + bg collect
    c.append(box("tl-xlu-bc", "BROADCAST", 80 + 6*62, 228, 60, 28, "xlu"))
    c.append(box("tl-xlu-bg", "bg collect", 80 + 7*62, 228, 60, 28, "xlu"))
    # VMEM placeholder
    c.append(text("tl-vmem", "(LOAD/STORE ops not shown)", 100, 300, 400, 18))
    # Annotations
    c.append(text("tl-ann1", "MXU dispatched (non-blocking)", 80 + 1*62, 92, 230, 16, color="#f2cc60"))
    c.append(text("tl-ann2", "XLU writeback happens during FETCH", 80 + 7*62, 212, 280, 16, color="#7ee787"))
    c.append(text("tl-ann3", "MXU done signal wakes SXU", 80 + 15*62, 92, 230, 16, color="#7ee787"))
    # Cycle rule
    for col in range(18):
        c.append(text(f"tl-c-{col}", str(col), 80 + col*62 + 30, 30, 20, 14))
    return wrap(c)

# ----- diagram 4: datapath ----------------------------------------------
def datapath():
    """Left-to-right flow: storage → engines → result regs → (writeback
    returns along the very top, via a dedicated bus above everything)."""
    c = []
    # SXU bar at top
    c.append(box("dp-sxu",
                 "SXU  (single microsequencer + background XLU collect rule)\n"
                 "LOAD / STORE / DISPATCH_{VPU, XLU_*, MXU, MXU_OS, SELECT} / WAIT_MXU / LOAD_{MXU,VPU,XLU}_RESULT /\n"
                 "PSUM_{WRITE,ACC,READ,READ_ROW,CLEAR,ACC_ROW} / MXU_CLEAR / SET_PRED_IF_ZERO / SKIP_IF_PRED / HALT",
                 40, 20, 1120, 60, "sxu"))

    # Column headers
    c.append(text("dp-storhdr",  "Storage (SBUF analog)", 40, 96, 160, 18, color="#e6edf3", bold=True))
    c.append(text("dp-enghdr",   "Engines",              280, 96, 440, 18, color="#e6edf3", bold=True))
    c.append(text("dp-acchdr",   "Accumulator + result regs", 820, 96, 340, 18, color="#e6edf3", bold=True))

    # Storage column — rows aligned with engines
    ROWS = {
        "mxu":  (130, 60),   # y, h
        "vpu":  (210, 70),
        "fpr":  (300, 60),
        "xlu":  (380, 70),
    }

    c.append(box("dp-vrf",   "VRegFile\n16 × (4×4 × Int32)",                     40, ROWS["mxu"][0], 160, 50, "mem"))
    c.append(box("dp-wdb",   "WeightSRAM_DB\n2-bank · active reads, inactive writes",  40, 210, 160, 60, "mem"))
    c.append(box("dp-adb",   "ActivationSRAM_DB\n2-bank · per-row Int8",         40, 280, 160, 60, "mem"))
    c.append(box("dp-vmem",  "VMEM\nunified tile scratchpad (depth 32)",         40, 390, 160, 60, "mem"))

    # Engines column
    c.append(box("dp-mxu", "MXU (Tensor Engine)\nSystolicArray 4×4 Int8 + Controller · WS + OS",
                 280, ROWS["mxu"][0], 440, ROWS["mxu"][1], "mxu"))
    c.append(box("dp-vpu", "VPU (Vector Engine)\nint32 + float32 ALU · 55 opcodes",
                 280, ROWS["vpu"][0], 440, ROWS["vpu"][1], "vpu"))
    c.append(box("dp-fpr",  "FpReducer\nFADD + FMUL + FCMP walker",
                 280, ROWS["fpr"][0], 215, ROWS["fpr"][1], "vpu"))
    c.append(box("dp-trs",  "TranscUnit\nEXP2 / LOG2 / SIN / COS (Remez)",
                 505, ROWS["fpr"][0], 215, ROWS["fpr"][1], "vpu"))
    c.append(box("dp-xlu",  "XLU (cross-lane / GPSIMD-like)\nTRANSPOSE · BROADCAST · PERMUTE",
                 280, ROWS["xlu"][0], 440, ROWS["xlu"][1], "xlu"))

    # Result regs (one per engine row, aligned)
    c.append(box("dp-psum",   "PSUMBank\n8 buckets × 4×4 Int32",        820, ROWS["mxu"][0], 160, ROWS["mxu"][1], "mxu"))
    c.append(box("dp-mxures", "MXU result reg\nInt32 row",               1000, ROWS["mxu"][0], 160, ROWS["mxu"][1], "mxu"))
    c.append(box("dp-vpures", "VPU resultReg\n4×4 Int32 buffer",         820, ROWS["vpu"][0], 340, ROWS["vpu"][1], "vpu"))
    c.append(box("dp-fpracc", "FpReducer.acc / TranscUnit.acc\nbroadcast-to-tile on isDone", 820, ROWS["fpr"][0], 340, ROWS["fpr"][1], "vpu"))
    c.append(box("dp-xlures", "XLU resultReg\nlane-permuted tile",       820, ROWS["xlu"][0], 340, ROWS["xlu"][1], "xlu"))

    # DMA engines (between storage and off-chip) — placed below storage column
    c.append(box("dp-wdma", "WeightDMA\nbackground preload",         250, 475, 170, 60, "sxu"))
    c.append(box("dp-adma", "ActivationDMA\nbackground preload",     440, 475, 170, 60, "sxu"))

    # Off-chip / chip-level
    c.append(box("dp-hbm",  "HBMModel\nconfigurable-latency DRAM",        40, 475, 160, 60, "mem"))
    c.append(box("dp-noc",  "ChipNoC (ring)\nbase for collectives",      635, 475, 170, 60, "xlu"))
    c.append(box("dp-sc",   "SparseCore\nembedding lookup",              820, 475, 160, 60, "sxu"))
    c.append(box("dp-otc",  "Other TensorCores × (N-1)",                 1000, 475, 160, 60, "box"))

    # === Edges — all orthogonal, explicit anchors ===
    # Storage → Engines (horizontal reads) — exit right (1, 0.5), enter left (0, 0.5)
    c.append(edge("dp-r-vpu", "dp-vrf", "dp-vpu", "hot",  "vs/vs2", exit=(1, 0.5), entry=(0, 0.3)))
    c.append(edge("dp-r-fpr", "dp-vrf", "dp-fpr", "hot",  "",       exit=(1, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-r-xlu", "dp-vrf", "dp-xlu", "hot",  "",       exit=(1, 0.5), entry=(0, 0.3)))
    c.append(edge("dp-w-mxu", "dp-wdb", "dp-mxu", "hot",  "weights",     exit=(1, 0.3), entry=(0, 0.7)))
    c.append(edge("dp-a-mxu", "dp-adb", "dp-mxu", "hot",  "activations", exit=(1, 0.3), entry=(0, 0.9)))
    # VMEM ↔ VRegFile — routed along left outer edge via waypoints
    c.append(edge("dp-vl", "dp-vmem", "dp-vrf", "solid", "LOAD", exit=(0, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-vs", "dp-vrf", "dp-vmem", "solid", "STORE", exit=(0.2, 1), entry=(0.2, 0)))

    # Engines → Result regs (horizontal writes into aligned row)
    c.append(edge("dp-e-mxu",  "dp-mxu", "dp-psum",   "hot", "", exit=(1, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-p-res",  "dp-psum","dp-mxures", "hot", "", exit=(1, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-e-vpu",  "dp-vpu", "dp-vpures", "hot", "", exit=(1, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-e-fpr",  "dp-fpr", "dp-fpracc", "hot", "", exit=(1, 0.5), entry=(0, 0.3)))
    c.append(edge("dp-e-trs",  "dp-trs", "dp-fpracc", "hot", "", exit=(1, 0.5), entry=(0, 0.7)))
    c.append(edge("dp-e-xlu",  "dp-xlu", "dp-xlures", "hot", "", exit=(1, 0.5), entry=(0, 0.5)))

    # Writeback bus (yellow, dashed) — single edge per result reg, routed
    # up the right side and back across the top to VRegFile's right edge.
    # All exit from right edge and enter VRegFile's top.
    c.append(edge("dp-wb-mxu", "dp-mxures", "dp-vrf", "fb", "writeback",
                  exit=(1, 0.3), entry=(1, 0.3)))
    c.append(edge("dp-wb-vpu", "dp-vpures", "dp-vrf", "fb", "",
                  exit=(1, 0.3), entry=(0.9, 0)))
    c.append(edge("dp-wb-xlu", "dp-xlures", "dp-vrf", "fb", "",
                  exit=(1, 0.3), entry=(0.7, 0)))

    # Epilogue bypass (purple, solid) — result regs → VPU left input
    c.append(edge("dp-ep-mxu", "dp-mxures", "dp-vpu", "epi", "LOAD_MXU_RESULT",
                  exit=(0.5, 1), entry=(0.9, 0)))
    c.append(edge("dp-ep-vpu", "dp-vpures", "dp-vpu", "epi", "LOAD_VPU_RESULT",
                  exit=(0, 0.5), entry=(1, 0.3)))
    c.append(edge("dp-ep-xlu", "dp-xlures", "dp-vpu", "epi", "LOAD_XLU_RESULT",
                  exit=(0, 0.3), entry=(1, 0.7)))

    # PSUM access from SXU — routed down-then-right
    c.append(edge("dp-ps-sxu", "dp-sxu", "dp-psum", "epi", "PSUM_*",
                  exit=(0.8, 1), entry=(0.2, 0)))

    # Dispatch bus (SXU → engines) — single representative edge with a
    # label, rather than three overlapping ones.
    c.append(edge("dp-d-mxu", "dp-sxu", "dp-mxu", "solid", "dispatch",
                  exit=(0.3, 1), entry=(0.6, 0)))
    c.append(edge("dp-d-vpu", "dp-sxu", "dp-vpu", "solid", "",
                  exit=(0.4, 1), entry=(0.6, 0)))
    c.append(edge("dp-d-xlu", "dp-sxu", "dp-xlu", "solid", "",
                  exit=(0.5, 1), entry=(0.6, 0)))

    # DMA preload → inactive bank of DB SRAM
    c.append(edge("dp-dma-w", "dp-wdma", "dp-wdb", "hot", "preload",
                  exit=(0.5, 0), entry=(0.6, 1)))
    c.append(edge("dp-dma-a", "dp-adma", "dp-adb", "hot", "",
                  exit=(0.5, 0), entry=(0.6, 1)))

    # HBM feeds
    c.append(edge("dp-hb-vmem", "dp-hbm", "dp-vmem", "solid", "",
                  exit=(0.5, 0), entry=(0.5, 1)))
    c.append(edge("dp-hb-wdma", "dp-hbm", "dp-wdma", "solid", "",
                  exit=(1, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-hb-adma", "dp-hbm", "dp-adma", "solid", "",
                  exit=(1, 0.8), entry=(0, 0.8)))

    # NoC chain (bottom row)
    c.append(edge("dp-nc-1", "dp-adma", "dp-noc", "solid", "",
                  exit=(1, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-nc-2", "dp-noc",  "dp-sc",  "solid", "",
                  exit=(1, 0.5), entry=(0, 0.5)))
    c.append(edge("dp-nc-3", "dp-sc",   "dp-otc", "solid", "",
                  exit=(1, 0.5), entry=(0, 0.5)))

    return wrap(c)

# ----- build viewer div -------------------------------------------------
def viewer_div(xml, height=480, toolbar="zoom"):
    payload = {
        "highlight": "#0000ff",
        "nav": True,
        "resize": True,
        "toolbar": toolbar,
        "edit": "_blank",
        "xml": xml,
    }
    attr = htmllib.escape(json.dumps(payload))
    return (f'<div class="mxgraph" style="max-width:100%;background:#161b22;border:1px solid #30363d;'
            f'border-radius:8px;min-height:{height}px;" data-mxgraph="{attr}"></div>')

# ----- write HTML -------------------------------------------------------
HEAD = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TinyTPU — Datapath &amp; Compilation Flow</title>
<style>
  :root {
    --bg: #0f1115;
    --panel: #161b22;
    --ink: #e6edf3;
    --dim: #8b949e;
    --accent: #58a6ff;
    --accent2: #d2a8ff;
    --accent3: #7ee787;
    --accent4: #f2cc60;
    --accent5: #ff7b72;
    --line: #30363d;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    padding: 32px;
    font: 14px/1.55 ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--ink);
  }
  h1 { font-size: 24px; margin: 0 0 6px; letter-spacing: -0.01em; }
  h2 { font-size: 15px; margin: 32px 0 10px; color: var(--dim);
       text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600; }
  .sub { color: var(--dim); font-size: 13px; margin: 0 0 24px; }
  .panel { background: var(--panel); border: 1px solid var(--line);
           border-radius: 8px; padding: 24px; }
  .cap { color: var(--dim); font-size: 12px; }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  code {
    background: #11151b;
    color: var(--accent3);
    padding: 1px 5px;
    border-radius: 4px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
  }
  ul { margin: 6px 0 0 18px; padding: 0; }
  li { margin: 3px 0; }
  em { color: var(--accent2); font-style: normal; }
  .mxgraph { margin: 8px 0; }
</style>
</head>
<body>
"""

TAIL = """
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body>
</html>
"""

def main():
    out = Path(__file__).parent / "architecture.html"
    parts = [HEAD]
    parts.append('<h1>TinyTPU — Top-Level Datapath &amp; Compilation Flow</h1>')
    parts.append('<p class="sub">4×4 tensor core with shared FP reducer, multi-cycle TranscUnit, '
                 '8-bucket PSUM bank, double-buffered Weight/Activation SRAMs + DMA engines, and a '
                 'dual-issue SXU sequencer (XLU overlaps with non-XLU ops). tinygrad <code>TINYTPU</code> '
                 'backend lowers UOp graphs to SXU microprograms executed on the BSV runtime. '
                 '<em>Diagrams rendered by draw.io viewer</em> — click any diagram to zoom, pan, or export.</p>')

    # compilation flow
    parts.append('<h2>Compilation Flow</h2>')
    parts.append('<div class="panel">')
    parts.append(viewer_div(compilation_flow(), height=300))
    parts.append('</div>')

    # SXU FSM
    parts.append('<h2>TensorCore — pipeline &amp; parallel execution</h2>')
    parts.append('<div class="panel" style="margin-bottom:24px">')
    parts.append('<div class="cap" style="margin-bottom:10px">'
                 '<strong>SXU pipeline FSM.</strong> The main sequencer is single-issue with a '
                 '<em>background XLU collect rule</em>: XLU dispatches advance pc immediately '
                 '(no XLU_COLLECT state), and a parallel rule writes the result one cycle '
                 'later while the main FSM fetches/executes the next non-XLU op. MXU dispatch '
                 'is non-blocking — it latches operand bases into Controller and SXU proceeds. '
                 'Multi-cycle collectors (VPU reducers, TranscUnit walker) guard on <code>isDone</code>.'
                 '</div>')
    parts.append(viewer_div(sxu_fsm(), height=640))
    parts.append('</div>')

    # Timeline
    parts.append('<div class="panel" style="margin-bottom:24px">')
    parts.append('<div class="cap" style="margin-bottom:10px">'
                 '<strong>Parallel execution timeline.</strong> Concurrency on a kernel like '
                 '<code>MXU-dispatch; VPU-add; XLU-broadcast; VPU-relu; WAIT_MXU; LOAD_MXU_RESULT</code>. '
                 'Time flows left → right in clock cycles. The main FSM is single-issue; '
                 'MXU overlaps via dispatch+wait; XLU also overlaps via the background collect rule '
                 'so its writeback happens while SXU fetches/executes the next op.'
                 '</div>')
    parts.append(viewer_div(timeline(), height=360))
    parts.append('</div>')

    # Datapath
    parts.append('<div class="panel">')
    parts.append('<div class="cap" style="margin-bottom:10px">'
                 '<strong>Datapath — TensorCore.</strong> Storage on the left, engines in the middle, '
                 'PSUM bucket bank + per-engine result regs on the right feeding VRegFile back. '
                 'Weight/Activation SRAMs are double-buffered; WeightDMA / ActivationDMA preload '
                 'the inactive bank in parallel with an MXU dispatch draining the active bank. '
                 'TranscUnit handles EXP2/LOG2/SIN/COS via a shared multi-cycle walker.'
                 '</div>')
    parts.append(viewer_div(datapath(), height=700))
    parts.append('</div>')

    # Prose sections below
    parts.append(r"""
<h2>How TinyTPU stacks up against AWS NeuronCore</h2>
<div class="panel">
<p class="cap" style="margin-top:0">
  Both are "matrix + vector + scalar + SIMD engines reading from a shared scratchpad."
  TinyTPU started with a single shared sequencer and no PSUM bank; recent architectural
  pushes closed several structural gaps (PSUM bucket bank, dual-issue XLU, output-stationary
  dataflow mode, double-buffered SRAMs with DMA engines, a transcendental unit). The
  remaining differences are mostly scale and a sync engine.
</p>
<table style="width:100%; border-collapse:collapse; font-size:12px; margin-top:8px">
  <thead>
    <tr style="color:var(--dim); text-align:left">
      <th style="padding:6px 8px; font-weight:600; width:22%">concept</th>
      <th style="padding:6px 8px; font-weight:600; width:39%">AWS NeuronCore</th>
      <th style="padding:6px 8px; font-weight:600; width:39%">TinyTPU TensorCore</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:6px 8px; color:var(--dim)">matrix engine</td>
        <td style="padding:6px 8px">Tensor Engine (large systolic, mixed-precision)</td>
        <td style="padding:6px 8px">MXU: 4×4 <code>Int8</code> systolic, Int32 accum.
             Two dataflow modes: weight-stationary (<code>start</code>) and
             output-stationary (<code>startOS</code> + <code>clearArray</code>).</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">vector engine</td>
        <td style="padding:6px 8px">Vector Engine (wide SIMD, FP32 / BF16)</td>
        <td style="padding:6px 8px">VPU: 4×4 int32/float32 ALU; multi-cycle reducers
             via <code>FpReducer</code>; transcendentals (EXP2/LOG2/SIN/COS) via
             <code>TranscUnit</code> walker.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">scalar / reduce engine</td>
        <td style="padding:6px 8px">Scalar Engine (ops on single values, softmax-style reductions)</td>
        <td style="padding:6px 8px">fused into VPU; <code>FpReducer</code> covers
             float SUM/PROD/MAX/MIN at tile / row / col granularity.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">SIMD / custom</td>
        <td style="padding:6px 8px">GPSIMD Engine (general SIMD, custom kernels, transcendentals)</td>
        <td style="padding:6px 8px">XLU: cross-lane broadcast / transpose / permute.
             Transcendentals now in TranscUnit; general custom SIMD still a gap.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">sequencer topology</td>
        <td style="padding:6px 8px"><strong>Per-engine SEQ</strong> — each engine has its own
             instruction stream; engines run independently</td>
        <td style="padding:6px 8px"><strong>Single SXU + dual-issue XLU</strong>:
             one microprogram fans out; MXU overlaps via dispatch/wait; XLU
             overlaps via a background <code>do_xlu_collect_bg</code> rule.
             VPU still single-issue.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">shared scratchpad</td>
        <td style="padding:6px 8px">SBUF — large on-chip buffer, every engine reads/writes</td>
        <td style="padding:6px 8px">VMEM + VRegFile — same role, smaller footprint</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">accumulator</td>
        <td style="padding:6px 8px"><strong>PSUM</strong> — dedicated partial-sum buffer
             shared by Tensor/Vector/Scalar</td>
        <td style="padding:6px 8px"><strong>PSUMBank</strong> (8 buckets × 4×4 Int32)
             with row-granular write/accumulate/read access, driven by both
             Controller (per-dispatch deposit via <code>startPsum</code>) and
             SXU opcodes. Multi-K-tile GEMM epilogue runs entirely in hardware.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">operand buffers</td>
        <td style="padding:6px 8px">direct HBM→SBUF→engine</td>
        <td style="padding:6px 8px"><code>WeightSRAM_DB</code> + <code>ActivationSRAM_DB</code>:
             2-bank ping-pong, <code>WeightDMA</code> / <code>ActivationDMA</code>
             preload inactive bank in parallel with an MXU dispatch draining the
             active bank.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">sync / barrier</td>
        <td style="padding:6px 8px">dedicated <strong>Sync Engine</strong></td>
        <td style="padding:6px 8px">no sync engine yet. <code>SET_PRED_IF_ZERO</code> +
             <code>SKIP_IF_PRED</code> are the baby IF/BARRIER primitives; real
             BARRIER / IF / ENDIF still TODO.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">off-chip</td>
        <td style="padding:6px 8px">HBM ↔ SBUF (high-bandwidth)</td>
        <td style="padding:6px 8px">HBMModel ↔ VMEM &amp; WeightSRAM_DB / ActivationSRAM_DB
             via DMA stubs; no compiler-managed spill/fill yet.</td></tr>
    <tr><td style="padding:6px 8px; color:var(--dim)">multi-device</td>
        <td style="padding:6px 8px">NeuronCores × N per chip, chips via NeuronLink, collectives in hardware</td>
        <td style="padding:6px 8px">TinyTPUChip has a ring NoC + SparseCore; no
             collective primitives wired yet</td></tr>
  </tbody>
</table>
</div>

<h2>Concurrency rules — what overlaps</h2>
<div class="panel" style="padding:18px 24px">
<ul>
  <li><strong>MXU runs in parallel with SXU.</strong>
      <code>DISPATCH_MXU</code> / <code>DISPATCH_MXU_OS</code> are issue-only —
      the Controller latches operand bases / tile length and starts streaming.
      SXU advances PC immediately; it synchronizes via <code>WAIT_MXU</code>,
      <code>LOAD_MXU_RESULT</code>, or a PSUM read.</li>
  <li><strong>XLU runs in parallel with the main FSM.</strong> Each XLU
      dispatch fires <code>xlu.execute(…)</code>, sets <code>xlu_busy</code>+
      <code>xlu_dst</code>, advances pc, and returns to FETCH. The next cycle,
      <code>do_xlu_collect_bg</code> writes <code>vrf[xlu_dst]</code> while the
      main FSM fetches/executes the next non-XLU op. Structural-hazard guard
      <code>!xlu_busy</code> blocks back-to-back XLU dispatches.</li>
  <li><strong>OS-mode MXU accumulator-hold.</strong> In
      <code>DF_OUTPUT_STATIONARY</code> mode, drain skips <code>array.clearAll</code>
      so consecutive <code>startOS</code> calls accumulate in-place.
      <code>SXU_MXU_CLEAR</code> starts a fresh epoch; WS dispatches pre-clear.</li>
  <li><strong>DB SRAM preload-parallel-to-dispatch.</strong> Writes target the
      inactive bank; reads serve the active bank. DMA kick runs concurrently
      with an ongoing Controller dispatch; an outer <code>swap</code> flips
      the pointer between tiles.</li>
  <li><strong>VPU collect blocks SXU for 1 cycle</strong> (single-cycle ALU).
      Multi-cycle FP reducers + TranscUnit walker extend this to N cycles via
      <code>vpu.isDone</code> guarding <code>EXEC_VPU_COLLECT</code>.</li>
  <li><strong>PSUM ops are 1-cycle</strong> except <code>PSUM_READ</code>
      (2-step: req + resp). Row-granular variants all single-cycle.</li>
  <li><strong>VMEM read latency = 1 cycle</strong>: <code>EXEC_LOAD_REQ</code>
      then <code>EXEC_LOAD_RESP</code>; stores are single-cycle.</li>
  <li><strong>What does NOT overlap today:</strong> two VPU ops; a VPU op
      reading its own prior result (no forwarding); back-to-back XLU ops;
      VPU + FpReducer / TranscUnit (all share the VPU issue slot).</li>
</ul>
</div>

<h2>Two slices side by side</h2>
<div class="grid2">

<div class="panel">
<div class="cap" style="margin-bottom:10px"><strong>Instruction encoding</strong> — one SXU instr per row, 10 fields</div>
<ul>
  <li>25 SXU opcodes: <code>LOAD_VREG</code>(0) … <code>HALT</code>(7) …
      <code>DISPATCH_MXU</code>(4), <code>PSUM_*</code>(15–19, 22),
      <code>DISPATCH_MXU_OS</code>(23), <code>MXU_CLEAR</code>(24)</li>
  <li><code>sxu_op</code>=2 (DISPATCH_VPU) indexes into 55 float/int ALUs +
      reducers + transcendentals</li>
  <li>SXU waits on <code>vpu.isDone</code> between VPU dispatches —
      FpReducer / TranscUnit take N cycles</li>
  <li>XLU dispatches fire the bg collect rule; structural guard
      <code>!xlu_busy</code> blocks back-to-back XLU</li>
  <li>data records (VMEM/WMEM/AMEM) live in a separate section before instrs</li>
  <li>PSUM bucket idx reuses <code>vmemAddr</code>;
      <code>vregDst[1:0]</code>=row;  <code>vregSrc2[1:0]</code>=psum_mode</li>
  <li>TASM mnemonics: <code>MXU_OS</code>, <code>MXU_CLEAR</code>,
      <code>PSUM_*</code>, <code>LOAD_{VPU,XLU}_RESULT</code></li>
</ul>
</div>

<div class="panel">
<div class="cap" style="margin-bottom:10px"><strong>Reducer opcode map</strong> — which opcode handles each shape × op</div>
<table style="width:100%; border-collapse:collapse; font-size:11px">
  <thead>
    <tr style="color:var(--dim); text-align:left">
      <th style="padding:4px 6px">op</th>
      <th style="padding:4px 6px">scalar</th>
      <th style="padding:4px 6px">row (ax=1)</th>
      <th style="padding:4px 6px">col (ax=0)</th>
      <th style="padding:4px 6px">combine</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding:3px 6px;color:var(--dim)">int sum</td>
        <td style="padding:3px 6px"><code>SUM_REDUCE_TILE</code></td>
        <td style="padding:3px 6px"><code>SUM_REDUCE</code></td>
        <td style="padding:3px 6px"><code>SUM_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>ADD</code></td></tr>
    <tr><td style="padding:3px 6px;color:var(--dim)">int max</td>
        <td style="padding:3px 6px"><code>MAX_REDUCE_TILE</code></td>
        <td style="padding:3px 6px"><code>MAX_REDUCE</code></td>
        <td style="padding:3px 6px"><code>MAX_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>MAX</code></td></tr>
    <tr><td style="padding:3px 6px;color:var(--dim)">int min</td>
        <td style="padding:3px 6px"><code>MIN_REDUCE_TILE</code></td>
        <td style="padding:3px 6px"><code>MIN_REDUCE</code></td>
        <td style="padding:3px 6px"><code>MIN_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>MIN</code></td></tr>
    <tr><td style="padding:3px 6px;color:var(--dim)">int prod</td>
        <td style="padding:3px 6px"><code>MUL_REDUCE_TILE</code></td>
        <td style="padding:3px 6px"><code>MUL_REDUCE</code></td>
        <td style="padding:3px 6px"><code>MUL_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>MUL</code></td></tr>
    <tr><td style="padding:3px 6px;color:var(--dim)">float sum</td>
        <td style="padding:3px 6px"><code style="color:var(--accent2)">FSUM_REDUCE_TILE*</code></td>
        <td style="padding:3px 6px"><code>FSUM_REDUCE</code></td>
        <td style="padding:3px 6px"><code>FSUM_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>FADD</code></td></tr>
    <tr><td style="padding:3px 6px;color:var(--dim)">float max</td>
        <td style="padding:3px 6px"><code style="color:var(--accent2)">FMAX_REDUCE_TILE*</code></td>
        <td style="padding:3px 6px"><code>FMAX_REDUCE</code></td>
        <td style="padding:3px 6px"><code>FMAX_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>FMAX</code></td></tr>
    <tr><td style="padding:3px 6px;color:var(--dim)">float min</td>
        <td style="padding:3px 6px"><code style="color:var(--accent2)">FMIN_REDUCE_TILE*</code></td>
        <td style="padding:3px 6px"><code>FMIN_REDUCE</code></td>
        <td style="padding:3px 6px"><code>FMIN_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>FMIN</code></td></tr>
    <tr><td style="padding:3px 6px;color:var(--dim)">float prod</td>
        <td style="padding:3px 6px"><code style="color:var(--accent2)">FPROD_REDUCE_TILE*</code></td>
        <td style="padding:3px 6px"><code>FPROD_REDUCE</code></td>
        <td style="padding:3px 6px"><code>FPROD_REDUCE_COL</code></td>
        <td style="padding:3px 6px"><code>FMUL</code></td></tr>
  </tbody>
</table>
<div class="cap" style="margin-top:8px; font-size:10px">* = multi-cycle via shared FpReducer.
  Float SUM reductions optionally fuse a post-reduction scalar FMUL
  (e.g. <code>mean = sum × (1/N)</code>) inside the same kernel.
  Transcendental VPU ops (<code>EXP2</code>, <code>LOG2</code>,
  <code>SIN</code>, <code>COS</code>) walk multi-cycle via <code>TranscUnit</code>.</div>
</div>

</div>

<h2>Where each piece lives</h2>
<div class="panel" style="padding:18px 24px">
<ul>
  <li><code>tinygrad/tinygrad/runtime/ops_tinytpu.py</code> — Python renderer + runtime:
      UOp → SXU bundle, SXU_PROGRAM executor, sim driver. ~40 pattern-match renderers
      including self-square / self-cube, swish / silu, softsign, clip / hardtanh,
      single-bound clamp, leaky_relu, all the transcendentals, plus silent-bug guards.</li>
  <li><code>scripts/tasm.py</code> — textual bundle assembler/disassembler; opcode
      table mirrors the Python renderer and the BSV <code>SxuOpCode</code> enum.</li>
  <li><code>src/VPU.bsv</code> — 4×4 lane integer + float ALU; delegates float
      reductions to <code>FpReducer</code> and transcendentals to <code>TranscUnit</code>.</li>
  <li><code>src/FpReducer.bsv</code> — 1 FADD + 1 FMUL + 1 FCMP FSM shared by all
      float SUM / MAX / MIN / PROD reducers (tile / row / col granularity).</li>
  <li><code>src/TranscUnit.bsv</code> — multi-cycle walker implementing EXP2 / LOG2 /
      SIN / COS via Remez minimax polynomials + range reduction.</li>
  <li><code>src/ScalarUnit.bsv</code> — microprogram sequencer; 25 SXU opcodes +
      background <code>do_xlu_collect_bg</code> dual-issue rule + 1-bit predicate.</li>
  <li><code>src/TensorCore.bsv</code>, <code>src/SystolicArray.bsv</code>,
      <code>src/Controller.bsv</code> — MXU stack + per-core integration. Controller
      carries a <code>DataflowMode</code> register; <code>startOS</code> preserves
      accum; <code>clearArray</code> is the explicit reset.</li>
  <li><code>src/PSUMBank.bsv</code> — 8-bucket × 4×4 Int32 accumulator with
      row-granular access. Shared by Controller + <code>PSUM_*</code> SXU opcodes.</li>
  <li><code>src/XLU.bsv</code> — cross-lane ops (broadcast / transpose / permute).</li>
  <li><code>src/VMEM.bsv</code>, <code>src/VRegFile.bsv</code> — unified scratchpad +
      engine scratch.</li>
  <li><code>src/WeightSRAM.bsv</code>, <code>src/ActivationSRAM.bsv</code> — plain
      MXU-local operand stores; embedded as <code>.plain</code> sub-interface of
      the DB variants.</li>
  <li><code>src/WeightSRAMDB.bsv</code>, <code>src/ActivationSRAMDB.bsv</code> —
      double-buffered 2-bank ping-pong wrappers with <code>swap</code> method.</li>
  <li><code>src/WeightDMA.bsv</code>, <code>src/ActivationDMA.bsv</code> — minimal
      DMA stub state machines that stream synthetic tile patterns into the inactive
      DB bank over N cycles.</li>
  <li><code>src/ChipNoC.bsv</code>, <code>src/HBMModel.bsv</code>,
      <code>src/SparseCore.bsv</code>, <code>src/TinyTPUChip.bsv</code> — chip-level.</li>
  <li><code>test/TbTinyTPURuntime.bsv</code> + <code>bdpi/tinytpu_io.c</code> — the
      bluesim runtime that Python tests dispatch bundles to.</li>
  <li>BSV TBs: <code>TbCtrlOS</code>, <code>TbCtrlDB</code>, <code>TbCtrlDBDMA</code>,
      <code>TbWeightDMA</code>, <code>TbActivationDMA</code>, <code>TbTranscUnit</code>,
      <code>TbSxuPSUM</code>, <code>TbCtrlPSUM</code> — 95 unit tests across all
      subsystems.</li>
</ul>
</div>
""")
    parts.append(TAIL)
    out.write_text("\n".join(parts))
    print(f"wrote {out} ({len(out.read_text().splitlines())} lines)")

if __name__ == "__main__":
    main()
