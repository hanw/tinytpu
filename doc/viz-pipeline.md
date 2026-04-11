# TinyTPU Pipeline Visualizer — Design Spec

**File:** `scripts/viz_pipeline.html`  
**Status:** Implemented  
**Depends on:** `scripts/profile_tpu.py` (optional — for real traces)

---

## 1. Goal

A single self-contained HTML file that renders a cycle-accurate Gantt chart
of TinyTPU's VLIW-style pipeline execution.  No server, no build step, no
external dependencies.

The primary audience is someone debugging a new SXU microprogram or tuning
MXU dispatch scheduling.  The chart should immediately answer:

- Which unit is active in each cycle?
- Where are the SXU stalls (`WAIT_MXU`)?
- How much of the MXU computation overlaps with the stall window?
- What instruction (PC) caused each event?

---

## 2. Pipeline model

TinyTPU's "VLIW" execution is actually a single-issue scalar SXU that dispatches
asynchronous work to three co-processors:

```
SXU ──dispatch──► VPU   (1-instruction round-trip: DISPATCH_VPU + VPU_COLLECT)
SXU ──dispatch──► MXU   (async: DISPATCH_MXU fires ctrl.start(), WAIT_MXU stalls)
SXU ──load/store─► VMEM (2-phase: READ_REQ + READ_RESP, or WRITE)
```

The visualizer shows four rows — one per unit — so the overlap between SXU
stalling and MXU executing is immediately visible.

### Event taxonomy

| Unit | Events |
|------|--------|
| SXU  | `FETCH` `LOAD_REQ` `LOAD_RESP` `STORE` `DISPATCH_VPU` `VPU_COLLECT` `DISPATCH_MXU` `WAIT_MXU` `HALT` |
| MXU  | `LOAD_W` `STREAM_A` `DRAIN` |
| VPU  | `EXEC` `RESULT` |
| VMEM | `READ_REQ` `READ_RESP` `WRITE` |

Each event maps to one color; `WAIT_MXU` gets a diagonal hatch pattern to
signal "stall, not useful work."

---

## 3. Data model

### Raw events

The internal representation is an array of per-cycle objects:

```js
{ cycle: Number, unit: "SXU"|"MXU"|"VPU"|"VMEM", ev: String, fields: Object }
```

This matches the structure parsed from BSV `TRACE cycle=N unit=U ev=E …` lines
by `scripts/profiler/trace_parser.py`.

### Coalesced blocks

Consecutive same-event cycles on the same unit are merged into a single
display block:

```js
{
  unit:       "SXU",
  ev:         "WAIT_MXU",
  startCycle: 15,
  endCycle:   21,
  count:      7,           // number of raw cycles
  fields:     { pc: 5 },  // fields from first raw event
  allFields:  { pc: Set{"5"} },  // union of all field values
}
```

Coalescing is done by `coalesce(rawEvents)` before the first render.
`WAIT_MXU` cycles 15–21 collapse into one "×7" hatched block — immediately
legible as a 7-cycle stall.

### Stats

`computeStats(rawEvents)` returns:

```js
{ totalCycles, util:{SXU,MXU,VPU,VMEM}, stallCycles, bubbles }
```

- `util[u]` — fraction of total cycles where unit `u` has at least one event
- `stallCycles` — count of `SXU WAIT_MXU` events (proxy for wasted SXU time)
- `bubbles` — cycles where **no** unit has any event (rare; MXU keeps SXU busy
  during systolic computation so this is typically 0)

---

## 4. Test case (embedded)

The HTML ships with a hard-coded 8-instruction program that exercises all four
units in 26 cycles:

```tasm
PC 0  LOAD  v0, VMEM[0]           ; SXU 3cy + VMEM 2cy
PC 1  LOAD  v1, VMEM[1]           ; SXU 3cy + VMEM 2cy
PC 2  VPU   v2 = ADD(v0, v1)      ; SXU 3cy + VPU 2cy
PC 3  VPU   v3 = RELU(v2)         ; SXU 3cy + VPU 2cy
PC 4  MXU   WMEM[0], AMEM[1], N=1 ; SXU 2cy → MXU 9cy async
PC 5  WAIT_MXU                    ; SXU 1+7 stall cy
PC 6  STORE VMEM[4], v3           ; SXU 2cy + VMEM 1cy
PC 7  HALT                        ; SXU 1cy
```

MXU timing (4×4 array, tileLen=1):

| MXU phase | Cycles | Meaning |
|-----------|--------|---------|
| `LOAD_W`  | 13–16  | load 4 weight rows, one/cycle |
| `STREAM_A`| 17–20  | stream activations: tileLen+rows−1 = 4 cycles |
| `DRAIN`   | 21     | accumulator latch |

SXU `WAIT_MXU` stall: cycles 15–21 (7 cycles).  
Key overlap: MXU `STREAM_A` (c17–c20) ⊂ SXU stall (c15–c21) — shown via the
dashed yellow arrow and red tint column.

Expected stats:

| Metric | Value |
|--------|-------|
| Total cycles | 26 |
| SXU utilization | 100% (every cycle) |
| MXU utilization | 34.6% (9 / 26) |
| VPU utilization | 15.4% (4 / 26) |
| VMEM utilization | 19.2% (5 / 26) |
| Stall cycles | 7 |
| Bubbles | 0 |

---

## 5. Rendering architecture

### Canvas renderer (`class Timeline`)

All timeline drawing uses an HTML5 Canvas element (not SVG) for
zoom/pan performance on large traces.

**DPR handling:** `canvas.width/height` are set to CSS size × `devicePixelRatio`.
`ctx.setTransform(dpr, 0, 0, dpr, 0, 0)` is called after every resize.
All measurements use CSS pixel units internally.

**Coordinate system:**

```
screenX = L + (cycle − viewStart) × pixPerCycle
screenY = T + unitIndex × ROW_H
```

- `L = 82`  — left margin for unit labels
- `T = 46`  — top margin for cycle ruler
- `ROW_H = 54`  — height of one unit row (including padding)
- `PAD = 7`  — vertical inset inside each block

**View state:**

| Variable | Meaning |
|----------|---------|
| `viewStart` | leftmost visible cycle (fractional) |
| `pixPerCycle` | zoom level in pixels per cycle |

**Zoom/pan:**

| Input | Action |
|-------|--------|
| Scroll wheel | zoom centered on cursor |
| Click-drag | pan left/right |
| `+` / `-` | zoom centered on midpoint |
| `←` / `→` | pan ±2 cycles |
| `R` | fit entire trace to window |

### Rendering layers (draw order)

1. Canvas background (`#0d0d1e`)
2. Stall zone tint (semi-transparent red column, cycles 13–21)
3. Vertical grid lines at major tick intervals
4. Horizontal row separator lines
5. Row background alternating stripes
6. **Blocks** — filled rounded rectangles, hatched for `WAIT_MXU`
7. Block labels (clipped to block width, minimum 22px to show)
8. Parallelism arrow (dashed yellow, connecting SXU stall ↔ MXU activity)
9. Cycle ruler (tick marks + labels)
10. Unit labels (left panel)
11. Hover cursor line (vertical, appears on hover)

### Block coloring

Each event type has a fixed color from `EV_COLOR`:

| Event | Color |
|-------|-------|
| FETCH | `#4A90D9` blue |
| LOAD_REQ | `#27AE60` green |
| LOAD_RESP | `#52C97F` light green |
| STORE | `#E67E22` orange |
| DISPATCH_VPU | `#8E44AD` purple |
| VPU_COLLECT | `#BA68C8` light purple |
| DISPATCH_MXU | `#C0392B` red |
| **WAIT_MXU** | `#E74C3C` hatched red (stall indicator) |
| HALT | `#546E7A` gray |
| LOAD_W | `#F39C12` amber |
| STREAM_A | `#1ABC9C` teal |
| DRAIN | `#16A085` dark teal |
| EXEC | `#E91E63` pink |
| RESULT | `#F48FB1` light pink |
| READ_REQ | `#2E7D32` dark green |
| READ_RESP | `#66BB6A` medium green |
| WRITE | `#FF5722` orange-red |

Text color is auto-selected (black/white) for contrast via luminance.

### Tooltip

On hover, a floating panel shows:
- Unit and event name
- Start cycle, duration
- The TASM assembly string for the current PC (from `PROGRAM[]`)
- Extra fields (addr, op name, etc.)
- Stall warning for `WAIT_MXU`

### Program listing panel

A fixed strip above the timeline shows all 8 instructions with:
- PC number
- TASM assembly
- Units touched
- Approximate cycle cost

On hover, the corresponding PC row is highlighted (`active` class).

---

## 6. Loading real traces

Click **Load trace.json** to ingest the Perfetto JSON emitted by:

```bash
python3 scripts/profile_tpu.py mybundle.txt --trace-out trace.json
```

The loader calls `rawFromPerfettoJSON(json)` which:
1. Iterates `traceEvents` where `ph === "X"` (duration events)
2. Maps `tid` → unit name via `{1:SXU, 2:MXU, 3:VPU, 4:VMEM}`
3. Expands coalesced Perfetto events back to per-cycle raw events
4. Calls `boot(rawEvents)` to re-render with real data

---

## 7. Extension points

### Adding a new functional unit

1. Add the unit name to `UNITS = ["SXU","MXU","VPU","VMEM","XLU"]`
2. Add its events to `EV_COLOR`
3. Add `tid=5` to `THREAD_IDS` in `perfetto_emitter.py`
4. Add a unit color to `UNIT_CLR` in `_drawUnitLabels`

### Annotating program sections

Add a `sectionLabel` field to `PROGRAM[]` entries.  Draw a vertical divider
at the boundary cycle and label it from the ruler.

### Longer traces / HBM access latency

The canvas renderer has no upper limit on `totalCycles`.  At the default zoom
level, 1000-cycle traces compress fine; the `_rulerStep()` function picks the
nearest "nice" step automatically.

### Chip-level tracing

When `TbTinyTPUChipRuntime` is available (see `profiler-design.md` §9), add
`NOC`, `HBM`, `SparseCore` units.  The HTML needs no structural changes —
only `UNITS`, `EV_COLOR`, and `THREAD_IDS` updates.

---

## 8. File structure

```
scripts/
  viz_pipeline.html      ← self-contained visualizer (this file)
  profile_tpu.py         ← generates trace.json from real simulator runs
  profiler/
    perfetto_emitter.py  ← Chrome trace JSON writer (feeds viz_pipeline.html)
    trace_parser.py      ← parses TRACE lines from traced BSV binary
```
