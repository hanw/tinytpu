from __future__ import annotations
from collections import Counter, defaultdict
from .bundle import Bundle
from .trace_parser import Event

UNITS = ("SXU", "MXU", "VPU", "VMEM")


def _line_value(lines:list[str], prefix:str) -> str | None:
  for line in lines:
    if line.startswith(prefix):
      return line[len(prefix):].strip()
  return None


def _total_cycles(events:list[Event], lines:list[str]) -> int:
  if (val := _line_value(lines, "cycles ")) is not None:
    return int(val)
  return max((ev.cycle for ev in events), default=-1) + 1


def _busy_cycles(events:list[Event]) -> dict[str, set[int]]:
  busy = {unit: set() for unit in UNITS}
  for ev in events:
    if ev.unit in busy: busy[ev.unit].add(ev.cycle)
  return busy


def _current_pc_by_cycle(events:list[Event]) -> dict[int, int | None]:
  sxu = sorted((ev for ev in events if ev.unit == "SXU"), key=lambda e: e.cycle)
  current_pc: int | None = None
  out: dict[int, int | None] = {}
  idx = 0
  if not sxu: return out
  for cycle in range(sxu[0].cycle, sxu[-1].cycle + 1):
    while idx < len(sxu) and sxu[idx].cycle <= cycle:
      if "pc" in sxu[idx].fields: current_pc = int(sxu[idx].fields["pc"])
      idx += 1
    out[cycle] = current_pc
  return out


def print_summary(bundle:Bundle, events:list[Event], lines:list[str]) -> None:
  total_cycles = _total_cycles(events, lines)
  busy = _busy_cycles(events)
  status = _line_value(lines, "status ") or "missing"
  mxu_result = _line_value(lines, "mxu_result ")
  print("== Summary ==")
  print(f"status: {status}")
  print(f"total_cycles: {total_cycles}")
  print(f"instruction_count: {len(bundle.instructions)}")
  if mxu_result is not None: print(f"mxu_result: {mxu_result}")
  for unit in UNITS:
    unit_busy = len(busy[unit])
    print(f"{unit}: busy={unit_busy} idle={max(total_cycles - unit_busy, 0)}")


def print_hotspots(bundle:Bundle, events:list[Event], top_n:int=5) -> None:
  pc_cycles: Counter[int] = Counter(int(ev.fields["pc"]) for ev in events if ev.unit == "SXU" and "pc" in ev.fields)
  print("\n== Hotspots ==")
  for pc, cycles in pc_cycles.most_common(top_n):
    opcode = bundle.instructions[pc].opcode_name if 0 <= pc < len(bundle.instructions) else "UNKNOWN"
    print(f"pc={pc} cycles={cycles} opcode={opcode}")


def print_instruction_mix(bundle:Bundle) -> None:
  counts = Counter(instr.opcode_name for instr in bundle.instructions)
  total = sum(counts.values()) or 1
  print("\n== Instruction Mix ==")
  for name, count in counts.most_common():
    print(f"{name}: {count} ({count*100.0/total:.1f}%)")


def print_utilization(events:list[Event], lines:list[str]) -> None:
  total_cycles = _total_cycles(events, lines)
  busy = _busy_cycles(events)
  print("\n== Utilization ==")
  busy_sum = 0
  for unit in UNITS:
    unit_busy = len(busy[unit])
    busy_sum += unit_busy
    util = 100.0 * unit_busy / total_cycles if total_cycles else 0.0
    print(f"{unit}: busy={unit_busy} idle={max(total_cycles-unit_busy, 0)} util={util:.1f}%")
  overall = 100.0 * busy_sum / (len(UNITS) * total_cycles) if total_cycles else 0.0
  print(f"overall: util={overall:.1f}%")


def print_mxu_breakdown(events:list[Event]) -> None:
  mxu = Counter(ev.ev for ev in events if ev.unit == "MXU")
  sxu_wait = sum(1 for ev in events if ev.unit == "SXU" and ev.ev == "WAIT_MXU")
  print("\n== MXU Breakdown ==")
  print(f"LOAD_W: {mxu.get('LOAD_W', 0)}")
  print(f"LOAD_W_RESP: {mxu.get('LOAD_W_RESP', 0)}")
  print(f"STREAM_A: {mxu.get('STREAM_A', 0)}")
  print(f"DRAIN: {mxu.get('DRAIN', 0)}")
  print(f"SXU_WAIT_MXU: {sxu_wait}")


def print_vpu_breakdown(events:list[Event]) -> None:
  vpu_exec = Counter(ev.fields.get("op", "unknown") for ev in events if ev.unit == "VPU" and ev.ev == "EXEC")
  print("\n== VPU Breakdown ==")
  for op, cycles in sorted(vpu_exec.items(), key=lambda item: item[0]):
    print(f"op={op}: exec_cycles={cycles}")


def print_bubbles(events:list[Event], lines:list[str], threshold:int=0) -> None:
  total_cycles = _total_cycles(events, lines)
  busy = _busy_cycles(events)
  current_pc = _current_pc_by_cycle(events)
  grouped: defaultdict[int | None, int] = defaultdict(int)
  for cycle in range(total_cycles):
    if any(cycle in busy[unit] for unit in UNITS): continue
    grouped[current_pc.get(cycle)] += 1
  print("\n== Bubbles ==")
  for pc, count in sorted(grouped.items(), key=lambda item: (-item[1], item[0] is None, item[0])):
    if count <= threshold: continue
    label = "unknown" if pc is None else f"pc={pc}"
    print(f"{label}: {count}")
