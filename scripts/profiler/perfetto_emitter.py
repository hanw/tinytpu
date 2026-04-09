from __future__ import annotations
from pathlib import Path
import json
from .bundle import VPU_OP_NAMES
from .trace_parser import Event

PID = 1
THREAD_IDS = {"MAIN": 0, "SXU": 1, "MXU": 2, "VPU": 3, "VMEM": 4}


def _event_name(ev:Event) -> str:
  if ev.ev == "DISPATCH_VPU" and (op := ev.fields.get("op")) is not None:
    return f"DISPATCH_VPU({VPU_OP_NAMES.get(int(op), op)})"
  if ev.ev == "EXEC" and (op := ev.fields.get("op")) is not None:
    return f"EXEC({VPU_OP_NAMES.get(int(op), op)})"
  return ev.ev


def _coalesce(events:list[Event]) -> list[dict]:
  trace_events: list[dict] = []
  sorted_events = sorted(events, key=lambda e: (THREAD_IDS.get(e.unit, 99), e.cycle, _event_name(e)))
  cur: dict | None = None
  for ev in sorted_events:
    tid = THREAD_IDS.get(ev.unit)
    if tid is None: continue
    name = _event_name(ev)
    if cur is not None and cur["tid"] == tid and cur["name"] == name and cur["ts"] + cur["dur"] == ev.cycle:
      cur["dur"] += 1
      continue
    cur = {"ph": "X", "name": name, "cat": ev.unit, "ts": ev.cycle, "dur": 1, "pid": PID, "tid": tid}
    trace_events.append(cur)
  return trace_events

def _metadata_events() -> list[dict]:
  return [
    {"ph": "M", "pid": PID, "name": "process_name", "args": {"name": "TinyTPU Trace"}},
    {"ph": "M", "pid": PID, "tid": THREAD_IDS["MAIN"], "name": "thread_name", "args": {"name": "Main"}},
    {"ph": "M", "pid": PID, "tid": THREAD_IDS["SXU"], "name": "thread_name", "args": {"name": "SXU"}},
    {"ph": "M", "pid": PID, "tid": THREAD_IDS["MXU"], "name": "thread_name", "args": {"name": "MXU"}},
    {"ph": "M", "pid": PID, "tid": THREAD_IDS["VPU"], "name": "thread_name", "args": {"name": "VPU"}},
    {"ph": "M", "pid": PID, "tid": THREAD_IDS["VMEM"], "name": "thread_name", "args": {"name": "VMEM"}},
  ]

def _main_block(events:list[Event]) -> list[dict]:
  if not events: return []
  start = min(ev.cycle for ev in events)
  end = max(ev.cycle for ev in events) + 1
  return [{
    "ph": "X", "name": "TinyTPU Run", "cat": "MAIN",
    "ts": start, "dur": end - start, "pid": PID, "tid": THREAD_IDS["MAIN"],
  }]

def emit_perfetto(events:list[Event]) -> dict:
  return {"traceEvents": [*_metadata_events(), *_main_block(events), *_coalesce(events)]}


def write_perfetto(path:str|Path, events:list[Event]) -> None:
  Path(path).write_text(json.dumps(emit_perfetto(events), indent=2), encoding="utf-8")
