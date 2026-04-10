from __future__ import annotations
from dataclasses import dataclass, field
import re

TRACE_RE = re.compile(r"^TRACE cycle=(?P<cycle>\d+) unit=(?P<unit>\w+) ev=(?P<ev>[A-Z_]+)(?P<rest>.*)$")


@dataclass(frozen=True)
class Event:
  cycle: int
  unit: str
  ev: str
  fields: dict[str, str] = field(default_factory=dict)


def parse_trace_output(stdout:str) -> tuple[list[Event], list[str]]:
  events: list[Event] = []
  other: list[str] = []
  for lineno, raw in enumerate(stdout.splitlines(), start=1):
    line = raw.strip()
    if not line: continue
    if (m := TRACE_RE.match(line)) is None:
      if line.startswith("TRACE "):
        raise ValueError(f"line {lineno}: malformed TRACE line")
      other.append(line)
      continue
    fields: dict[str, str] = {}
    rest = m.group("rest").strip()
    if rest:
      for tok in rest.split():
        if "=" not in tok:
          raise ValueError(f"line {lineno}: malformed TRACE field {tok!r}")
        k, v = tok.split("=", 1)
        if not v:
          raise ValueError(f"line {lineno}: empty TRACE field value for {k}")
        fields[k] = v
    events.append(Event(cycle=int(m.group("cycle")), unit=m.group("unit"), ev=m.group("ev"), fields=fields))
  return events, other
