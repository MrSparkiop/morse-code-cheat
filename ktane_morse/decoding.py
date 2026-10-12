"""The interface every Morse decoder implements.

The capture side calls ``feed(t, level)`` once per captured frame, where
``t`` is a monotonic timestamp in seconds (time.perf_counter()) and
``level`` is one brightness number for the watched light (higher = brighter),
or None when the frame is known to be unusable (camera moving, capture
failed, ...). It calls ``status()`` whenever it wants to refresh the display.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Status:
    answer: str | None = None      # committed word, once the decoder is sure; else None
    confidence: float = 0.0        # 0..1, belief in ranking[0]
    ranking: list[tuple[str, float]] = field(default_factory=list)  # best candidates first, (word, probability)
    morse: str = ""                # decoded stream for display, e.g. "... - .-. / --- -..." ("/" = word gap, "?" = unreadable)
    letters: str = ""              # same stream as letters, e.g. "ST R/OB" ("?" for unknown)
    signal: bool = False           # a blinking light is currently being seen
    unit: float | None = None      # estimated dot length in seconds


class Decoder(Protocol):
    def feed(self, t: float, level: float | None) -> None: ...
    def status(self) -> Status: ...
    def reset(self) -> None: ...
