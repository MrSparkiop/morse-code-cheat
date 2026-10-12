"""Naive reference decoder: fixed game timing, answers only after reading one whole word.

Exists to sanity-check the simulator and benchmark; not meant to be clever.
"""

from __future__ import annotations

from collections import deque

from ..decoding import Status
from ..morse import REVERSE_MORSE, WORDS

UNIT = 0.27


class Decoder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.hist: deque[float] = deque(maxlen=2000)
        self.state: bool | None = None
        self.edge_t: float | None = None
        self.symbols = ""        # current letter
        self.letters: list[str] = []
        self.synced = False      # saw a word gap, so letters start at a word boundary
        self.answer: str | None = None
        self.last_t = None

    def feed(self, t: float, level: float | None) -> None:
        if level is None or (self.last_t is not None and t - self.last_t > 0.4):
            self.state, self.edge_t, self.synced, self.symbols, self.letters = None, None, False, "", []
            self.last_t = t
            if level is None:
                return
        self.last_t = t
        self.hist.append(level)
        lo, hi = min(self.hist), max(self.hist)
        if hi - lo < 10:
            return
        thr = (lo + hi) / 2
        on = level > thr
        if self.state is None:
            self.state = on
            return
        if on == self.state:
            return
        if self.edge_t is not None:
            d = t - self.edge_t
            if self.state:  # an on-run ended
                self.symbols += "." if d < 2 * UNIT else "-"
            else:           # an off-run ended
                if d > 7 * UNIT:
                    self._end_letter()
                    if self.synced and self.letters:
                        word = "".join(self.letters).lower()
                        if word in WORDS:
                            self.answer = word
                    self.synced, self.letters = True, []
                elif d > 2.5 * UNIT:
                    self._end_letter()
        self.state, self.edge_t = on, t

    def _end_letter(self) -> None:
        if self.symbols:
            self.letters.append(REVERSE_MORSE.get(self.symbols, "?"))
        self.symbols = ""

    def status(self) -> Status:
        return Status(answer=self.answer, confidence=1.0 if self.answer else 0.0,
                      letters="".join(self.letters), signal=self.state is not None, unit=UNIT)
