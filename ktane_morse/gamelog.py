"""Optional cross-check against the game's own log file.

The game writes lines such as
    DEBUG ... [MorseCodeComponent #2] Chosen word is: MorseComponent/rule_Option18 ("bistro") (freq=552).
to <game folder>/logs/ktane.log. Reading it lets the tool show whether the
decoded answer matches what the game picked (useful to check the decoder;
the screen decoder never uses it to produce answers).
"""

from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path

CHOSEN = re.compile(r'\[MorseCodeComponent #(\d+)\] Chosen word is: .*?\("(\w+)"\)')
NEW_BOMB = re.compile(r"\[BombGenerator\] Bomb component list")

DEFAULT_DIRS = [
    r"C:\Program Files (x86)\Steam\steamapps\common\Keep Talking and Nobody Explodes",
    r"C:\Program Files\Steam\steamapps\common\Keep Talking and Nobody Explodes",
]


def find_log(game_dir: str | None = None) -> Path | None:
    for d in ([game_dir] if game_dir else []) + DEFAULT_DIRS:
        p = Path(d) / "logs" / "ktane.log"
        if p.is_file():
            return p
    return None


class GameLogWatcher(threading.Thread):
    """Follows ktane.log and keeps the Morse words chosen for the current bomb."""

    def __init__(self, path: Path) -> None:
        super().__init__(daemon=True, name="gamelog")
        self.path = path
        self.words: list[str] = []
        self._stop = threading.Event()

    def run(self) -> None:
        pos = 0
        try:
            pos = self._scan_existing()
        except OSError:
            pass
        while not self._stop.wait(0.5):
            try:
                size = os.path.getsize(self.path)
                if size < pos:  # log rotated
                    pos = 0
                if size > pos:
                    with open(self.path, "r", encoding="utf-8", errors="replace") as f:
                        f.seek(pos)
                        chunk = f.read()
                        pos = f.tell()
                    self._consume(chunk)
            except OSError:
                time.sleep(1)

    def _scan_existing(self) -> int:
        # Pick up the words of a bomb that is already running when we start.
        with open(self.path, "r", encoding="utf-8", errors="replace") as f:
            data = f.read()
            end = f.tell()
        last_bomb = None
        for m in NEW_BOMB.finditer(data):
            last_bomb = m.start()
        if last_bomb is not None:
            self._consume(data[last_bomb:])
        return end

    def _consume(self, text: str) -> None:
        for line in text.splitlines():
            if NEW_BOMB.search(line):
                self.words = []
            m = CHOSEN.search(line)
            if m:
                self.words = self.words + [m.group(2).lower()]

    def stop(self) -> None:
        self._stop.set()
