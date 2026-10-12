"""A stand-in for the game's Morse module, for trying the tool without playing.

    python -m ktane_morse.fake_module                 # random word
    python -m ktane_morse.fake_module --word strobe --size 80

Opens a window with an amber lamp that flashes a word with the game's exact
timing (0.27 s unit, 1/3 on, 1/4/10 off). The word is printed to the console
so you can check the tool's answer.
"""

from __future__ import annotations

import argparse
import ctypes
import random
import sys
import time
import tkinter as tk

from .capture import enable_dpi_awareness
from .morse import WORDS
from .simulate import GAME_TIMING, word_schedule

UNLIT, LIT = "#753e29", "#fbde47"
GLOW_OFF, GLOW_ON = "#3b3a33", "#8c7a2a"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--word", choices=sorted(WORDS), help="word to flash (default: random)")
    ap.add_argument("--size", type=int, default=60, help="lamp diameter in pixels (default 60)")
    ap.add_argument("--x", type=int, default=200)
    ap.add_argument("--y", type=int, default=200)
    a = ap.parse_args(argv)

    enable_dpi_awareness()
    if sys.platform == "win32":
        ctypes.windll.winmm.timeBeginPeriod(1)  # 1 ms timer resolution for accurate flashes
    word = a.word or random.choice(list(WORDS))
    print(f"Flashing '{word}' (answer {WORDS[word]} MHz). Close the window to stop.", flush=True)

    s = a.size
    W, H = int(s * 6), int(s * 6)
    root = tk.Tk()
    root.title("Fake Morse Module")
    root.geometry(f"{W}x{H}+{a.x}+{a.y}")
    root.configure(bg="#26272b")
    root.attributes("-topmost", True)
    c = tk.Canvas(root, width=W, height=H, bg="#26272b", highlightthickness=0)
    c.pack(fill="both", expand=True)
    m = s // 3
    c.create_rectangle(m, m, W - m, H - m, fill="#5d6064", outline="#3a3c40", width=3)
    cx, cy = int(W * 0.35), int(H * 0.2)
    glow = c.create_oval(cx - s, cy - s * 0.8, cx + s, cy + s * 0.8, fill=GLOW_OFF, outline="")
    lamp = c.create_oval(cx - s / 2, cy - s * 0.38, cx + s / 2, cy + s * 0.38, fill=UNLIT, outline="#2a2a2a", width=2)
    c.create_oval(W * 0.82, H * 0.13, W * 0.82 + s * 0.3, H * 0.13 + s * 0.3, fill="#222", outline="#111")  # status LED
    c.create_rectangle(W * 0.25, H * 0.45, W * 0.75, H * 0.58, fill="#111", outline="#000")
    c.create_text(W / 2, H * 0.515, text="3.505 MHz", fill="#e04020", font=("Consolas", max(8, s // 4), "bold"))
    c.create_rectangle(W * 0.4, H * 0.7, W * 0.6, H * 0.8, fill="#c8c8c8", outline="#333")
    c.create_text(W / 2, H * 0.75, text="TX", font=("Segoe UI", max(8, s // 5), "bold"))

    sched = word_schedule(word, GAME_TIMING)
    state = {"i": 0, "deadline": time.perf_counter() + 1.0}

    def step():
        on, d = sched[state["i"]]
        c.itemconfigure(lamp, fill=LIT if on else UNLIT)
        c.itemconfigure(glow, fill=GLOW_ON if on else GLOW_OFF)
        state["deadline"] += d
        state["i"] = (state["i"] + 1) % len(sched)
        root.after(max(1, int((state["deadline"] - time.perf_counter()) * 1000)), step)

    root.after(1000, step)
    root.mainloop()


if __name__ == "__main__":
    main()
