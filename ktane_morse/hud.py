"""Small always-on-top window that shows the answer while you play."""

from __future__ import annotations

import ctypes
import sys
import tkinter as tk
from typing import Callable

BG = "#16181d"
FG = "#d8dce4"
DIM = "#7d8594"
AMBER = "#ffcf3f"
GREEN = "#62d26f"
RED = "#ff6b6b"


class Hud:
    """The UI. `on_command(name)` is called with "auto", "select", "reset", "speak" or "quit"."""

    def __init__(self, root: tk.Tk, on_command: Callable[[str], None], hotkeys: dict[str, str]) -> None:
        self.root = root
        self.on_command = on_command
        root.title("KTANE Morse")
        root.configure(bg=BG)
        root.attributes("-topmost", True)
        root.resizable(False, False)
        root.protocol("WM_DELETE_WINDOW", lambda: on_command("quit"))

        pad = {"padx": 12}
        self.freq = tk.Label(root, text="—", font=("Segoe UI", 30, "bold"), fg=DIM, bg=BG, anchor="w")
        self.freq.grid(row=0, column=0, sticky="w", **pad, pady=(8, 0))
        self.word = tk.Label(root, text="", font=("Segoe UI", 16, "bold"), fg=FG, bg=BG, anchor="e")
        self.word.grid(row=0, column=1, sticky="e", **pad, pady=(8, 0))
        self.hint = tk.Label(root, text="", font=("Segoe UI", 11), fg=FG, bg=BG, anchor="w")
        self.hint.grid(row=1, column=0, columnspan=2, sticky="w", **pad)
        self.morse = tk.Label(root, text="", font=("Consolas", 11), fg=AMBER, bg=BG, anchor="w", width=44)
        self.morse.grid(row=2, column=0, columnspan=2, sticky="w", **pad, pady=(6, 0))
        self.status = tk.Label(root, text="Starting…", font=("Segoe UI", 10), fg=DIM, bg=BG, anchor="w",
                               justify="left", wraplength=400)
        self.status.grid(row=3, column=0, columnspan=2, sticky="w", **pad, pady=(2, 6))

        bar = tk.Frame(root, bg=BG)
        bar.grid(row=4, column=0, columnspan=2, sticky="we", padx=8, pady=(0, 10))
        self.speak_var = tk.BooleanVar(value=False)

        def button(text, cmd):
            b = tk.Button(bar, text=text, command=lambda: on_command(cmd), bg="#262a33", fg=FG,
                          activebackground="#343a46", activeforeground=FG, relief="flat", padx=8, pady=2,
                          font=("Segoe UI", 9))
            b.pack(side="left", padx=4)
            return b

        button(f"Auto find ({hotkeys.get('auto', '')})".replace(" ()", ""), "auto")
        button(f"Select light ({hotkeys.get('select', '')})".replace(" ()", ""), "select")
        button("Reset", "reset")
        tk.Checkbutton(bar, text="Speak", variable=self.speak_var, command=lambda: on_command("speak"),
                       bg=BG, fg=FG, selectcolor="#262a33", activebackground=BG, activeforeground=FG,
                       font=("Segoe UI", 9)).pack(side="left", padx=6)

        root.update_idletasks()
        sw = root.winfo_screenwidth()
        root.geometry(f"+{max(0, sw - root.winfo_reqwidth() - 40)}+40")
        root.after(200, self._exclude_from_capture)

    def _exclude_from_capture(self) -> None:
        """Keep this window out of screen captures so the tool never reads its own display."""
        if sys.platform != "win32":
            return
        try:
            hwnd = int(self.root.wm_frame(), 16)
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)  # WDA_EXCLUDEFROMCAPTURE
        except Exception:
            pass

    def show_answer(self, freq: str | None, word: str | None, hint: str, fresh: bool) -> None:
        if freq is None:
            self.freq.configure(text="—", fg=DIM)
            self.word.configure(text="")
            self.hint.configure(text=hint)
            return
        self.freq.configure(text=f"{freq} MHz", fg=GREEN if fresh else DIM)
        self.word.configure(text=(word or "").upper(), fg=FG if fresh else DIM)
        self.hint.configure(text=hint)

    def show_progress(self, morse: str, status: str, colour: str = DIM) -> None:
        self.morse.configure(text=morse[-44:])
        self.status.configure(text=status, fg=colour)
