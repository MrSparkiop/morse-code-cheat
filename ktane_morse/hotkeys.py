"""System-wide hotkeys (work while the game has focus) via RegisterHotKey."""

from __future__ import annotations

import ctypes
import sys
import threading
from ctypes import wintypes
from typing import Callable

MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

VK = {f"F{i}": 0x6F + i for i in range(1, 13)}  # F1=0x70 ... F12=0x7B


class HotkeyThread(threading.Thread):
    """Calls callback(name) from this thread when a registered key is pressed.

    keys maps a name to a key such as "F8". Keys another program already owns
    are skipped and listed in .failed.
    """

    def __init__(self, keys: dict[str, str], callback: Callable[[str], None]) -> None:
        super().__init__(daemon=True, name="hotkeys")
        self.keys = keys
        self.callback = callback
        self.failed: list[str] = []
        self._tid = 0
        self._ready = threading.Event()

    def run(self) -> None:
        if sys.platform != "win32":
            self._ready.set()
            return
        user32 = ctypes.windll.user32
        self._tid = ctypes.windll.kernel32.GetCurrentThreadId()
        ids: dict[int, str] = {}
        for i, (name, key) in enumerate(self.keys.items(), start=1):
            vk = VK.get(key.upper())
            if vk is None or not user32.RegisterHotKey(None, i, MOD_NOREPEAT, vk):
                self.failed.append(key)
                continue
            ids[i] = name
        self._ready.set()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            if msg.message == WM_HOTKEY and msg.wParam in ids:
                try:
                    self.callback(ids[msg.wParam])
                except Exception:
                    pass
        for i in ids:
            user32.UnregisterHotKey(None, i)

    def start_and_wait(self) -> "HotkeyThread":
        self.start()
        self._ready.wait(2)
        return self

    def stop(self) -> None:
        if self._tid:
            ctypes.windll.user32.PostThreadMessageW(self._tid, WM_QUIT, 0, 0)
