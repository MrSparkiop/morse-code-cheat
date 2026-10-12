"""Screen capture: DPI awareness, monitor geometry and fast region grabs.

All coordinates are physical screen pixels in the Windows virtual-screen
space (the primary monitor's top-left is (0, 0); other monitors may have
negative coordinates).
"""

from __future__ import annotations

import ctypes
import sys
from dataclasses import dataclass

import numpy as np

_dpi_done = False


def enable_dpi_awareness() -> None:
    """Make this process per-monitor DPI aware so every API speaks physical pixels.

    Must run before any window (Tk) is created. Safe to call repeatedly.
    """
    global _dpi_done
    if _dpi_done or sys.platform != "win32":
        return
    _dpi_done = True
    try:  # Windows 10 1703+: per-monitor v2
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:  # Windows 8.1+
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


@dataclass(frozen=True)
class Region:
    left: int
    top: int
    width: int
    height: int

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    def as_mss(self) -> dict:
        return {"left": self.left, "top": self.top, "width": self.width, "height": self.height}

    def clamp(self, bounds: "Region") -> "Region":
        l = max(self.left, bounds.left)
        t = max(self.top, bounds.top)
        r = min(self.right, bounds.right)
        b = min(self.bottom, bounds.bottom)
        return Region(l, t, max(1, r - l), max(1, b - t))

    def __str__(self) -> str:
        return f"{self.width}x{self.height}+{self.left}+{self.top}"


def _mss_factory():
    import mss

    return getattr(mss, "MSS", None) or mss.mss  # mss >= 10 renamed the class


class Grabber:
    """Grabs screen regions as RGB uint8 arrays of shape (h, w, 3).

    An mss handle is bound to the thread that created it, so create a Grabber
    inside the thread that will use it.
    """

    def __init__(self) -> None:
        enable_dpi_awareness()
        self._sct = _mss_factory()()
        mons = self._sct.monitors
        self.virtual = Region(mons[0]["left"], mons[0]["top"], mons[0]["width"], mons[0]["height"])
        self.monitors = [Region(m["left"], m["top"], m["width"], m["height"]) for m in mons[1:]]

    def primary(self) -> Region:
        for m in self.monitors:
            if m.left == 0 and m.top == 0:
                return m
        return self.monitors[0]

    def monitor_at(self, x: int, y: int) -> Region:
        for m in self.monitors:
            if m.left <= x < m.right and m.top <= y < m.bottom:
                return m
        return self.primary()

    def grab(self, region: Region) -> np.ndarray:
        shot = self._sct.grab(region.as_mss())
        bgra = np.frombuffer(shot.raw if hasattr(shot, "raw") else shot.bgra, dtype=np.uint8)
        bgra = bgra.reshape(shot.height, shot.width, 4)
        return bgra[:, :, 2::-1]  # BGRA -> RGB view

    def close(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass

    def __enter__(self) -> "Grabber":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


GAME_TITLE = "Keep Talking and Nobody Explodes"


def find_window(title: str = GAME_TITLE) -> Region | None:
    """Client area (physical pixels) of the visible, non-minimised window with this exact title."""
    if sys.platform != "win32":
        return None
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, title)
    if not hwnd or not user32.IsWindowVisible(hwnd) or user32.IsIconic(hwnd):
        return None
    rect = wintypes.RECT()
    if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return None
    origin = wintypes.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(origin))
    w, h = rect.right - rect.left, rect.bottom - rect.top
    if w < 50 or h < 50:
        return None
    return Region(origin.x, origin.y, w, h)


def foreground_window_title() -> str:
    """Title of the focused window ('' if unavailable)."""
    if sys.platform != "win32":
        return ""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    n = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(n + 1)
    user32.GetWindowTextW(hwnd, buf, n + 1)
    return buf.value
