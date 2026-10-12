"""Click-and-drag region picker drawn over a frozen screenshot of one monitor."""

from __future__ import annotations

import tkinter as tk

import numpy as np

from .capture import Grabber, Region, enable_dpi_awareness


def _photo_from_rgb(master: tk.Misc, rgb: np.ndarray) -> tk.PhotoImage:
    h, w = rgb.shape[:2]
    ppm = b"P6 %d %d 255\n" % (w, h) + np.ascontiguousarray(rgb).tobytes()
    return tk.PhotoImage(master=master, data=ppm, format="PPM")


def pick_region(master: tk.Misc | None = None, monitor: Region | None = None) -> Region | None:
    """Show a fullscreen frozen screenshot and let the user drag a box around the Morse light.

    Returns the region in physical screen pixels, or None if cancelled (Esc / right-click).
    """
    enable_dpi_awareness()
    with Grabber() as g:
        if monitor is None:
            pt = _cursor_pos()
            monitor = g.monitor_at(*pt) if pt else g.primary()
        shot = np.ascontiguousarray(g.grab(monitor))

    own_root = master is None
    root = tk.Tk() if own_root else master
    if own_root:
        root.withdraw()
    top = tk.Toplevel(root)
    top.overrideredirect(True)
    top.geometry(f"{monitor.width}x{monitor.height}+{monitor.left}+{monitor.top}")
    top.attributes("-topmost", True)
    top.configure(cursor="crosshair")

    canvas = tk.Canvas(top, width=monitor.width, height=monitor.height, highlightthickness=0, bd=0)
    canvas.pack(fill="both", expand=True)
    # Dim the screenshot so the selection box stands out.
    photo = _photo_from_rgb(top, (shot.astype(np.uint16) * 3 // 5).astype(np.uint8))
    canvas.create_image(0, 0, image=photo, anchor="nw")
    canvas.image = photo  # keep a reference
    canvas.create_text(
        monitor.width // 2, 40, fill="#ffd54a", font=("Segoe UI", 18, "bold"),
        text="Drag a box around the Morse code light  ·  Esc / right-click to cancel",
    )

    result: dict[str, Region | None] = {"region": None}
    state = {"x0": 0, "y0": 0, "rect": None}

    def on_press(e):
        state["x0"], state["y0"] = e.x, e.y
        if state["rect"]:
            canvas.delete(state["rect"])
        state["rect"] = canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="#00e5ff", width=2)

    def on_drag(e):
        if state["rect"]:
            canvas.coords(state["rect"], state["x0"], state["y0"], e.x, e.y)

    def on_release(e):
        x0, x1 = sorted((state["x0"], e.x))
        y0, y1 = sorted((state["y0"], e.y))
        if x1 - x0 < 4 or y1 - y0 < 4:  # treat a click as "box of 40 px around this point"
            cx, cy = e.x, e.y
            x0, y0, x1, y1 = cx - 20, cy - 20, cx + 20, cy + 20
        region = Region(monitor.left + x0, monitor.top + y0, x1 - x0, y1 - y0).clamp(monitor)
        result["region"] = region
        top.destroy()

    def cancel(_e=None):
        top.destroy()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    canvas.bind("<ButtonPress-3>", cancel)
    top.bind("<Escape>", cancel)
    top.focus_force()
    top.grab_set()
    root.wait_window(top)
    if own_root:
        root.destroy()
    return result["region"]


def _cursor_pos() -> tuple[int, int] | None:
    try:
        import ctypes
        from ctypes import wintypes

        pt = wintypes.POINT()
        if ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return pt.x, pt.y
    except Exception:
        pass
    return None


if __name__ == "__main__":
    print(pick_region())
