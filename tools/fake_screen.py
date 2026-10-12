"""Synthetic KTANE-like screen frames for exercising the Morse-lamp locator (numpy only).

    from tools.fake_screen import FakeScreen, SceneSpec, Event, ALL_DISTRACTORS

    scr = FakeScreen(SceneSpec(width=1280, height=720, word="shell", zoomed=False,
                               distractors=ALL_DISTRACTORS,
                               events=[Event("pan", 8.0, 1.0, dx=40, dy=-25)]))
    for t, frame in scr.frames(fps=30, duration=12):   # frame: (H, W, 3) uint8 RGB, a new array each time
        ...
    scr.lamp_center(t)            # ground truth (x, y) in frame pixels, None without a Morse lamp
    scr.object_center("timer", t) # also "simon_yellow", "highlight", "status_led", "yellow_wire", "solved_led"

    python -m tools.fake_screen bench [--width 2560 --height 1440 --seconds 10 --zoomed]
    python -m tools.fake_screen snap out.png [--t 1.0 --zoomed --width ...]

Scene: a textured wooden table, a dark bomb casing with grey modules (whole
bomb in view, lamp ~14 px across at 1440p), or zoomed onto the Morse module
(lamp ~92 px). The Morse lamp is unlit RGB(117,62,41) (room-lit) and lit
RGB(251,222,71) with an additive (1.0, 0.835, 0.0) glow about half a module
wide; lit parts ignore the room light. Morse timing comes from
ktane_morse.simulate.morse_segments.

Distractors (``SceneSpec.distractors``):
    "timer"        red 7-segment M:SS timer, changes every second
    "panic"        the timer digits blink every 0.1 s
    "simon"        Simon Says: red/blue/green/yellow buttons flashing 0.65 s apart, 4.5 s pause
    "highlight"    static white specular highlights
    "status_led"   the Morse module's status LED flashes red for 1 s (strikes) every ~7 s
    "solved_led"   another module's status LED is green
    "yellow_wire"  a yellow (room-lit) wire on a Wires module

Events (``SceneSpec.events``), times in seconds:
    Event("emergency", start, length)   red room light pulses: 1 s pulse, 0.75-2 s apart
    Event("lights_out", start, length)  flicker 1.6 s, dark `length` s (gain 0.12), flicker 0.6 s
    Event("pan", start, length, dx, dy) camera moves for `length` s, then settles shifted by (dx, dy) px
    Event("zoom", start, length, dx, dy, scale)  camera zooms in smoothly to `scale` x about the
                                        frame point (W/2 + dx, H/2 + dy) (e.g. selecting a module)
    Event("pause", start, length)       pause menu over ~85 % of the screen; game time freezes
    Event("strike", start)              status LED red for 1 s
    Event("solved", start)              Morse lamp off from then on
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from ktane_morse.simulate import morse_segments

ALL_DISTRACTORS = ("timer", "simon", "highlight", "status_led", "solved_led", "yellow_wire")

LIT = np.array([251, 222, 71], np.float32)
UNLIT = np.array([117, 62, 41], np.float32)
GLOW = np.array([255, 213, 0], np.float32)     # (1.0, 0.835, 0.0) * 255
FACE = (158, 158, 152)
CASING = (52, 53, 58)

SEGMENTS = {0: "abcdef", 1: "bc", 2: "abged", 3: "abgcd", 4: "fgbc", 5: "afgcd",
            6: "afgedc", 7: "abc", 8: "abcdefg", 9: "abcdfg"}


@dataclass
class Event:
    kind: str
    start: float
    length: float = 0.0
    dx: float = 0.0
    dy: float = 0.0
    scale: float = 1.0


@dataclass
class SceneSpec:
    width: int = 1280
    height: int = 720
    word: str | None = "shell"           # None: no Morse lamp at all
    zoomed: bool = False
    distractors: tuple[str, ...] = ()
    events: list[Event] = field(default_factory=list)
    seed: int = 0
    phase: float | None = 0.0            # seconds into the Morse cycle at t=0 (None = random)
    morse_start: float = 0.0             # lamp stays off before this time
    glow: float = 0.55                   # glow intensity at the lamp's edge (0..1)
    noise: float = 0.0                   # per-frame gaussian-ish noise amplitude (slow at 1440p)
    module_scale: float = 1.0            # multiplies the module (and lamp) size


class _Sprite:
    """A dynamic overlay: out = region*(1-alpha) + alpha*color + add (lit parts ignore room light)."""

    def __init__(self, x0: int, y0: int, alpha: np.ndarray | None, color, add: np.ndarray | None = None) -> None:
        self.x0, self.y0 = x0, y0
        shape = (alpha if alpha is not None else add).shape[:2]
        self.h, self.w = shape
        self.alpha = None if alpha is None else alpha.astype(np.float32)[..., None]
        self.color = np.asarray(color, np.float32)
        self.add = add


def _grid(x0: int, y0: int, w: int, h: int):
    ys = np.arange(y0, y0 + h, dtype=np.float32)[:, None] + 0.5
    xs = np.arange(x0, x0 + w, dtype=np.float32)[None, :] + 0.5
    return xs, ys


def _disc_alpha(cx: float, cy: float, r: float, pad: int = 2):
    x0, y0 = int(math.floor(cx - r - pad)), int(math.floor(cy - r - pad))
    w = h = int(math.ceil(2 * r + 2 * pad)) + 1
    xs, ys = _grid(x0, y0, w, h)
    d = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    return x0, y0, np.clip(r + 0.5 - d, 0, 1)


class FakeScreen:
    def __init__(self, spec: SceneSpec) -> None:
        self.spec = spec
        self.W, self.H = spec.width, spec.height
        self.S = spec.height / 1440.0
        self.M = int(0.12 * max(self.W, self.H))
        self.rng = np.random.default_rng(spec.seed)
        self.prng = random.Random(spec.seed)
        self._objects: dict[str, tuple[float, float]] = {}   # world coords
        self._build()
        self._timeline()

    # ------------------------------------------------------------------ scene
    def _build(self) -> None:
        sp, S, M = self.spec, self.S, self.M
        Hw, Ww = self.H + 2 * M, self.W + 2 * M
        self.world_shape = (Hw, Ww)
        # wooden table
        xs = np.arange(Ww, dtype=np.float32)[None, :]
        ys = np.arange(Hw, dtype=np.float32)[:, None]
        grain = np.sin(2 * np.pi * (ys / (23 * S + 3) + 0.35 * np.sin(xs / (310 * S + 10)) + 0.1 * np.sin(xs / 41)))
        coarse = self.rng.uniform(-1, 1, (Hw // 64 + 2, Ww // 64 + 2)).astype(np.float32)
        coarse = np.repeat(np.repeat(coarse, 64, 0), 64, 1)[:Hw, :Ww]
        base = 0.85 + 0.08 * grain + 0.07 * coarse
        img = np.empty((Hw, Ww, 3), np.float32)
        for c, v in enumerate((122, 86, 55)):
            img[..., c] = v * base
        img += self.rng.uniform(-7, 7, (Hw, Ww, 1)).astype(np.float32)
        self.img = img
        self.sprites: dict[str, _Sprite] = {}

        if sp.zoomed:
            mw = 1150 * S * sp.module_scale
            gap = 0.03 * mw
            cx, cy = M + self.W / 2, M + self.H / 2
            mx0, my0 = cx - mw / 2, cy - mw / 2
            self._rect(mx0 - 3 * gap, my0 - 3 * gap, mx0 + mw + 3 * gap, my0 + mw + 3 * gap, CASING)
            self._rect(mx0 - 2 * mw - 4 * gap, my0 - gap, mx0 + 3 * mw + 4 * gap, my0 + mw + gap, CASING)
            layout = {"morse" if sp.word else "blank": (mx0, my0),
                      "timer": (mx0 - mw - gap, my0), "simon": (mx0 + mw + gap, my0)}
            highlights = [(mx0 + 0.62 * mw, my0 + 0.955 * mw, 0.06 * mw), (mx0 + 0.5 * mw, my0 + 0.25 * mw, 0.03 * mw)]
        else:
            mw = 175 * S * sp.module_scale
            gap, pad = 0.07 * mw, 0.12 * mw
            cw, ch = 3 * mw + 2 * gap + 2 * pad, 2 * mw + gap + 2 * pad
            cx0, cy0 = M + (self.W - cw) / 2, M + (self.H - ch) / 2
            self._rect(cx0, cy0, cx0 + cw, cy0 + ch, CASING)

            def at(r, c):
                return cx0 + pad + c * (mw + gap), cy0 + pad + r * (mw + gap)

            layout = {"wires": at(0, 0), "timer": at(0, 1), "morse" if sp.word else "blank": at(0, 2),
                      "simon": at(1, 0), "keypad": at(1, 1), "button": at(1, 2)}
            kx, ky = at(1, 1)
            highlights = [(cx0 + 0.1 * cw, cy0 + 0.05 * pad, 0.10 * mw), (kx + 0.7 * mw, ky + 0.2 * mw, 0.07 * mw)]
        self.mw = mw
        for kind, (x0, y0) in layout.items():
            getattr(self, "_mod_" + kind)(x0, y0, mw)
        if "highlight" in sp.distractors:
            for hx, hy, hr in highlights:
                self._highlight(hx, hy, hr)
        self.world = np.clip(self.img, 0, 255).astype(np.uint8)
        del self.img
        # pause screen (frame-sized): a menu panel over most of the screen, dimmed scene at the edges
        p = self.world[M:M + self.H, M:M + self.W].astype(np.float32) * 0.25
        x0, y0 = int(self.W * 0.08), int(self.H * 0.08)
        p[y0:self.H - y0, x0:self.W - x0] = (30, 32, 40)
        for i in range(5):
            yy = y0 + int((0.1 + 0.15 * i) * (self.H - 2 * y0))
            p[yy:yy + int(0.04 * self.H), x0 + int(0.25 * self.W):self.W - x0 - int(0.25 * self.W)] = (200, 200, 205)
        self.pause_img = p.astype(np.uint8)

    def _rect(self, x0, y0, x1, y1, color) -> None:
        Hw, Ww = self.world_shape
        a, b = max(0, int(round(y0))), min(Hw, int(round(y1)))
        c, d = max(0, int(round(x0))), min(Ww, int(round(x1)))
        if a < b and c < d:
            self.img[a:b, c:d] = color

    def _blend(self, x0: int, y0: int, al: np.ndarray, color) -> None:
        """img = img*(1-al) + al*color over the world region at (x0, y0), clipped to the world."""
        Hw, Ww = self.world_shape
        a0, b0 = max(0, -y0), max(0, -x0)
        a1, b1 = min(al.shape[0], Hw - y0), min(al.shape[1], Ww - x0)
        if a0 >= a1 or b0 >= b1:
            return
        reg = self.img[y0 + a0:y0 + a1, x0 + b0:x0 + b1]
        a = al[a0:a1, b0:b1, None]
        reg[...] = reg * (1 - a) + a * np.asarray(color, np.float32)

    def _disc(self, cx, cy, r, color) -> None:
        x0, y0, al = _disc_alpha(cx, cy, r)
        self._blend(x0, y0, al, color)

    def _highlight(self, cx, cy, r) -> None:
        x0, y0 = int(cx - 2 * r), int(cy - 2 * r)
        xs, ys = _grid(x0, y0, int(4 * r), int(4 * r))
        al = np.exp(-(((xs - cx) / r) ** 2 + ((ys - cy) / (0.5 * r)) ** 2))
        self._blend(x0, y0, al, (255, 255, 250))
        n = sum(1 for k in self._objects if k.startswith("highlight"))
        self._objects["highlight" if n == 0 else f"highlight{n}"] = (cx, cy)

    def _face(self, x0, y0, mw) -> None:
        self._rect(x0, y0, x0 + mw, y0 + mw, (110, 110, 106))
        b = 0.03 * mw
        self._rect(x0 + b, y0 + b, x0 + mw - b, y0 + mw - b, FACE)
        for sx in (0.06, 0.94):
            for sy in (0.06, 0.94):
                self._disc(x0 + sx * mw, y0 + sy * mw, 0.018 * mw, (90, 90, 92))

    def _led(self, x0, y0, mw):
        cx, cy, r = x0 + 0.88 * mw, y0 + 0.12 * mw, 0.045 * mw
        self._disc(cx, cy, r * 1.3, (70, 70, 72))
        self._disc(cx, cy, r, (28, 30, 30))
        return cx, cy, r

    def _mod_blank(self, x0, y0, mw) -> None:
        self._face(x0, y0, mw)
        for i in range(3):
            self._rect(x0 + 0.2 * mw, y0 + (0.3 + 0.18 * i) * mw, x0 + 0.8 * mw, y0 + (0.38 + 0.18 * i) * mw, (60, 62, 66))
        self._led(x0, y0, mw)

    _mod_keypad = _mod_blank

    def _mod_button(self, x0, y0, mw) -> None:
        self._face(x0, y0, mw)
        self._disc(x0 + 0.45 * mw, y0 + 0.55 * mw, 0.3 * mw, (40, 70, 170))
        cx, cy, r = self._led(x0, y0, mw)
        if "solved_led" in self.spec.distractors:
            sx, sy, al = _disc_alpha(cx, cy, r)
            self.sprites["solved_led"] = _Sprite(sx, sy, al, (40, 230, 70))
            self._objects["solved_led"] = (cx, cy)

    def _mod_wires(self, x0, y0, mw) -> None:
        self._face(x0, y0, mw)
        cols = [(200, 30, 30), (40, 60, 200), (235, 235, 235), (25, 25, 25)]
        if "yellow_wire" in self.spec.distractors:
            cols.insert(2, (222, 200, 40))
        th = 0.045 * mw
        for i, c in enumerate(cols):
            yy = y0 + (0.22 + 0.13 * i) * mw
            self._rect(x0 + 0.15 * mw, yy, x0 + 0.85 * mw, yy + th, c)
            if c == (222, 200, 40):
                self._objects["yellow_wire"] = (x0 + 0.5 * mw, yy + th / 2)
        self._led(x0, y0, mw)

    def _mod_timer(self, x0, y0, mw) -> None:
        self._rect(x0, y0, x0 + mw, y0 + mw, (110, 110, 106))
        self._rect(x0 + 0.03 * mw, y0 + 0.03 * mw, x0 + 0.97 * mw, y0 + 0.97 * mw, (70, 70, 72))
        dx0, dy0, dx1, dy1 = x0 + 0.1 * mw, y0 + 0.35 * mw, x0 + 0.9 * mw, y0 + 0.68 * mw
        self._rect(dx0, dy0, dx1, dy1, (14, 10, 10))
        if "timer" not in self.spec.distractors and "panic" not in self.spec.distractors:
            return
        dw, dh = 0.15 * mw, 0.26 * mw
        th = max(1.0, 0.16 * dw)
        xs = [dx0 + 0.04 * mw + k * 0.19 * mw for k in range(4)]   # M, colon, S, S
        ytop = dy0 + 0.035 * mw
        self._digit_boxes = []
        for k, xx in enumerate(xs):
            if k == 1:
                for fy in (0.32, 0.68):
                    self._rect(xx + 0.4 * dw, ytop + fy * dh, xx + 0.4 * dw + th, ytop + fy * dh + th, (235, 30, 20))
                continue
            masks = {}
            for dgt, segs in SEGMENTS.items():
                m = np.zeros((int(dh) + 1, int(dw) + 1), np.float32)
                for s in segs:
                    self._seg(m, s, dw, dh, th)
                masks[dgt] = m
            unl = masks[8]
            self._blend(int(xx), int(ytop), unl, (42, 12, 10))
            self._digit_boxes.append((int(xx), int(ytop), masks))
        self._objects["timer"] = ((dx0 + dx1) / 2, (dy0 + dy1) / 2)

    @staticmethod
    def _seg(m, s, dw, dh, th) -> None:
        h2 = dh / 2
        rects = {"a": (0, 0, dw, th), "d": (0, dh - th, dw, dh), "g": (0, h2 - th / 2, dw, h2 + th / 2),
                 "f": (0, 0, th, h2), "e": (0, h2, th, dh), "b": (dw - th, 0, dw, h2), "c": (dw - th, h2, dw, dh)}
        x0, y0, x1, y1 = rects[s]
        m[int(y0):int(math.ceil(y1)), int(x0):int(math.ceil(x1))] = 1.0

    def _mod_simon(self, x0, y0, mw) -> None:
        self._face(x0, y0, mw)
        if "simon" not in self.spec.distractors:
            return
        cx, cy, half = x0 + 0.5 * mw, y0 + 0.52 * mw, 0.11 * mw
        unlit = [(125, 112, 28), (120, 25, 25), (25, 35, 120), (25, 105, 35)]   # yellow(0), red, blue, green
        lit = [(255, 240, 70), (255, 70, 55), (90, 130, 255), (80, 255, 100)]
        pos = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        self._simon = []
        for k in range(4):
            px, py = cx + pos[k][0] * 2.3 * half, cy + pos[k][1] * 2.3 * half
            self._rect(px - half, py - half, px + half, py + half, unlit[k])
            a = np.ones((int(2 * half), int(2 * half)), np.float32)
            self.sprites[f"simon{k}"] = _Sprite(int(px - half), int(py - half), a, lit[k])
            if k == 0:
                self._objects["simon_yellow"] = (px, py)
        self._led(x0, y0, mw)

    def _mod_morse(self, x0, y0, mw) -> None:
        sp = self.spec
        self._face(x0, y0, mw)
        lx, ly, r = x0 + 0.22 * mw, y0 + 0.17 * mw, 0.04 * mw
        self.lamp_radius = r
        self._disc(lx, ly, r * 1.25, (80, 78, 74))       # bezel
        self._disc(lx, ly, r, UNLIT)
        # frequency display, arrows, TX button
        self._rect(x0 + 0.2 * mw, y0 + 0.45 * mw, x0 + 0.8 * mw, y0 + 0.62 * mw, (20, 22, 20))
        for i in range(4):
            self._rect(x0 + (0.27 + 0.12 * i) * mw, y0 + 0.49 * mw, x0 + (0.34 + 0.12 * i) * mw, y0 + 0.58 * mw, (70, 72, 70))
        self._rect(x0 + 0.08 * mw, y0 + 0.48 * mw, x0 + 0.16 * mw, y0 + 0.59 * mw, (70, 70, 72))
        self._rect(x0 + 0.84 * mw, y0 + 0.48 * mw, x0 + 0.92 * mw, y0 + 0.59 * mw, (70, 70, 72))
        self._rect(x0 + 0.38 * mw, y0 + 0.75 * mw, x0 + 0.62 * mw, y0 + 0.88 * mw, (85, 85, 88))
        self._rect(x0 + 0.40 * mw, y0 + 0.30 * mw, x0 + 0.85 * mw, y0 + 0.36 * mw, (40, 40, 40))
        # lit lamp sprite: lit disc + additive glow (half a module wide)
        R = 0.25 * mw
        gx0, gy0 = int(lx - R), int(ly - R)
        n = int(2 * R) + 1
        xs, ys = _grid(gx0, gy0, n, n)
        d = np.sqrt((xs - lx) ** 2 + (ys - ly) ** 2)
        alpha = np.clip(r + 0.5 - d, 0, 1)
        sigma = 0.09 * mw
        inten = sp.glow * np.exp(-np.maximum(0, d - r) ** 2 / (2 * sigma ** 2)) * np.clip((R - d) / (0.2 * R), 0, 1)
        add = (inten * (1 - alpha))[..., None] * GLOW
        self.sprites["lamp"] = _Sprite(gx0, gy0, alpha, LIT, add.astype(np.float32))
        self._objects["lamp"] = (lx, ly)
        cx, cy, rr = self._led(x0, y0, mw)
        sx, sy, al = _disc_alpha(cx, cy, rr)
        self.sprites["strike_led"] = _Sprite(sx, sy, al, (255, 45, 30))
        self._objects["status_led"] = (cx, cy)

    # ------------------------------------------------------------------ time
    def _timeline(self) -> None:
        sp = self.spec
        rng = self.prng
        dur = 600.0
        if sp.word:
            segs = morse_segments(sp.word, dur, random.Random(sp.seed + 1), phase=sp.phase)
            self._mstart = np.array([s[0] for s in segs])
            self._mend = np.array([s[1] for s in segs])
            self._mon = np.array([s[2] for s in segs])
        # Simon flashes: (start, end, colour)
        fl = []
        colours = [rng.randrange(4) for _ in range(5)]
        if 0 not in colours:
            colours[rng.randrange(5)] = 0
        flash = rng.uniform(0.25, 0.5)
        t, stage = rng.uniform(-3, 0), rng.randint(2, 3)
        while t < dur:
            for c in colours[:stage]:
                fl.append((t, t + flash, c))
                t += 0.65
            t += 4.5
            if rng.random() < 0.3 and stage < 5:
                stage += 1
        self._simon_fl = fl
        self._simon_start = np.array([f[0] for f in fl])
        # strikes
        strikes = [e.start for e in sp.events if e.kind == "strike"]
        if "status_led" in sp.distractors:
            strikes += [2.5 + 7.0 * k for k in range(int(dur / 7))]
        self._strikes = sorted(strikes)
        # emergency pulse starts
        self._emerg = []
        for e in sp.events:
            if e.kind == "emergency":
                p = e.start
                while p < e.start + e.length:
                    self._emerg.append(p)
                    p += 1.0 + rng.uniform(0.75, 2.0)
        self._emerg_arr = np.array(sorted(self._emerg)) if self._emerg else np.zeros(0)
        self._lut_cache: dict[int, np.ndarray] = {}

    def game_time(self, t: float) -> float:
        g = t
        for e in self.spec.events:
            if e.kind == "pause" and t > e.start:
                g -= min(e.length, t - e.start)
        return g

    def lamp_on(self, t: float) -> bool:
        sp = self.spec
        if not sp.word or t < sp.morse_start:
            return False
        if any(e.kind == "solved" and t >= e.start for e in sp.events):
            return False
        g = self.game_time(t) - sp.morse_start
        i = int(np.searchsorted(self._mstart, g, "right")) - 1
        return bool(i >= 0 and self._mon[i] and g < self._mend[i])

    def offset(self, t: float) -> tuple[int, int]:
        ox = oy = 0.0
        for e in self.spec.events:
            if e.kind != "pan" or t < e.start:
                continue
            u = min(1.0, (t - e.start) / max(1e-6, e.length))
            sm = u * u * (3 - 2 * u)
            wob = math.sin(math.pi * u) if u < 1 else 0.0
            ox += e.dx * sm + 0.05 * self.W * math.sin(2 * math.pi * 1.4 * u) * wob
            oy += e.dy * sm + 0.04 * self.H * math.sin(2 * math.pi * 1.1 * u + 0.7) * wob
        lim = self.M - 2
        return int(round(max(-lim, min(lim, ox)))), int(round(max(-lim, min(lim, oy))))

    def zoom(self, t: float) -> tuple[float, float, float]:
        """(magnification, centre x, centre y) of the camera zoom at time t (frame coords)."""
        m, cx, cy = 1.0, self.W / 2, self.H / 2
        for e in self.spec.events:
            if e.kind != "zoom" or t < e.start:
                continue
            u = min(1.0, (t - e.start) / max(1e-6, e.length))
            sm = u * u * (3 - 2 * u)
            m *= 1 + (max(1.0, e.scale) - 1) * sm
            cx, cy = self.W / 2 + e.dx, self.H / 2 + e.dy
        return m, cx, cy

    def is_moving(self, t: float) -> bool:
        return any(e.kind in ("pan", "pause", "zoom") and e.start <= t < e.start + e.length for e in self.spec.events)

    def room_gain(self, t: float) -> tuple[float, float, float]:
        L = 1.0
        for e in self.spec.events:
            if e.kind != "lights_out":
                continue
            a, b = e.start, e.start + 1.6
            c, d = b + e.length, b + e.length + 0.6
            if a <= t < b or c <= t < d:
                slot = int(t / 0.06)
                L = min(L, 0.2 + 0.8 * random.Random(self.spec.seed * 7919 + slot).random())
            elif b <= t < c:
                L = min(L, 0.12)
        gr = gg = gb = L
        if len(self._emerg_arr):
            i = int(np.searchsorted(self._emerg_arr, t, "right")) - 1
            if i >= 0 and t - self._emerg_arr[i] < 1.0:
                e = math.sin(math.pi * (t - self._emerg_arr[i])) ** 2
                gg *= 1 - 0.45 * e
                gb *= 1 - 0.45 * e
        return gr, gg, gb

    def _active(self, t: float) -> dict[str, bool]:
        sp = self.spec
        act = {}
        if "lamp" in self.sprites:
            act["lamp"] = self.lamp_on(t)
        if "strike_led" in self.sprites:
            act["strike_led"] = any(s <= t < s + 1.0 for s in self._strikes)
        if "solved_led" in self.sprites:
            act["solved_led"] = True
        if "simon0" in self.sprites:
            gt = self.game_time(t)
            i = int(np.searchsorted(self._simon_start, gt, "right")) - 1
            lit = self._simon_fl[i][2] if i >= 0 and gt < self._simon_fl[i][1] else -1
            for k in range(4):
                act[f"simon{k}"] = lit == k
        return act

    # ------------------------------------------------------------------ geometry (frame coords)
    def _to_frame(self, p, t):
        if p is None:
            return None
        ox, oy = self.offset(t)
        m, zx, zy = self.zoom(t)
        return zx + (p[0] - self.M - ox - zx) * m, zy + (p[1] - self.M - oy - zy) * m

    def lamp_size(self, t: float = 0.0) -> float:
        """Lamp diameter in frame pixels at time t (zoom included)."""
        return self.lamp_diameter * self.zoom(t)[0]

    def lamp_center(self, t: float = 0.0):
        return self._to_frame(self._objects.get("lamp"), t)

    def object_center(self, name: str, t: float = 0.0):
        return self._to_frame(self._objects.get(name), t)

    @property
    def lamp_diameter(self) -> float:
        return 2 * getattr(self, "lamp_radius", 0.0)

    # ------------------------------------------------------------------ rendering
    def _lut(self, g: float) -> np.ndarray:
        q = int(round(g * 1000))
        lut = self._lut_cache.get(q)
        if lut is None:
            lut = np.clip(np.round(np.arange(256) * (q / 1000.0)), 0, 255).astype(np.uint8)
            if len(self._lut_cache) > 2000:
                self._lut_cache.clear()
            self._lut_cache[q] = lut
        return lut

    def render(self, t: float) -> np.ndarray:
        sp = self.spec
        W, H, M = self.W, self.H, self.M
        for e in sp.events:
            if e.kind == "pause" and e.start <= t < e.start + e.length:
                return self.pause_img.copy()
        ox, oy = self.offset(t)
        X0, Y0 = M + ox, M + oy
        m, zx, zy = self.zoom(t)
        if m == 1.0:
            frame = self._compose(t, X0, Y0, W, H)
        else:   # render the visible world patch, then nearest-neighbour upscale it
            Hw, Ww = self.world_shape
            wx0, wy0 = X0 + zx - zx / m, Y0 + zy - zy / m
            ix0 = int(min(max(0, math.floor(wx0)), Ww - 2))
            iy0 = int(min(max(0, math.floor(wy0)), Hw - 2))
            iw, ih = min(Ww - ix0, int(math.ceil(W / m)) + 2), min(Hw - iy0, int(math.ceil(H / m)) + 2)
            patch = self._compose(t, ix0, iy0, iw, ih)
            xs = np.clip((wx0 + (np.arange(W) + 0.5) / m - ix0).astype(np.int64), 0, iw - 1)
            ys = np.clip((wy0 + (np.arange(H) + 0.5) / m - iy0).astype(np.int64), 0, ih - 1)
            frame = patch[ys[:, None], xs[None, :]]
        if sp.noise > 0:
            if not hasattr(self, "_noise_bank"):
                self._noise_bank = [self.rng.normal(0, sp.noise, (H, W, 3)).astype(np.int16) for _ in range(3)]
            nz = self._noise_bank[int(t * 997) % 3]
            frame = np.clip(frame.astype(np.int16) + nz, 0, 255).astype(np.uint8)
        return frame

    def _compose(self, t: float, X0: int, Y0: int, W: int, H: int) -> np.ndarray:
        """The (H, W) world patch at (X0, Y0) with room light, active sprites and timer digits."""
        crop = self.world[Y0:Y0 + H, X0:X0 + W]
        gains = self.room_gain(t)
        if gains == (1.0, 1.0, 1.0):
            frame = crop.copy()
        else:
            frame = np.empty(crop.shape, np.uint8)
            for c in range(3):
                if gains[c] == 1.0:
                    frame[..., c] = crop[..., c]
                else:
                    frame[..., c] = self._lut(gains[c])[crop[..., c]]
        for name, on in self._active(t).items():
            if on:
                self._paste(frame, self.sprites[name], X0, Y0)
        self._timer(frame, t, X0, Y0)
        return frame

    def _paste(self, frame: np.ndarray, sp: _Sprite, X0: int, Y0: int, alpha=None) -> None:
        H, W = frame.shape[:2]
        x0, y0 = sp.x0 - X0, sp.y0 - Y0
        a0, b0 = max(0, -y0), max(0, -x0)
        a1, b1 = min(sp.h, H - y0), min(sp.w, W - x0)
        if a0 >= a1 or b0 >= b1:
            return
        reg = frame[y0 + a0:y0 + a1, x0 + b0:x0 + b1]
        out = reg.astype(np.float32)
        al = (sp.alpha if alpha is None else alpha[..., None])
        if al is not None:
            al = al[a0:a1, b0:b1]
            out = out * (1 - al) + al * sp.color
        if sp.add is not None:
            out += sp.add[a0:a1, b0:b1]
        np.clip(out, 0, 255, out=out)
        reg[...] = out.astype(np.uint8)

    def _timer(self, frame: np.ndarray, t: float, X0: int, Y0: int) -> None:
        boxes = getattr(self, "_digit_boxes", None)
        if not boxes:
            return
        if "panic" in self.spec.distractors and int(t / 0.1) % 2 == 1:
            return
        rem = max(0, 300 - int(self.game_time(t)))
        digits = [rem // 60 % 10, rem % 60 // 10, rem % 10]
        for (xx, yy, masks), dgt in zip(boxes, digits):
            m = masks[dgt]
            self._paste(frame, _Sprite(xx, yy, m, (235, 30, 20)), X0, Y0)

    def frames(self, fps: float = 30.0, duration: float = 10.0, t0: float = 0.0, jitter: float = 0.1,
               drop: float = 0.0, hitch: float = 0.0, seed: int | None = None):
        """Yield (t, frame) like a capture loop: jittered intervals, optional drops and stalls."""
        rng = random.Random(self.spec.seed + 99 if seed is None else seed)
        t = t0
        while t < t0 + duration:
            if rng.random() >= drop:
                yield t, self.render(t)
            dt = (1.0 / fps) * max(0.3, 1 + rng.gauss(0, jitter))
            if hitch and rng.random() < hitch:
                dt += rng.uniform(0.05, 0.25)
            t += dt


# --------------------------------------------------------------------------- #
# CLI: speed benchmark and snapshots

def _bench(a) -> int:
    from ktane_morse.locator import LampLocator

    spec = SceneSpec(width=a.width, height=a.height, word=a.word, zoomed=a.zoomed,
                     distractors=ALL_DISTRACTORS, seed=a.seed)
    scr = FakeScreen(spec)
    loc = LampLocator()
    times, found_at = [], None
    n = 0
    gen_t = 0.0
    g0 = time.perf_counter()
    for t, frame in scr.frames(fps=a.fps, duration=a.seconds):
        gen_t += time.perf_counter() - g0
        if a.bgra:   # emulate mss: BGRA buffer, RGB view with negative channel stride
            bgra = np.empty((frame.shape[0], frame.shape[1], 4), np.uint8)
            bgra[..., :3] = frame[..., ::-1]
            bgra[..., 3] = 255
            frame = bgra[:, :, 2::-1]
        c0 = time.perf_counter()
        loc.update(t, frame)
        times.append(time.perf_counter() - c0)
        n += 1
        if found_at is None and spec.word:
            lc = scr.lamp_center(t)
            if any(c.contains(*lc) for c in loc.candidates(3)):
                found_at = t
        g0 = time.perf_counter()
    ms = np.array(times[5:]) * 1000
    print(f"{a.width}x{a.height} {'zoomed' if a.zoomed else 'bomb view'} input={'bgra-view' if a.bgra else 'rgb'} "
          f"factor={loc.factor} small={loc.small_shape} frames={n}")
    print(f"  locator ms/frame: mean={ms.mean():.2f} median={np.median(ms):.2f} p95={np.percentile(ms, 95):.2f} "
          f"max={ms.max():.2f}   (render {1000 * gen_t / n:.1f} ms/frame, not included)")
    print(f"  lamp in top-3 from t={found_at}")
    for c in loc.candidates(5):
        print(f"   id={c.id} bbox={c.bbox} score={c.score:.3f} morse={c.morse:.2f} good={c.good_runs} "
              f"contrast={c.contrast:.0f} rate={c.rate:.2f}")
    return 0


def write_png(path: str, rgb: np.ndarray) -> None:
    """Minimal PNG writer (stdlib zlib) for snapshots."""
    import struct
    import zlib

    rgb = np.ascontiguousarray(rgb, np.uint8)
    h, w = rgb.shape[:2]
    raw = np.concatenate([np.zeros((h, 1), np.uint8), rgb.reshape(h, w * 3)], axis=1).tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    with open(path, "wb") as fh:
        fh.write(bytes([137, 80, 78, 71, 13, 10, 26, 10]) + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                 + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def _snap(a) -> int:
    spec = SceneSpec(width=a.width, height=a.height, word=a.word, zoomed=a.zoomed,
                     distractors=ALL_DISTRACTORS, seed=a.seed)
    scr = FakeScreen(spec)
    t = a.t
    if not scr.lamp_on(t) and spec.word:   # find a lit moment for the snapshot
        for k in range(200):
            if scr.lamp_on(t + 0.05 * k):
                t += 0.05 * k
                break
    f = scr.render(t)
    write_png(a.out, f)
    print(f"wrote {a.out} t={t:.2f} lamp at {scr.lamp_center(t)} diameter {scr.lamp_diameter:.1f}px")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bench", "snap"):
        p = sub.add_parser(name)
        p.add_argument("--width", type=int, default=2560)
        p.add_argument("--height", type=int, default=1440)
        p.add_argument("--zoomed", action="store_true")
        p.add_argument("--word", default="shell")
        p.add_argument("--seed", type=int, default=0)
        if name == "bench":
            p.add_argument("--seconds", type=float, default=10.0)
            p.add_argument("--fps", type=float, default=40.0)
            p.add_argument("--bgra", action="store_true", help="feed the RGB view of a BGRA buffer, like mss")
        else:
            p.add_argument("out")
            p.add_argument("--t", type=float, default=1.0)
    a = ap.parse_args(argv)
    return _bench(a) if a.cmd == "bench" else _snap(a)


if __name__ == "__main__":
    sys.exit(main())
