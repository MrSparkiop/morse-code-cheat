"""Hidden holdout benchmark suite for the Morse decoders.

    python -m tools.bench MODULE --suite holdout [--quick] [--seed N] [--show-failures]

The public suite (ktane_morse.simulate) samples a continuous on/off signal with
continuous element jitter. This suite instead simulates the whole chain frame
by frame, the way it happens on a real PC:

  game        frames on a vsync grid (60 / 59.94 / 120 / 144 / 165 Hz), behind a
              60 fps frame cap on a high-refresh monitor, unlocked, or a slow
              machine (38-58 fps); deltaTime noise, missed vsyncs, stalls. The
              Morse coroutine is resolved on those frames with WaitForSeconds
              semantics (resume on the first frame whose scaled time is >= start
              + duration), so every element is a whole number of game frames,
              exactly as the game makes them (at 60 fps: dot 17 frames, dash 49,
              letter gap 65, word gap 162/163). Pauses freeze scaled time.
  compositor  frames reach the screen on the monitor refresh grid.
  capture     free-running, paced (incl. 60/60, 59/60, 144/60 beats), 15.6 ms
              timer-tick, and desktop-duplication capture loops; jitter, dropped
              frames, stalls, optional 1 ms timestamp quantisation; each sample
              shows the newest frame on screen a few ms before its timestamp.
  level       ON = lit lamp (unlit shader) plus a little room-light leak, OFF =
              room-lit lamp and module; lights-out / emergency lights, red LED
              strike flashes, hover highlights, frequency-display changes, cursor
              passes, camera settling, bloom, zoom, hand sway, noise, quantisation;
              garbage while the camera moves / the bomb is turned away / the
              game is paused; None, black, frozen or foreign frames on alt-tab.

Every positive gets an oracle verdict that decides ``must_answer``. The oracle
knows the true element boundaries on screen and marks an element readable when
it lies in a stretch with valid samples (no gap > GAP_MAX = 0.6 unit, no game
stall), no garbage/pause/dropout/cursor/solved, and a single level threshold
that separates on from off with a noise-dependent margin within +-1 s. The
scenario is must_answer when >= 1.5 consecutive cycles are readable, or when
the readable pieces (each >= 0.1 cycle) of one game-time block (pauses split
blocks) cover every element of the cycle and add up to >= 1.75 cycles: the
timestamps keep running through dropouts / hitches / looks away while the
light keeps looping, so an ideal decoder can stitch them. Otherwise silence is
fine but a wrong answer is still a failure. (Checked independently at sample
level: >99.7 % of readable elements classify correctly with a genie threshold,
and every must_answer positive keeps >= 1.5 consecutive or >= 1.75 stitched
genie-verified cycles.)

Families, named holdout:<family>:<word> (negatives holdout:<family>:none);
full counts in FAMILIES, quick = the first quarter (rounded up) of each family:

  positives (must_answer from the oracle; the rest may stay silent)
    beat60    capture at / near the game rate: 60/60 paced grabbing at the flip,
              59-59.94/60, 59.94 Hz monitor, desktop duplication, 1 ms stamps
    beat144   60 fps game seen by 144/120/30 Hz capture, 60 fps cap on 144 Hz
    tick      capture loop paced by 15.6 ms Windows timer ticks (+1 ms stamps)
    lowfps    slow machine: 38-58 fps game with stalls, 18-30 fps capture
    phase     first sample at an awkward phase (early/mid/late word gap,
              mid-dash, mid-dot, right after a dot, mid letter gap, ...)
    short     1-2.1 cycles only (mostly optional: tests restraint)
    lowcon    contrast 8-15 level units with noise, quantised levels
    drift     camera settling, bloom, zoom, slow ramps
    strike    1-4 red status-LED strike flashes (1 s) leaking into the region
    hover     hover highlight steps, frequency-display changes, cursor passes
    solved    light goes dark forever after 0.6-3.4 cycles (+green LED, look away)
    alttab    short dropouts every 1.5-8 s (None / black / frozen / constant)
    pause     game paused 2-25 s (Morse timing frozen, pause screen in region)
    lookaway  bomb turned to another module (Simon, timer, alarm, Button...)
              and back, incl. traces that start on the other module
    motion    camera/bomb motion garbage (flagged or not), new framing settling
    room      lights-out and emergency lights over the whole decode
    scaled    DotLength x0.85-0.96 / x1.04-1.2 (minority)
    combo     2-4 of the above at once
    long      3-5 min sessions: strikes, hovers, room events, dropouts, pauses,
              looks away; 40 % solved before the end (stability, flips)
  negatives (any answer = FP)
    simon, button (strip held 1-5 s), needy (0.1 s panic + digits), vent (0.4 s),
    alarm (0.25 s), panic (0.1 s), capacitor (slow glow ramps), timer digits,
    emergency lights, lights out, solvedstatic (dark lamp + hovers/room),
    hoveruser (hover highlight blinking on a dark lamp), noise
  soft negatives (reported separately)
    gibberish (non-repeating Morse-timed letters), gibloop (a looping
    non-dictionary word, half of them one letter away from a real word)
"""

from __future__ import annotations

import math
import random
import zlib
from dataclasses import dataclass, field

import numpy as np

from ktane_morse.morse import WORDS
from ktane_morse.simulate import (GAME_TIMING, Scenario, cycle_length, pattern_segments,
                                  random_morse_segments, simon_segments, word_schedule)

UNIT = GAME_TIMING.dot
WORD_LIST = list(WORDS)
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# oracle (see _oracle): an element is readable when no gap between valid samples
# around it exceeds GAP_MAX units and one threshold separates ON from OFF with
# margin; must_answer needs MUST_RUN consecutive readable cycles, or readable
# pieces (each >= STITCH_MIN cycles) of one game-time block covering every
# element and adding up to STITCH_TOTAL cycles.
GAP_MAX = 0.6
MARGIN_SIGMA = 4.5
MUST_RUN = 1.5
STITCH_MIN = 0.1
STITCH_TOTAL = 1.75


# --------------------------------------------------------------------------- #
# Scenario description

@dataclass
class Game:
    mode: str = "vsync"            # "vsync": frames on a refresh grid; "free": capped / unlocked frame rate
    hz: float = 60.0
    jitter: float = 0.0            # free: relative std-dev of the frame time
    miss_prob: float = 0.0         # vsync: chance a frame misses its refresh
    hitch_prob: float = 0.0        # chance per frame of a 50-300 ms stall
    dt_noise: float = 2e-4         # std-dev of the frame timestamp noise seen by Time.time (s)
    present_lat: float = 0.012     # render + present latency (s)
    monitor_hz: float | None = None  # free mode: compositor refresh grid (None: shown immediately)
    tear: float = 0.0              # chance that the first capture of a new state is a half-updated frame

    def desc(self) -> str:
        s = f"game={self.mode}{self.hz:g}"
        if self.monitor_hz:
            s += f"/mon{self.monitor_hz:g}"
        if self.hitch_prob:
            s += "+stalls"
        return s


@dataclass
class Cap:
    fps: float = 60.0
    mode: str = "free"             # free | paced | tick | dxgi
    jitter: float = 0.1            # free: relative std-dev; paced/tick: absolute std-dev (s)
    drop_prob: float = 0.0
    hitch_prob: float = 0.0
    ts_quantum: float = 0.0
    grab_delay: tuple[float, float] = (0.0005, 0.004)
    ppm: float = 0.0               # paced: clock-rate error of the capture loop
    align: float | None = None     # paced: first grab this long after a screen update (s)

    def desc(self) -> str:
        s = f"cap={self.mode}{self.fps:g}"
        if self.ts_quantum:
            s += f"/q{self.ts_quantum * 1000:g}ms"
        return s


@dataclass
class Ev:
    kind: str
    start: float
    length: float = 0.0
    p: dict = field(default_factory=dict)

    def desc(self) -> str:
        extra = self.p.get("mode") or self.p.get("style") or ""
        return f"{self.kind}{'/' + extra if extra else ''}@{self.start:.1f}+{self.length:.1f}"


@dataclass
class Spec:
    family: str
    word: str | None               # the real word flashed (scored); None for negatives
    duration: float
    game: Game
    cap: Cap
    off0: float = 40.0             # OFF level under normal room light
    contrast: float = 100.0        # ON - OFF under normal room light
    k_on: float = 0.1              # how much of the room-light loss also shows in the ON level
    noise: float = 1.0
    level_q: float = 0.0           # level quantisation (mean of few pixels)
    scale: float = 1.0             # DotLength / 0.27
    start: str = "random"          # phase of the first sample (see _choose_t0)
    events: list[Ev] = field(default_factory=list)
    fake: str | None = None        # looping non-dictionary word (soft negative)
    gib: bool = False              # non-repeating Morse gibberish (soft negative)
    segs: list | None = None       # on/off distractor, in capture time
    soft: bool = False
    notes: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Small helpers

def _schedule(word: str, scale: float = 1.0):
    sched = word_schedule(word, GAME_TIMING.scaled(scale))
    u = UNIT * scale
    ons = [on for on, _ in sched]
    durs = [d for _, d in sched]
    kinds = []
    for on, d in sched:
        r = d / u
        kinds.append(("dot" if r < 2 else "dash") if on else ("intra" if r < 2 else "letter" if r < 7 else "word"))
    return ons, durs, kinds


def _cyc(word: str, scale: float = 1.0) -> float:
    """Approximate on-screen cycle length (nominal + average frame quantisation)."""
    return cycle_length(word, GAME_TIMING.scaled(scale)) + 0.0105 * len(word_schedule(word))


def _fill(on_iv, pad: float = 1e9) -> list[tuple[float, float, bool]]:
    """Sorted, merged ON intervals -> contiguous (start, end, on) segments."""
    segs = []
    t = -pad
    for a, b in sorted(on_iv):
        a = max(a, t)
        if b <= a:
            continue
        if a > t:
            segs.append((t, a, False))
        segs.append((a, b, True))
        t = b
    segs.append((t, pad, False))
    return segs


def _on_parts(segs) -> list[tuple[float, float]]:
    return [(a, b) for a, b, on in segs if on]


def _segstate(segs, x: np.ndarray) -> np.ndarray:
    st = np.array([s for s, _, _ in segs])
    en = np.array([e for _, e, _ in segs])
    on = np.array([o for _, _, o in segs], dtype=bool)
    i = np.searchsorted(st, x, side="right") - 1
    ic = np.clip(i, 0, len(st) - 1)
    return (i >= 0) & on[ic] & (x < en[ic])


def _rw(n: int, nrng: np.random.Generator, start: float | None = None) -> np.ndarray:
    """Garbage levels of a moving picture: a reflected random walk in 0..255."""
    if n <= 0:
        return np.zeros(0)
    start = nrng.uniform(0, 255) if start is None or not np.isfinite(start) else start
    y = start + np.cumsum(nrng.normal(0, nrng.uniform(4, 30), n))
    y = np.mod(y, 510.0)
    return np.where(y > 255, 510 - y, y)


def _merge(iv) -> np.ndarray:
    out: list[list[float]] = []
    for s, e in sorted(iv):
        if e <= s:
            continue
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return np.array(out, dtype=float).reshape(-1, 2)


def _hits(M: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Does [a, b] overlap any merged interval in M?"""
    if len(M) == 0:
        return np.zeros(len(a), dtype=bool)
    i = np.searchsorted(M[:, 1], a, side="right")
    ic = np.minimum(i, len(M) - 1)
    return (i < len(M)) & (M[ic, 0] < b)


def _place(rng: random.Random, busy: list, length: float, lo: float, hi: float, margin: float = 1.5):
    """Random start in [lo, hi - length] not overlapping `busy` (with margin); None if no room."""
    for _ in range(60):
        if hi - length <= lo:
            return None
        s = rng.uniform(lo, hi - length)
        if all(s + length + margin <= a or s >= b + margin for a, b in busy):
            busy.append((s, s + length))
            return s
    return None


# --------------------------------------------------------------------------- #
# Game frames, Morse coroutine, capture

def _game_frames(g: Game, W: float, nrng: np.random.Generator):
    """Frame times F (wall), scaled game time T (no pauses) and screen times S over [0, W]."""
    if g.mode == "vsync":
        R = 1.0 / g.hz
        n = int(W * g.hz) + 64
        inc = np.ones(n, dtype=np.int64)
        if g.miss_prob:
            inc += (nrng.random(n) < g.miss_prob).astype(np.int64)
        if g.hitch_prob:
            h = nrng.random(n) < g.hitch_prob
            inc[h] += np.round(nrng.uniform(0.05, 0.3, int(h.sum())) / R).astype(np.int64)
        idx = np.cumsum(inc) - inc[0]
        F = idx * R + nrng.uniform(0, R)
        F = F[F < W]
        S = F + g.present_lat
    else:
        n = int(W * g.hz * 1.4) + 64
        dt = (1.0 / g.hz) * np.maximum(0.3, 1 + nrng.normal(0, g.jitter, n))
        if g.hitch_prob:
            h = nrng.random(n) < g.hitch_prob
            dt[h] += nrng.uniform(0.05, 0.3, int(h.sum()))
        F = np.concatenate([[0.0], np.cumsum(dt)[:-1]]) + nrng.uniform(0, 1.0 / g.hz)
        F = F[F < W]
        S = F + g.present_lat + nrng.uniform(0, 0.25 / g.hz, len(F))
        if g.monitor_hz:
            Rm = 1.0 / g.monitor_hz
            ph = nrng.uniform(0, Rm)
            S = np.ceil((S - ph) / Rm) * Rm + ph
        S = np.maximum.accumulate(S)
    assert F[-1] > W - 1.0
    T = np.maximum.accumulate(F - F[0] + nrng.normal(0, g.dt_noise, len(F)))
    DT = np.clip(np.diff(T, prepend=T[0]), 0, 1 / 3)  # Unity clamps deltaTime (maximumDeltaTime)
    return F, np.cumsum(DT), S


def _coroutine(T: np.ndarray, durs) -> np.ndarray:
    """Frame index at which each element starts.

    WaitForSeconds(d) yielded on frame k resumes on the first frame j > k with
    T[j] >= T[k] + d; the light changes on that frame.
    """
    n = len(durs)
    out = []
    k, e, N = 0, 0, len(T)
    while k < N:
        out.append(k)
        j = int(np.searchsorted(T, T[k] + durs[e % n], side="left"))
        k = max(k + 1, j)
        e += 1
    return np.array(out, dtype=np.int64)


PHASES = ["gap_early", "gap_mid", "gap_late", "mid_dash", "mid_dot", "after_dot", "mid_letter_gap",
          "before_word_gap", "letter_start"]


def _choose_t0(sp: Spec, rng: random.Random, starts, S, ons, kinds, cyc, t0_max) -> float:
    if sp.start == "random" or cyc is None:
        return rng.uniform(4.0, 4.0 + 1.25 * (cyc or 10.0))
    n = len(kinds)
    cand = []
    for e in range(len(starts) - 1):
        a = S[starts[e]]
        if a < 5.0 or a > t0_max - 1.0:
            continue
        k, prev, nxt = kinds[e % n], kinds[(e - 1) % n], kinds[(e + 1) % n]
        ok = {
            "gap_early": k == "word", "gap_mid": k == "word", "gap_late": k == "word",
            "mid_dash": k == "dash", "mid_dot": k == "dot",
            "after_dot": (not ons[e % n]) and prev == "dot",
            "mid_letter_gap": k == "letter", "before_word_gap": nxt == "word",
            "letter_start": ons[e % n] and prev in ("letter", "word"),
        }[sp.start]
        if ok:
            cand.append(e)
    if not cand:
        return rng.uniform(4.0, 4.0 + cyc)
    e = rng.choice(cand)
    a, b = S[starts[e]], S[starts[e + 1]]
    d = b - a
    off = {
        "gap_early": lambda: rng.uniform(0.05, 0.6),
        "gap_mid": lambda: rng.uniform(0.9, 1.8),
        "gap_late": lambda: d - rng.uniform(0.02, 0.3),
        "mid_dash": lambda: rng.uniform(0.1, d - 0.08),
        "mid_dot": lambda: rng.uniform(0.04, d - 0.04),
        "after_dot": lambda: rng.uniform(0.0, 0.07),
        "mid_letter_gap": lambda: rng.uniform(0.15, d - 0.1),
        "before_word_gap": lambda: rng.uniform(0.0, d - 0.01),
        "letter_start": lambda: -rng.uniform(0.0, 0.04),
    }[sp.start]()
    return a + off


def _capture_times(cap: Cap, dur: float, S_rel: np.ndarray, nrng: np.random.Generator):
    """Capture timestamps c (relative to capture start) and the content time of each grab."""
    gd0, gd1 = cap.grab_delay
    if cap.mode == "free":
        n = int(dur * cap.fps * 1.3) + 100
        dt = (1.0 / cap.fps) * np.maximum(0.2, 1 + nrng.normal(0, cap.jitter, n))
        if cap.hitch_prob:
            h = nrng.random(n) < cap.hitch_prob
            dt[h] += nrng.uniform(0.05, 0.25, int(h.sum()))
        c = nrng.uniform(0, 1.0 / cap.fps) + np.concatenate([[0.0], np.cumsum(dt)[:-1]])
    elif cap.mode == "paced":
        P = (1.0 / cap.fps) * (1 + cap.ppm)
        if cap.align is not None:
            first = S_rel[min(len(S_rel) - 1, int(np.searchsorted(S_rel, 0.0)))]
            c0 = first + cap.align + 0.5 * (gd0 + gd1)
            while c0 < 0:
                c0 += P
        else:
            c0 = nrng.uniform(0, P)
        n = int(dur / P) + 2
        c = c0 + np.arange(n) * P + nrng.normal(0, cap.jitter, n)
        if cap.hitch_prob:
            keep = np.ones(n, dtype=bool)
            for j in np.nonzero(nrng.random(n) < cap.hitch_prob)[0]:
                h = nrng.uniform(0.05, 0.25)
                keep[(c > c[j]) & (c < c[j] + h)] = False  # paced loop skips the slots it missed
            c = c[keep]
    elif cap.mode == "tick":
        tick = 1 / 64  # default Windows timer resolution: sleeps end on 15.625 ms ticks
        p2 = min(1.0, max(0.0, 64 / cap.fps - 1))
        n = int(dur / tick) + 4
        k = 1 + (nrng.random(n) < p2).astype(np.int64)
        if cap.hitch_prob:
            h = nrng.random(n) < cap.hitch_prob
            k[h] += np.ceil(nrng.uniform(0.05, 0.25, int(h.sum())) / tick).astype(np.int64)
        grid = np.cumsum(k) * tick - nrng.uniform(0, tick)
        c = grid + nrng.uniform(0.0004, 0.003, n) + nrng.normal(0, cap.jitter, n)
    elif cap.mode == "dxgi":
        u = np.unique(S_rel)
        u = u[(u >= -0.002) & (u < dur)]
        keep, last = [], -1.0
        for v in u:  # desktop duplication: one frame per screen update, at most cap.fps
            if v - last >= 0.98 / cap.fps:
                keep.append(v)
                last = v
        c = np.array(keep) + nrng.uniform(0.0003, 0.002, len(keep))
    else:
        raise ValueError(cap.mode)
    c = np.sort(c)
    c = c[(c >= 0) & (c < dur)]
    if cap.drop_prob:
        c = c[nrng.random(len(c)) >= cap.drop_prob]
    if cap.mode == "dxgi":
        gd = np.full(len(c), 0.0001)
    else:
        gd = nrng.uniform(gd0, gd1, len(c))
    return c, c - gd


# --------------------------------------------------------------------------- #
# Level model

def _room(sp: Spec, x: np.ndarray) -> np.ndarray:
    """Room light factor (1 = normal) at relative times x."""
    L = np.ones_like(x)
    for ev in sp.events:
        if ev.kind == "lights_out":
            s = ev.start
            a = s + 1.57
            b = a + ev.length
            e = b + 0.61
            m = (x >= s) & (x < e)
            if m.any():
                xm = x[m]
                tab = ev.p["tab"]
                v = tab[np.floor((xm - s) / 0.06).astype(np.int64) % len(tab)]
                v = np.where((xm >= a) & (xm < b), ev.p["dark"], v)
                L[m] = np.minimum(L[m], v)
        elif ev.kind == "emergency":
            ps = ev.p["pulses"]
            if len(ps) == 0:
                continue
            edge = 0.12
            idx = np.searchsorted(ps, x, side="right") - 1
            u = x - ps[np.clip(idx, 0, None)]
            inside = (idx >= 0) & (u < 1.0)
            sh = np.where(u < edge, 0.5 - 0.5 * np.cos(np.pi * np.clip(u, 0, 1) / edge),
                          np.where(u > 1 - edge, 0.5 - 0.5 * np.cos(np.pi * np.clip(1 - u, 0, 1) / edge), 1.0))
            L = L * (1 - ev.p["pol"] * ev.p["depth"] * np.where(inside, sh, 0.0))
    return L


def _adds(sp: Spec, x: np.ndarray):
    """Additive level terms for OFF and ON, and a multiplicative framing scale."""
    off = np.zeros_like(x)
    on = np.zeros_like(x)
    sc = np.ones_like(x)
    for ev in sp.events:
        k, s, l, p = ev.kind, ev.start, ev.length, ev.p
        if k in ("hover", "strike"):
            m = (x >= s) & (x < s + l)
            off[m] += p["amp"]
            on[m] += p["amp"] * p["k"]
        elif k == "led":
            m = x >= s
            off[m] += p["amp"]
            on[m] += p["amp"]
        elif k == "steps":
            cum = np.concatenate([[0.0], np.cumsum(p["deltas"])])
            v = cum[np.searchsorted(p["times"], x, side="right")]
            off += v
            on += v * p["k"]
        elif k == "settle":
            v = np.where(x >= s, p["amp"] * np.exp(-np.clip(x - s, 0, None) / p["tau"]), 0.0)
            off += v
            on += v * p["k"]
        elif k == "bloom":
            v = p["amp"] * np.sin(2 * np.pi * x / p["period"] + p["ph"])
            off += v
            on += v * p["k"]
        elif k == "ramp":
            v = p["rate"] * np.clip(x - s, 0, l)
            off += v
            on += v
        elif k == "zoom":
            u = np.clip((x - s) / max(l, 1e-6), 0, 1)
            sc *= 1 + (p["f"] - 1) * u * u * (3 - 2 * u)
        elif k == "sway":
            v = p["amp"] * (np.sin(2 * np.pi * p["f"] * x + p["ph"]) + 0.4 * np.sin(2 * np.pi * 2.37 * p["f"] * x + p["ph2"]))
            off += v
            on += v * p["k"]
        elif k == "func":
            v = p["fn"](x)
            off += v
            on += v * p.get("k", 1.0)
    return off, on, sc


def _epochs(sp: Spec):
    """Level parameters change after the camera/bomb moves (the region frames a different view)."""
    times, offs, cons = [-1e18], [sp.off0], [sp.contrast]
    for ev in sorted((e for e in sp.events if e.kind in ("motion", "motion_flagged", "look_away")), key=lambda e: e.start):
        times.append(ev.start + ev.length)
        offs.append(offs[-1] * ev.p["off_f"])
        cons.append(cons[-1] * ev.p["con_f"])
    return np.array(times), np.array(offs), np.array(cons)


def _levels(sp: Spec, x: np.ndarray, ep):
    L = _room(sp, x)
    aoff, aon, sc = _adds(sp, x)
    i = np.searchsorted(ep[0], x, side="right") - 1
    o0, c0 = ep[1][i], ep[2][i]
    on = sc * (o0 + c0 - sp.k_on * o0 * (1 - L)) + aon
    off = sc * o0 * L + aoff
    return np.clip(on, 0, 255), np.clip(off, 0, 255)


# --------------------------------------------------------------------------- #
# Rendering + oracle

def _render(sp: Spec, rng: random.Random, nrng: np.random.Generator):
    dur = sp.duration
    looping = sp.word is not None or sp.fake is not None
    morse = looping or sp.gib
    cyc = None
    if looping:
        ons, durs, kinds = _schedule(sp.word or sp.fake, sp.scale)
        cyc = float(sum(durs))
    elif sp.gib:
        ons, durs, kinds = [], [], []
        while sum(durs) < dur + 120:
            w = "".join(rng.choice(LETTERS) for _ in range(rng.randint(3, 7)))
            o, d, k = _schedule(w, sp.scale)
            ons += o
            durs += d
            kinds += k
    t0_max = 6.0 + 2.6 * (cyc or 12.0)
    W = t0_max + dur + 2.0
    F, Tn, S = _game_frames(sp.game, W, nrng)
    N = len(F)
    pauses = sorted((e for e in sp.events if e.kind == "pause"), key=lambda e: e.start)
    starts = None
    if morse:
        starts = _coroutine(Tn, durs)
        t0 = _choose_t0(sp, rng, starts, S, ons, kinds, cyc, t0_max)
        if pauses:
            DT = np.diff(Tn, prepend=Tn[0])
            for ev in pauses:  # Time.timeScale = 0: WaitForSeconds does not advance
                DT[(F >= t0 + ev.start) & (F < t0 + ev.start + ev.length)] = 0.0
            starts = _coroutine(np.cumsum(DT), durs)
        eidx = np.searchsorted(starts, np.arange(N), side="right") - 1
        state = np.asarray(ons, dtype=bool)[eidx % len(ons)]
    else:
        t0 = rng.uniform(4.0, 10.0)
        state = _segstate(sp.segs, F - t0) if sp.segs else np.zeros(N, dtype=bool)
    solved = min((e.start for e in sp.events if e.kind == "solved"), default=math.inf)
    if solved < math.inf:
        state = state & (F < t0 + solved)

    S_rel = S - t0
    c, x = _capture_times(sp.cap, dur, S_rel, nrng)
    kf = np.clip(np.searchsorted(S_rel, x, side="right") - 1, 0, N - 1)
    st = state[kf]
    ep = _epochs(sp)
    ON, OFF = _levels(sp, x, ep)
    lvl = np.where(st, ON, OFF)
    if sp.game.tear > 0 and len(kf) > 1:
        prevk = np.concatenate([[-1], kf[:-1]])
        new = (kf > 0) & (prevk < kf) & (state[kf] != state[np.maximum(kf - 1, 0)])
        cand = new & (nrng.random(len(kf)) < sp.game.tear)
        fr = nrng.uniform(0.15, 0.85, int(cand.sum()))
        lvl[cand] = fr * lvl[cand] + (1 - fr) * np.where(st[cand], OFF[cand], ON[cand])
    lvl = lvl + nrng.normal(0, sp.noise, len(lvl))

    for ev in sorted(sp.events, key=lambda e: ["cursor", "pause", "look_away", "motion", "motion_flagged", "dropout"].index(e.kind)
                     if e.kind in ("cursor", "pause", "look_away", "motion", "motion_flagged", "dropout") else -1):
        s, e = ev.start, ev.start + ev.length
        if ev.kind == "cursor":
            m = (x >= s) & (x < e)
            lvl[m] += ev.p["amp"]
        elif ev.kind == "pause":
            m = (x >= s) & (x < e)
            if m.any():
                xm = x[m]
                fade = np.clip(np.minimum(xm - s, e - xm) / 0.25, 0, 1)
                style = ev.p["style"]
                if style == "dim":
                    over = lvl[m] * ev.p["f"]
                elif style == "const":
                    over = ev.p["lvl"] + nrng.normal(0, 0.4, int(m.sum()))
                else:
                    over = _rw(int(m.sum()), nrng, float(lvl[m][0]))
                lvl[m] = fade * over + (1 - fade) * lvl[m]
        elif ev.kind == "look_away":
            m = (x >= s) & (x < e)
            if m.any():
                xm = x[m]
                v = ev.p["away"](xm) + nrng.normal(0, max(0.3, sp.noise), len(xm))
                garb = (xm < s + ev.p["g1"]) | (xm >= e - ev.p["g2"])
                v[garb] = _rw(int(garb.sum()), nrng)
                lvl[m] = v
        elif ev.kind in ("motion", "motion_flagged"):
            m = (x >= s) & (x < e)
            lvl[m] = _rw(int(m.sum()), nrng) if ev.kind == "motion" else np.nan
        elif ev.kind == "dropout":
            m = (c >= s) & (c < e)
            mode = ev.p["mode"]
            if mode == "none":
                lvl[m] = np.nan
            elif mode == "black":
                lvl[m] = nrng.uniform(0, 1.0, int(m.sum()))
            elif mode == "const":
                lvl[m] = ev.p["lvl"] + nrng.normal(0, 0.3, int(m.sum()))
            elif mode == "stale":
                prior = np.nonzero((c < s) & ~np.isnan(lvl))[0]
                lvl[m] = lvl[prior[-1]] if len(prior) else np.nan

    lvl = np.clip(lvl, 0, 255)
    if sp.level_q:
        lvl = np.round(lvl / sp.level_q) * sp.level_q
    t = c
    if sp.cap.ts_quantum:
        t = np.round(t / sp.cap.ts_quantum) * sp.cap.ts_quantum
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, lvl, x, c = t[keep], lvl[keep], x[keep], c[keep]
    trace = [(float(a), None if b != b else float(b)) for a, b in zip(t, lvl)]

    info = {"t": t, "lvl": lvl, "t0": t0, "cyc": cyc}
    if sp.word is not None:
        info.update(_oracle(sp, t, lvl, ON[keep], OFF[keep], S_rel, starts, durs, cyc, ep, pauses, solved))
    return trace, info


def _oracle(sp: Spec, t, lvl, ON_s, OFF_s, S_rel, starts, durs, cyc, ep, pauses, solved) -> dict:
    dur = sp.duration
    n = len(durs)
    a = S_rel[starts[:-1]]
    b = S_rel[starts[1:]]
    eid = np.arange(len(starts) - 1)
    sel = (b > 0) & (a < dur)
    a, b, eid = a[sel], b[sel], eid[sel]
    frac = np.asarray(durs)[eid % n] / cyc

    bad = []
    for ev in sp.events:
        if ev.kind in ("motion", "motion_flagged", "look_away", "dropout", "cursor"):
            bad.append((ev.start - 0.02, ev.start + ev.length + 0.02))
        elif ev.kind == "pause":
            bad.append((ev.start - 0.05, ev.start + ev.length + 0.05))
    if solved < math.inf:
        bad.append((solved - 0.01, math.inf))
    for g in np.nonzero(np.diff(S_rel) > 0.06)[0]:  # game stall: screen frozen, element stretched
        if S_rel[g + 1] > -1 and S_rel[g] < dur + 1:
            bad.append((S_rel[g] - 0.02, S_rel[g + 1] + 0.02))
    M0 = _merge(bad)
    valid = ~np.isnan(lvl) & ~_hits(M0, t, t)
    tv = t[valid]
    res = {"best": 0.0, "total": 0.0, "stitched": False, "must": False, "bad": M0}
    if len(tv) < 10:
        return res
    gmax = GAP_MAX * UNIT * sp.scale
    d = np.diff(tv)
    for i in np.nonzero(d > gmax)[0]:
        bad.append((tv[i], tv[i + 1]))
    bad.append((-math.inf, tv[0]))
    bad.append((tv[-1], math.inf))

    # a single threshold must separate ON and OFF (noise-free) within +-1 s, with margin
    xg = np.arange(0.0, dur, 0.02)
    ONg, OFFg = _levels(sp, xg, ep)
    h = 50
    lo = np.lib.stride_tricks.sliding_window_view(np.pad(ONg, h, mode="edge"), 2 * h + 1).min(-1)
    hi = np.lib.stride_tricks.sliding_window_view(np.pad(OFFg, h, mode="edge"), 2 * h + 1).max(-1)
    fps_eff = 1.0 / max(1e-4, float(np.median(d)))
    n_avg = min(6.0, max(1.0, fps_eff * 0.06))
    sig = math.sqrt(sp.noise ** 2 + sp.level_q ** 2 / 12)
    need = max(3.0, MARGIN_SIGMA * sig * math.sqrt(2.0 / n_avg))
    fail = (lo - hi) < need
    if fail.any():
        edges = np.diff(np.concatenate([[0], fail.astype(np.int8), [0]]))
        for i0, i1 in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            bad.append((xg[i0] - 0.011, xg[i1 - 1] + 0.011))
    M = _merge(bad)
    obs = ~_hits(M, a - 0.02, b + 0.02)

    # runs of consecutive readable elements; a pause (game time frozen behind the
    # pause screen) splits the stream into blocks that cannot be stitched together
    block = np.searchsorted(np.array([p.start for p in pauses] or [math.inf]), a, side="right")
    runs: list[list] = []  # [block, cycles, element ids]
    prev = False
    for k in range(len(obs)):
        if obs[k] and prev and runs[-1][0] == block[k]:
            runs[-1][1] += frac[k]
            runs[-1][2].add(int(eid[k] % n))
        elif obs[k]:
            runs.append([block[k], frac[k], {int(eid[k] % n)}])
        prev = bool(obs[k])
    best = max((r[1] for r in runs), default=0.0)
    # Stitching across short interruptions: the timestamps keep running while the
    # light keeps looping, so an ideal decoder can align readable pieces of one
    # block. Pieces shorter than STITCH_MIN cycles are ignored (too little to align).
    total, stitched = 0.0, False
    for bk in np.unique(block):
        pieces = [r for r in runs if r[0] == bk and r[1] >= STITCH_MIN]
        tot = float(sum(r[1] for r in pieces))
        total = max(total, tot)
        cover = set().union(*(r[2] for r in pieces)) if pieces else set()
        if len(cover) == n and tot >= STITCH_TOTAL:
            stitched = True
    res.update(best=best, total=total, stitched=stitched, must=bool(best >= MUST_RUN or stitched), bad=M,
               need=need, elems=(a, b, eid % n, obs))
    return res


# --------------------------------------------------------------------------- #
# Building blocks for the families

def _game(rng: random.Random, kind: str | None = None) -> Game:
    kind = kind or rng.choices(["v60", "cap60", "v144", "free", "low"], [50, 15, 10, 15, 10])[0]
    lat = rng.uniform(0.006, 0.035)
    dtn = rng.uniform(5e-5, 8e-4)
    if kind == "v60":
        return Game("vsync", rng.choice([60.0, 60.0, 60.0, 59.94, 60.02]), miss_prob=rng.choice([0.0, 0.002, 0.01]),
                    hitch_prob=rng.choice([0.0, 0.0, 3e-4]), dt_noise=dtn, present_lat=lat)
    if kind == "cap60":  # 60 fps frame limiter on a high-refresh monitor
        return Game("free", 60.0, jitter=rng.uniform(0.01, 0.06), monitor_hz=rng.choice([120.0, 144.0, 165.0]),
                    dt_noise=dtn, present_lat=lat)
    if kind == "v144":
        return Game("vsync", rng.choice([144.0, 120.0, 165.0, 143.86]), miss_prob=rng.choice([0.0, 0.005, 0.02]),
                    dt_noise=dtn, present_lat=lat)
    if kind == "free":
        mon = rng.choice([None, 60.0, 144.0, 240.0])
        return Game("free", rng.uniform(90, 240), jitter=rng.uniform(0.05, 0.25), monitor_hz=mon,
                    tear=rng.uniform(0, 0.4) if mon is None else 0.0, dt_noise=dtn, present_lat=lat)
    if kind == "low":
        return Game("free", rng.uniform(38, 58), jitter=rng.uniform(0.08, 0.3), monitor_hz=60.0,
                    hitch_prob=rng.uniform(2e-4, 2e-3), dt_noise=dtn, present_lat=lat)
    raise ValueError(kind)


def _cap(rng: random.Random, kind: str | None = None, max_fps: float = 1e9) -> Cap:
    kind = kind or rng.choices(["free", "paced", "dxgi"], [60, 30, 10])[0]
    # stalls of the capture loop (GC, mss/GDI hiccups): 0-0.18 per second whatever the frame rate
    if kind == "free":
        fps = min(rng.choice([30, 45, 60, 60, 60, 75, 90, 120, 144, 240]), max_fps)
        return Cap(fps, "free", jitter=rng.uniform(0.03, 0.25), drop_prob=rng.uniform(0, 0.012),
                   hitch_prob=rng.uniform(0, 0.18) / fps)
    if kind == "paced":
        fps = min(rng.choice([60.0, 60.0, 59.94, 30.0, 120.0, 144.0]), max_fps)
        return Cap(fps, "paced", jitter=rng.uniform(1e-4, 1e-3), drop_prob=rng.uniform(0, 0.008),
                   hitch_prob=rng.uniform(0, 0.12) / fps, ppm=rng.uniform(-300, 300) * 1e-6)
    if kind == "dxgi":
        return Cap(min(rng.choice([60.0, 144.0, 240.0]), max_fps), "dxgi", drop_prob=rng.uniform(0, 0.005))
    raise ValueError(kind)


def _sway(rng: random.Random, amp: float) -> Ev:
    return Ev("sway", 0.0, 0.0, {"amp": amp, "f": rng.uniform(0.15, 1.2), "ph": rng.uniform(0, 6.3),
                                 "ph2": rng.uniform(0, 6.3), "k": rng.uniform(0.3, 1.0)})


def _base(rng: random.Random, fam: str, word: str | None, cycles: float = 1.0, scale: float = 1.0) -> Spec:
    sp = Spec(fam, word, 0.0, _game(rng), _cap(rng))
    sp.off0 = rng.uniform(10, 80)
    sp.contrast = rng.uniform(35, 160)
    sp.k_on = rng.uniform(0, 0.25)
    sp.noise = rng.uniform(0.3, 2.2)
    sp.level_q = rng.choice([0.0, 0.0, 0.0, 0.25, 1 / 16, 1 / 32])
    sp.scale = scale
    sp.events.append(_sway(rng, sp.contrast * rng.uniform(0, 0.05)))
    if word is not None:
        sp.duration = cycles * _cyc(word, scale)
    return sp


def _lights_out(rng: random.Random, s: float) -> Ev:
    return Ev("lights_out", s, rng.uniform(5, 10), {"tab": np.array([rng.uniform(0.2, 1.0) for _ in range(256)]),
                                                    "dark": rng.uniform(0.08, 0.18)})


def _emergency(rng: random.Random, s: float, length: float, depth: float | None = None, pol: float = 1.0) -> Ev:
    ps, t = [], s
    while t < s + length:
        ps.append(t)
        off = 2.0 - 1.25 * min(1.0, (t - s) / max(1.0, length)) if rng.random() < 0.5 else rng.uniform(0.75, 2.0)
        t += 1.0 * (1 + rng.gauss(0, 0.01)) + off
    return Ev("emergency", s, length, {"pulses": np.array(ps), "depth": depth or rng.uniform(0.3, 0.6), "pol": pol})


def _strike(rng: random.Random, s: float, con: float) -> Ev:
    return Ev("strike", s, rng.uniform(0.95, 1.05), {"amp": rng.uniform(0.15, 0.9) * con, "k": rng.uniform(0, 0.4)})


def _hover(rng: random.Random, s: float, con: float, length: float | None = None) -> Ev:
    amp = rng.uniform(0.1, 0.8) * con * (1 if rng.random() < 0.7 else -0.4)
    return Ev("hover", s, length if length is not None else rng.uniform(0.8, 6.0), {"amp": amp, "k": rng.uniform(0, 0.5)})


def _steps(rng: random.Random, s: float, con: float) -> Ev:
    n = rng.randint(1, 8)
    times = s + np.cumsum([rng.uniform(0.12, 0.5) for _ in range(n)])
    deltas = np.array([rng.choice([-1, 1]) * rng.uniform(0.03, 0.2) * con for _ in range(n)])
    return Ev("steps", s, float(times[-1] - s), {"times": times, "deltas": deltas, "k": rng.uniform(0.3, 1.0)})


def _cursor(rng: random.Random, s: float, con: float) -> Ev:
    return Ev("cursor", s, rng.uniform(0.03, 0.15), {"amp": rng.choice([-1, 1]) * rng.uniform(0.3, 1.2) * con})


def _dropout(rng: random.Random, s: float, mode: str | None = None) -> Ev:
    r = rng.random()
    length = rng.uniform(0.03, 0.25) if r < 0.7 else rng.uniform(0.25, 0.8) if r < 0.95 else rng.uniform(1.0, 3.0)
    mode = mode or rng.choice(["none", "none", "none", "black", "stale", "const"])
    return Ev("dropout", s, length, {"mode": mode, "lvl": rng.uniform(0, 200)})


def _pause(rng: random.Random, s: float, length: float) -> Ev:
    return Ev("pause", s, length, {"style": rng.choice(["dim", "const", "garbage"]), "f": rng.uniform(0.25, 0.5),
                                   "lvl": rng.uniform(0, 120)})


def _distractor_fn(rng: random.Random, D: float, con: float):
    """Something else in the region (another module) as a level function of relative time."""
    base = rng.uniform(0, 120)
    kind = rng.choice(["static", "simon", "timer", "alarm", "button"])
    if kind == "static":
        return kind, lambda x: np.full_like(x, base)
    if kind == "timer":
        fn = _timer_fn(rng, D + 10)
        return kind, lambda x: base + fn(x)
    if kind == "simon":
        segs = _fill(_on_parts(simon_segments(D + 10, rng)))
    elif kind == "alarm":
        segs = _bursts(rng, D + 10, 0.25, 0.25, (3, 15), (2, 10))
    else:
        segs = _fill(_holds(rng, D + 10))
    amp = rng.uniform(0.5, 1.5) * con
    return kind, lambda x: base + amp * _segstate(segs, x)


def _look_away(rng: random.Random, s: float, length: float, sp: Spec) -> Ev:
    kind, fn = _distractor_fn(rng, length + 1, sp.contrast)
    shift = s  # the away function is defined in time since the look started
    return Ev("look_away", s, length, {"away": lambda x, fn=fn, shift=shift: fn(x - shift), "mode": kind,
                                       "g1": rng.uniform(0.2, 0.8), "g2": rng.uniform(0.2, 0.8),
                                       "off_f": rng.uniform(0.75, 1.3), "con_f": rng.uniform(0.75, 1.15)})


def _motion(rng: random.Random, s: float, flagged: bool | None = None) -> Ev:
    flagged = rng.random() < 0.4 if flagged is None else flagged
    return Ev("motion_flagged" if flagged else "motion", s, rng.uniform(0.3, 2.5),
              {"off_f": rng.uniform(0.75, 1.3), "con_f": rng.uniform(0.8, 1.15)})


def _drift_events(rng: random.Random, sp: Spec, n: int | None = None) -> list[Ev]:
    con, D = sp.contrast, sp.duration
    out = []
    kinds = rng.sample(["settle", "bloom", "zoom", "ramp"], n or rng.randint(1, 3))
    for k in kinds:
        if k == "settle":
            out.append(Ev("settle", 0.0, 0.0, {"amp": rng.choice([-1, 1]) * rng.uniform(0.4, 1.5) * con,
                                               "tau": rng.uniform(1.5, 6.0), "k": rng.uniform(0.5, 1.0)}))
        elif k == "bloom":
            out.append(Ev("bloom", 0.0, 0.0, {"amp": rng.uniform(0.1, 0.4) * con, "period": rng.uniform(4, 20),
                                              "ph": rng.uniform(0, 6.3), "k": rng.uniform(0.0, 1.0)}))
        elif k == "zoom":
            l = rng.uniform(2, 8)
            out.append(Ev("zoom", rng.uniform(2, max(3, D - l - 2)), l, {"f": rng.uniform(0.55, 1.5)}))
        else:
            out.append(Ev("ramp", 0.0, D, {"rate": rng.choice([-1, 1]) * rng.uniform(0.3, 0.8) * con / max(D, 1)}))
    return out


def _add_strikes(rng, sp, n=None, lo=2.0):
    for s in sorted(rng.uniform(lo, max(lo + 1, sp.duration - 2)) for _ in range(n or rng.randint(1, 4))):
        if rng.random() < 0.6:  # cursor resting on TX just before pressing it
            sp.events.append(_hover(rng, s - rng.uniform(0.3, 1.5), sp.contrast * 0.3, rng.uniform(0.5, 2.5)))
        sp.events.append(_strike(rng, s, sp.contrast))


def _add_hovers(rng, sp, n=None):
    D = sp.duration
    for _ in range(n or rng.randint(2, 5)):
        sp.events.append(_hover(rng, rng.uniform(0, D - 1), sp.contrast))
    for _ in range(rng.randint(0, 3)):
        sp.events.append(_steps(rng, rng.uniform(0, D - 3), sp.contrast))
    for _ in range(rng.randint(0, 3)):
        sp.events.append(_cursor(rng, rng.uniform(0, D - 0.2), sp.contrast))


def _add_dropouts(rng, sp, busy, every=(1.5, 8.0)):
    t = rng.uniform(0.5, 4)
    mode = rng.choice([None, "none", "none", "black", "stale", "const"])
    while t < sp.duration - 0.5:
        ev = _dropout(rng, t, mode)
        sp.events.append(ev)
        busy.append((t, t + ev.length))
        t += ev.length + rng.uniform(*every)


def _add_room(rng, sp, busy=None):
    D = sp.duration
    n_lo = rng.randint(0, 2)
    n_em = rng.randint(0 if n_lo else 1, 1)
    for _ in range(n_lo):
        sp.events.append(_lights_out(rng, rng.uniform(0, max(1, D - 12))))
    for _ in range(n_em):
        L = rng.uniform(10, min(60, max(12, D * 0.6)))
        sp.events.append(_emergency(rng, rng.uniform(0, max(1, D - L)), L))


# --------------------------------------------------------------------------- #
# Distractor sources (negatives)

def _bursts(rng, D, on, off, blen, gap):
    iv = []
    t = rng.uniform(-blen[1], gap[1] * 0.5)
    while t < D + 1:
        end = t + rng.uniform(*blen)
        u = t
        while u < end:
            iv.append((u, u + on * (1 + rng.gauss(0, 0.01))))
            u += on + off
        t = u + rng.uniform(*gap)
    return _fill(iv)


def _holds(rng, D):
    """The Button: coloured strip lit while the button is held (1-5 s)."""
    iv = []
    t = rng.uniform(0.5, 8)
    while t < D + 1:
        if rng.random() < 0.2:  # quick tap: the strip never lights
            t += rng.uniform(0.1, 0.3)
        else:
            L = rng.uniform(1.0, 5.0)
            iv.append((t, t + L))
            t += L
        t += rng.uniform(3, 25)
    return iv


def _simon(rng, D):
    """Simon Says yellow button: sequence playback (0.65 s spacing), 4.5 s repeats, player presses."""
    colours = [rng.randrange(4) for _ in range(5)]
    if 0 not in colours[:3]:
        colours[rng.randrange(3)] = 0
    flash = rng.uniform(0.25, 0.45)
    iv = []
    t = rng.uniform(-3, 1)
    stage = rng.randint(1, 3)
    while t < D + 1 and stage <= 5:
        reps = rng.choice([0, 0, 1, 1, 2, 3])
        for r in range(reps + 1):
            for col in colours[:stage]:
                if col == 0:
                    iv.append((t, t + flash))
                t += 0.65
            t += 4.5 if r < reps else rng.uniform(0.8, 3.0)
        for col in colours[:stage]:
            press = rng.uniform(0.12, 0.35)
            if col == 0:
                iv.append((t, t + press))
            t += press + rng.uniform(0.2, 1.0)
        t += rng.uniform(0.6, 1.2)
        stage += 1
    return _fill(iv)


SEG7 = np.array([6, 2, 5, 5, 4, 5, 6, 3, 7, 6], dtype=float)


def _timer_fn(rng, D):
    """Bomb timer digits in the region: lit-segment count of the seconds digits; strikes speed it up."""
    R0 = rng.uniform(70 + D, 600)
    strikes = sorted(rng.uniform(5, D) for _ in range(rng.randint(0, 2)))
    speeds = [1.0, 1.25, 1.5]
    amp = rng.uniform(5, 60)
    w = rng.uniform(0, 1)
    seg_amp = [amp * rng.uniform(0.6, 1.0) for _ in range(2)]

    def fn(x):
        x = np.asarray(x, dtype=float)
        el = x.copy()
        for i, ts in enumerate(strikes):
            el += (speeds[i + 1] - speeds[i]) * np.clip(x - ts, 0, None)
        sec = np.floor(R0 - el).astype(np.int64) % 60
        return (seg_amp[0] * SEG7[sec % 10] + w * seg_amp[1] * SEG7[sec // 10]) / 7
    return fn


def _capacitor_fn(rng, D, amp):
    """Capacitor glow: slow fill (30-60 s), discharged by holding the lever (1-4 s)."""
    bt, bv = [0.0], [rng.uniform(0, 0.6)]
    t, v = 0.0, bv[0]
    while t < D + 5:
        rate = 1 / rng.uniform(30, 60)
        target = rng.uniform(0.4, 1.0)
        dt = max(0.5, (target - v) / rate)
        t += dt
        v = min(1.0, v + rate * dt)
        bt.append(t)
        bv.append(v)
        hold = rng.uniform(1, 4)
        t += hold
        v = max(0.0, v - rng.uniform(0.5, 1.0) * v)
        bt.append(t)
        bv.append(v)
    bt, bv = np.array(bt), np.array(bv)
    return lambda x: amp * np.interp(x, bt, bv)


def _neg_base(rng, fam, D=None) -> Spec:
    sp = _base(rng, fam, None)
    sp.duration = D or rng.uniform(45, 90)
    return sp


# --------------------------------------------------------------------------- #
# Families. Each builder: (rng, word, j) -> Spec

def f_beat60(rng, word, j):
    sp = _base(rng, "beat60", word, rng.uniform(1.8, 3.2))
    sub = j % 5
    g = Game("vsync", 60.0, miss_prob=rng.choice([0.0, 0.0, 0.003]), dt_noise=rng.uniform(5e-5, 4e-4),
             present_lat=rng.uniform(0.008, 0.03))
    if sub == 0:  # capture paced at the game rate, grabbing right at the screen update
        sp.cap = Cap(60.0, "paced", jitter=rng.uniform(1e-4, 6e-4), grab_delay=(0.0015, 0.002),
                     ppm=rng.uniform(-150, 150) * 1e-6, align=rng.uniform(-0.0008, 0.0008), drop_prob=rng.uniform(0, 0.003))
    elif sub == 1:  # capture slightly slower than the game: a game frame is skipped every ~second
        sp.cap = Cap(rng.choice([59.0, 59.5, 59.94, 58.0]), "paced", jitter=rng.uniform(1e-4, 8e-4))
    elif sub == 2:  # 59.94 Hz monitor, capture paced at 60
        g.hz = 59.94
        sp.cap = Cap(60.0, "paced", jitter=rng.uniform(1e-4, 8e-4), align=rng.uniform(-0.001, 0.001))
    elif sub == 3:  # desktop duplication: exactly one sample per game frame
        g.hz = rng.choice([60.0, 59.94])
        sp.cap = Cap(60.0, "dxgi", drop_prob=rng.uniform(0, 0.004))
    else:
        sp.cap = Cap(60.0, "free", jitter=rng.uniform(0.02, 0.1), ts_quantum=rng.choice([0.0, 0.001]),
                     drop_prob=rng.uniform(0, 0.01), hitch_prob=rng.uniform(0, 0.002))
    sp.game = g
    return sp


def f_beat144(rng, word, j):
    sp = _base(rng, "beat144", word, rng.uniform(1.8, 3.2))
    sub = j % 5
    if sub == 0:
        sp.game = Game("vsync", 60.0, dt_noise=rng.uniform(5e-5, 4e-4))
        sp.cap = Cap(144.0, "paced", jitter=rng.uniform(1e-4, 6e-4), ppm=rng.uniform(-200, 200) * 1e-6)
    elif sub == 1:  # 60 fps cap on a 144 Hz monitor: uneven 13.9/20.8 ms screen updates
        sp.game = _game(rng, "cap60")
        sp.game.monitor_hz = 144.0
        sp.cap = Cap(144.0, "free", jitter=rng.uniform(0.05, 0.15), drop_prob=rng.uniform(0, 0.01))
    elif sub == 2:
        sp.game = Game("vsync", 144.0, dt_noise=rng.uniform(5e-5, 4e-4))
        sp.cap = Cap(60.0, "paced", jitter=rng.uniform(1e-4, 8e-4))
    elif sub == 3:
        sp.game = Game("vsync", 60.0, dt_noise=rng.uniform(5e-5, 4e-4))
        sp.cap = Cap(120.0, "paced", jitter=rng.uniform(1e-4, 4e-4), align=rng.uniform(-0.0006, 0.0006),
                     grab_delay=(0.0015, 0.002))
    else:
        sp.game = Game("vsync", 60.0, dt_noise=rng.uniform(5e-5, 4e-4))
        sp.cap = Cap(30.0, "paced", jitter=rng.uniform(1e-4, 8e-4), ppm=rng.uniform(-300, 300) * 1e-6)
    return sp


def f_tick(rng, word, j):
    sp = _base(rng, "tick", word, rng.uniform(2.0, 3.2))
    sp.cap = Cap(rng.choice([64.0, 60.0, 50.0, 40.0, 32.0]), "tick", jitter=rng.uniform(5e-5, 5e-4),
                 ts_quantum=0.001 if j % 2 else 0.0, drop_prob=rng.uniform(0, 0.01), hitch_prob=rng.uniform(0, 0.003))
    return sp


def f_lowfps(rng, word, j):
    sp = _base(rng, "lowfps", word, rng.uniform(2.5, 4.0))
    sp.game = _game(rng, "low")
    sp.cap = Cap(rng.uniform(18, 30), "free", jitter=rng.uniform(0.1, 0.35), drop_prob=rng.uniform(0.01, 0.05),
                 hitch_prob=rng.uniform(0.002, 0.01))
    return sp


def f_phase(rng, word, j):
    sp = _base(rng, "phase", word, rng.uniform(1.55, 2.6))
    sp.start = PHASES[j % len(PHASES)]
    return sp


def f_short(rng, word, j):
    sp = _base(rng, "short", word, rng.uniform(1.0, 2.1))
    return sp


def f_lowcon(rng, word, j):
    sp = _base(rng, "lowcon", word, rng.uniform(2.0, 3.5))
    sp.contrast = rng.uniform(8, 15)
    sp.noise = rng.uniform(0.4, min(2.5, 0.18 * sp.contrast))
    sp.off0 = rng.uniform(5, 45)
    sp.k_on = rng.uniform(0, 0.1)
    sp.level_q = rng.choice([0.0, 0.0, 0.25, 1 / 16])
    sp.events = [_sway(rng, sp.contrast * rng.uniform(0, 0.08))]
    if sp.cap.fps < 30:
        sp.cap.fps = 60.0
    if rng.random() < 0.3:
        sp.events.append(_lights_out(rng, rng.uniform(0, max(1, sp.duration - 12))))
    if rng.random() < 0.25:
        sp.events.append(Ev("bloom", 0.0, 0.0, {"amp": rng.uniform(0.1, 0.2) * sp.contrast, "period": rng.uniform(5, 20),
                                                "ph": rng.uniform(0, 6.3), "k": 1.0}))
    return sp


def f_drift(rng, word, j):
    sp = _base(rng, "drift", word, rng.uniform(2.5, 3.5))
    sp.contrast = rng.uniform(25, 120)
    sp.events += _drift_events(rng, sp)
    return sp


def f_strike(rng, word, j):
    sp = _base(rng, "strike", word, rng.uniform(2.5, 4.0))
    _add_strikes(rng, sp)
    return sp


def f_hover(rng, word, j):
    sp = _base(rng, "hover", word, rng.uniform(2.5, 4.0))
    _add_hovers(rng, sp)
    return sp


def f_solved(rng, word, j):
    sp = _base(rng, "solved", word)
    cyc = _cyc(word)
    s = rng.uniform(0.6, 3.4) * cyc
    sp.events.append(Ev("solved", s))
    if rng.random() < 0.5:  # green status LED leaking into the region
        sp.events.append(Ev("led", s, 0.0, {"amp": rng.uniform(0, 0.15) * sp.contrast}))
    sp.duration = s + rng.uniform(15, 60)
    if rng.random() < 0.3:
        _add_strikes(rng, sp, n=rng.randint(1, 2), lo=1.0)
        sp.events = [e for e in sp.events if e.kind != "strike" or e.start < s - 1.2]
    if rng.random() < 0.3:  # user turns to another module after solving
        a = s + rng.uniform(2, 8)
        if a < sp.duration - 3:
            sp.events.append(_look_away(rng, a, sp.duration - a + 1, sp))
    return sp


def f_alttab(rng, word, j):
    sp = _base(rng, "alttab", word, rng.uniform(3.0, 5.0))
    _add_dropouts(rng, sp, [])
    return sp


def f_pause(rng, word, j):
    sp = _base(rng, "pause", word)
    cyc = _cyc(word)
    t = rng.uniform(0.3, 1.8) * cyc
    for _ in range(1 if rng.random() < 0.6 else 2):
        L = rng.uniform(2, 25)
        sp.events.append(_pause(rng, t, L))
        t += L + rng.uniform(0.5, 1.6) * cyc
    sp.duration = t + rng.uniform(0.3, 1.5) * cyc
    return sp


def f_lookaway(rng, word, j):
    sp = _base(rng, "lookaway", word)
    cyc = _cyc(word)
    t = rng.uniform(0.4, 2.0) * cyc
    if j % 4 == 0:  # the region first shows another module; the Morse module is turned into view later
        t = 0.0
    for _ in range(rng.randint(1, 2)):
        L = rng.uniform(2, 15)
        sp.events.append(_look_away(rng, t, L, sp))
        t += L + rng.uniform(0.6, 1.8) * cyc
    sp.duration = t + rng.uniform(0.2, 1.2) * cyc
    return sp


def f_motion(rng, word, j):
    """Bomb turned / camera moved: unflagged (or flagged) garbage, then a slightly different framing that settles."""
    sp = _base(rng, "motion", word, rng.uniform(3.0, 4.5))
    busy: list = []
    for _ in range(rng.randint(1, 3)):
        ev = _motion(rng, 0.0, flagged=(j % 3 == 0) or None)
        s = _place(rng, busy, ev.length, 1.0, sp.duration - 1.0)
        if s is None:
            continue
        ev.start = s
        sp.events.append(ev)
        sp.events.append(Ev("settle", s + ev.length, 0.0, {"amp": rng.choice([-1, 1]) * rng.uniform(0.1, 0.6) * sp.contrast,
                                                          "tau": rng.uniform(0.4, 3.0), "k": rng.uniform(0.5, 1.0)}))
    if rng.random() < 0.3:
        l = rng.uniform(2, 6)
        sp.events.append(Ev("zoom", rng.uniform(1, max(2, sp.duration - l - 1)), l, {"f": rng.uniform(0.6, 1.4)}))
    return sp


def f_room(rng, word, j):
    sp = _base(rng, "room", word, rng.uniform(3.0, 4.2))
    sp.off0 = rng.uniform(20, 110)
    sp.contrast = rng.uniform(30, 140)
    sp.k_on = rng.uniform(0, 0.5)
    _add_room(rng, sp)
    return sp


def f_scaled(rng, word, j):
    scale = rng.uniform(0.85, 0.96) if j % 2 else rng.uniform(1.04, 1.2)
    sp = _base(rng, "scaled", word, rng.uniform(2.0, 3.5), scale=scale)
    if rng.random() < 0.3:
        _add_hovers(rng, sp, n=rng.randint(1, 2))
    if rng.random() < 0.3:
        _add_strikes(rng, sp, n=1)
    return sp


def f_combo(rng, word, j):
    sp = _base(rng, "combo", word, rng.uniform(2.5, 4.5))
    if rng.random() < 0.3:
        sp.game = _game(rng, "low")
    busy: list = []
    picks = rng.sample(["strike", "hover", "alttab", "room", "drift", "lowcon", "motion", "pause", "lookaway"],
                       rng.randint(2, 4))
    if "lowcon" in picks:
        sp.contrast = rng.uniform(10, 25)
        sp.noise = rng.uniform(0.4, 0.15 * sp.contrast)
    cyc = _cyc(word)
    for p in picks:
        if p == "pause":
            L = rng.uniform(2, 15)
            sp.duration += L
            s = _place(rng, busy, L, 0.4 * cyc, sp.duration - 0.4 * cyc)
            if s is not None:
                sp.events.append(_pause(rng, s, L))
        elif p == "lookaway":
            L = rng.uniform(2, 12)
            sp.duration += L
            s = _place(rng, busy, L, 0.4 * cyc, sp.duration - 0.4 * cyc)
            if s is not None:
                sp.events.append(_look_away(rng, s, L, sp))
        elif p == "motion":
            for _ in range(rng.randint(1, 2)):
                ev = _motion(rng, 0.0)
                s = _place(rng, busy, ev.length, 1.0, sp.duration - 1.0)
                if s is not None:
                    ev.start = s
                    sp.events.append(ev)
                    sp.events.append(Ev("settle", s + ev.length, 0.0, {"amp": rng.choice([-1, 1]) * rng.uniform(0.1, 0.5) * sp.contrast,
                                                                      "tau": rng.uniform(0.5, 3.0), "k": 1.0}))
    for p in picks:
        if p == "strike":
            _add_strikes(rng, sp)
        elif p == "hover":
            _add_hovers(rng, sp)
        elif p == "alttab":
            _add_dropouts(rng, sp, busy, every=(2.0, 10.0))
        elif p == "room":
            _add_room(rng, sp)
        elif p == "drift":
            sp.events += _drift_events(rng, sp, n=rng.randint(1, 2))
    sp.notes.append("+".join(picks))
    return sp


def f_long(rng, word, j):
    sp = _base(rng, "long", word)
    sp.duration = D = rng.uniform(180, 300)
    sp.cap = _cap(rng, max_fps=90.0)
    if sp.cap.mode == "dxgi":
        sp.cap.fps = 60.0
    con = sp.contrast
    busy: list = []
    solved = None
    if rng.random() < 0.4:
        solved = D - rng.uniform(10, 60)
        sp.events.append(Ev("solved", solved))
        busy.append((solved - 1, D))
    hi = solved if solved is not None else D
    _add_strikes(rng, sp, n=rng.randint(1, 4), lo=10.0)
    sp.events = [e for e in sp.events if e.kind not in ("strike", "hover") or e.start < hi - 1.5]
    for _ in range(rng.randint(3, 10)):
        sp.events.append(_hover(rng, rng.uniform(0, D - 1), con))
    for _ in range(rng.randint(1, 4)):
        sp.events.append(_steps(rng, rng.uniform(0, D - 3), con))
    for _ in range(rng.randint(0, 6)):
        sp.events.append(_cursor(rng, rng.uniform(0, D - 0.2), con))
    for _ in range(rng.randint(0, 2)):
        L = rng.uniform(20, 60)
        sp.events.append(_emergency(rng, rng.uniform(0, D - L), L))
    for _ in range(rng.randint(0, 2)):
        sp.events.append(_lights_out(rng, rng.uniform(0, D - 12)))
    for _ in range(rng.randint(0, 4)):
        ev = _dropout(rng, 0.0)
        s = _place(rng, busy, ev.length, 1.0, hi)
        if s is not None:
            ev.start = s
            sp.events.append(ev)
    if rng.random() < 0.4:
        L = rng.uniform(5, 20)
        s = _place(rng, busy, L, 10.0, hi)
        if s is not None:
            sp.events.append(_pause(rng, s, L))
    for _ in range(rng.randint(0, 2)):
        L = rng.uniform(3, 15)
        s = _place(rng, busy, L, 10.0, hi)
        if s is not None:
            sp.events.append(_look_away(rng, s, L, sp))
    if rng.random() < 0.5:
        sp.events.append(Ev("bloom", 0.0, 0.0, {"amp": rng.uniform(0.05, 0.25) * con, "period": rng.uniform(8, 30),
                                                "ph": rng.uniform(0, 6.3), "k": rng.uniform(0, 1)}))
    return sp


# negatives --------------------------------------------------------------- #

def n_simon(rng, word, j):
    sp = _neg_base(rng, "simon")
    sp.segs = _simon(rng, sp.duration) if j % 2 == 0 else _fill(_on_parts(simon_segments(sp.duration + 5, rng)))
    return sp


def n_button(rng, word, j):
    sp = _neg_base(rng, "button")
    sp.segs = _fill(_holds(rng, sp.duration))
    return sp


def n_needy(rng, word, j):
    sp = _neg_base(rng, "needy")
    sp.segs = _bursts(rng, sp.duration, 0.1, 0.1, (2, 5), (12, 35))
    fn = _timer_fn(rng, sp.duration)  # the needy countdown digits next to it
    sp.events.append(Ev("func", 0.0, 0.0, {"fn": lambda x: 0.3 * fn(x), "k": 1.0}))
    return sp


def n_vent(rng, word, j):
    sp = _neg_base(rng, "vent")
    sp.segs = _bursts(rng, sp.duration, 0.4, 0.4, (5, 30), (8, 30))
    return sp


def n_alarm(rng, word, j):
    sp = _neg_base(rng, "alarm")
    sp.segs = _bursts(rng, sp.duration, 0.25, 0.25, (5, 20), (5, 30))
    return sp


def n_panic(rng, word, j):
    sp = _neg_base(rng, "panic")
    sp.segs = _bursts(rng, sp.duration, 0.1, 0.1, (1, 5), (4, 20))
    return sp


def n_capacitor(rng, word, j):
    sp = _neg_base(rng, "capacitor", rng.uniform(60, 120))
    sp.events.append(Ev("func", 0.0, 0.0, {"fn": _capacitor_fn(rng, sp.duration, rng.uniform(0.5, 1.5) * sp.contrast)}))
    return sp


def n_timer(rng, word, j):
    sp = _neg_base(rng, "timer")
    sp.events.append(Ev("func", 0.0, 0.0, {"fn": _timer_fn(rng, sp.duration)}))
    return sp


def n_emergency(rng, word, j):
    sp = _neg_base(rng, "emergency", rng.uniform(60, 120))
    sp.off0 = rng.uniform(30, 120)
    sp.events.append(_emergency(rng, rng.uniform(-5, 5), sp.duration + 10, pol=rng.choice([1.0, 1.0, -0.6])))
    return sp


def n_lightsout(rng, word, j):
    sp = _neg_base(rng, "lightsout", rng.uniform(60, 100))
    sp.off0 = rng.uniform(30, 120)
    if rng.random() < 0.3:
        sp.segs = [(-1e9, 1e9, True)]  # a steadily lit yellow thing in the region
    t = rng.uniform(0, 10)
    while t < sp.duration:
        ev = _lights_out(rng, t)
        sp.events.append(ev)
        t += ev.length + 2.2 + rng.uniform(3, 25)
    return sp


def n_solvedstatic(rng, word, j):
    sp = _neg_base(rng, "solvedstatic")
    if rng.random() < 0.5:
        sp.events.append(Ev("led", -1.0, 0.0, {"amp": rng.uniform(0, 0.2) * sp.contrast}))
    _add_hovers(rng, sp)
    if rng.random() < 0.5:
        _add_room(rng, sp)
    return sp


def n_hoveruser(rng, word, j):
    sp = _neg_base(rng, "hoveruser")
    t = rng.uniform(0, 5)
    while t < sp.duration:
        L = rng.uniform(1.2, 6.0)
        sp.events.append(Ev("hover", t, L, {"amp": rng.uniform(0.3, 1.0) * sp.contrast, "k": 1.0}))
        t += L + rng.uniform(0.3, 8.0)
    return sp


def n_noise(rng, word, j):
    sp = _neg_base(rng, "noise")
    sp.noise = rng.uniform(4, 15)
    sp.events = [_sway(rng, sp.contrast * rng.uniform(0.05, 0.3))]
    if rng.random() < 0.5:
        sp.events += _drift_events(rng, sp, n=1)
    return sp


def n_gibberish(rng, word, j):
    sp = _neg_base(rng, "gibberish")
    sp.gib = True
    sp.soft = True
    if j % 2:  # also the public generator's flavour (continuous timing), re-rendered on game frames
        sp.gib = False
        sp.segs = _fill(_on_parts(random_morse_segments(sp.duration + 5, rng)))
    return sp


def n_gibloop(rng, word, j):
    sp = _neg_base(rng, "gibloop")
    sp.soft = True
    while True:
        if j % 2 == 0:  # one-letter mutation of a real word
            w = list(rng.choice(WORD_LIST))
            i = rng.randrange(len(w))
            w[i] = rng.choice([ch for ch in LETTERS if ch != w[i]])
            w = "".join(w)
        else:
            w = "".join(rng.choice(LETTERS) for _ in range(rng.randint(4, 6)))
        if w not in WORDS:
            break
    sp.fake = w
    sp.notes.append(f"fake={w}")
    return sp


FAMILIES: list[tuple[str, int, object, bool]] = [
    # name, full count, builder, negative
    ("beat60", 20, f_beat60, False),
    ("beat144", 20, f_beat144, False),
    ("tick", 12, f_tick, False),
    ("lowfps", 16, f_lowfps, False),
    ("phase", 36, f_phase, False),
    ("short", 12, f_short, False),
    ("lowcon", 20, f_lowcon, False),
    ("drift", 20, f_drift, False),
    ("strike", 20, f_strike, False),
    ("hover", 20, f_hover, False),
    ("solved", 20, f_solved, False),
    ("alttab", 20, f_alttab, False),
    ("pause", 16, f_pause, False),
    ("lookaway", 16, f_lookaway, False),
    ("motion", 12, f_motion, False),
    ("room", 20, f_room, False),
    ("scaled", 12, f_scaled, False),
    ("combo", 20, f_combo, False),
    ("long", 16, f_long, False),
    ("simon", 10, n_simon, True),
    ("button", 8, n_button, True),
    ("needy", 6, n_needy, True),
    ("vent", 6, n_vent, True),
    ("alarm", 4, n_alarm, True),
    ("panic", 4, n_panic, True),
    ("capacitor", 6, n_capacitor, True),
    ("timer", 8, n_timer, True),
    ("emergency", 8, n_emergency, True),
    ("lightsout", 6, n_lightsout, True),
    ("solvedstatic", 8, n_solvedstatic, True),
    ("hoveruser", 6, n_hoveruser, True),
    ("noise", 6, n_noise, True),
    ("gibberish", 8, n_gibberish, True),
    ("gibloop", 8, n_gibloop, True),
]


def _q(n: int, quick: bool) -> int:
    return (n + 3) // 4 if quick else n


def holdout_count(quick: bool) -> int:
    return sum(_q(n, quick) for _, n, _, _ in FAMILIES)


def _locate(i: int, quick: bool) -> tuple[int, int]:
    if i < 0:
        raise IndexError(i)
    for fi, (_, n, _, _) in enumerate(FAMILIES):
        m = _q(n, quick)
        if i < m:
            return fi, i
        i -= m
    raise IndexError(i)


def _build(i: int, quick: bool = False, seed: int = 1):
    """(Scenario, Spec, oracle info) for scenario i. Quick scenarios are the first quarter of each family."""
    fi, j = _locate(i, quick)
    name, _, fn, neg = FAMILIES[fi]
    s = zlib.crc32(f"ktane-holdout/{seed}/{name}/{j}".encode())
    rng = random.Random(s)
    nrng = np.random.default_rng(s)
    word = None if neg else WORD_LIST[(j + 5 * fi) % len(WORD_LIST)]
    sp = fn(rng, word, j)
    trace, info = _render(sp, rng, nrng)
    must = bool(info.get("must", False)) if word is not None else False
    notes = [sp.game.desc(), sp.cap.desc(), f"off={sp.off0:.0f}", f"con={sp.contrast:.0f}", f"noise={sp.noise:.1f}"]
    if sp.start != "random":
        notes.append(f"start={sp.start}")
    if sp.scale != 1.0:
        notes.append(f"scale={sp.scale:.3f}")
    notes += sp.notes
    evs = [e.desc() for e in sp.events if e.kind not in ("sway", "func")]
    if evs:
        notes.append("ev=" + ",".join(evs[:8]) + (f",+{len(evs) - 8}" if len(evs) > 8 else ""))
    if word is not None:
        notes.append(f"readable={info['best']:.2f}/{info['total']:.2f}cyc{' stitch' if info['stitched'] else ''}")
    label = word or "none"
    sc = Scenario(f"holdout:{name}:{label}", word, trace, " ".join(notes), must_answer=must, soft=sp.soft)
    return sc, sp, info


def holdout_scenario(i: int, quick: bool = False, seed: int = 1) -> Scenario:
    """Scenario number i of the holdout suite (deterministic in i, quick and seed)."""
    return _build(i, quick, seed)[0]
