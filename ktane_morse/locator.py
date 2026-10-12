"""Automatic locator for the blinking Morse lamp of KTANE's vanilla Morse Code module.

The app captures the game area (up to 2560x1440 RGB at ~25-45 fps), hands every
frame to ``LampLocator.update`` and asks for the regions that look like a
blinking amber lamp. It attaches a decoder to each of the top candidates
(back-filling it from ``history``), keeps feeding it ``FrameInfo.levels`` and
keeps the one that decodes to a word.

Public API
----------
::

    from ktane_morse.locator import LampLocator, Candidate, FrameInfo

    loc = LampLocator()                    # keyword-only tuning parameters, see __init__
    info = loc.update(t, frame_rgb)        # once per captured frame -> FrameInfo
    for c in loc.candidates(3):            # best first -> list[Candidate]
        past = loc.history(c.id)           # [(t, level | None), ...] for buffered frames of this view
        now = info.levels.get(c.id)        # this frame's level (None = unusable frame)
    loc.level(cid)                         # latest frame's level of a candidate (None if unknown)
    loc.candidate(cid)                     # Candidate | None (live candidates only)
    loc.reset()                            # forget everything

Typical app loop (one decoder per candidate id)::

    info = loc.update(t, frame)
    for c in loc.candidates(3):
        d = decoders.get(c.id)
        if d is None or d.version != c.version:          # new candidate or re-defined box
            d = decoders[c.id] = make_decoder(); d.version = c.version
            for tt, lv in loc.history(c.id):             # back-fill the buffered frames
                d.feed(tt, lv)
        else:
            d.feed(t, info.levels.get(c.id))             # None while the camera moves
    # drop decoders whose id is no longer among loc.candidates(0) for a while

``LampLocator.update(t, frame) -> FrameInfo``
    ``t``: monotonic seconds (``time.perf_counter()``), non-decreasing.
    ``frame``: ``(H, W, 3)`` uint8 RGB with any strides (the ``[:, :, 2::-1]``
    view of an mss BGRA grab is fine), or None when the capture failed. A new
    frame size resets the locator.

``FrameInfo`` (returned by every update)
    ``t``             the frame's timestamp
    ``ok``            False when the frame should be fed to decoders as None
                      (capture failed, camera/bomb moving, full-screen change,
                      garbage frame). Room-light changes (lights out, emergency
                      lights) do NOT make a frame bad.
    ``moving``        a global scene change is in progress (motion, pause menu, ...)
    ``view_changed``  True on the frame where a new, different view settled
                      (zoom, bomb turned, pause menu, ...): buffered statistics
                      were reset and the candidates became "stale" (absent from
                      ``candidates()`` / ``levels``) until re-found; a re-found
                      lamp usually keeps its id, but back-fill again from ``history``
    ``shift``         ``(dx, dy)`` full-resolution px when the view settled at a
                      translated position (camera pan): statistics, history and
                      candidate boxes moved with it, ids and versions are kept and
                      the levels continue (only the moving frames are None); None
                      otherwise
    ``epoch``         number of view changes so far (history covers one epoch)
    ``agreement``     0..1 edge-structure agreement with the previous frame
                      (1 = same picture up to lighting); nan if not computed
    ``gain``          room-light gain relative to the reference view (1 normal,
                      ~0.55 in an emergency-light pulse, ~0.1 lights out); nan if unknown
    ``levels``        ``{candidate id: level or None}`` for every live candidate

``Candidate`` (``candidates()`` returns copies; ids are stable across updates)
    ``id``            int, unique for the lifetime of the locator
    ``bbox``          ``(x, y, w, h)`` full-resolution frame pixels of the box the
                      level is measured in (lamp plus a little glow)
    ``center``        ``(x, y)`` full-resolution centre of that box
    ``small_bbox``    ``(r0, r1, c0, c1)`` the same box in downscaled-map cells
    ``score``         0..1 ranking score (amplitude x blink plausibility x Morse timing fit)
    ``morse``         0..1 fraction of observed on/off runs whose length fits Morse timing
    ``good_runs``     number of complete runs that fit (dots, dashes, 1/4/10-unit gaps)
    ``runs``          number of complete runs analysed
    ``rate``          level transitions per second
    ``contrast``      p95 - p10 of the level over the window (yellowness units)
    ``on_level`` / ``off_level``  p95 / p10 of the level
    ``first_seen`` / ``last_seen``  times of the first / latest analysis that saw it
    ``version``       increments whenever ``bbox`` is re-defined (levels change meaning:
                      re-back-fill from ``history``); a pure camera shift
                      (``FrameInfo.shift``) moves ``bbox`` without a new version
    ``region(pad=0.5, min_size=16)`` -> padded ``(x, y, w, h)`` clipped to the frame,
                      for capturing just the lamp at high fps (``features.lamp_level``)

Levels
    A level is ``features.lamp_level`` applied to the block-MAX downscaled
    ``features.yellowness`` map: the mean of the top 15 % (at least 4) cells of
    the candidate's box. Block-max keeps a 10 px lamp intact at a 4x downscale,
    and a lit lamp scores the same (~150) as ``lamp_level`` on the full-resolution
    region would, so a decoder can move to a high-fps capture of ``region()``.
    ``history`` and ``FrameInfo.levels`` use the same buffered maps, so they are
    identical for the same frame.

How it works (per frame, vectorised numpy; see ``tools/fake_screen.py bench``)
    1. yellowness = min(R, G) - B on the full frame, block-max downscaled to
       <= ``max_width`` cells wide, stored in a ring buffer (~8 s).
    2. scene check on a subsampled green channel: the signs of strong local
       gradients survive any room-light gain but not motion, so "agreement" with
       the previous frame / the frame 0.3 s ago flags motion and garbage. When
       motion stops (or the picture drifts from the view's reference frame),
       phase correlation against the reference tells a translated view (shift
       everything, keep the history) from the same view and from a new view
       (reset). The median green ratio against the reference gives the
       room-light gain.
    3. per-cell max/min of yellowness, and of yellowness divided by the gain,
       in 0.5 s chunks. Every ``analysis_interval`` the window amplitude
       min(amp, gain-compensated amp) is thresholded, connected components are
       labelled (run-length union-find), each blob's level series is built from
       the ring buffer and scored on Morse timing (dot 1u / dash 3u on, 1u / 4u /
       10u off, at most 4 elements per letter, u = 0.27 s), blink rate,
       contrast and correlation with the room-light gain; blobs are matched to
       tracks (stable ids).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

import numpy as np

__all__ = ["LampLocator", "Candidate", "FrameInfo", "yellow_blockmax", "top_mean", "morse_evidence",
           "phase_shift"]

UNIT = 0.27
# Allowed lengths (seconds) of complete runs, measured between frame midpoints.
ON_WINDOWS = ((0.16, 0.42), (0.62, 1.02))                    # dot (1u), dash (3u)
OFF_WINDOWS = ((0.16, 0.42), (0.84, 1.45), (2.25, 3.30))     # intra-letter (1u), letter (4u), word (10u)
LONG_OFF = 0.60                                               # an off run at least this long ends a letter


# --------------------------------------------------------------------------- #
# public data

@dataclass
class Candidate:
    id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    small_bbox: tuple[int, int, int, int]
    score: float = 0.0
    morse: float = 0.0
    good_runs: int = 0
    runs: int = 0
    rate: float = 0.0
    contrast: float = 0.0
    on_level: float = 0.0
    off_level: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    version: int = 0
    frame_size: tuple[int, int] = (0, 0)   # (width, height) of the frames it refers to

    def region(self, pad: float = 0.5, min_size: int = 16) -> tuple[int, int, int, int]:
        """bbox grown by ``pad`` x its size on every side (at least ``min_size`` px), clipped to the frame."""
        x, y, w, h = self.bbox
        cx, cy = x + w / 2, y + h / 2
        ww = max(min_size, w * (1 + 2 * pad))
        hh = max(min_size, h * (1 + 2 * pad))
        fw, fh = self.frame_size
        x0, y0 = int(math.floor(cx - ww / 2)), int(math.floor(cy - hh / 2))
        x1, y1 = int(math.ceil(cx + ww / 2)), int(math.ceil(cy + hh / 2))
        if fw and fh:
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(fw, x1), min(fh, y1)
        return x0, y0, max(1, x1 - x0), max(1, y1 - y0)

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        bx, by, bw, bh = self.bbox
        return bx - margin <= x < bx + bw + margin and by - margin <= y < by + bh + margin


@dataclass
class FrameInfo:
    t: float
    ok: bool
    moving: bool = False
    view_changed: bool = False
    epoch: int = 0
    agreement: float = float("nan")
    gain: float = float("nan")
    shift: tuple[int, int] | None = None
    levels: dict[int, float | None] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# vectorised building blocks

def yellow_blockmax(frame: np.ndarray, factor: int, out: np.ndarray | None = None,
                    tmp: np.ndarray | None = None) -> np.ndarray:
    """Block-MAX downscale by ``factor`` of features.yellowness (min(R, G) - B, clipped at 0).

    Returns uint8 of shape (H // factor, W // factor); trailing rows/columns that
    do not fill a block are ignored. ``tmp`` may be a (H//f*f, W//f*f) uint8
    scratch array to avoid an allocation.
    """
    f = int(factor)
    H, W = frame.shape[:2]
    h, w = H // f, W // f
    fr = frame[: h * f, : w * f]
    r, g, b = fr[..., 0], fr[..., 1], fr[..., 2]
    m = np.minimum(r, g, out=tmp)
    np.maximum(m, b, out=m)       # max(min(r, g), b) - b == clip(min(r, g) - b, 0, 255)
    np.subtract(m, b, out=m)
    if out is None:
        out = np.empty((h, w), np.uint8)
    if f == 1:
        out[...] = m
        return out
    rows = np.maximum(m[0::f], m[1::f])
    for k in range(2, f):
        np.maximum(rows, m[k::f], out=rows)
    np.maximum(rows[:, 0::f], rows[:, 1::f], out=out)
    for k in range(2, f):
        np.maximum(out, rows[:, k::f], out=out)
    return out


def top_mean(a2d: np.ndarray, top_fraction: float = 0.15, min_pixels: int = 4) -> np.ndarray:
    """Row-wise features.lamp_level reduction: mean of the top ``top_fraction`` (>= min_pixels) values."""
    n = a2d.shape[1]
    k = min(n, max(min_pixels, int(n * top_fraction)))
    if k >= n:
        return a2d.mean(axis=1, dtype=np.float64)
    return np.partition(a2d, n - k, axis=1)[:, n - k:].mean(axis=1, dtype=np.float64)


def _in_windows(d: np.ndarray, windows) -> np.ndarray:
    ok = np.zeros(d.shape, bool)
    for lo, hi in windows:
        ok |= (d >= lo) & (d <= hi)
    return ok


def morse_evidence(ts, vals, *, max_gap: float = 0.16, hard_gap: float = 0.25, min_contrast: float = 30.0) -> dict:
    """How much a level series looks like the Morse lamp.

    ``vals`` may contain nan for unusable frames (they are skipped; gaps longer
    than ``hard_gap`` between usable frames split the series).
    Returns a dict with contrast/on/off levels, edge count and rate, complete
    run counts, the fraction of runs fitting Morse timing and a 0..1 ``score``
    (before the room-light correlation penalty).
    """
    ts = np.asarray(ts, np.float64)
    v_all = np.asarray(vals, np.float64)
    res = {"contrast": 0.0, "on": 0.0, "off": 0.0, "edges": 0, "rate": 0.0,
           "good": 0, "bad": 0, "runs": 0, "morse": 0.0, "score": 0.0}
    oi = np.flatnonzero(np.isfinite(v_all))
    n = len(oi)
    if n < 8:
        return res
    v, tt = v_all[oi], ts[oi]
    lo, hi = np.percentile(v, [10, 95])
    # A lamp that only started blinking (or a word gap filling most of the
    # window) is on for < 5 % of the frames: also accept the k-th highest
    # value, k ~ 0.1 s of frames (single-frame glitches stay excluded).
    span0 = float(tt[-1] - tt[0])
    k = max(3, int(round(0.1 * n / span0))) if span0 > 0 else 3
    if k < n:
        hi = max(hi, float(np.partition(v, n - k)[n - k]))
    c = float(hi - lo)
    res.update(contrast=c, on=float(hi), off=float(lo))
    if c < min_contrast:
        return res
    s = np.full(n, -1, np.int8)
    s[v >= lo + 0.6 * c] = 1
    s[v <= lo + 0.4 * c] = 0
    # Between valid samples i and i+1 (unusable frames are simply skipped):
    # a gap longer than ``hard_gap`` may hide whole elements and splits the
    # series; a shorter one cannot hide an element (>= 0.27 s), so the same
    # state on both sides is one run, and a change is one edge whose time is
    # only trusted when the gap is <= max_gap.
    dt = np.diff(tt)
    gap = dt > hard_gap
    seg = np.concatenate(([0], np.cumsum(gap)))
    idx = np.where(s >= 0, np.arange(n), -1)
    np.maximum.accumulate(idx, out=idx)                       # forward fill (hysteresis)
    safe = np.maximum(idx, 0)
    st = np.where((idx >= 0) & (seg[safe] == seg), s[safe], -1)

    a, b = st[:-1], st[1:]
    change = ~gap & (a >= 0) & (b >= 0) & (a != b)
    edge = change & (dt <= max_gap)
    other = gap | ((a < 0) & (b >= 0)) | (change & (dt > max_gap))
    bpos = np.flatnonzero(edge | other) + 1
    pos = np.concatenate(([0], bpos, [n]))
    typ = np.concatenate(([False], edge[bpos - 1], [False]))
    tim = np.concatenate(([tt[0]], (tt[bpos - 1] + tt[bpos]) / 2, [tt[-1]]))
    rstate = st[pos[:-1]]
    complete = typ[:-1] & typ[1:] & (rstate >= 0)
    dur = np.diff(tim)

    n_edges = int(edge.sum())
    span = max(1e-6, float(tt[-1] - tt[0]))
    rate = n_edges / span
    on = rstate == 1
    good_on = on & _in_windows(dur, ON_WINDOWS)
    good_off = (rstate == 0) & _in_windows(dur, OFF_WINDOWS)
    good = complete & (good_on | good_off)
    badw = np.where(complete & ~good, np.where(on & (dur > 1.2), 2, 1), 0)
    n_good = int(good.sum())
    n_bad = int(badw.sum())

    # a letter has at most 4 elements: penalise longer chains of short-gap blinks
    chain = 0
    prev_short_off = False
    for k in range(len(rstate)):
        if not complete[k]:
            chain, prev_short_off = 0, False
            continue
        if rstate[k] == 1:
            chain = chain + 1 if prev_short_off or chain == 0 else 1
            if chain > 4:
                n_bad += 2
            prev_short_off = False
        else:
            if dur[k] >= LONG_OFF:
                chain = 0
                prev_short_off = False
            else:
                prev_short_off = True

    total = n_good + n_bad
    morse = n_good / total if total else 0.0
    evidence = 1.0 - math.exp(-n_good / 5.0)
    blink = min(1.0, n_edges / 6.0) * (1.0 if 0.3 <= rate <= 5.0 else 0.3)
    prior = morse if total >= 3 else 0.5
    cf = min(1.0, c / 80.0)
    score = cf * (0.2 * blink * prior + 0.8 * morse * evidence)
    res.update(edges=n_edges, rate=rate, good=n_good, bad=n_bad, runs=int(complete.sum()),
               morse=morse, score=score)
    return res


def _components(mask: np.ndarray):
    """8-connected components of a boolean mask (run-length + vectorised union-find).

    Returns (pix, lab, ncomp): flat indices of the mask pixels in row-major
    order, their component label (0..ncomp-1) and the number of components.
    """
    H, W = mask.shape
    p = np.zeros((H, W + 2), np.int8)
    p[:, 1:-1] = mask
    d = np.diff(p, axis=1)
    sr, sc = np.nonzero(d == 1)
    er, ec = np.nonzero(d == -1)
    n = len(sr)
    if n == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), 0
    big = W + 4
    key_s = sr * big + sc
    key_e = sr * big + ec            # ec is exclusive
    lo = np.searchsorted(key_e, (sr - 1) * big + sc, "left")
    hi = np.searchsorted(key_s, (sr - 1) * big + ec, "right")
    cnt = np.maximum(hi - lo, 0)
    labels = np.arange(n)
    tot = int(cnt.sum())
    if tot:
        bb = np.repeat(np.arange(n), cnt)
        starts = np.cumsum(cnt) - cnt
        aa = np.repeat(lo, cnt) + (np.arange(tot) - np.repeat(starts, cnt))
        for _ in range(64):
            la, lb = labels[aa], labels[bb]
            if np.array_equal(la, lb):
                break
            m = np.minimum(la, lb)
            np.minimum.at(labels, la, m)
            np.minimum.at(labels, lb, m)
            while True:
                nl = labels[labels]
                if np.array_equal(nl, labels):
                    break
                labels = nl
    uniq, comp = np.unique(labels, return_inverse=True)
    lengths = ec - sc
    pix_run = np.repeat(np.arange(n), lengths)
    starts = np.cumsum(lengths) - lengths
    pix = (sr * W + sc)[pix_run] + (np.arange(int(lengths.sum())) - starts[pix_run])
    return pix, comp[pix_run], len(uniq)


_HANN: dict[tuple[int, int], np.ndarray] = {}


def phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Translation between two same-size grey images by phase correlation.

    Returns ``(dy, dx, peak)`` with sub-sample precision such that
    ``b[y, x] ~ a[y - dy, x - dx]``. ``peak`` (0..1) is the height of the
    correlation peak: ~0.2-1 for the same scene moved (whatever the lighting),
    < 0.1 for unrelated pictures.
    """
    h, w = a.shape
    win = _HANN.get((h, w))
    if win is None:
        win = (np.hanning(h)[:, None] * np.hanning(w)[None, :]).astype(np.float32)
        _HANN[(h, w)] = win
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    fa = np.fft.rfft2((a - a.mean()) * win)
    fb = np.fft.rfft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-6
    c = np.fft.irfft2(r, s=(h, w))
    py, px = divmod(int(np.argmax(c)), w)

    def sub(m: float, z: float, p: float) -> float:
        d = m - 2 * z + p
        return 0.0 if d >= 0 else float(np.clip(0.5 * (m - p) / d, -0.5, 0.5))

    fy = sub(c[(py - 1) % h, px], c[py, px], c[(py + 1) % h, px])
    fx = sub(c[py, (px - 1) % w], c[py, px], c[py, (px + 1) % w])
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    return dy + fy, dx + fx, float(c[py, px])


def _shift2d(a: np.ndarray, dy: int, dx: int, fill: int) -> np.ndarray:
    """out[..., y, x] = a[..., y - dy, x - dx]; uncovered cells get ``fill``."""
    h, w = a.shape[-2:]
    out = np.full_like(a, fill)
    if abs(dy) >= h or abs(dx) >= w:
        return out
    out[..., max(0, dy):h + min(0, dy), max(0, dx):w + min(0, dx)] = \
        a[..., max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def _iou(a, b) -> float:
    r0, r1 = max(a[0], b[0]), min(a[1], b[1])
    c0, c1 = max(a[2], b[2]), min(a[3], b[3])
    inter = max(0, r1 - r0) * max(0, c1 - c0)
    ua = (a[1] - a[0]) * (a[3] - a[2]) + (b[1] - b[0]) * (b[3] - b[2]) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- #

class _Ref:
    """Signs of the strong gradients of a scene signature (for structure agreement)."""

    __slots__ = ("idx", "sgn", "t")

    def __init__(self, e: np.ndarray, t: float, max_n: int = 12000) -> None:
        a = np.abs(e)
        sample = a[::7]
        q = np.partition(sample, int(0.8 * (len(sample) - 1)))[int(0.8 * (len(sample) - 1))]
        thr = int(min(40, max(8, q)))
        idx = np.flatnonzero(a >= thr)
        if len(idx) > max_n:
            idx = idx[:: len(idx) // max_n + 1]
        self.idx = idx
        self.sgn = np.sign(e[idx]).astype(np.int16)
        self.t = t

    def agree(self, e: np.ndarray) -> float:
        if len(self.idx) < 40:
            return 1.0
        return float(np.count_nonzero(e[self.idx] * self.sgn >= 1)) / len(self.idx)


@dataclass
class _Track:
    id: int
    box: tuple[int, int, int, int]       # level region in small-map cells (r0, r1, c0, c1)
    first_seen: float
    last_seen: float
    score: float = 0.0
    ev: dict = field(default_factory=dict)
    stale_since: float | None = None
    version: int = 0
    level: float | None = None


class LampLocator:
    """Finds blinking amber lamps in full-screen captures. See the module docstring for the API."""

    def __init__(self, *, max_width: int = 640, window: float = 8.0, max_frames: int = 360,
                 chunk: float = 0.5, analysis_interval: float = 0.25, min_epoch_time: float = 1.0,
                 amp_threshold: int = 40, max_tracks: int = 10, max_blobs: int = 12,
                 top_fraction: float = 0.15, min_pixels: int = 4,
                 move_threshold: float = 0.72, view_threshold: float = 0.62,
                 drift_threshold: float = 0.8, shift_peak: float = 0.15, same_peak: float = 0.25,
                 settle_time: float = 0.3, min_score: float = 0.002) -> None:
        self.max_width = max_width
        self.window = window
        self.max_frames = max_frames
        self.chunk = chunk
        self.analysis_interval = analysis_interval
        self.min_epoch_time = min_epoch_time
        self.amp_threshold = amp_threshold
        self.max_tracks = max_tracks
        self.max_blobs = max_blobs
        self.top_fraction = top_fraction
        self.min_pixels = min_pixels
        self.move_threshold = move_threshold
        self.view_threshold = view_threshold
        self.drift_threshold = drift_threshold
        self.shift_peak = shift_peak
        self.same_peak = same_peak
        self.settle_time = settle_time
        self.min_score = min_score
        self._shape: tuple[int, int] | None = None
        self._next_id = 1
        self.reset()

    # ----------------------------------------------------------------- setup
    def reset(self) -> None:
        self._shape = None
        self._tracks: dict[int, _Track] = {}
        self.epoch = 0
        self._epoch_t0 = -math.inf
        self._last_t = -math.inf
        self._next_analysis = -math.inf
        self._prev_ref: _Ref | None = None
        self._old_refs: deque[_Ref] = deque()
        self._anchor: _Ref | None = None
        self._gain_pos: np.ndarray | None = None
        self._gain_ref: np.ndarray | None = None
        self._anchor_g: np.ndarray | None = None
        self._was_moving = False
        self._next_check = -math.inf
        self._lut_cache: dict[int, np.ndarray] = {}
        self._last_levels: dict[int, float | None] = {}
        self._last_ok = False

    def _init_shape(self, H: int, W: int) -> None:
        f = max(1, math.ceil(W / self.max_width))
        self.factor = f
        self.frame_shape = (H, W)
        h, w = H // f, W // f
        self.small_shape = (h, w)
        self._shape = (H, W)
        cap = self.max_frames
        self._buf = np.zeros((cap, h, w), np.uint8)
        self._bt = np.full(cap, np.nan)
        self._bok = np.zeros(cap, bool)
        self._bgain = np.full(cap, np.nan)
        self._bepoch = np.full(cap, -1, np.int64)
        self._boff = np.zeros((cap, 2), np.int64)     # view offset (cells) of each buffered frame
        self._off = np.zeros(2, np.int64)             # current view offset within the epoch
        self._count = 0
        self._tmp = np.empty((h * f, w * f), np.uint8)
        nch = int(math.ceil(self.window / self.chunk)) + 1
        self._nch = nch
        self._cmax = np.zeros((nch, h, w), np.uint8)
        self._cmin = np.full((nch, h, w), 255, np.uint8)
        self._cmaxc = np.zeros((nch, h, w), np.uint8)
        self._cminc = np.full((nch, h, w), 255, np.uint8)
        self._chunk_no = None            # absolute chunk number currently being filled
        self._sig_step = max(2, 2 * f)

    def _clear_chunks(self) -> None:
        self._cmax.fill(0)
        self._cmin.fill(255)
        self._cmaxc.fill(0)
        self._cminc.fill(255)
        self._chunk_no = None

    # ----------------------------------------------------------------- scene check
    def _signature(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self._sig_step
        g = frame[::s, ::s, 1].astype(np.int16)
        e = np.concatenate((np.diff(g, axis=1).ravel(), np.diff(g, axis=0).ravel()))
        return g, e

    def _set_anchor(self, g: np.ndarray, e: np.ndarray, t: float, gain: float = 1.0) -> None:
        """Make this frame the view's reference; ``gain`` = its room light relative to the old reference."""
        self._anchor = _Ref(e, t)
        self._anchor_g = g
        flat = g.ravel()
        pos = np.flatnonzero(flat >= 24)
        if len(pos) > 6000:
            pos = pos[:: len(pos) // 6000 + 1]
        if len(pos) >= 100:
            self._gain_pos = pos
            k = gain if math.isfinite(gain) and gain > 0.02 else 1.0
            self._gain_ref = flat[pos].astype(np.float32) / k
        else:
            self._gain_pos = self._gain_ref = None

    def _gain_after_shift(self, g: np.ndarray, sy: int, sx: int) -> float:
        """Room-light gain of ``g`` vs the reference, the reference moved by (sy, sx) signature cells."""
        if self._gain_pos is None:
            return float("nan")
        h, w = g.shape
        r, c = np.divmod(self._gain_pos, w)
        r, c = r + sy, c + sx
        ok = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        if np.count_nonzero(ok) < 50:
            return float("nan")
        return float(np.median(g[r[ok], c[ok]].astype(np.float32) / self._gain_ref[ok]))

    def _apply_shift(self, dy: int, dx: int) -> None:
        """The view moved by (dy, dx) cells: move the statistics and the tracks with it."""
        self._off += (dy, dx)
        self._cmax = _shift2d(self._cmax, dy, dx, 0)
        self._cmin = _shift2d(self._cmin, dy, dx, 255)
        self._cmaxc = _shift2d(self._cmaxc, dy, dx, 0)
        self._cminc = _shift2d(self._cminc, dy, dx, 255)
        h, w = self.small_shape
        for tr in list(self._tracks.values()):
            r0, r1, c0, c1 = tr.box
            area = (r1 - r0) * (c1 - c0)
            r0, r1 = max(0, r0 + dy), min(h, r1 + dy)
            c0, c1 = max(0, c0 + dx), min(w, c1 + dx)
            if r1 <= r0 or c1 <= c0 or (r1 - r0) * (c1 - c0) < 0.5 * area:
                del self._tracks[tr.id]            # left the picture
                continue
            if (r1 - r0) * (c1 - c0) != area:
                tr.version += 1
            tr.box = (r0, r1, c0, c1)

    def _gain(self, g: np.ndarray) -> float:
        if self._gain_pos is None:
            return float("nan")
        cur = g.ravel()[self._gain_pos].astype(np.float32)
        return float(np.median(cur / self._gain_ref))

    def _new_epoch(self, t: float) -> None:
        self.epoch += 1
        self._epoch_t0 = t
        self._off[:] = 0
        self._clear_chunks()
        self._next_analysis = t + self.min_epoch_time
        for tr in self._tracks.values():
            if tr.stale_since is None:
                tr.stale_since = t

    # ----------------------------------------------------------------- per frame
    def update(self, t: float, frame: np.ndarray | None) -> FrameInfo:
        t = float(t)
        if frame is not None:
            if frame.ndim != 3 or frame.shape[2] < 3 or frame.dtype != np.uint8:
                raise ValueError("frame must be an (H, W, 3) uint8 RGB array")
            H, W = frame.shape[:2]
            if self._shape != (H, W):
                if self._shape is not None:
                    keep_id = self._next_id
                    self.reset()
                    self._next_id = keep_id
                self._init_shape(H, W)
        if self._shape is None:            # nothing seen yet
            return FrameInfo(t, False)

        slot = self._count % self.max_frames
        self._count += 1
        self._bt[slot] = t
        self._bepoch[slot] = self.epoch
        self._bok[slot] = False
        self._bgain[slot] = np.nan
        info = FrameInfo(t, False, epoch=self.epoch)
        self._last_t = t

        if frame is None:
            self._last_ok = False
            info.levels = self._levels(None)
            return info

        y = yellow_blockmax(frame, self.factor, out=self._buf[slot], tmp=self._tmp)
        g, e = self._signature(frame)
        ref = _Ref(e, t)
        a_prev = self._prev_ref.agree(e) if self._prev_ref is not None else 1.0
        old = None
        while len(self._old_refs) > 1 and self._old_refs[1].t <= t - self.settle_time:
            self._old_refs.popleft()
        if self._old_refs and self._old_refs[0].t <= t - self.settle_time:
            old = self._old_refs[0]
        a_old = old.agree(e) if old is not None else 1.0
        moving = a_prev < self.move_threshold or a_old < self.move_threshold
        if a_prev >= self.move_threshold:
            self._old_refs.append(ref)
            while self._old_refs and self._old_refs[0].t < t - 2.0:
                self._old_refs.popleft()
        self._prev_ref = ref
        info.agreement = a_prev
        info.moving = moving

        if moving:
            self._was_moving = True
        else:
            if self._anchor is None:
                self._set_anchor(g, e, t)
                self._new_epoch(t)
                info.view_changed = self.epoch > 1
            else:
                self._check_view(t, g, e, info)
            self._was_moving = False
            self._bepoch[slot] = self.epoch
            self._boff[slot] = self._off
            info.epoch = self.epoch
            k = self._gain(g)
            info.gain = k
            self._bgain[slot] = k
            self._bok[slot] = True
            info.ok = True
            self._accumulate(t, y, k)
        self._last_ok = info.ok

        if info.ok and t >= self._next_analysis:
            self._next_analysis = t + self.analysis_interval
            self._analyse(t)
        info.levels = self._levels(y if info.ok else None)
        return info

    def _check_view(self, t: float, g: np.ndarray, e: np.ndarray, info: FrameInfo) -> None:
        """After motion (or when the picture drifted from the reference): same view, shifted view or new view?"""
        agr = self._anchor.agree(e)
        if not (self._was_moving or agr < self.view_threshold
                or (agr < self.drift_threshold and t >= self._next_check)):
            return
        self._next_check = t + 1.0
        sy, sx, peak = phase_shift(self._anchor_g, g)
        s, f = self._sig_step, self.factor
        dy, dx = int(round(sy * s / f)), int(round(sx * s / f))       # in small-map cells
        if peak >= self.shift_peak and (dy, dx) != (0, 0):
            k = self._gain_after_shift(g, int(round(sy)), int(round(sx)))
            self._apply_shift(dy, dx)
            self._set_anchor(g, e, t, gain=k)
            info.shift = (dx * f, dy * f)
        elif peak >= self.same_peak or agr >= self.view_threshold:
            return                                                    # same view
        else:
            self._set_anchor(g, e, t)
            self._new_epoch(t)
            info.view_changed = True

    def _accumulate(self, t: float, y: np.ndarray, k: float) -> None:
        no = int(math.floor(t / self.chunk))
        if self._chunk_no is None or no != self._chunk_no:
            first = no if self._chunk_no is None else max(self._chunk_no + 1, no - self._nch + 1)
            for c in range(first, no + 1):
                i = c % self._nch
                self._cmax[i].fill(0)
                self._cmin[i].fill(255)
                self._cmaxc[i].fill(0)
                self._cminc[i].fill(255)
            self._chunk_no = no
        i = no % self._nch
        np.maximum(self._cmax[i], y, out=self._cmax[i])
        np.minimum(self._cmin[i], y, out=self._cmin[i])
        if math.isfinite(k) and abs(k - 1.0) > 0.04 and k > 0.02:
            q = int(round(k * 100))
            lut = self._lut_cache.get(q)
            if lut is None:
                lut = np.clip(np.round(np.arange(256) / (q / 100.0)), 0, 255).astype(np.uint8)
                if len(self._lut_cache) > 512:
                    self._lut_cache.clear()
                self._lut_cache[q] = lut
            yc = lut[y]
        else:
            yc = y
        np.maximum(self._cmaxc[i], yc, out=self._cmaxc[i])
        np.minimum(self._cminc[i], yc, out=self._cminc[i])

    def _levels(self, y: np.ndarray | None) -> dict[int, float | None]:
        out: dict[int, float | None] = {}
        for tid, tr in self._tracks.items():
            if tr.stale_since is not None:
                continue
            if y is None:
                lvl = None
            else:
                r0, r1, c0, c1 = tr.box
                lvl = float(top_mean(y[r0:r1, c0:c1].reshape(1, -1), self.top_fraction, self.min_pixels)[0])
            tr.level = lvl
            out[tid] = lvl
        self._last_levels = out
        return out

    # ----------------------------------------------------------------- buffered data
    def _epoch_slots(self) -> np.ndarray:
        n = min(self._count, self.max_frames)
        if self._count <= self.max_frames:
            slots = np.arange(n)
        else:
            slots = (np.arange(n) + self._count) % self.max_frames
        return slots[self._bepoch[slots] == self.epoch]

    def _series(self, box, slots: np.ndarray) -> np.ndarray:
        """Levels of a box (current view cells) for buffered frames, following view shifts."""
        r0, r1, c0, c1 = box
        d = self._off[None, :] - self._boff[slots]
        ok = self._bok[slots]
        if not d[ok].any():
            sub = self._buf[slots, r0:r1, c0:c1].reshape(len(slots), -1)
            vals = top_mean(sub, self.top_fraction, self.min_pixels)
        else:
            h, w = self.small_shape
            vals = np.full(len(slots), np.nan)
            uniq, inv = np.unique(d, axis=0, return_inverse=True)
            inv = inv.ravel()
            for j, (oy, ox) in enumerate(uniq):
                a0, a1 = max(0, r0 - oy), min(h, r1 - oy)
                b0, b1 = max(0, c0 - ox), min(w, c1 - ox)
                if a1 <= a0 or b1 <= b0 or (a1 - a0) * (b1 - b0) < 0.5 * (r1 - r0) * (c1 - c0):
                    continue
                sel = np.flatnonzero(inv == j)
                sub = self._buf[slots[sel], a0:a1, b0:b1].reshape(len(sel), -1)
                vals[sel] = top_mean(sub, self.top_fraction, self.min_pixels)
        vals[~ok] = np.nan
        return vals

    # ----------------------------------------------------------------- analysis
    def _analyse(self, t: float) -> None:
        if t - self._epoch_t0 < self.min_epoch_time:
            return
        amp = self._cmax.max(axis=0) - self._cmin.min(axis=0)
        ampc = self._cmaxc.max(axis=0)
        np.subtract(ampc, np.minimum(ampc, self._cminc.min(axis=0)), out=ampc)
        np.minimum(amp, ampc, out=amp)
        h, w = amp.shape
        thr = self.amp_threshold
        mask = amp >= thr
        nmask = int(np.count_nonzero(mask))
        if nmask > 0.12 * h * w:          # something global is going on: only take the strongest
            thr = max(thr, int(np.partition(amp.ravel(), int(0.97 * h * w))[int(0.97 * h * w)]) + 1)
            mask = amp >= thr
        blobs = self._blobs(amp, mask, thr) if mask.any() else []

        slots = self._epoch_slots()
        ts = self._bt[slots]
        gains = self._bgain[slots]
        dets = []
        for blob in blobs:
            vals = self._series(blob["box"], slots)
            ev = morse_evidence(ts, vals)
            ev["score"] *= self._light_factor(vals, gains)
            dets.append((ev["score"], blob, ev))
        dets.sort(key=lambda d: -d[0])
        self._match(t, dets)

    def _blobs(self, amp: np.ndarray, mask: np.ndarray, thr: int) -> list[dict]:
        h, w = amp.shape
        pix, lab, nc = _components(mask)
        if nc == 0:
            return []
        vals = amp.ravel()[pix].astype(np.int32)
        area = np.bincount(lab, minlength=nc)
        peak = np.zeros(nc, np.int32)
        np.maximum.at(peak, lab, vals)
        order = np.lexsort((-vals, lab))                 # by label, then value desc
        first = np.ones(len(order), bool)
        first[1:] = lab[order][1:] != lab[order][:-1]
        peak_pix = np.zeros(nc, np.int64)
        peak_pix[lab[order][first]] = pix[order][first]
        rows, cols = pix // w, pix % w
        rmin = np.full(nc, h); rmax = np.zeros(nc, np.int64)
        cmin = np.full(nc, w); cmax = np.zeros(nc, np.int64)
        np.minimum.at(rmin, lab, rows); np.maximum.at(rmax, lab, rows)
        np.minimum.at(cmin, lab, cols); np.maximum.at(cmax, lab, cols)
        strength = peak * np.sqrt(area)
        keep = np.flatnonzero(area <= 0.2 * h * w)
        keep = keep[np.argsort(-strength[keep])][: self.max_blobs]
        labimg = None
        out = []
        for c in keep:
            r0, r1, c0, c1 = int(rmin[c]), int(rmax[c]) + 1, int(cmin[c]), int(cmax[c]) + 1
            pr, pc = int(peak_pix[c] // w), int(peak_pix[c] % w)
            core_thr = max(thr, int(0.5 * peak[c]))
            if area[c] > 4:
                if labimg is None:
                    labimg = np.full(h * w, -1, np.int64)
                    labimg[pix] = lab
                    labimg = labimg.reshape(h, w)
                sub = (amp[r0:r1, c0:c1] >= core_thr) & (labimg[r0:r1, c0:c1] == c)
                spix, slab, snc = _components(sub)
                if snc > 1:
                    sw = c1 - c0
                    target = (pr - r0) * sw + (pc - c0)
                    hit = slab[np.flatnonzero(spix == target)]
                    if len(hit):
                        spix = spix[slab == hit[0]]
                if len(spix):
                    sw = c1 - c0
                    srow, scol = spix // sw, spix % sw
                    r0, r1 = r0 + int(srow.min()), r0 + int(srow.max()) + 1
                    c0, c1 = c0 + int(scol.min()), c0 + int(scol.max()) + 1
            box = (max(0, r0 - 1), min(h, r1 + 1), max(0, c0 - 1), min(w, c1 + 1))
            out.append({"box": box, "peak": int(peak[c]), "area": int(area[c])})
        return out

    @staticmethod
    def _light_factor(vals: np.ndarray, gains: np.ndarray) -> float:
        """Penalty for series that just follow the room light (emergency pulses, lights out)."""
        ok = np.isfinite(vals) & np.isfinite(gains)
        if np.count_nonzero(ok) < 10:
            return 1.0
        g = gains[ok]
        lo, hi = np.percentile(g, [5, 95])
        if hi - lo < 0.08:
            return 1.0
        v = vals[ok]
        if v.std() < 1e-6:
            return 1.0
        r = float(np.corrcoef(v, g)[0, 1])
        if not math.isfinite(r) or r <= 0.4:
            return 1.0
        return max(0.0, 1.0 - (r - 0.4) / 0.4)

    def _match(self, t: float, dets) -> None:
        h, w = self.small_shape
        used: set[int] = set()
        for score, blob, ev in dets:
            box = blob["box"]
            cy, cx = (box[0] + box[1]) / 2, (box[2] + box[3]) / 2
            size = max(box[1] - box[0], box[3] - box[2])
            best, best_d = None, math.inf
            for tr in self._tracks.values():
                if tr.id in used:
                    continue
                tb = tr.box
                d = math.hypot((tb[0] + tb[1]) / 2 - cy, (tb[2] + tb[3]) / 2 - cx)
                tsize = max(size, tb[1] - tb[0], tb[3] - tb[2])
                if tr.stale_since is None:
                    okm = _iou(tb, box) >= 0.3 or d <= max(2.5, 0.6 * tsize)
                else:
                    okm = d <= max(12.0, 2.0 * tsize)
                if okm and d < best_d:
                    best, best_d = tr, d
            if best is None:
                if score < self.min_score:
                    continue
                best = _Track(self._next_id, box, t, t)
                self._next_id += 1
                self._tracks[best.id] = best
                best.score = score
            else:
                if best.stale_since is not None:          # re-found in a new view: history restarted
                    best.box = box
                    best.version += 1
                elif _iou(best.box, box) < 0.5 and best.box != box:
                    best.box = box
                    best.version += 1
                best.stale_since = None
                best.score = 0.3 * best.score + 0.7 * score
            best.last_seen = t
            best.ev = ev
            used.add(best.id)
        for tr in list(self._tracks.values()):
            if tr.id in used:
                continue
            if tr.stale_since is not None:
                if t - tr.stale_since > 10.0:
                    del self._tracks[tr.id]
                continue
            tr.score *= 0.6
            if t - tr.last_seen > 2.5:
                del self._tracks[tr.id]
        if len(self._tracks) > self.max_tracks:
            ranked = sorted(self._tracks.values(), key=lambda tr: (tr.stale_since is None, tr.score), reverse=True)
            for tr in ranked[self.max_tracks:]:
                del self._tracks[tr.id]

    # ----------------------------------------------------------------- queries
    def _to_candidate(self, tr: _Track) -> Candidate:
        f = self.factor
        r0, r1, c0, c1 = tr.box
        x, y, ww, hh = c0 * f, r0 * f, (c1 - c0) * f, (r1 - r0) * f
        ev = tr.ev
        H, W = self.frame_shape
        return Candidate(
            id=tr.id, bbox=(x, y, ww, hh), center=(x + ww / 2, y + hh / 2), small_bbox=tr.box,
            score=float(tr.score), morse=float(ev.get("morse", 0.0)), good_runs=int(ev.get("good", 0)),
            runs=int(ev.get("runs", 0)), rate=float(ev.get("rate", 0.0)), contrast=float(ev.get("contrast", 0.0)),
            on_level=float(ev.get("on", 0.0)), off_level=float(ev.get("off", 0.0)),
            first_seen=tr.first_seen, last_seen=tr.last_seen, version=tr.version, frame_size=(W, H))

    def candidates(self, k: int = 3) -> list[Candidate]:
        """Live candidates, best first (at most ``k``; ``k=0`` for all)."""
        if self._shape is None:
            return []
        live = [tr for tr in self._tracks.values() if tr.stale_since is None and tr.score >= self.min_score]
        live.sort(key=lambda tr: (-tr.score, tr.id))
        if k:
            live = live[:k]
        return [self._to_candidate(tr) for tr in live]

    def candidate(self, cid: int) -> Candidate | None:
        tr = self._tracks.get(cid)
        if tr is None or tr.stale_since is not None:
            return None
        return self._to_candidate(tr)

    def level(self, cid: int) -> float | None:
        """Level of candidate ``cid`` in the latest frame (None if unusable or unknown)."""
        return self._last_levels.get(cid)

    def history(self, cid: int) -> list[tuple[float, float | None]]:
        """Levels of candidate ``cid`` for every buffered frame of the current view, oldest first."""
        tr = self._tracks.get(cid)
        if tr is None or tr.stale_since is not None or self._shape is None:
            return []
        slots = self._epoch_slots()
        if len(slots) == 0:
            return []
        vals = self._series(tr.box, slots)
        ts = self._bt[slots]
        return [(float(a), None if not math.isfinite(b) else float(b)) for a, b in zip(ts, vals)]

    def small_map(self) -> np.ndarray | None:
        """The latest downscaled yellowness map (debugging)."""
        if self._shape is None or self._count == 0:
            return None
        return self._buf[(self._count - 1) % self.max_frames]
