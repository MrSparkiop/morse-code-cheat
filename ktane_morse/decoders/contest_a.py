"""Contestant A: discrete symbols + cyclic constraint matching.

placeholder docstring (rewritten at the end)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..decoding import Status
from ..morse import MORSE, REVERSE_MORSE, WORDS

# --------------------------------------------------------------------------- #
# Symbol classes. ON runs: dot / dash. OFF runs: intra-letter / letter / word gap.
DOT, DASH = 1, 2
INTRA, LETTER, WGAP = 1, 2, 4
NOMINAL = {1: {DOT: 1.0, DASH: 3.0}, 0: {INTRA: 1.0, LETTER: 4.0, WGAP: 10.0}}
# acceptance bands, in units of the estimated dot length (generous: covers a
# 0.8-1.25 time scale before the unit estimate has converged)
BANDS = {
    1: ((DOT, 0.45, 1.85), (DASH, 2.15, 4.7)),
    0: ((INTRA, 0.45, 2.1), (LETTER, 2.55, 6.3), (WGAP, 6.9, 15.5)),
}
UNIT_PRIOR = 0.27
UNIT_MIN, UNIT_MAX = 0.20, 0.36
FRAME_BIAS = 0.006          # every element runs up to one game frame long
BIG = 1000
MM_CAP = 3


def _word_runs(word: str) -> list[tuple[int, int]]:
    out = []
    codes = [MORSE[c] for c in word.upper()]
    for li, code in enumerate(codes):
        for si, s in enumerate(code):
            out.append((1, DOT if s == "." else DASH))
            if si < len(code) - 1:
                out.append((0, INTRA))
        out.append((0, LETTER if li < len(codes) - 1 else WGAP))
    return out


class _Tables:
    """All cyclic word run-strings concatenated into one position array."""

    def __init__(self) -> None:
        self.words = list(WORDS)
        runs = [_word_runs(w) for w in self.words]
        self.lengths = np.array([len(r) for r in runs])
        self.starts = np.concatenate([[0], np.cumsum(self.lengths)[:-1]]).astype(np.intp)
        n = int(self.lengths.sum())
        self.n = n
        pred = np.empty(n, np.intp)
        st = np.empty(n, np.int8)
        cl = np.empty(n, np.int8)
        for wi, r in enumerate(runs):
            s0 = self.starts[wi]
            L = len(r)
            for p, (s, c) in enumerate(r):
                pred[s0 + p] = s0 + (p - 1) % L
                st[s0 + p] = s
                cl[s0 + p] = c
        self.pred = pred
        self.mm = np.zeros((2, 8, n), np.int32)
        for s in (0, 1):
            for mask in range(8):
                self.mm[s, mask] = np.where(st != s, BIG, np.where((cl & mask) != 0, 0, 1))
        # null model ("Morse-like gibberish" with the word table's class frequencies)
        freq = {}
        for s in (0, 1):
            tot = float(np.sum(st == s))
            freq[s] = {c: float(np.sum((st == s) & (cl == c))) / tot for c in (1, 2, 4)}
        self.info = {}
        for s in (0, 1):
            for m in range(8):
                p = sum(v for c, v in freq[s].items() if c & m)
                self.info[(s, m)] = -math.log2(p) if p > 0 else 0.0
        self.log_len = np.log2(self.lengths.astype(float))

    def advance(self, cost: np.ndarray, state: int, mask: int) -> np.ndarray:
        out = cost[self.pred] + self.mm[state, mask]
        np.minimum(out, BIG, out=out)
        return out

    def summary(self, cost: np.ndarray):
        wmin = np.minimum.reduceat(cost, self.starts)
        cnt0 = np.add.reduceat((cost == 0).astype(np.int32), self.starts)
        return np.minimum(wmin, MM_CAP), cnt0


TABLES = _Tables()


def _classify(state: int, lo_r: float, hi_r: float) -> int:
    """Class mask for a run whose length (in units) lies in [lo_r, hi_r]; 0 = invalid."""
    bands = BANDS[state]
    if hi_r < bands[0][1] or lo_r > bands[-1][2]:
        return 0
    mask = 0
    for cls, a, b in bands:
        if hi_r >= a and lo_r <= b:
            mask |= cls
    if mask == 0:  # falls in the gap between two bands: ambiguous
        for i in range(len(bands) - 1):
            if lo_r > bands[i][2] and hi_r < bands[i + 1][1]:
                mask = bands[i][0] | bands[i + 1][0]
    return mask


def _classify_lb(state: int, lo_r: float) -> int:
    """Class mask for a run known to last at least lo_r units."""
    mask = 0
    for cls, a, b in BANDS[state]:
        if b >= lo_r:
            mask |= cls
    return mask


def _fmt(paths) -> str:
    def one(st, m):
        if st:
            return {DOT: ".", DASH: "-"}.get(m, "?")
        return {INTRA: "", LETTER: " ", WGAP: " / "}.get(m, "~")
    if len(paths) == 1:
        return "".join(one(st, m) for st, m in paths[0])
    return "{" + "|".join("".join(one(st, m) for st, m in p) for p in paths) + "}"


class _Segment:
    """Consecutive symbols with no erasure inside.

    Each added item is a list of alternative symbol paths (usually one path of
    one symbol); the cost array keeps, for every cyclic position of every
    word, the fewest mismatches of any alignment ending there.
    """

    __slots__ = ("cost", "info", "items", "hist", "nruns")

    def __init__(self) -> None:
        self.cost = np.zeros(TABLES.n, np.int32)
        self.info = 0.0
        self.items: deque = deque(maxlen=64)   # (paths, t_end)
        self.hist: deque = deque(maxlen=3)     # (cost, info) before each of the last items
        self.nruns = 0

    def add(self, paths, t_end: float) -> None:
        self.hist.append((self.cost, self.info))
        cost = None
        info = None
        for path in paths:
            c = self.cost
            inf = 0.0
            for st, m in path:
                c = TABLES.advance(c, st, m)
                inf += TABLES.info[(st, m)]
            cost = c if cost is None else np.minimum(cost, c)
            info = inf if info is None else min(info, inf)
        self.cost = cost
        self.info += info - (1.0 if len(paths) > 1 else 0.0)
        self.items.append((paths, t_end))
        self.nruns += 1

    def pop(self) -> bool:
        if not self.hist or not self.items:
            return False
        self.cost, self.info = self.hist.pop()
        self.items.pop()
        self.nruns -= 1
        return True

    def summary(self, extra=None):
        cost, info = self.cost, self.info
        if extra is not None:
            cost = TABLES.advance(cost, extra[0], extra[1])
            info += TABLES.info[extra]
        wmin, cnt0 = TABLES.summary(cost)
        return wmin, cnt0, info, self.nruns + (extra is not None)


class _Piece:
    """A stretch of samples with a constant level (between two detected changes)."""

    __slots__ = ("ta", "tb", "tl", "s", "n", "m", "mn", "ad", "nd", "done", "lab")

    def __init__(self, ta, tb, tl, s, n, ad=0.0, nd=0) -> None:
        self.ta = ta        # last sample time before the piece (None: unknown)
        self.tb = tb        # first sample time
        self.tl = tl        # last sample time
        self.s = s          # sum of samples
        self.n = n
        self.m = s / n      # running (robust) mean used by the change detector
        self.mn = n
        self.ad = ad        # sum of |successive differences| inside the piece
        self.nd = nd
        self.done = False   # handed to the labeller
        self.lab = None


class Decoder:
    H = 8.0                 # CUSUM decision threshold (nats)
    W_RETRACT = 1.25        # symbols ending this close before an anomaly are retracted (s)
    T_TOTAL = 11.0          # commit: total evidence (bits over the null) for the winner
    T_SEG = 6.0             # commit: evidence of the best single segment
    MARGIN = 2              # commit: every other word needs this many mismatches (low SNR)
    log = None              # debugging: a list receives events

    def __init__(self) -> None:
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        self.now = 0.0
        self.t_prev: float | None = None
        self.x_prev = 0.0
        self.diffs = deque(maxlen=256)
        self.dts = deque(maxlen=64)
        self.nsamp = 0
        self.sig_glob = 1.0
        self.sig_hist = deque(maxlen=4)      # (t_close, sigma) of recent solid pieces
        self.sig_ref = 1.0
        self.sig_ok = False
        self.dt_med = 1 / 60
        # change detector
        self.p: _Piece | None = None
        self.gu = self.gd = 0.0
        self.cu = None          # [t_before, t_first, sum, n, ad, nd, edge_dx]
        self.cd = None
        # labeller
        self.lo: float | None = None
        self.hi: float | None = None
        self.last_solid: _Piece | None = None
        self.glitches = deque(maxlen=8)
        self.outliers = deque(maxlen=4)
        self.midliers = deque(maxlen=4)
        self.relearn: list = []         # [mean, n, t_first, last_piece]
        self.drop_first = False
        self.run_lab: int | None = None
        self.run_start = None           # (ta, tb) edge or None (partial)
        self.run_t0 = 0.0
        self.run_long = False
        self.run_hole = None            # (t1, t2): a bridged hole inside the run
        self.hole_fresh = False         # a bridged hole was just passed, next label pending
        self.lb_mask = 0
        # symbols
        self.units = deque(maxlen=31)
        self.unit = UNIT_PRIOR
        self.seg: _Segment | None = None
        self.closed: deque = deque(maxlen=60)   # summaries of closed segments
        self.edges = deque(maxlen=16)
        # output
        self.answer: str | None = None
        self.ranking: list[tuple[str, float]] = []
        self.confidence = 0.0
        self.morse_tokens = deque(maxlen=160)
        self.letter_tokens = deque(maxlen=80)
        self.cur_letter = ""
        self._status: Status | None = None
        self._status_sig = None
        self.dirty = True

    def _log(self, *a) -> None:
        if self.log is not None:
            self.log.append((round(self.now, 3),) + a)

    # ------------------------------------------------------------------ input
    def feed(self, t: float, level: float | None) -> None:
        self.now = t
        if level is None:
            return
        x = float(level)
        if x != x or x in (math.inf, -math.inf):
            return
        tp = self.t_prev
        dx = None
        if tp is not None:
            dt = t - tp
            if dt <= 0:
                return
            if dt > self._hole_limit():
                if not self._hole(tp, t):
                    tp = None
            else:
                self.dts.append(dt)
                dx = abs(x - self.x_prev)
                self.diffs.append(dx)
        self.nsamp += 1
        self.t_prev = t
        self.x_prev = x
        if self.nsamp % 32 == 0 or (not self.sig_ok and len(self.diffs) >= 24):
            self._update_noise()
        if not self.sig_ok:
            return
        p = self.p
        if p is None:
            self.p = _Piece(tp, t, t, x, 1)
            self.gu = self.gd = 0.0
            self.cu = self.cd = None
            return
        if self.lo is not None:
            lo, hi = self.lo, self.hi
            dev = x - hi if x > hi else lo - x if x < lo else 0.0
            if dev > 0.0 and dev > max(0.4 * (hi - lo), 5.0 * self.sig_ref):
                ol = self.outliers
                ol.append(t)
                if len(ol) >= 3 and t - ol[-3] < 0.3:
                    ol.clear()
                    self._anomaly("outliers", min(self.last_solid.tl if self.last_solid is not None else t, t - 0.3))
                    self.p = None
                    return
            elif dev == 0.0 and min(x - lo, hi - x) > max(0.3 * (hi - lo), 3.5 * self.sig_ref):
                # between the two levels: also suspicious when it persists
                ol = self.midliers
                ol.append(t)
                if len(ol) >= 4 and t - ol[-4] < 0.3 and self.dt_med < 0.3 / 5:
                    ol.clear()
                    self._anomaly("midlevel", min(self.last_solid.tl if self.last_solid is not None else t, t - 0.3))
                    self.p = None
                    return
        self._sample(p, tp, t, x, dx)
        p = self.p
        if not p.done and p.n >= 2 and (p.tl - p.tb) + self.dt_med >= self._t_solid() \
                and self.gu < 0.5 * self.H and self.gd < 0.5 * self.H:
            self._solid(p)
        self._check_run(t)

    def _unit_safe(self) -> float:
        return self.unit if len(self.units) >= 5 else UNIT_MIN

    def _hole_limit(self) -> float:
        # a hole shorter than the shortest possible run cannot hide a whole run
        return 0.85 * self._unit_safe()

    def _t_solid(self) -> float:
        return max(0.07, 0.28 * self.unit)

    def _update_noise(self) -> None:
        if len(self.dts) >= 8:
            self.dt_med = float(np.median(np.fromiter(self.dts, float, len(self.dts))))
        n = len(self.diffs)
        if n >= 24:
            k = min(n, max(24, int(1.5 / max(self.dt_med, 1e-3))))
            a = np.fromiter(self.diffs, float, n)[n - k:]
            j = (3 * k) // 4
            self.sig_glob = float(np.partition(a, j)[j]) / 1.63
            self.sig_ok = True
        now = self.now
        rec = [s for t, s in self.sig_hist if now - t < 3.0]
        ref = max(rec) if len(rec) >= 2 else self.sig_glob
        self.sig_ref = max(min(ref, 2.0 * self.sig_glob), self._floor())

    def _floor(self) -> float:
        c = (self.hi - self.lo) if self.lo is not None else 0.0
        return max(0.15, 0.015 * c)

    def _psig(self, p: _Piece) -> float:
        """Noise level for piece p (p.ad is a running mean of |successive differences|)."""
        ref = self.sig_ref
        if p.nd >= 8:
            s = max(p.ad / 1.128, 0.5 * ref)
        elif p.nd > 0:
            s = max(ref, p.ad / 1.128)
        else:
            s = ref
        return max(s, self._floor())

    # --------------------------------------------------------- change detector
    def _sample(self, p: _Piece, tp, t: float, x: float, dx) -> None:
        s = self._psig(p)
        if self.lo is not None:
            d = max(0.5 * (self.hi - self.lo), 1.5 * s)
        else:
            d = 3.5 * s
        a = d / s
        z = (x - p.m) / s
        zc = a + 3.0
        if z > zc:
            z = zc
        elif z < -zc:
            z = -zc
        cap = 0.6 * self.H
        iu = a * (z - 0.5 * a)
        idn = a * (-z - 0.5 * a)
        if iu > cap:
            iu = cap
        if idn > cap:
            idn = cap
        if dx is not None:
            lim = 4.0 * self.sig_ref
            if dx > lim:
                dx = lim
            p.nd += 1
            p.ad += (dx - p.ad) / (p.nd if p.nd < 20 else 20)
        else:
            dx = 0.0
        g = self.gu + iu
        if g <= 0:
            self.gu = 0.0
            self.cu = None
        else:
            c = self.cu
            if c is None:
                self.cu = [tp, t, x, 1, 0.0, 0, dx]
            else:
                c[2] += x
                c[3] += 1
                c[4] += dx
                c[5] += 1
            self.gu = g
        g = self.gd + idn
        if g <= 0:
            self.gd = 0.0
            self.cd = None
        else:
            c = self.cd
            if c is None:
                self.cd = [tp, t, x, 1, 0.0, 0, dx]
            else:
                c[2] += x
                c[3] += 1
                c[4] += dx
                c[5] += 1
            self.gd = g
        p.s += x
        p.n += 1
        p.tl = t
        if self.gu < 0.3 * self.H and self.gd < 0.3 * self.H:
            if p.mn < 40:
                p.mn += 1
            inn = x - p.m
            if inn > d:
                inn = d
            elif inn < -d:
                inn = -d
            p.m += inn / p.mn
        if self.gu >= self.H:
            self._change(self.cu)
        elif self.gd >= self.H:
            self._change(self.cd)

    def _change(self, cand) -> None:
        old = self.p
        tb4, tf, cs, cn, cad, cnd, edx = cand
        old.s -= cs
        old.n -= cn
        new = _Piece(tb4, tf, self.t_prev, cs, cn, cad / cnd if cnd else 0.0, cnd)
        if old.n > 0 and tb4 is not None:
            old.tl = tb4
            self.p = new
            self._close(old)
        else:  # the whole old piece belongs to the new one (its mean was badly initialised)
            new.ta = old.ta
            new.tb = old.tb
            new.s += old.s
            new.n += old.n
            new.ad, new.nd = old.ad, old.nd
            new.m = new.s / new.n
            new.done, new.lab = old.done, old.lab
            if self.last_solid is old:
                self.last_solid = new
            for e in self.relearn:
                if e[3] is old:
                    e[3] = new
            self.p = new
        self.gu = self.gd = 0.0
        self.cu = self.cd = None

    def _close(self, p: _Piece) -> None:
        """A piece ended (the next one started)."""
        if not p.done:
            if p.n >= 2 and (p.tl - p.tb) + self.dt_med >= self._t_solid():
                self._solid(p, final=True)
            else:
                self._glitch(p)
                return
        if p.nd >= 6:
            self.sig_hist.append((p.tl, p.ad / 1.128))
        if p.lab is not None and self.lo is not None:
            mean = p.s / p.n
            lvl = self.hi if p.lab == 1 else self.lo
            c = self.hi - self.lo
            if abs(mean - lvl) > self._tol(c, p) * 1.15:
                self._anomaly("drift", p.tb)
                return
            w = 0.35 * min(1.0, p.n / 6.0)
            if p.lab == 1:
                self.hi += w * (mean - self.hi)
            else:
                self.lo += w * (mean - self.lo)

    def _bridge_limit(self) -> float:
        return 1.6 * self._unit_safe()

    def _hole(self, t1: float, t2: float) -> bool:
        """No usable sample between t1 and t2. Returns True if the hole is bridged."""
        p = self.p
        if p is not None:
            self.p = None
            self._close(p)
        self.gu = self.gd = 0.0
        self.cu = self.cd = None
        h = t2 - t1
        if (self.lo is not None and self.run_lab is not None and not self.run_long
                and self.run_start is not None and self.run_hole is None and h <= self._bridge_limit()):
            self._log("BRIDGE", round(h, 3))
            self.run_hole = (t1, t2)
            self.hole_fresh = True
            return True
        self._log("HOLE", round(h, 3))
        if self.run_lab is not None and not self.run_long:
            self._end_run_partial(t1)
        self._break("hole")
        self.run_lab = None
        self.run_hole = None
        self.last_solid = None
        if self.lo is None:
            self.relearn = []
        return False

    # ----------------------------------------------------------------- labeller
    def _tol(self, c: float, p: _Piece) -> float:
        se = self._psig(p) / math.sqrt(max(1, p.n))
        return min(max(0.3 * c, 3.0 * se), 0.6 * c)

    def _label(self, m: float, p: _Piece):
        c = self.hi - self.lo
        dl = abs(m - self.lo)
        dh = abs(m - self.hi)
        if min(dl, dh) > self._tol(c, p):
            return None
        return 0 if dl < dh else 1

    def _glitch(self, p: _Piece) -> None:
        t = p.tb
        self.glitches.append(t)
        g = self.glitches
        n = len(g)
        if (n >= 3 and t - g[-3] < 0.35) or (n >= 5 and t - g[-5] < 1.0):
            onset = self.last_solid.tl if self.last_solid is not None else g[-3]
            self.glitches.clear()
            self._anomaly("garbage", onset)

    def _solid(self, p: _Piece, final: bool = False) -> None:
        p.done = True
        m = p.s / p.n if final else p.m
        if self.lo is None:
            self._relearn_add(p, m)
            return
        lab = self._label(m, p)
        if lab is None:
            onset = self.last_solid.tl if self.last_solid is not None else p.tb
            self._anomaly("level", onset)
            self._relearn_add(p, m)
            return
        p.lab = lab
        if self.hole_fresh:
            self.hole_fresh = False
            if lab != self.run_lab:
                self.run_hole = None   # the single edge lies inside the hole
        if self.run_lab is None:
            self._start_run(lab, None, p.tb)
        elif lab != self.run_lab:
            ls = self.last_solid
            edge = (ls.tl if ls is not None else None, p.tb)
            self._end_run(edge)
            self._start_run(lab, edge, p.tb)
        self.last_solid = p

    def _cmin(self, n1: int, n2: int) -> float:
        """Smallest level difference that counts as two different levels."""
        s = self.sig_ref
        return max(2.0, 2.5 * s, 4.0 * s * math.sqrt(1.0 / n1 + 1.0 / n2))

    def _relearn_add(self, p: _Piece, m: float) -> None:
        buf = self.relearn
        if buf:
            e = buf[-1]
            if abs(m - e[0]) < self._cmin(e[1], p.n):
                e[0] = (e[0] * e[1] + m * p.n) / (e[1] + p.n)
                e[1] += p.n
                e[3] = p
                return
        buf.append([m, p.n, p.tb, p])
        if len(buf) > 3:
            del buf[0]
        if len(buf) < 3:
            return
        A, B, C = buf
        cAB = abs(B[0] - A[0])
        tolAC = max(0.3 * cAB, 3.0 * self.sig_ref * math.sqrt(1.0 / A[1] + 1.0 / C[1]))
        if abs(C[0] - A[0]) > tolAC:
            del buf[0]
            return
        # confirmed two-level model
        la = 1 if A[0] > B[0] else 0
        lvl_a = (A[0] * A[1] + C[0] * C[1]) / (A[1] + C[1])
        if la == 1:
            self.hi, self.lo = lvl_a, B[0]
        else:
            self.lo, self.hi = lvl_a, B[0]
        self.relearn = []
        self._log("LEARN", round(self.lo, 1), round(self.hi, 1), "drop" if self.drop_first else "")
        eAB = (A[3].tl, B[2])
        eBC = (B[3].tl, C[2])
        if not self.drop_first:
            # A: partial run, B: complete run
            self.run_lab = la
            self.run_start = None
            self.run_t0 = A[2]
            self.run_long = False
            self.run_hole = None
            self._end_run(eAB)
            self._start_run(1 - la, eAB, B[2])
            self._end_run(eBC)
        self.drop_first = False
        self._start_run(la, eBC, C[2])
        C[3].lab = la
        self.last_solid = C[3]

    def _anomaly(self, why: str, onset: float) -> None:
        self._log("ANOMALY", why, round(onset, 3))
        seg = self.seg
        if seg is not None:
            k = 0
            while seg.items and k < 2 and seg.items[-1][1] > onset - self.W_RETRACT:
                if not seg.pop():
                    seg = self.seg = None
                    break
                k += 1
                self._log("RETRACT")
        self._break(why)
        self.lo = self.hi = None
        self.run_lab = None
        self.run_hole = None
        self.hole_fresh = False
        self.last_solid = None
        self.relearn = []
        self.drop_first = True

    # ------------------------------------------------------------------- runs
    def _start_run(self, lab: int, edge, t0: float) -> None:
        self.run_lab = lab
        self.run_start = edge if (edge is not None and edge[0] is not None) else None
        self.run_t0 = t0
        self.run_long = False
        self.run_hole = None
        self.hole_fresh = False
        self.lb_mask = 7 if lab == 0 else 3
        self.edges.append(t0)

    def _slop(self) -> float:
        if self.lo is not None and (self.hi - self.lo) < 8 * self.sig_ref:
            return 0.5 * self.dt_med + 0.012
        return 0.012

    def _mask(self, lab: int, dlo: float, dhi: float) -> int:
        u = self.unit
        return _classify(lab, (dlo - FRAME_BIAS) / u, (dhi - FRAME_BIAS) / u)

    def _end_run(self, edge) -> None:
        """The current run ended at `edge` = (last sample of it, first sample after)."""
        lab = self.run_lab
        if lab is None or self.run_long or edge[0] is None:
            return
        u = self.unit
        if self.run_start is None:
            d_lo = edge[0] - self.run_t0
            mask = _classify_lb(lab, (d_lo - 0.01) / u)
            if mask:
                self._add(lab, [[(lab, mask)]], edge[0])
            return
        s0, s1 = self.run_start
        e0, e1 = edge
        d = 0.5 * (e0 + e1) - 0.5 * (s0 + s1)
        w = 0.85 * 0.5 * ((e1 - e0) + (s1 - s0)) + self._slop()
        mask = self._mask(lab, d - w, d + w)
        if self.run_hole is not None:
            # either one run, or the run was split by a hidden short run inside the hole
            t1, t2 = self.run_hole
            rmin = 0.75 * self._unit_safe()
            sl = self._slop()
            m1 = self._mask(lab, t1 - s1 - sl, t2 - rmin - s0 + sl)
            mh = self._mask(1 - lab, rmin, t2 - t1)
            m2 = self._mask(lab, e0 - t2 - sl, e1 - t1 - rmin + sl)
            paths = []
            if mask:
                paths.append([(lab, mask)])
            if m1 and mh and m2:
                paths.append([(lab, m1), (1 - lab, mh), (lab, m2)])
            if not paths:
                self._log("INVALID-H", lab, round(d, 3))
                self._break("invalid")
                return
            self._add(lab, paths, e0)
            return
        if mask == 0:
            self._log("INVALID", lab, round(d, 3))
            if (d + w - FRAME_BIAS) / u < BANDS[lab][0][1]:
                self._anomaly("short", s0)
            else:
                self._break("long")
            return
        if mask in (DOT, DASH, INTRA, LETTER, WGAP) and w / u < 0.3:
            self.units.append((d - FRAME_BIAS) / NOMINAL[lab][mask])
            if len(self.units) >= 3:
                us = sorted(self.units)
                self.unit = min(UNIT_MAX, max(UNIT_MIN, us[len(us) // 2]))
        self._add(lab, [[(lab, mask)]], e0)

    def _end_run_partial(self, t_last: float) -> None:
        """The current run was cut by an erasure at t_last: its length is a lower bound."""
        lab = self.run_lab
        if self.run_start is None or self.run_hole is not None:
            return  # no usable bound
        d_lo = t_last - self.run_start[1]
        mask = _classify_lb(lab, (d_lo - 0.01) / self.unit)
        if mask:
            self._add(lab, [[(lab, mask)]], t_last)

    def _check_run(self, t: float) -> None:
        lab = self.run_lab
        if lab is None or self.run_long:
            return
        # last time surely inside the run
        ts = t
        if lab == 1 and self.cd is not None:
            ts = self.cd[0] if self.cd[0] is not None else t
        elif lab == 0 and self.cu is not None:
            ts = self.cu[0] if self.cu[0] is not None else t
        if self.run_hole is not None:
            t0 = self.run_hole[1]
        else:
            t0 = self.run_start[1] if self.run_start is not None else self.run_t0
        r = (ts - t0 - 0.01) / self.unit
        if r > BANDS[lab][-1][2]:
            self.run_long = True
            self._log("LONG", lab)
            if self.run_start is not None:
                self._break("long")
            return
        if self.run_start is None or self.run_hole is not None:
            return
        mask = _classify_lb(lab, r)
        if mask != self.lb_mask:
            self.lb_mask = mask
            self._decide()

    # ---------------------------------------------------------------- symbols
    def _add(self, lab: int, paths, t_end: float) -> None:
        if self.log is not None:
            self._log("SYM", "ON " if lab else "off", paths if len(paths) > 1 else paths[0][0][1], round(self.unit, 3))
        seg = self.seg
        if seg is None:
            seg = self.seg = _Segment()
        seg.add(paths, t_end)
        if len(paths) == 1 and len(paths[0]) == 1:
            self._display(lab, paths[0][0][1])
        else:
            self._display(lab, 0)
        self.dirty = True
        self.lb_mask = 0
        self._decide()

    def _break(self, why: str) -> None:
        seg = self.seg
        if seg is not None:
            if self.log is not None:
                wmin = seg.summary()[0]
                self._log("BREAK", why, seg.nruns, "".join(_fmt(it) for it, _ in seg.items),
                          {TABLES.words[i]: int(wmin[i]) for i in np.argsort(wmin)[:3]})
            if seg.nruns >= 1:
                self.closed.append(seg.summary())
            self.seg = None
            self._display_break()
            self.dirty = True
            self._decide()

    # ------------------------------------------------------------- decision
    def _decide(self) -> None:
        summ = list(self.closed)
        seg = self.seg
        extra = None
        if self.run_lab is not None and self.run_start is not None and not self.run_long \
                and self.run_hole is None and self.lb_mask not in (0, 7, 3):
            extra = (self.run_lab, self.lb_mask)
        if seg is not None:
            summ.append(seg.summary(extra))
        elif extra is not None:
            summ.append(_Segment().summary(extra))
        if not summ:
            self.ranking = []
            self.confidence = 0.0
            return
        nw = len(TABLES.words)
        tot = np.zeros(nw, np.int64)
        E = np.zeros(nw)
        best = np.full(nw, -1e9)
        for wmin, cnt0, info, n in summ:
            tot += wmin
            ok = wmin == 0
            bits = info + np.log2(np.maximum(cnt0, 1)) - TABLES.log_len
            E += np.where(ok, bits, 0.0)
            best = np.maximum(best, np.where(ok, bits, -1e9))
        score = E - 7.0 * tot
        score -= score.max()
        p = np.exp2(score)
        p /= p.sum()
        order = np.argsort(-p)
        self.ranking = [(TABLES.words[i], float(p[i])) for i in order[:5]]
        self.confidence = float(p[order[0]])
        if self.answer is not None:
            return
        cands = np.flatnonzero(tot == 0)
        if len(cands) != 1:
            return
        b = int(cands[0])
        others = np.delete(tot, b)
        snr = (self.hi - self.lo) / self.sig_ref if self.lo is not None else 0.0
        if self.log is not None:
            self._log("DEC", TABLES.words[b], round(float(E[b]), 1), round(float(best[b]), 1), int(others.min()),
                      round(snr, 1))
        if others.min() < (1 if snr >= 8 else self.MARGIN):
            return
        tt, ts = self.T_TOTAL, self.T_SEG
        if snr < 6:
            tt, ts = tt + 3, ts + 2
        if E[b] < tt or best[b] < ts:
            return
        self.answer = TABLES.words[b]
        self._log("COMMIT", self.answer, round(float(E[b]), 1), round(float(best[b]), 1))
        self.dirty = True

    # -------------------------------------------------------------- display
    def _display(self, state: int, mask: int) -> None:
        if state == 1:
            ch = "." if mask == DOT else "-" if mask == DASH else "?"
            self.morse_tokens.append(ch)
            self.cur_letter += ch
        else:
            if mask == INTRA:
                return
            if mask == WGAP:
                self.morse_tokens.append(" / ")
                self._end_letter()
                self.letter_tokens.append("/")
            else:
                self.morse_tokens.append(" ")
                self._end_letter()

    def _end_letter(self) -> None:
        if self.cur_letter:
            self.letter_tokens.append(REVERSE_MORSE.get(self.cur_letter, "?"))
        self.cur_letter = ""

    def _display_break(self) -> None:
        if self.morse_tokens and self.morse_tokens[-1] != " ? ":
            self.morse_tokens.append(" ? ")
            self.cur_letter = ""
            self.letter_tokens.append("?")

    # --------------------------------------------------------------- output
    def status(self) -> Status:
        now = self.now
        e = self.edges
        sig = bool(e) and now - e[-1] < 3.8 and sum(1 for t in e if now - t < 8.0) >= 3
        if self._status is not None and not self.dirty and sig == self._status_sig:
            return self._status
        self.dirty = False
        self._status_sig = sig
        ranking = list(self.ranking)
        conf = self.confidence
        if self.answer is not None:
            conf = 1.0
            ranking = [(self.answer, 1.0)] + [(w, 0.0) for w, _ in ranking if w != self.answer][:4]
        morse = "".join(self.morse_tokens).strip()
        letters = "".join(self.letter_tokens) + (REVERSE_MORSE.get(self.cur_letter, "?") if self.cur_letter else "")
        self._status = Status(answer=self.answer, confidence=conf, ranking=ranking, morse=morse,
                              letters=letters, signal=sig, unit=self.unit)
        return self._status
