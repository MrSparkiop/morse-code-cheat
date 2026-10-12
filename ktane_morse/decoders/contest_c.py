"""Contestant C: time-domain Bayesian template matching (forward HMM over word/scale/phase).

(docstring is finalised at the end)
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..decoding import Status
from ..morse import MORSE, REVERSE_MORSE, WORDS

# --------------------------------------------------------------------------- #
# Model constants

UNIT = 0.27            # game DotLength (s)
EXTRA = 0.005          # mean extra time per element (game frame quantisation)
DT = 0.08              # HMM bin width (s)
S_MIN, S_MAX, S_STEP = 0.76, 1.31, 1.03   # time-scale hypotheses (geometric grid)
P_SLIP = 0.012         # per-bin phase stay/skip probability (absorbs scale-grid error and jitter)
BETA = 2.5             # |log-likelihood ratio| of one fully observed bin
LAG = 0.24             # evidence enters the HMM this long after real time
HOLD = 0.18            # a sample's value is trusted at most HOLD/2 away from it
FLOOR = 1e-16          # fixed-share floor of every segment, relative to total
RESET_RHO = 0.35       # phase re-acquisition mix after an unknown stretch
RESET_MIN_BINS = 3     # ... of at least this many bins
NULL_Q = (0.05, 0.15, 0.35)   # switching probabilities of the free-blinker null models
D_WINDOW = 12.0        # window (s) of the predictive-likelihood comparison against the null
ALT_PRIOR = 0.5        # prior mass of the periodic "distractor" hypotheses
ALT_PERIODS = (0.2, 4.6, 1.06)
ALT_DUTIES = (0.3, 0.5, 0.7)
ALT_MIN_LEN = 4.0      # alternatives are rendered over several periods (integer-bin accuracy)
MAXBACK = 6.0          # evidence older than this is never fed (start-up backlog limit)
CATCH = 25             # at most this many HMM steps per feed() call (bounded call time)
CONST_MAX = 4.5        # a light that has not changed for this long is "not blinking": no evidence
FIT_WIN = 5.0          # level-model window (s)
EPS_DEAD = 1e-4        # per-bin probability that the light dies for good

WORD_LIST = list(WORDS)
NW = len(WORD_LIST)


def _elements(word: str) -> list[tuple[bool, int]]:
    out: list[tuple[bool, int]] = []
    codes = [MORSE[c] for c in word.upper()]
    for li, code in enumerate(codes):
        for si, sym in enumerate(code):
            out.append((True, 1 if sym == "." else 3))
            if si < len(code) - 1:
                out.append((False, 1))
        out.append((False, 4 if li < len(codes) - 1 else 10))
    return out


def _render(durs: list[float], ons: list[bool], reps: int = 1) -> np.ndarray:
    """ON-fraction of each DT bin over `reps` cycles of the (on, duration) pattern."""
    d = np.tile(np.array(durs, float), reps)
    on = np.tile(np.array(ons, float), reps)
    period = d.sum()
    n = max(4, int(round(period / DT)))
    edges = np.concatenate([[0.0], np.cumsum(d)]) * (n / period)
    cum_on = np.concatenate([[0.0], np.cumsum(on * np.diff(edges))])
    f = np.interp(np.arange(n + 1, dtype=float), edges, cum_on)
    return np.clip(np.diff(f), 0.0, 1.0)


class _Model:
    """Concatenated cyclic templates: every (scale, word) segment, then the alternatives."""

    def __init__(self) -> None:
        scales = []
        s = S_MIN
        while s <= S_MAX * 1.0001:
            scales.append(s)
            s *= S_STEP
        self.scales = np.array(scales)
        tpl, lens, cls, ss, prior = [], [], [], [], []
        nword_seg = len(scales) * NW
        for si, sc in enumerate(scales):
            for wi, w in enumerate(WORD_LIST):
                els = _elements(w)
                t = _render([u * UNIT * sc + EXTRA for _, u in els], [o for o, _ in els])
                tpl.append(t); lens.append(len(t)); cls.append(wi); ss.append(si)
                prior.append((1.0 - ALT_PRIOR) / nword_seg)
        alts = []
        p = ALT_PERIODS[0]
        while p <= ALT_PERIODS[1]:
            for duty in ALT_DUTIES:
                alts.append((p, duty))
            p *= ALT_PERIODS[2]
        for p, duty in alts:
            reps = max(1, int(math.ceil(ALT_MIN_LEN / p)))
            t = _render([p * duty, p * (1 - duty)], [True, False], reps)
            tpl.append(t); lens.append(len(t)); cls.append(NW); ss.append(-1)
            prior.append(ALT_PRIOR / len(alts))
        self.tpl = np.concatenate(tpl).astype(np.float32)
        self.n = len(self.tpl)
        self.lens = np.array(lens)
        self.starts = np.concatenate([[0], np.cumsum(self.lens)[:-1]])
        self.ends = self.starts + self.lens
        self.seg_cls = np.array(cls)
        self.seg_scale = np.array(ss)
        self.nseg = len(lens)
        self.nword_seg = nword_seg
        self.word_end = int(self.starts[nword_seg])      # word states are alpha[:word_end]
        prior = np.array(prior)
        self.init = np.repeat(prior / self.lens, self.lens).astype(np.float32)
        self.init /= self.init.sum()
        # boundary fix-ups for the cyclic shift / diffusion
        self.last = self.ends - 1
        self.left_of_start = self.starts - 1           # index -1 only for segment 0 (handled)
        self.right_of_last = self.ends % self.n


_MODEL: _Model | None = None


def _model() -> _Model:
    global _MODEL
    if _MODEL is None:
        _MODEL = _Model()
    return _MODEL


# --------------------------------------------------------------------------- #

class Decoder:
    record: list | None = None   # diagnostics hook: set to a list to log per-step internals

    def __init__(self) -> None:
        self.m = _model()
        self.reset()

    # ------------------------------------------------------------------ reset
    def reset(self) -> None:
        m = self.m
        self.alpha = m.init.copy()
        self.buf = np.empty_like(self.alpha)
        self.tmp = np.empty_like(self.alpha)
        self.logz = 0.0
        self.null = [[0.5, 0.5] for _ in NULL_Q]
        self.null_logz = [0.0 for _ in NULL_Q]
        self.t0: float | None = None
        self.last_t: float | None = None
        # sample buffer (relative times, values, usable flags); absolute index of st[0] is `base`
        self.st: list[float] = []
        self.sv: list[float] = []
        self.sfl: list[int] = []
        self.prev_out = False
        self.out_start = -1e9
        self.dirty_t = -1e9
        self.base = 0
        self.p = 0                # absolute index of the first sample the next bin may need
        self.prev: tuple[float, float] | None = None   # (t, v) of last valid sample
        self.dv: deque[float] = deque(maxlen=400)
        self.noise_raw = 0.0
        self.noise = 0.0
        self.lo = self.hi = None
        self.last_fit = -1e9
        self.fit_from = -1e9
        self.jump_times: deque[float] = deque()
        self.garbage_until = -1e9
        self.garbage_start = -1e9
        self.none_since: float | None = None
        # display runs: [state, start, end] with state 1 = on, 0 = off, -1 = unknown
        self.runs: deque[list] = deque(maxlen=80)
        self.edges: deque[float] = deque(maxlen=64)
        # bins
        self.b0 = 0
        self.pristine = True
        self.bin_state = -1
        self.bin_edge_t = 0.0
        self.unk_bins = 0
        self.steps = 0
        self.Lhist: deque[float] = deque()
        self.lam_abs = 0.0
        self.logz_prev = 0.0
        self.sur: deque[float] = deque(maxlen=40)
        self.dead = 0.0
        # status
        self.answer: str | None = None
        self.hold = 0
        self.unit = UNIT
        self._status = Status()
        self.diag: dict = {}

    # ------------------------------------------------------------------- feed
    def feed(self, t: float, level: float | None) -> None:
        if self.t0 is None:
            self.t0 = t
        t = t - self.t0
        if self.last_t is not None and t <= self.last_t:
            return
        self.last_t = t
        ok = level is not None and math.isfinite(level)
        v = float(level) if ok else 0.0
        self.st.append(t)
        self.sv.append(v)
        self.sfl.append(0 if ok else 2)
        if not ok:
            if self.none_since is None:
                self.none_since = t
            self.dirty_t = t
            self.prev = None
            self.prev_out = False
            self._run(-1, t)
        else:
            if self.none_since is not None:
                if t - self.none_since > 0.3:
                    self.fit_from = t          # the view may have moved: refit levels
                self.none_since = None
            self._classify(t, v)
        if t - self.last_fit >= 0.2:
            self.last_fit = t
            self._fit(t)
        self._advance(t)

    # ------------------------------------------------------------ front end
    def _classify(self, t: float, v: float) -> None:
        """Feed-time bookkeeping of one valid sample: garbage / level-shift detection, display runs.

        Sample flags: 0 = usable, 1 = outside the current level model (still used for re-fitting),
        2 = garbage or missing (never used).
        """
        prev = self.prev
        close = prev is not None and t - prev[0] < 0.12
        state = -1
        if self.lo is not None:
            lo, hi = self.lo, self.hi
            c = hi - lo
            tol = max(4.0 * self.noise, 0.3 * c)
            inr = lo - tol <= v <= hi + tol
            # random-looking data (many big frame-to-frame jumps) is garbage
            if close and abs(v - prev[1]) > 0.5 * c:
                self.jump_times.append(t)
            while self.jump_times and self.jump_times[0] < t - 0.25:
                self.jump_times.popleft()
            if len(self.jump_times) >= 3 or (not inr and self.prev_out and close and abs(v - prev[1]) > 0.5 * c):
                if self.garbage_until < t - 0.05:
                    self.garbage_start = t
                    self._void(t - 0.15)
                self.garbage_until = t + 0.15
            if t < self.garbage_until:
                self.sfl[-1] = 2
                self.dirty_t = t
            else:
                if self.garbage_start > 0 and t - self.garbage_start > 0.45:
                    self.fit_from = t          # after a long garbage stretch the view may differ
                self.garbage_start = -1e9
                if not inr:
                    # a steady value the model cannot explain: the levels have shifted -> refit
                    self.sfl[-1] = 1
                    self.dirty_t = t
                    if not (self.prev_out and close and abs(v - prev[1]) <= 0.25 * c):
                        self.out_start = t
                    elif t - self.out_start >= 0.25 and self.fit_from < self.out_start:
                        self.fit_from = self.out_start - 0.02
                else:
                    if close:
                        self.dv.append(abs(v - prev[1]))
                    state = 1 if v > 0.5 * (lo + hi) else 0
            self.prev_out = not inr
        elif close:
            self.dv.append(abs(v - prev[1]))
        self._run(state, t)
        self.prev = (t, v)

    def _void(self, since: float) -> None:
        """Mark buffered samples from `since` on as garbage (detected late)."""
        st, sfl = self.st, self.sfl
        i = len(st) - 1
        while i >= 0 and st[i] >= since:
            sfl[i] = 2
            i -= 1

    def _fit(self, t: float) -> None:
        """(Re)fit the two-level model (off/on level, noise) on the recent non-garbage samples."""
        if len(self.dv) >= 20:
            self.noise_raw = float(np.median(np.fromiter(self.dv, float))) / 0.954
        st = self.st
        restricted = self.fit_from > t - FIT_WIN
        t_from = self.fit_from if restricted else t - FIT_WIN
        if restricted and t - t_from < 0.6 and self.lo is not None:
            return
        i = len(st)
        while i > 0 and st[i - 1] >= t_from:
            i -= 1
        if len(st) - i < 16:
            return
        vv = np.array(self.sv[i:])
        fl = np.array(self.sfl[i:])
        x = np.sort(vv[fl != 2])
        n = len(x)
        if n < 16:
            return
        cs = np.cumsum(x)
        k = np.arange(1, n)
        m0 = cs[:-1] / k
        m1 = (cs[-1] - cs[:-1]) / (n - k)
        between = k * (n - k) * (m1 - m0) ** 2
        kmin = max(4, int(0.05 * n))
        between[: kmin - 1] = -1
        between[n - kmin:] = -1
        j = int(np.argmax(between)) + 1
        lo_c, hi_c = x[:j], x[j:]
        if len(lo_c) < kmin or len(hi_c) < kmin:
            return
        lo = float(np.median(lo_c))
        hi = float(np.median(hi_c))
        spread = max(float(np.median(np.abs(lo_c - lo))), float(np.median(np.abs(hi_c - hi)))) / 0.674
        sep = hi - lo
        nraw = max(self.noise_raw, 1e-3 * max(1.0, abs(hi)))
        if sep < 4.0 * max(nraw, spread):
            # low contrast-to-noise: accept only if the variance is clearly more than the frame-to-frame
            # noise explains, then refine the two levels by EM with the noise fixed
            if len(self.dv) < 20 or float(np.var(x[n // 50: n - n // 50])) < 2.0 * nraw * nraw:
                return
            w1 = len(hi_c) / n
            mu0, mu1 = lo, hi
            k2 = -0.5 / (nraw * nraw)
            for _ in range(8):
                l0 = (1.0 - w1) * np.exp(k2 * (x - mu0) ** 2)
                l1 = w1 * np.exp(k2 * (x - mu1) ** 2)
                r1 = l1 / (l0 + l1 + 1e-300)
                s1 = float(r1.sum())
                if s1 < 4 or n - s1 < 4:
                    return
                w1 = s1 / n
                mu1 = float((r1 * x).sum()) / s1
                mu0 = float(((1.0 - r1) * x).sum()) / (n - s1)
            lo, hi = mu0, mu1
            sep = hi - lo
            if sep < 2.0 * nraw or min(w1, 1.0 - w1) < 0.05:
                return
        if sep < 0.03 * max(abs(hi), abs(lo)):
            return
        noise = max(nraw, 0.04 * sep)
        tol = max(4.0 * noise, 0.3 * sep)
        if np.count_nonzero((x < lo - tol) | (x > hi + tol)) > 0.2 * n:
            return                                # does not explain the window (mixed regimes)
        self.lo, self.hi, self.noise = lo, hi, noise

    def _run(self, state: int, t: float) -> None:
        runs = self.runs
        if runs and runs[-1][0] == state:
            runs[-1][2] = t
            return
        if runs and state != -1:
            # ignore a single short blip: merge it back into the run before it
            r = runs[-1]
            if r[2] - r[1] < 0.06 and len(runs) >= 2 and runs[-2][0] == state:
                runs.pop()
                runs[-1][2] = t
                return
        if runs and runs[-1][0] in (0, 1) and state in (0, 1):
            self.edges.append(t)
        start = runs[-1][2] if runs else t
        runs.append([state, start, t])

    def _bin_lambda(self, k: int) -> float:
        """Evidence (log-likelihood ratio on/off, in [-BETA, BETA]) of bin k from buffered samples."""
        if self.lo is None:
            return 0.0
        a = k * DT
        b = a + DT
        h = 0.5 * HOLD
        st, sv, sfl = self.st, self.sv, self.sfl
        n = len(st)
        i = self.p - self.base
        if i < 0:
            i = 0
        while i < n and st[i] < a - h:
            i += 1
        self.p = i + self.base
        lo, hi = self.lo, self.hi
        c = hi - lo
        mid = 0.5 * (lo + hi)
        tol = max(4.0 * self.noise, 0.3 * c)
        sig = max(self.noise, 0.08 * c)
        gain = 0.5 * c / (sig * sig)
        acc = 0.0
        j = i
        while j < n:
            tj = st[j]
            if tj > b + h:
                break
            if sfl[j] != 2:
                vj = sv[j]
                if lo - tol <= vj <= hi + tol:
                    l0 = tj - h
                    if j > 0:
                        q = 0.5 * (st[j - 1] + tj)
                        if q > l0:
                            l0 = q
                    h0 = tj + h
                    if j + 1 < n:
                        q = 0.5 * (tj + st[j + 1])
                        if q < h0:
                            h0 = q
                    if l0 < a:
                        l0 = a
                    if h0 > b:
                        h0 = b
                    if h0 > l0:
                        acc += (h0 - l0) * math.tanh(gain * (vj - mid))
            j += 1
        return BETA * acc / DT

    def _advance(self, t: float) -> None:
        if self.pristine:
            # nothing has been learnt yet: the HMM state is shift-invariant, so old bins can be skipped
            kmin = int((t - MAXBACK) / DT)
            if kmin > self.b0:
                self.b0 = kmin
            if self.lo is None:
                self._trim(t)
                return
        steps = 0
        while (self.b0 + 1) * DT <= t - LAG and steps < CATCH:
            lam = self._bin_lambda(self.b0)
            center = (self.b0 + 0.5) * DT
            # a light that has not changed for a long time is not blinking (solved / static)
            if lam > 0.5 * BETA:
                bs = 1
            elif lam < -0.5 * BETA:
                bs = 0
            else:
                bs = -1
            if bs >= 0:
                if bs != self.bin_state:
                    self.bin_state = bs
                    self.bin_edge_t = center
                elif center - self.bin_edge_t > CONST_MAX:
                    lam = 0.0
            self.b0 += 1
            self._step(lam)
            steps += 1
        self._trim(t)

    def _trim(self, t: float) -> None:
        """Bound the sample buffer: keep the fit window and whatever the next bin needs."""
        st = self.st
        if len(st) < 3000:
            return
        need = self.p - self.base - 2
        lim = t - FIT_WIN - 1.0
        cut = 0
        while cut < need and st[cut] < lim:
            cut += 1
        if self.pristine and self.lo is None:
            # backlog only matters up to MAXBACK
            while cut < len(st) - 1 and st[cut] < t - MAXBACK - 1.0:
                cut += 1
            if self.p - self.base < cut:
                self.p = self.base + cut
        if cut > 500:
            del st[:cut]
            del self.sv[:cut]
            del self.sfl[:cut]
            self.base += cut

    # ----------------------------------------------------------------- HMM
    def _step(self, lam: float) -> None:
        m = self.m
        if lam > BETA:
            lam = BETA
        elif lam < -BETA:
            lam = -BETA
        a, buf = self.alpha, self.buf
        # deterministic phase advance (cyclic inside every segment)
        buf[1:] = a[:-1]
        buf[m.starts] = a[m.last]
        a, buf = buf, a
        if not self.pristine:
            self.dead += EPS_DEAD     # "the light died (module solved / view lost)" absorbs a little mass
        if abs(lam) > 0.05:
            self.pristine = False
            if self.unk_bins >= RESET_MIN_BINS:
                tot = np.add.reduceat(a, m.starts)
                a *= (1.0 - RESET_RHO)
                a += np.repeat((RESET_RHO * tot / m.lens).astype(np.float32), m.lens)
            self.unk_bins = 0
            r = math.exp(lam) - 1.0
            em = self.tmp
            np.multiply(m.tpl, r, out=em)
            em += 1.0
            a *= em
            self.logz -= 0.5 * lam
            self.lam_abs += abs(lam)
            e_on, e_off = math.exp(0.5 * lam), math.exp(-0.5 * lam)
            for i, q in enumerate(NULL_Q):
                p = self.null[i]
                pon = (p[0] * (1 - q) + p[1] * q) * e_on
                poff = (p[1] * (1 - q) + p[0] * q) * e_off
                z = pon + poff
                p[0], p[1] = pon / z, poff / z
                self.null_logz[i] += math.log(z)
        else:
            self.unk_bins += 1
        self.steps += 1
        if self.steps % 2 == 0:
            # phase diffusion (stay / skip one bin), applied every other step
            p2 = 2.0 * P_SLIP
            tmp = self.tmp
            np.multiply(a, 1.0 - 2.0 * p2, out=buf)
            np.multiply(a, p2, out=tmp)
            buf[1:] += tmp[:-1]
            buf[:-1] += tmp[1:]
            lft = tmp[m.left_of_start]
            lft[0] = 0.0
            buf[m.starts] += tmp[m.last] - lft
            rgt = tmp[m.right_of_last]
            rgt[-1] = 0.0
            buf[m.last] += tmp[m.starts] - rgt
            a, buf = buf, a
            z = float(a.sum())
            a *= np.float32(1.0 / z)
            self.dead /= z
            self.logz += math.log(z)
            # surprise of the last two bins: |lam|/2 summed minus the predictive log-likelihood gained
            sur = 0.5 * self.lam_abs - (self.logz - self.logz_prev)
            self.logz_prev = self.logz
            self.lam_abs = 0.0
            self.sur.append(max(0.0, sur))
            self.alpha, self.buf = a, buf
            self._update_status()
        else:
            self.alpha, self.buf = a, buf

    # -------------------------------------------------------------- status
    def _null_logz(self) -> float:
        mx = max(self.null_logz)
        return mx + math.log(sum(math.exp(z - mx) for z in self.null_logz) / len(self.null_logz))

    def _update_status(self) -> None:
        m = self.m
        L = self.logz - self._null_logz()
        self.Lhist.append(L)
        while len(self.Lhist) > int(D_WINDOW / (2 * DT)):
            self.Lhist.popleft()
        D = L - min(self.Lhist)
        tot = np.add.reduceat(self.alpha, m.starts).astype(float)
        if self.steps % 6 == 0:
            low = tot < FLOOR
            if low.any():
                add = np.where(low, (FLOOR - tot) / m.lens, 0.0).astype(np.float32)
                self.alpha += np.repeat(add, m.lens)
        cp = np.bincount(m.seg_cls, weights=tot, minlength=NW + 1)
        cp /= cp.sum()
        wp = cp[:NW]
        order = np.argsort(-wp)
        b, b2 = int(order[0]), int(order[1])
        lp = math.log(max(wp[b], 1e-300))
        Mw = lp - math.log(max(wp[b2], 1e-300))
        Ma = lp - math.log(max(cp[NW], 1e-300))
        Md = math.log(max(float(tot.sum()) * float(wp[b]), 1e-300)) - math.log(max(self.dead, 1e-300))
        wsum = tot[: m.nword_seg]
        sp = np.bincount(m.seg_scale[: m.nword_seg], weights=wsum, minlength=len(m.scales))
        sp /= max(sp.sum(), 1e-300)
        scale = float(np.exp((sp * np.log(m.scales)).sum()))
        self.unit = UNIT * scale
        self.diag = {"Mw": Mw, "Ma": Ma, "D": D, "best": WORD_LIST[b], "p": float(wp[b]), "alt": float(cp[NW]), "scale": scale}
        if self.record is not None:
            sl = list(self.sur)
            self.record.append((self.last_t + self.t0, b, Mw, Ma, D, sum(sl[-10:]), sum(sl[-20:]), sum(sl[-30:]), self.last_t - self.dirty_t, (self.hi - self.lo) / self.noise if self.lo is not None else 0.0, self.last_t - self.bin_edge_t - LAG, Md))
        if self.answer is None:
            if Mw >= 20.0 and Ma >= 20.0 and D >= 15.0:
                self.hold += 1
                if self.hold >= 2:
                    self.answer = WORD_LIST[b]
            else:
                self.hold = 0
        ranking = [(WORD_LIST[int(i)], float(wp[int(i)])) for i in order[:5]]
        self._status = Status(answer=self.answer, confidence=float(wp[b]), ranking=ranking,
                              signal=self.lo is not None, unit=self.unit)

    def status(self) -> Status:
        return self._status
