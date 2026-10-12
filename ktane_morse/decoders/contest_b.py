"""Contestant B: probabilistic closed-vocabulary Morse matcher.

Pipeline: brightness samples -> on/off runs (binariser) -> forward algorithm
over every word's cyclic element sequence (matcher) -> posterior -> commit.

Binariser.  Two levels (lamp off / on) are found in a short window by a
two-class split that must look clean (well separated, little time in the
middle, coherent runs).  They are then tracked slowly; a CUSUM on the
per-sample log-likelihood ratio confirms lamp edges.  The noise estimate only
learns from samples that sit on a level, so it cannot be inflated by room
light changes.  Samples that fit neither level (room light dims/brightens,
camera motion, pause screens), None samples, capture gaps and absurdly long
runs break the chain: the run in progress becomes right-censored.  Lock is
regained either at the first clean step of about the old contrast between two
flat plateaus (fast; the run that starts there has a known start) or by a new
window search.

Matcher.  For each word, each cyclic element position and a 50-point grid of
unit lengths u (flat prior 0.21-0.345 s, so a 0.8-1.25x time scaling of the
game's 0.27 s is fine), alpha holds log P(runs so far, last run = element).
Durations are log-normal-ish (gaussian around units*u + frame bias, relative
sigma 5% plus edge-time uncertainty), mixed with a 2% broad outlier density.
Left/right censored runs use survival functions.  Extra transitions: one run
covering element+gap+element (a missed pulse, likelier across capture
stalls), three runs making one element (a spurious blip), and a rare jump to
any position (resync after garbage).  A break is bridged: the time from the
last edge before the erasure to the first edge after it must be filled by a
whole number of elements (gaussian on their summed length), mixed with a
restart branch (independent alignment of the new segment), so short erasures
cost almost nothing and pauses do not break the model.  Two null models
compete with the 16 words: random Morse (i.i.d. element classes) and
"not Morse" (i.i.d. broad durations).

Decision.  Posterior by Bayes over words and nulls with the unit integrated
out.  Commit when the leader has posterior >= 0.995, beats the runner-up by
>= 7 nats and both nulls by >= 6 nats, its unit estimate is inside the grid,
and this holds on consecutive updates.  Committed answers never change.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from ..decoding import Status
from ..morse import MORSE, REVERSE_MORSE, WORDS

NEG = -np.inf
_LOG2PI = math.log(2.0 * math.pi)

# ----------------------------------------------------------------------------- timing model
N_U = 50
U_GRID = np.exp(np.linspace(math.log(0.17), math.log(0.43), N_U))   # candidate dot lengths (s)
U_FLAT = (0.21, 0.345)        # flat prior on the unit (0.78x .. 1.28x the game's 0.27 s)
U_TAIL = 0.05                 # log-space sd of the prior's tails outside the flat band
SIG_REL = 0.05                # relative timing noise of one element
SIG_BASE = 0.008              # absolute timing noise (s): game frames, display refresh
BIAS = 0.005                  # every element runs up to one game frame long
EPS = 0.02                    # probability a run's duration is an outlier ...
EPS_CLEAN = 0.003             # ... when the run was seen cleanly (steady levels, no stalls, good SNR)
G_MIN, G_MAX = 0.02, 8.0      # support of the broad (log-uniform) duration density
_LOG_G_RANGE = math.log(G_MAX / G_MIN)
P_MERGE = 0.003               # one observed run covers element+gap+element (a missed gap / dot)
P_MERGE_GAP = 0.2             # ... when the run contained a capture stall long enough to hide one
P_SPLIT = 0.003               # three observed runs make up one element (a spurious short blip)
P_JUMP = 0.001                # the alignment jumps anywhere in the word (resync after garbage) ...
P_JUMP_CLEAN = 1e-5          # ... when this run and the one before it were seen cleanly
P_STOP = 0.05                 # display only: the light may have stopped during the open run
P_BRIDGE = 0.8                # after an erasure: timing continues (vs. restart, e.g. a pause)
T_BRIDGE = 5.0                # longest erasure (last edge before -> first edge after) bridged
J_MAX = 24                    # most elements a bridge may span

PRIOR_WORDS, PRIOR_NM, PRIOR_NN, PRIOR_BL = 0.5, 0.25, 0.125, 0.125
BL_ALPHA = 1.0                # blinker null: weight of a brand-new duration vs. repeating an old one
BL_REL = 0.07                 # blinker null: relative width of the "same duration again" kernel
BL_KEEP = 24                  # blinker null: durations remembered per polarity

# ----------------------------------------------------------------------------- decision
M_WORD = 7.0                  # nats the leader must beat the runner-up word by
M_NULL = 6.0                  # nats the leader must beat both null models by
P_COMMIT = 0.995
K_STABLE = 2                  # consecutive updates satisfying the rule

# ----------------------------------------------------------------------------- binariser
GAP_BREAK = 0.40              # a capture gap longer than this breaks the chain
H_CUSUM = 10.0                # evidence (nats) needed to accept a lamp edge
LLR_CLIP = 4.0                # per-sample evidence clip (robust to single garbage samples)
T_MIN_EXC = 0.03              # an edge needs the new state to last at least this long
CONF_T = 0.05                 # an edge is held back this long before it is believed
TAU_LVL = 0.5                 # level tracking time constant (s)
TAU_SD = 2.0
EXPL_SD, EXPL_C = 3.5, 0.22   # a sample "fits" a level within max(3.5 sd, 0.22 contrast)
CAP_ON, CAP_OFF = 3.5, 6.0    # an open run longer than this breaks the chain (light stuck / stopped)
INIT_WINDOWS = (3.0, 2.0, 1.2)
INIT_EVERY = 0.1
BUF_SPAN = 3.2
MILESTONES = {1: (1.9, 2.8), 0: (1.9, 2.8, 5.2, 7.0)}   # open-run decision checks (units)
PL_MIN = 0.09                 # re-lock: both plateaus must last this long ...
PL_N = 3                      # ... and hold this many samples
RELOCK_RATIO = (0.55, 1.8)    # ... and the step must be this fraction of the old contrast


def _word_elements(word: str) -> list[tuple[int, int]]:
    """One cycle as (polarity, units) pairs, polarity 1 = lamp on."""
    els: list[tuple[int, int]] = []
    codes = [MORSE[c] for c in word.upper()]
    for li, code in enumerate(codes):
        for si, s in enumerate(code):
            els.append((1, 1 if s == "." else 3))
            if si < len(code) - 1:
                els.append((0, 1))
        els.append((0, 4 if li < len(codes) - 1 else 10))
    return els


_AS = (0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429)


def _log_sf(z: np.ndarray) -> np.ndarray:
    """log P(N(0,1) > z), vectorised (Abramowitz-Stegun 7.1.26)."""
    x = np.abs(z) * (1.0 / math.sqrt(2.0))
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (_AS[0] + t * (_AS[1] + t * (_AS[2] + t * (_AS[3] + t * _AS[4]))))
    lh = math.log(0.5) + np.log(poly) - x * x
    return np.where(z >= 0, lh, np.log1p(-np.exp(lh)))


def _lse0(a: np.ndarray) -> np.ndarray:
    m = a.max(axis=0)
    with np.errstate(invalid="ignore"):
        r = m + np.log(np.exp(a - m).sum(axis=0))
    return np.where(np.isfinite(m), r, NEG)


def _log_g(d: float) -> float:
    d = min(max(d, G_MIN), G_MAX)
    return -math.log(d) - math.log(_LOG_G_RANGE)


def _log_gs(d: float) -> float:
    d = min(max(d, G_MIN), G_MAX * 0.999)
    return math.log(math.log(G_MAX / d) / _LOG_G_RANGE)


class _Model:
    """Static tables: every word's cyclic element sequence plus the null pseudo-word."""

    def __init__(self) -> None:
        self.words = list(WORDS)
        W = len(self.words)
        pol, units, span, prev1, prev3, starts, lpi, pos, wlen = [], [], [], [], [], [], [], [], []
        frac_n = {0: [], 1: []}
        frac_s = {0: [], 1: []}
        for w in self.words:
            els = _word_elements(w)
            n = len(els)
            base = len(pol)
            starts.append(base)
            sp = [els[(k - 2) % n][1] + els[(k - 1) % n][1] + els[k][1] for k in range(n)]
            for k, (p, c) in enumerate(els):
                pol.append(p)
                units.append(c)
                span.append(sp[k])
                prev1.append(base + (k - 1) % n)
                prev3.append(base + (k - 3) % n)
                lpi.append(-math.log(n / 2))
                pos.append(k)
                wlen.append(n)
            for p in (0, 1):
                cs = [c for (q, c) in els if q == p]
                ss = [sp[k] for k, (q, _) in enumerate(els) if q == p]
                frac_n[p].append({v: cs.count(v) / len(cs) for v in set(cs)})
                frac_s[p].append({v: ss.count(v) / len(ss) for v in set(ss)})
        E = len(pol)
        self.W, self.E, self.R = W, E, E + 2
        self.nv = np.array([1.0, 3.0, 4.0, 10.0])
        self.sv = np.array(sorted(set(span)), dtype=float)
        nidx = {v: i for i, v in enumerate(self.nv)}
        sidx = {v: i for i, v in enumerate(self.sv)}
        # null pseudo-elements: row E = lamp on, row E+1 = lamp off
        self.pol = np.array(pol + [1, 0])
        self.prev1 = np.array(prev1 + [E + 1, E])
        self.prev3 = np.array(prev3 + [E + 1, E])
        self.uidx = np.array([nidx[float(c)] for c in units] + [0, 0])
        self.sidx = np.array([sidx[float(c)] for c in span] + [0, 0])
        self.lpi = np.array(lpi + [0.0, 0.0])[:, None]
        self.starts = np.array(starts + [E])
        rm = np.zeros(self.R, dtype=int)
        for wi in range(W):
            rm[starts[wi]:(starts[wi + 1] if wi + 1 < W else E)] = wi
        rm[E:] = W
        self.row_model = rm
        self.mask = {p: (self.pol != p) for p in (0, 1)}
        # null class weights: i.i.d. classes with the vocabulary's average frequencies
        self.null_n = {}
        self.null_s = {}
        for p in (0, 1):
            wn = np.full(len(self.nv), NEG)
            for v in self.nv:
                f = sum(fr.get(int(v), 0.0) for fr in frac_n[p]) / W
                if f > 0:
                    wn[nidx[v]] = math.log(f)
            ws = np.full(len(self.sv), NEG)
            for v in self.sv:
                f = sum(fr.get(int(v), 0.0) for fr in frac_s[p]) / W
                if f > 0:
                    ws[sidx[v]] = math.log(f)
            self.null_n[p] = wn[:, None]
            self.null_s[p] = ws[:, None]
        # unit grid
        u = U_GRID[None, :]
        self.mu_n = self.nv[:, None] * u + BIAS
        self.mu_s = self.sv[:, None] * u + 3 * BIAS
        self.s2_n = (SIG_REL * self.mu_n) ** 2 + SIG_BASE ** 2
        self.s2_s = (SIG_REL * self.mu_s) ** 2 + SIG_BASE ** 2
        lu = np.log(U_GRID)
        lo, hi = math.log(U_FLAT[0]), math.log(U_FLAT[1])
        z = np.where(lu < lo, (lu - lo) / U_TAIL, np.where(lu > hi, (lu - hi) / U_TAIL, 0.0))
        lp = -0.5 * z * z
        self.lpu = lp - (lp.max() + math.log(np.exp(lp - lp.max()).sum()))
        self.lprior = np.array([math.log(PRIOR_WORDS / W)] * W + [math.log(PRIOR_NM), math.log(PRIOR_NN), math.log(PRIOR_BL)])
        # bridges: element d is the last of j consecutive elements starting at src[j-1, d]
        units_a = np.array(units, dtype=float)
        pos_a = np.array(pos)
        wl = np.array(wlen)
        base_a = np.array([starts[rm[r]] for r in range(E)])
        self.br_src = np.zeros((J_MAX, E), dtype=int)
        self.br_s1 = np.zeros((J_MAX, E))
        self.br_s2 = np.zeros((J_MAX, E))
        for j in range(1, J_MAX + 1):
            src = base_a + (pos_a - j + 1) % wl
            s1 = np.zeros(E)
            s2 = np.zeros(E)
            for i in range(j):
                e = base_a + (pos_a - i) % wl
                s1 += units_a[e]
                s2 += units_a[e] ** 2
            self.br_src[j - 1] = src
            self.br_s1[j - 1] = s1
            self.br_s2[j - 1] = s2
        self.br_j = np.arange(1, J_MAX + 1, dtype=float)
        self.br_srcpol = self.pol[self.br_src]
        self.wcount = np.bincount(rm[:E], minlength=W).astype(float)

        self.lpi_lin = np.exp(self.lpi)
        self.jump_w = self.lpi_lin.copy()
        self.jump_w[E:] = 0.0
        self.null_n_lin = {p: np.exp(self.null_n[p]) for p in (0, 1)}
        self.null_s_lin = {p: np.exp(self.null_s[p]) for p in (0, 1)}

    # -- emissions -------------------------------------------------------------
    def tables(self, d: float, var: float, survival: bool, eps: float = EPS):
        out = []
        for mu, s2 in ((self.mu_n, self.s2_n), (self.mu_s, self.s2_s)):
            s2 = s2 + var
            z = (d - mu) / np.sqrt(s2)
            if survival:
                core = _log_sf(z)
                alt = math.log(eps) + _log_gs(d)
            else:
                core = -0.5 * z * z - 0.5 * np.log(s2) - 0.5 * _LOG2PI
                alt = math.log(eps) + _log_g(d)
            out.append(np.logaddexp(math.log(1 - eps) + core, alt))
        return out

    def rows(self, tn: np.ndarray, ts: np.ndarray):
        en = tn[self.uidx]
        es = ts[self.sidx]
        E = self.E
        en[E] = _lse0(self.null_n[1] + tn)
        en[E + 1] = _lse0(self.null_n[0] + tn)
        es[E] = _lse0(self.null_s[1] + ts)
        es[E + 1] = _lse0(self.null_s[0] + ts)
        return en, es

    def lin_rows(self, d: float, var: float, survival: bool, eps: float = EPS):
        """Linear-domain emissions per row, scaled by exp(-shift): (EN, ES, shift)."""
        tn, ts = self.tables(d, var, survival, eps)
        sh = float(tn.max())
        TN = np.exp(tn - sh)
        TS = np.exp(ts - sh)
        EN = TN[self.uidx]
        ES = TS[self.sidx]
        E = self.E
        EN[E] = (self.null_n_lin[1] * TN).sum(axis=0)
        EN[E + 1] = (self.null_n_lin[0] * TN).sum(axis=0)
        ES[E] = (self.null_s_lin[1] * TS).sum(axis=0)
        ES[E + 1] = (self.null_s_lin[0] * TS).sum(axis=0)
        return EN, ES, sh

    def lin_n(self, d: float, var: float, sh: float) -> np.ndarray:
        """Single-element emissions only, linear, scaled by exp(-sh)."""
        mu, s2 = self.mu_n, self.s2_n + var
        z = (d - mu) / np.sqrt(s2)
        core = -0.5 * z * z - 0.5 * np.log(s2) - 0.5 * _LOG2PI
        tn = np.logaddexp(math.log(1 - EPS) + core, math.log(EPS) + _log_g(d))
        TN = np.exp(tn - sh)
        EN = TN[self.uidx]
        E = self.E
        EN[E] = (self.null_n_lin[1] * TN).sum(axis=0)
        EN[E + 1] = (self.null_n_lin[0] * TN).sum(axis=0)
        return EN

    def per_model(self, alpha: np.ndarray) -> np.ndarray:
        """log-sum over each model's rows -> (W+1, U)."""
        m = np.maximum.reduceat(alpha, self.starts, axis=0)
        mf = np.where(np.isfinite(m), m, 0.0)
        s = np.add.reduceat(np.exp(alpha - mf[self.row_model]), self.starts, axis=0)
        with np.errstate(divide="ignore"):
            return np.log(s) + mf

    def bridge_kernel(self, T: float, v_e: float, p: int, q: int):
        """log density that elements src..d (src of polarity p, d followed by polarity q) last T.

        Returns (E, U) over destination rows d (logsumexp over spans) and the
        same averaged over words and start positions (the random-Morse null), (U,).
        """
        u = U_GRID
        lo = T / (U_GRID[-1] * 1.25) - 1.0
        hi = T / (U_GRID[0] * 0.8) + 1.0
        js = [j for j in range(J_MAX) if self.br_s1[j].max() >= lo and self.br_s1[j].min() <= hi]
        E = self.E
        if not js:
            return np.full((E, N_U), NEG), np.full(N_U, NEG)
        ja = np.array(js)
        s1 = self.br_s1[ja][:, :, None]
        s2 = self.br_s2[ja][:, :, None]
        jj = self.br_j[ja][:, None, None]
        mean = s1 * u + jj * BIAS
        var = (SIG_REL * SIG_REL) * s2 * (u * u) + jj * (SIG_BASE * SIG_BASE) + v_e
        ll = -0.5 * (T - mean) ** 2 / var - 0.5 * np.log(var) - 0.5 * _LOG2PI
        bad = (self.br_srcpol[ja] != p) | (self.pol[None, :E] == q)
        ll[bad] = NEG
        return ll, ja


_MODEL: _Model | None = None


def _model() -> _Model:
    global _MODEL
    if _MODEL is None:
        _MODEL = _Model()
    return _MODEL


class _Blinker:
    """Null model for regular non-Morse blinkers (Simon, timers, alarms): each run repeats the
    duration of an earlier run of the same polarity (kernel), or is new (broad density)."""

    def __init__(self) -> None:
        self.d = {0: deque(maxlen=BL_KEEP), 1: deque(maxlen=BL_KEEP)}

    def logp(self, pol: int, d: float, var: float, survival: bool = False) -> float:
        xs = self.d[pol]
        n = len(xs)
        base = math.exp(_log_gs(d) if survival else _log_g(d))
        if n == 0:
            return math.log(base)
        acc = 0.0
        for x in xs:
            sd = math.sqrt((BL_REL * x) ** 2 + 1e-4 + var)
            z = (d - x) / sd
            if survival:
                acc += 0.5 * math.erfc(z * 0.7071067811865476)
            elif -8.0 < z < 8.0:
                acc += math.exp(-0.5 * z * z) / (sd * 2.5066282746310002)
        return math.log((acc + BL_ALPHA * base) / (n + BL_ALPHA))

    def add(self, pol: int, d: float) -> None:
        self.d[pol].append(d)


# ============================================================================= matcher
class _Matcher:
    """Forward algorithm over every word's cyclic alignment on a unit grid, one chain for all evidence.

    The forward variable is kept in the linear domain, scaled so its maximum is 1
    (P), with the log of the scale in ``off``: log alpha = log P + off.
    """

    def __init__(self) -> None:
        self.M = _model()
        self.P: np.ndarray | None = None
        self.off = 0.0
        self.nn = np.zeros(2)                  # log-lik of the "not Morse" and "blinker" nulls
        self.bl = _Blinker()
        self.TW: np.ndarray | None = None      # per-model sums of P, (W+1, U)
        self.pending: tuple | None = None      # chain state at an erasure
        self.hist: deque = deque(maxlen=3)     # (P, off) before each of the last runs
        self.rhist: deque = deque(maxlen=3)    # (pol, d, var) of the last runs
        self.open_run: tuple[int, float, float] | None = None
        self.disp: deque = deque(maxlen=120)   # (pol, d) or None for a break
        self.nruns = 0
        self.dirty = True
        self.answer: str | None = None
        self.streak = 0
        self.streak_word = -1
        self.last_lp: np.ndarray | None = None
        self.u_est = 0.27

    def has_chain(self) -> bool:
        return self.P is not None

    def _set_lin(self, a: np.ndarray, log_scale: float) -> None:
        m = float(a.max())
        if not m > 0.0:
            m = 1.0
        a *= 1.0 / m
        self.off = log_scale + math.log(m)
        self.P = a
        self.TW = np.add.reduceat(a, self.M.starts, axis=0)
        self.dirty = True

    def _set_log(self, a: np.ndarray, log_scale: float) -> None:
        m = float(a.max())
        if not math.isfinite(m):
            m = 0.0
        self._set_lin(np.exp(a - m), log_scale + m)

    def totals(self) -> np.ndarray:
        """log-likelihood of each model (16 words + random Morse) per unit, (W+1, U)."""
        with np.errstate(divide="ignore"):
            return np.log(self.TW) + self.off

    # -- chain control -----------------------------------------------------------
    def first_censored(self, pol: int, d: float, var: float) -> None:
        """First run of the very first segment: it started before we were watching."""
        M = self.M
        EN, ES, sh = M.lin_rows(d, var, survival=True)
        a = M.lpi_lin * (EN + P_MERGE * ES)
        a[M.mask[pol]] = 0.0
        self.nn = np.array([_log_gs(d), self.bl.logp(pol, d, var, True)])
        self._set_lin(a, sh)
        self.hist.clear()
        self.rhist.clear()
        self.disp.append((pol, d))

    def break_(self, e1: float, pol: int, d1: float, var1: float) -> None:
        """The run of polarity pol that began at edge e1 was seen for d1 s, then the view broke."""
        if self.P is None or self.pending is not None:
            return
        self.pending = (self.P, self.off, self.nn.copy(), e1, pol, d1, var1)
        self.hist.clear()
        self.rhist.clear()
        if self.disp and self.disp[-1] is not None:
            self.disp.append((pol, -d1))   # negative = censored, for display
            self.disp.append(None)
        self.dirty = True

    def start(self, e2: float, q: int, w2: float) -> None:
        """A run of polarity q starts at the clean edge e2 (after an erasure, or the first edge seen)."""
        M = self.M
        E = M.E
        if self.P is None:
            a = np.zeros((M.R, N_U))
            sel = M.pol == 1 - q
            a[sel] = np.broadcast_to(M.lpi_lin, (M.R, N_U))[sel]
            self.nn = np.zeros(2)
            self._set_lin(a, 0.0)
            return
        if self.pending is None:
            return
        P_b, off_b, nn_b, e1, p, d1, var1 = self.pending
        self.pending = None
        with np.errstate(divide="ignore"):
            alpha_b = np.log(P_b)
        T = e2 - e1
        tn, ts = M.tables(d1, var1, survival=True)
        en, _ = M.rows(tn, ts)
        c = alpha_b[M.prev1] + en
        c[M.mask[p]] = NEG
        tot = M.per_model(c)                           # (W+1, U)
        restart = tot[M.row_model] + M.lpi
        nn = nn_b + np.array([_log_gs(d1), self.bl.logp(p, d1, var1, True)])
        if T <= T_BRIDGE:
            ll, ja = M.bridge_kernel(T, var1 + w2 * w2 / 12.0, p, q)
            if len(ja):
                src = M.br_src[ja]                     # (J, E)
                x = c[src] + ll
                br = np.full((M.R, N_U), NEG)
                br[:E] = _lse0(x)
                # random-Morse null: average span density over words and starting positions
                k = _lse0(ll) + M.lpi[:E]              # (E, U)
                kw = M.per_model(np.vstack([k, np.full((2, N_U), NEG)]))[:M.W]
                kn = _lse0(kw - np.log(M.W))
                br[E + q] = _lse0(c[E:]) + kn
                a = np.logaddexp(math.log(P_BRIDGE) + br, math.log(1 - P_BRIDGE) + restart)
            else:
                a = math.log(1 - P_BRIDGE) + restart
            nn += float(np.logaddexp(math.log(P_BRIDGE) + _log_g(T), math.log(1 - P_BRIDGE)))
        else:
            a = restart
        a[M.pol != 1 - q] = NEG
        self.nn = nn
        self._set_log(a, off_b)
        self.hist.clear()
        self.rhist.clear()
        self._decide()

    # -- runs ------------------------------------------------------------------
    def run(self, pol: int, d: float, var: float, lpm: float, eps: float = EPS, pj: float = P_JUMP) -> None:
        if self.P is None or self.pending is not None:
            return
        M = self.M
        EN, ES, sh = M.lin_rows(d, var, survival=False, eps=eps)
        P = self.P
        a = P[M.prev1] * EN
        a += P[M.prev3] * (math.exp(lpm) * ES)
        # resync: jump to any position of the same word
        a += (pj * self.TW[M.row_model] * M.jump_w) * EN
        if len(self.hist) == 3 and len(self.rhist) == 3:
            (p2, d2, v2), (p1, d1, v1) = self.rhist[1], self.rhist[2]
            P3, o3 = self.hist[1]
            if p2 == pol and d1 < 0.5 * U_GRID[-1] and P3 is not None:
                ok = d1 < 0.5 * U_GRID
                EN3 = M.lin_n(d2 + d1 + d, v2 + v1 + var, sh)
                sp = P3[M.prev1] * EN3
                sp *= P_SPLIT * math.exp(o3 - self.off)
                sp[:, ~ok] = 0.0
                a += sp
        a[M.mask[pol]] = 0.0
        self.hist.append((self.P, self.off))
        self.rhist.append((pol, d, var))
        self.nn = self.nn + np.array([_log_g(d), self.bl.logp(pol, d, var)])
        self.bl.add(pol, d)
        self._set_lin(a, self.off + sh)
        self.nruns += 1
        self.disp.append((pol, d))
        self._decide()

    def _censored(self, run: tuple[int, float, float]):
        """Totals after appending a right-censored run (with the 'light stopped' option)."""
        M = self.M
        pol, d, var = run
        EN, ES, sh = M.lin_rows(d, var, survival=True)
        P = self.P
        P1 = P[M.prev1]
        a = P1 * EN
        a += P[M.prev3] * (P_MERGE * ES)
        a *= 1 - P_STOP
        a += (P_STOP * math.exp(-sh)) * P1
        a[M.mask[pol]] = 0.0
        ls = math.log(1 - P_STOP)
        nn = self.nn + np.logaddexp(ls + np.array([_log_gs(d), self.bl.logp(pol, d, var, True)]), math.log(P_STOP))
        with np.errstate(divide="ignore"):
            return np.log(np.add.reduceat(a, M.starts, axis=0)) + (self.off + sh), nn

    # -- posterior ---------------------------------------------------------------
    def _posterior(self, ll: np.ndarray, nn: np.ndarray) -> np.ndarray:
        M = self.M
        x = ll + M.lpu[None, :]
        mx = x.max(axis=1, keepdims=True)
        mx = np.where(np.isfinite(mx), mx, 0.0)
        with np.errstate(divide="ignore"):
            lm = (mx + np.log(np.exp(x - mx).sum(axis=1, keepdims=True)))[:, 0]
        lp = M.lprior + np.concatenate([lm, nn])
        lp -= lp.max()
        lp -= math.log(np.exp(lp).sum())
        return lp

    def censored_decide(self, pol: int, d: float, var: float) -> None:
        """Decision update from the open run, known to have lasted at least d so far."""
        if self.answer is not None or self.P is None or self.pending is not None:
            return
        ll, nn = self._censored((pol, d, var))
        self._decide(ll, nn)

    def _decide(self, ll: np.ndarray | None = None, nn: np.ndarray | None = None) -> None:
        if self.answer is not None:
            return
        M = self.M
        if ll is None:
            ll, nn = self.totals(), self.nn
        lp = self._posterior(ll, nn)
        self.last_lp = lp
        W = M.W
        order = np.argsort(-lp[:W])
        b, s = int(order[0]), int(order[1])
        ok = (lp[b] >= math.log(P_COMMIT) and lp[b] - lp[s] >= M_WORD
              and lp[b] - lp[W:].max() >= M_NULL)
        k = int(np.argmax(ll[b] + M.lpu))
        self.u_est = float(U_GRID[k])
        if ok and (k < 2 or k > N_U - 3):
            ok = False   # the unit must be well inside the grid
        if ok and b == self.streak_word:
            self.streak += 1
        elif ok:
            self.streak, self.streak_word = 1, b
        else:
            self.streak, self.streak_word = 0, -1
        if self.streak >= K_STABLE:
            self.answer = M.words[b]

    def display(self):
        """(log posterior over words+nulls, unit estimate) from all evidence incl. the open run."""
        M = self.M
        if self.P is None:
            return None, None
        if self.pending is None and self.open_run is not None and self.open_run[1] > 0:
            ll, nn = self._censored(self.open_run)
        else:
            ll, nn = self.totals(), self.nn
        lp = self._posterior(ll, nn)
        b = int(np.argmax(lp[:M.W + 1]))
        x = ll[b] + M.lpu
        w = np.exp(x - x.max())
        unit = float((w * U_GRID).sum() / w.sum())
        return lp, unit


# ============================================================================= binariser
INIT, TRACK, LOST = 0, 1, 2


class _Binariser:
    """Adaptive two-level tracker with CUSUM edges, anomaly breaks and fast re-lock."""

    def __init__(self, sink: _Matcher) -> None:
        self.sink = sink
        self.mode = INIT
        self.bt: deque = deque()
        self.bx: deque = deque()
        self.prev_t: float | None = None
        self.next_try = -1e18
        self.lo = self.hi = 0.0
        self.var = 1.0
        self.sd = 1.0
        self.state = 0
        self.S = 0.0
        self.exc_n = 0
        self.exc_t0 = 0.0
        self.exc_tp = 0.0
        self.unx: deque = deque(maxlen=5)
        self.pend: deque = deque()
        self.run_t = 0.0
        self.run_w = 0.0
        self.run_pol = 0
        self.run_gap = 0.0
        self.run_unx = 0
        self.run_known = False
        self.prev_clean = False
        self.pstart: tuple | None = None
        self.ms_i = 0
        self.ms_next = 1e9
        self.edges: deque = deque(maxlen=8)
        self.fdt = 0.03
        # plateaus for re-locking: [t0, t_last, sum, n]
        self.pl: list | None = None
        self.plist: deque = deque(maxlen=4)

    # -- public ------------------------------------------------------------------
    def feed(self, t: float, x: float | None) -> None:
        if x is None or (x != x):
            if self.mode == TRACK:
                if self.prev_t is not None:
                    self._break(self.prev_t, soft=True)
            else:
                self._lost_reset()
            return
        if self.prev_t is not None:
            dt = t - self.prev_t
            if dt > GAP_BREAK:
                if self.mode == TRACK:
                    self._break(self.prev_t, soft=True)
                else:
                    self._lost_reset()
            elif dt > 0:
                self.fdt += 0.05 * (min(dt, 0.1) - self.fdt)
        if self.mode == TRACK:
            if self.prev_t is None or t - self.prev_t > GAP_BREAK:
                self._track_resume(t, x)
            else:
                self._track(t, x, t - self.prev_t)
        elif self.mode == LOST:
            self._lost_feed(t, x)
        else:
            self._init_feed(t, x)
        self.prev_t = t

    def open_run(self, t: float):
        if self.mode == TRACK and self.run_known and self.pstart is None:
            end = self.pend[0][0] if self.pend else t
            return (self.run_pol, max(1e-3, end - self.run_t), self.run_w ** 2 / 12)
        return None

    # -- chain helpers ------------------------------------------------------------
    def _confirm(self, e) -> None:
        edge_t, w, new_state = e[0], e[1], e[2]
        d = max(edge_t - self.run_t, 1e-3)
        var = (self.run_w ** 2 + w * w) / 12.0
        if self.run_known:
            if self.pstart is not None:
                self.sink.start(*self.pstart)
                self.pstart = None
            lpm = math.log(P_MERGE_GAP) if self.run_gap > 0.15 else math.log(P_MERGE)
            clean = (self.run_gap <= 0.12 and self.run_unx == 0 and w <= 0.1 and self.run_w <= 0.1
                     and self.hi - self.lo >= 5.0 * self.sd)
            pj = P_JUMP_CLEAN if clean and self.prev_clean else P_JUMP
            self.sink.run(self.run_pol, d, var, lpm, EPS_CLEAN if clean else EPS, pj)
            self.prev_clean = clean
        elif not self.sink.has_chain():
            self.sink.first_censored(self.run_pol, d, var)
        else:
            self.pstart = (edge_t, new_state, w)
        self.run_t, self.run_w, self.run_pol, self.run_gap = edge_t, w, new_state, 0.0
        self.run_unx = 0
        self.run_known = True
        self._ms_reset()
        self.edges.append(edge_t)

    def _break(self, t_end: float, soft: bool) -> None:
        """The view broke at t_end: censor the open run, then wait for levels again."""
        while self.pend and self.pend[0][0] < t_end - 0.02:
            self._confirm(self.pend.popleft())
        self.pend.clear()
        if self.run_known and self.pstart is None:
            self.sink.break_(self.run_t, self.run_pol, max(1e-3, t_end - self.run_t), self.run_w ** 2 / 12)
        self.pstart = None
        self.run_known = False
        self.prev_clean = False
        self.S = 0.0
        self.exc_n = 0
        self.unx.clear()
        if soft:
            # levels are probably still valid: try to resume directly on the next sample
            self.mode = TRACK
            self.prev_t = None
        else:
            self._to_lost(t_end)

    def _to_lost(self, t: float) -> None:
        self.mode = LOST
        self._lost_reset()
        self.next_try = t + INIT_EVERY

    def _lost_reset(self) -> None:
        self.bt.clear()
        self.bx.clear()
        self.pl = None
        self.plist.clear()

    # -- tracking -------------------------------------------------------------------
    def _track_resume(self, t: float, x: float) -> None:
        C = self.hi - self.lo
        tol = max(EXPL_SD * self.sd, EXPL_C * C)
        if abs(x - self.lo) <= tol:
            self.state = 0
        elif abs(x - self.hi) <= tol:
            self.state = 1
        else:
            self._to_lost(t)
            self._lost_feed(t, x)
            return
        self.S = 0.0
        self.exc_n = 0
        self.unx.clear()
        self.pend.clear()
        self.run_t, self.run_w, self.run_pol, self.run_gap = t, 0.0, self.state, 0.0
        self.run_known = False
        self.pstart = None

    def _track(self, t: float, x: float, dt: float) -> None:
        lo, hi, sd = self.lo, self.hi, self.sd
        C = hi - lo
        tol = EXPL_SD * sd if EXPL_SD * sd > EXPL_C * C else EXPL_C * C
        dl = x - lo
        dh = x - hi
        unexplained = abs(dl) > tol and abs(dh) > tol
        if unexplained:
            self.run_unx += 1
        self.unx.append((t, unexplained))
        if unexplained:
            n = 0
            first = None
            for tt, f in self.unx:
                if f:
                    n += 1
                    if first is None:
                        first = tt
            if n >= 3:
                self._anomaly(first, t)
                return
        if dt > self.run_gap:
            self.run_gap = dt
        mid = 0.5 * (lo + hi)
        k = C / (sd * sd)
        llr = k * (x - mid) if self.state == 0 else k * (mid - x)
        if llr > LLR_CLIP:
            llr = LLR_CLIP
        elif llr < -LLR_CLIP:
            llr = -LLR_CLIP
        S = self.S + llr
        if S <= 0.0:
            S = 0.0
            self.exc_n = 0
        else:
            if self.exc_n == 0:
                self.exc_t0 = t
                self.exc_tp = self.prev_t
            self.exc_n += 1
            if S >= H_CUSUM and self.exc_n >= 2 and t - self.exc_t0 >= T_MIN_EXC:
                self.state ^= 1
                self.pend.append((0.5 * (self.exc_t0 + self.exc_tp), self.exc_t0 - self.exc_tp, self.state, t))
                S = 0.0
                self.exc_n = 0
        self.S = S
        if self.exc_n == 0 and not unexplained:
            a = dt / (TAU_LVL + dt)
            r = dh if self.state else dl
            lim = 2.5 * sd
            if r > lim:
                r = lim
            elif r < -lim:
                r = -lim
            if self.state:
                self.hi = hi + a * r
            else:
                self.lo = lo + a * r
            b = dt / (TAU_SD + dt)
            self.var += b * (r * r - self.var)
            fl = max(0.3, 0.02 * (self.hi - self.lo))
            self.sd = math.sqrt(self.var) if self.var > fl * fl else fl
            if self.hi - self.lo < 2.5 * self.sd:
                self._anomaly(t, t)
                return
        while self.pend and t - self.pend[0][3] >= CONF_T:
            self._confirm(self.pend.popleft())
        if not self.pend:
            cap = CAP_ON if self.run_pol else CAP_OFF
            if t - self.run_t > cap:
                self._break(t, soft=False)
                return
            if self.run_known and self.pstart is None:
                d = (self.exc_tp if self.exc_n > 0 else t) - self.run_t
                if d >= self.ms_next:
                    self._ms_advance()
                    self.sink.censored_decide(self.run_pol, d, self.run_w ** 2 / 12)

    def _ms_reset(self) -> None:
        self.ms_i = 0
        self.ms_next = MILESTONES[self.run_pol][0] * self.sink.u_est

    def _ms_advance(self) -> None:
        ms = MILESTONES[self.run_pol]
        self.ms_i += 1
        self.ms_next = ms[self.ms_i] * self.sink.u_est if self.ms_i < len(ms) else 1e9

    def _anomaly(self, t_a: float, t: float) -> None:
        self._break(t_a, soft=False)

    # -- re-lock after an anomaly ---------------------------------------------------------
    def _lost_feed(self, t: float, x: float) -> None:
        C = self.hi - self.lo
        tol = max(3.0 * self.sd, 0.15 * C)
        pl = self.pl
        if pl is None or abs(x - pl[2] / pl[3]) > tol:
            if pl is not None:
                self.plist.append(pl)
            pl = self.pl = [t, t, x, 1]
        else:
            pl[1] = t
            pl[2] += x
            pl[3] += 1
        if pl[3] >= PL_N and pl[1] - pl[0] >= PL_MIN:
            prev = None
            trans = 0
            for q in reversed(self.plist):
                if q[3] >= PL_N and q[1] - q[0] >= PL_MIN:
                    prev = q
                    break
                trans += q[3]
                if trans > 2:
                    break
            if prev is not None and pl[0] - prev[1] <= max(0.07, 3.0 * self.fdt):
                m1 = prev[2] / prev[3]
                m2 = pl[2] / pl[3]
                step = m2 - m1
                if RELOCK_RATIO[0] * C <= abs(step) <= RELOCK_RATIO[1] * C:
                    self._relock(prev, pl, m1, m2)
                    return
        # fallback: a fresh window search over what was seen since the break
        self._init_feed(t, x)

    def _relock(self, prev, pl, m1: float, m2: float) -> None:
        up = m2 > m1
        self.lo, self.hi = (m1, m2) if up else (m2, m1)
        self.var = self.sd * self.sd
        self.mode = TRACK
        self.state = 1 if up else 0
        self.S = 0.0
        self.exc_n = 0
        self.unx.clear()
        self.pend.clear()
        w = max(pl[0] - prev[1], 1e-3)
        e2 = 0.5 * (pl[0] + prev[1])
        self.run_t, self.run_w, self.run_pol, self.run_gap = e2, w, self.state, 0.0
        self.run_unx = 1   # the run after a re-lock is not trusted fully
        self.run_known = True
        self._ms_reset()
        self.pstart = (e2, self.state, w)
        self.edges.append(e2)
        self._lost_reset()

    # -- level search ------------------------------------------------------------------
    def _init_feed(self, t: float, x: float) -> None:
        self.bt.append(t)
        self.bx.append(x)
        while self.bt and t - self.bt[0] > BUF_SPAN:
            self.bt.popleft()
            self.bx.popleft()
        if t < self.next_try or t - self.bt[0] < INIT_WINDOWS[-1]:
            return
        self.next_try = t + INIT_EVERY
        ts = np.fromiter(self.bt, float, len(self.bt))
        xs = np.fromiter(self.bx, float, len(self.bx))
        for win in INIT_WINDOWS:
            if ts[-1] - ts[0] < win * 0.95 and win != INIT_WINDOWS[-1]:
                continue
            k = int(np.searchsorted(ts, ts[-1] - win))
            res = _find_levels(ts[k:], xs[k:])
            if res is not None:
                self._start_tracking(ts[k:], xs[k:], *res)
                return

    def _start_tracking(self, ts: np.ndarray, xs: np.ndarray, lo: float, hi: float, sd: float) -> None:
        self.mode = TRACK
        self.lo, self.hi, self.sd, self.var = lo, hi, sd, sd * sd
        self._lost_reset()
        self.unx.clear()
        self.pend.clear()
        t0 = float(ts[0])
        self.state = 1 if xs[0] > 0.5 * (lo + hi) else 0
        self.S = 0.0
        self.exc_n = 0
        self.run_t, self.run_w, self.run_pol, self.run_gap = t0, 0.0, self.state, 0.0
        self.run_known = False
        self.pstart = None
        self.prev_t = t0
        for i in range(1, len(ts)):
            t = float(ts[i])
            self._track(t, float(xs[i]), t - self.prev_t)
            self.prev_t = t
            if self.mode != TRACK:
                # lost the levels again during replay: keep the rest for the next search
                for j in range(i + 1, len(ts)):
                    self.bt.append(float(ts[j]))
                    self.bx.append(float(xs[j]))
                return


def _find_levels(ts: np.ndarray, xs: np.ndarray):
    """Two clean levels in a window? -> (lo, hi, noise_sd) or None."""
    n = len(xs)
    if n < 8:
        return None
    d = np.diff(xs)
    sd = 1.4826 * float(np.median(np.abs(d - np.median(d)))) / math.sqrt(2.0)
    sd = max(sd, 0.3)
    srt = np.sort(xs)
    c = np.cumsum(srt)
    i = np.arange(1, n)
    mu0 = c[:-1] / i
    mu1 = (c[-1] - c[:-1]) / (n - i)
    bcv = i * (n - i) * (mu1 - mu0) ** 2
    k = int(np.argmax(bcv)) + 1
    thr = 0.5 * (srt[k - 1] + srt[k])
    low = xs[xs < thr]
    high = xs[xs >= thr]
    if len(low) < 3 or len(high) < 3:
        return None
    lo = float(np.median(low))
    hi = float(np.median(high))
    C = hi - lo
    if C < 3.5 * sd or C < 1.0:
        return None
    mid = 0.5 * (lo + hi)
    if float(np.mean(np.abs(xs - mid) < 0.25 * C)) > 0.12:
        return None
    # within-class spread must look like the sample noise
    for cls, m in ((low, lo), (high, hi)):
        spread = 1.4826 * float(np.median(np.abs(cls - m)))
        if spread > max(2.5 * sd, 0.12 * C):
            return None
    # temporal coherence: hysteresis classification gives few, not-too-short runs
    b = np.empty(n, dtype=bool)
    s = xs[0] > mid
    h = 0.2 * C
    for j in range(n):
        v = xs[j]
        if s and v < mid - h:
            s = False
        elif not s and v > mid + h:
            s = True
        b[j] = s
    ch = np.flatnonzero(b[1:] != b[:-1]) + 1
    if len(ch) == 0:
        return None
    bounds = np.concatenate([[ts[0]], ts[ch], [ts[-1]]])
    durs = np.diff(bounds)
    inner = durs[1:-1]
    if len(inner) and (np.sum(inner < 0.06) > 1):
        return None
    on_time = float(np.sum(np.where(b[:-1], np.diff(ts), 0.0)))
    off_time = float(ts[-1] - ts[0]) - on_time
    if on_time < 0.1 or off_time < 0.1:
        return None
    return lo, hi, sd


# ============================================================================= decoder
class Decoder:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._m = _Matcher()
        self._b = _Binariser(self._m)
        self._t = 0.0
        self._status: Status | None = None
        self._status_t = -1e18
        self._sig = False

    def feed(self, t: float, level: float | None) -> None:
        self._t = t
        self._b.feed(t, level)

    def _signal(self) -> bool:
        e = self._b.edges
        if len(e) < 3:
            return False
        return self._t - e[-1] < 3.6 and self._t - e[-3] < 8.0

    def status(self) -> Status:
        sig = self._signal()
        m = self._m
        age = self._t - self._status_t
        if self._status is not None and sig == self._sig and (age < 0.2 or (age < 0.5 and not m.dirty)):
            if self._status.answer != m.answer:
                self._status.answer = m.answer
            return self._status
        m.dirty = False
        self._sig = sig
        self._status_t = self._t
        m.open_run = self._b.open_run(self._t)
        lp, unit = m.display()
        M = m.M
        W = M.W
        if lp is None:
            self._status = Status(answer=None, signal=sig)
            return self._status
        p = np.exp(lp)
        order = np.argsort(-lp[:W])
        ranking = [(M.words[i], float(p[i])) for i in order[:5]]
        if m.answer is not None:
            conf = 1.0 if m.answer == ranking[0][0] else float(p[M.words.index(m.answer)])
            if m.answer != ranking[0][0]:
                ranking = [(m.answer, conf)] + [r for r in ranking if r[0] != m.answer][:4]
        else:
            conf = ranking[0][1]
        morse, letters = _render(m.disp, unit or 0.27)
        self._status = Status(answer=m.answer, confidence=conf, ranking=ranking, morse=morse,
                              letters=letters, signal=sig, unit=unit)
        return self._status


def _render(disp, u: float) -> tuple[str, str]:
    """Readable Morse and letters from the recent runs."""
    morse: list[str] = []
    cur = ""
    letters: list[str] = []

    def flush_letter():
        nonlocal cur
        if cur:
            letters.append(REVERSE_MORSE.get(cur, "?"))
            morse.append(cur)
        cur = ""

    for item in disp:
        if item is None:
            flush_letter()
            if morse and morse[-1] != "?":
                morse.append("?")
                letters.append("?")
            continue
        pol, d = item
        if d < 0:
            continue
        if pol:
            cur += "." if d < 2.0 * u else "-"
        elif d < 2.5 * u:
            pass
        elif d < 7.0 * u:
            flush_letter()
        else:
            flush_letter()
            if morse and morse[-1] != "/":
                morse.append("/")
                letters.append("/")
    flush_letter()
    out_m = " ".join(morse)
    out_l = "".join(letters)
    return out_m[-80:], out_l[-24:]
