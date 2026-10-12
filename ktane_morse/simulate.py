"""Synthetic brightness traces of the Morse light, for tests and benchmarks.

A trace is what the capture loop hands to a decoder: a list of
(timestamp_seconds, level) pairs, where level is the brightness of the watched
region, or None for a frame the capture layer flagged as unusable.

The model follows the vanilla game (read from MorseCodeComponent and its
prefab): DotLength 0.27 s, dash = 3 units, gap inside a letter = 1 unit,
gap between letters = 4 units, gap before the word repeats = 10 units. Each
wait ends on the first rendered frame at or after its deadline, so every
element runs up to one game frame long. The lit bulb uses an unlit shader
(its brightness ignores room lighting); the unlit bulb and the module around
it are lit by the room, so lights-out / emergency-light events change the
"off" level and the background but not the "on" contribution.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from .morse import MORSE, WORDS


@dataclass(frozen=True)
class Timing:
    """Durations in seconds of each part of the signal."""

    dot: float = 0.27
    dash: float = 0.81
    intra_gap: float = 0.27   # between dots/dashes of one letter
    letter_gap: float = 1.08  # between letters
    word_gap: float = 2.70    # before the word repeats

    def scaled(self, k: float) -> "Timing":
        return Timing(self.dot * k, self.dash * k, self.intra_gap * k, self.letter_gap * k, self.word_gap * k)


GAME_TIMING = Timing()


def word_schedule(word: str, timing: Timing = GAME_TIMING) -> list[tuple[bool, float]]:
    """One cycle of the looping signal as (light_on, duration) pairs, ending with the word gap."""
    out: list[tuple[bool, float]] = []
    letters = [MORSE[c] for c in word.upper()]
    for li, code in enumerate(letters):
        for si, sym in enumerate(code):
            out.append((True, timing.dot if sym == "." else timing.dash))
            if si < len(code) - 1:
                out.append((False, timing.intra_gap))
        out.append((False, timing.letter_gap if li < len(letters) - 1 else timing.word_gap))
    return out


def cycle_length(word: str, timing: Timing = GAME_TIMING) -> float:
    return sum(d for _, d in word_schedule(word, timing))


# --------------------------------------------------------------------------- #
# On/off sources. Each returns absolute (start, end, on) segments in "game
# time" covering [0, duration].

def morse_segments(word: str, duration: float, rng: random.Random, timing: Timing = GAME_TIMING,
                   game_fps: float = 60.0, jitter: float = 0.003, phase: float | None = None) -> list[tuple[float, float, bool]]:
    sched = word_schedule(word, timing)
    cyc = sum(d for _, d in sched)
    phase = rng.uniform(0, cyc) if phase is None else phase
    t = -phase
    segs: list[tuple[float, float, bool]] = []
    frame = 1.0 / game_fps
    while t < duration + 1:
        for on, d in sched:
            d = d + rng.uniform(0, frame) + rng.gauss(0, jitter)
            segs.append((t, t + max(0.01, d), on))
            t += max(0.01, d)
    return segs


def pattern_segments(pattern: list[tuple[bool, float]], duration: float, rng: random.Random,
                     jitter: float = 0.01, loop: bool = True) -> list[tuple[float, float, bool]]:
    """Generic looping on/off pattern with random phase (for distractors)."""
    cyc = sum(d for _, d in pattern)
    t = -rng.uniform(0, cyc)
    segs = []
    while t < duration + 1:
        for on, d in pattern:
            d = max(0.01, d * (1 + rng.gauss(0, jitter)))
            segs.append((t, t + d, on))
            t += d
        if not loop:
            segs.append((t, duration + 2, False))
            break
    return segs


def simon_segments(duration: float, rng: random.Random) -> list[tuple[float, float, bool]]:
    """Yellow button of Simon Says: sequences of flashes every 0.65 s, repeated after 4.5 s.

    The sequence grows by one colour after each stage the player passes.
    """
    colours = [rng.randrange(4) for _ in range(5)]
    if 0 not in colours:
        colours[rng.randrange(5)] = 0  # make sure yellow (0) flashes at least once
    flash = rng.uniform(0.25, 0.5)
    segs = []
    t = rng.uniform(-3, 0)
    stage = rng.randint(1, 3)
    while t < duration + 1:
        for c in colours[:stage]:
            if c == 0:
                segs.append((t, t + flash, True))
                segs.append((t + flash, t + 0.65, False))
            else:
                segs.append((t, t + 0.65, False))
            t += 0.65
        segs.append((t, t + 4.5, False))
        t += 4.5
        if rng.random() < 0.3 and stage < 5:
            stage += 1
    return segs


def random_morse_segments(duration: float, rng: random.Random, timing: Timing = GAME_TIMING) -> list[tuple[float, float, bool]]:
    """Morse-timed gibberish (random letters, no repetition) -- a worst-case null."""
    segs = []
    t = 0.0
    letters = list(MORSE.values())
    while t < duration + 1:
        n = rng.randint(3, 7)
        word = [rng.choice(letters) for _ in range(n)]
        for li, code in enumerate(word):
            for si, sym in enumerate(code):
                d = timing.dot if sym == "." else timing.dash
                segs.append((t, t + d, True)); t += d
                if si < len(code) - 1:
                    segs.append((t, t + timing.intra_gap, False)); t += timing.intra_gap
            g = timing.letter_gap if li < n - 1 else timing.word_gap
            segs.append((t, t + g, False)); t += g
    return segs


# --------------------------------------------------------------------------- #

@dataclass
class Event:
    start: float
    length: float
    kind: str
    # kinds:
    #   "lights_out" room light flickers, goes dark, flickers back (length = dark part)
    #   "emergency"  room light pulses (1 s on / 0.75-2 s off) for `length` seconds
    #   "motion"     camera/bomb moves: garbage levels, then the region frames a slightly different view
    #   "motion_flagged" same, but the capture layer reports None while moving
    #   "dropout"    capture reports None (e.g. window lost focus)
    #   "pause"      game paused: pause screen shown (garbage), Morse timing freezes
    #   "solved"     from `start` on the light is off forever (module solved / bomb over)


@dataclass
class TraceConfig:
    duration: float = 45.0
    fps: float = 60.0               # capture rate
    fps_jitter: float = 0.15        # relative std-dev of the frame interval
    drop_prob: float = 0.01         # chance a frame is silently missing
    hitch_prob: float = 0.003       # chance per frame of a 50-250 ms stall
    noise: float = 1.5              # gaussian noise std-dev on the level
    bg: float = 50.0                # static background contribution (scaled by room light)
    lamp_off: float = 25.0          # unlit bulb contribution (scaled by room light)
    lamp_on: float = 140.0          # lit bulb contribution (not affected by room light)
    events: list[Event] = field(default_factory=list)


def _room_light(t: float, events: list[Event], rng_seed: int) -> float:
    """Room light factor at time t (1 = normal)."""
    L = 1.0
    for ev in events:
        if ev.kind == "lights_out":
            a, b = ev.start, ev.start + 1.57
            c, d = b + ev.length, b + ev.length + 0.61
            if a <= t < b or c <= t < d:
                # flicker: pseudo-random but deterministic per 60 ms slot
                slot = int(t / 0.06)
                L = min(L, 0.2 + 0.8 * random.Random(rng_seed * 7919 + slot).random())
            elif b <= t < c:
                L = min(L, 0.12)
        elif ev.kind == "emergency" and ev.start <= t < ev.start + ev.length:
            off_duty = 2.0 - 1.25 * min(1.0, (t - ev.start) / max(1.0, ev.length))
            period = 1.0 + off_duty
            if ((t - ev.start) % period) < 1.0:
                L = min(L, 0.55)  # red tint: green/brightness of lit-by-room surfaces drops
    return L


def make_trace(segs: list[tuple[float, float, bool]], cfg: TraceConfig, seed: int = 0) -> list[tuple[float, float | None]]:
    """Sample an on/off segment list the way the capture loop would."""
    rng = random.Random(seed)
    # game-time warp for pauses: wall time -> game time
    pauses = sorted((e.start, e.length) for e in cfg.events if e.kind == "pause")

    def game_time(t: float) -> float:
        g = t
        for s, l in pauses:
            if t > s:
                g -= min(l, t - s)
        return g

    solved_at = min((e.start for e in cfg.events if e.kind == "solved"), default=math.inf)
    samples: list[tuple[float, float | None]] = []
    t = 0.0
    si = 0
    bg, lamp_off, lamp_on = cfg.bg, cfg.lamp_off, cfg.lamp_on
    moved: set[int] = set()
    while t < cfg.duration:
        g = game_time(t)
        while si < len(segs) - 1 and segs[si][1] <= g:
            si += 1
        on = segs[si][2] and segs[si][0] <= g and t < solved_at
        L = _room_light(t, cfg.events, seed)

        value: float | None = None
        garbage = False
        for i, ev in enumerate(cfg.events):
            inside = ev.start <= t < ev.start + ev.length
            if ev.kind in ("motion", "motion_flagged"):
                if inside:
                    garbage = True
                    if ev.kind == "motion_flagged":
                        value = None
                        garbage = False
                        break
                elif t >= ev.start + ev.length and i not in moved:
                    moved.add(i)  # after the move the region frames things a bit differently
                    bg *= rng.uniform(0.7, 1.3)
                    lamp_off *= rng.uniform(0.8, 1.2)
                    lamp_on *= rng.uniform(0.8, 1.1)
            elif ev.kind == "pause" and inside:
                garbage = True
            elif ev.kind == "dropout" and inside:
                value = None
                break
        else:
            if garbage:
                value = rng.uniform(0, 255)
            else:
                value = bg * L + (lamp_on if on else lamp_off * L) + rng.gauss(0, cfg.noise)
                value = min(255.0, max(0.0, value))
        if garbage and value is not None:
            value = rng.uniform(0, 255)

        if rng.random() >= cfg.drop_prob:
            samples.append((t, value))
        dt = (1 / cfg.fps) * max(0.2, 1 + rng.gauss(0, cfg.fps_jitter))
        if rng.random() < cfg.hitch_prob:
            dt += rng.uniform(0.05, 0.25)
        t += dt
    return samples


# --------------------------------------------------------------------------- #
# Scenario suites

@dataclass
class Scenario:
    name: str
    word: str | None              # the word actually flashed (None = not a Morse module at all)
    trace: list[tuple[float, float | None]]
    notes: str = ""
    must_answer: bool = True      # False: staying silent is fine, but any answer given must equal `word`
    soft: bool = False            # a stress test that is reported separately from the hard requirements


def _base_cfg(rng: random.Random, hard: bool) -> TraceConfig:
    cfg = TraceConfig(
        fps=rng.choice([30, 60, 60, 120, 240, 360] if not hard else [20, 25, 30, 60, 144]),
        fps_jitter=rng.uniform(0.03, 0.2 if not hard else 0.45),
        drop_prob=rng.uniform(0, 0.02 if not hard else 0.08),
        hitch_prob=rng.uniform(0, 0.004 if not hard else 0.015),
        noise=rng.uniform(0.3, 2.5 if not hard else 6.0),
        bg=rng.uniform(10, 120),
        lamp_off=rng.uniform(5, 50),
    )
    contrast = rng.uniform(40, 200) if not hard else rng.uniform(12, 60)
    cfg.lamp_on = cfg.lamp_off + contrast
    return cfg


def positive_scenario(rng: random.Random, seed: int, hard: bool = False, word: str | None = None) -> Scenario:
    word = word or rng.choice(list(WORDS))
    cfg = _base_cfg(rng, hard)
    scale = rng.uniform(0.97, 1.05) if not hard else rng.uniform(0.85, 1.2)
    game_fps = rng.choice([60, 60, 60, 144, 240])
    notes = [f"fps={cfg.fps}", f"scale={scale:.2f}", f"game_fps={game_fps}"]
    must = True
    if hard:
        kinds = ["lights_out", "emergency", "motion", "motion_flagged", "dropout", "pause"]
        for _ in range(rng.randint(1, 2)):
            k = rng.choice(kinds)
            start = rng.uniform(2, 30)
            length = {"lights_out": rng.uniform(5, 10), "emergency": rng.uniform(5, 20)}.get(k, rng.uniform(0.3, 2.5))
            cfg.events.append(Event(start, length, k))
            notes.append(f"{k}@{start:.1f}+{length:.1f}")
        if rng.random() < 0.15:
            s = rng.uniform(8, 30)
            cfg.events.append(Event(s, 0, "solved"))
            must = False
            notes.append(f"solved@{s:.1f}")
    segs = morse_segments(word, cfg.duration + 30, rng, GAME_TIMING.scaled(scale), game_fps=game_fps)
    return Scenario(f"{'hard' if hard else 'normal'}:{word}", word, make_trace(segs, cfg, seed), " ".join(notes), must_answer=must)


def negative_scenario(rng: random.Random, seed: int, kind: str | None = None) -> Scenario:
    kinds = ["static", "solved", "simon", "timer", "panic", "alarm", "emergency_only", "lights_out_only", "random_morse", "noise"]
    kind = kind or rng.choice(kinds)
    cfg = _base_cfg(rng, hard=rng.random() < 0.3)
    cfg.duration = 60.0
    D = cfg.duration + 5
    word = None
    if kind == "static":
        segs = [(0, D, False)]
    elif kind == "solved":
        word = rng.choice(list(WORDS))
        segs = morse_segments(word, D, rng)
        # light dies after a short fragment: decoder may stay silent, but must not guess wrong
        cut = rng.uniform(1.0, 3.5)
        cfg.events.append(Event(cut, 0, "solved"))
        kind = f"solved_after_{cut:.1f}s"
    elif kind == "simon":
        segs = simon_segments(D, rng)
    elif kind == "timer":
        pat = rng.choice([[(True, 0.5), (False, 0.5)], [(True, 1.0), (False, 1.0)], [(True, 2.0), (False, 1.0)], [(True, 1.0), (False, 2.0)]])
        segs = pattern_segments(pat, D, rng, jitter=0.002)
    elif kind == "panic":
        segs = pattern_segments([(True, 0.1), (False, 0.1)], D, rng, jitter=0.02)
    elif kind == "alarm":
        segs = pattern_segments([(True, 0.25), (False, 0.25)], D, rng, jitter=0.01)
    elif kind == "emergency_only":
        segs = [(0, D, rng.random() < 0.5)]
        cfg.events.append(Event(0, D, "emergency"))
    elif kind == "lights_out_only":
        segs = [(0, D, rng.random() < 0.5)]
        cfg.events.append(Event(rng.uniform(0, 20), rng.uniform(5, 10), "lights_out"))
        cfg.events.append(Event(rng.uniform(25, 45), rng.uniform(5, 10), "lights_out"))
    elif kind == "random_morse":
        segs = random_morse_segments(D, rng)
    elif kind == "noise":
        segs = [(0, D, False)]
        cfg.noise = rng.uniform(5, 25)
    else:
        raise ValueError(kind)
    return Scenario(f"neg:{kind}", word, make_trace(segs, cfg, seed), f"fps={cfg.fps}",
                    must_answer=False, soft=kind == "random_morse")


NEGATIVE_KINDS = ["static", "solved", "simon", "timer", "panic", "alarm", "emergency_only",
                  "lights_out_only", "random_morse", "noise"]


def public_scenario(i: int, n_pos: int = 160, n_hard: int = 160, n_neg: int = 120, seed: int = 1) -> Scenario:
    """Scenario number i of the public suite (deterministic; lets workers rebuild traces in parallel)."""
    words = list(WORDS)
    if i < n_pos:
        s = seed * 100_000 + i
        return positive_scenario(random.Random(s), s, hard=False, word=words[i % len(words)])
    i -= n_pos
    if i < n_hard:
        s = seed * 100_000 + 20_000 + i
        return positive_scenario(random.Random(s), s, hard=True, word=words[i % len(words)])
    i -= n_hard
    if i < n_neg:
        s = seed * 100_000 + 40_000 + i
        return negative_scenario(random.Random(s), s, kind=NEGATIVE_KINDS[i % len(NEGATIVE_KINDS)])
    raise IndexError(i)


def public_suite(n_pos: int = 160, n_hard: int = 160, n_neg: int = 120, seed: int = 1) -> list[Scenario]:
    """Deterministic benchmark suite. Every word appears equally often among positives."""
    return [public_scenario(i, n_pos, n_hard, n_neg, seed) for i in range(n_pos + n_hard + n_neg)]
