"""Tests for ktane_morse.locator.LampLocator on synthetic screens from tools/fake_screen.py.

Run from the project root:  .venv\\Scripts\\python.exe -m pytest tests -q
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # project root (plain `pytest` works too)

from ktane_morse.features import lamp_level
from ktane_morse.locator import LampLocator, phase_shift
from tools.fake_screen import ALL_DISTRACTORS, Event, FakeScreen, SceneSpec

FPS = 30.0
STATIC_OBJECTS = ("timer", "highlight", "highlight1", "status_led", "solved_led", "yellow_wire")


@dataclass
class Run:
    scr: FakeScreen
    loc: LampLocator
    times: list = field(default_factory=list)
    infos: list = field(default_factory=list)
    top3: list = field(default_factory=list)       # candidates(3) after every frame
    levels: list = field(default_factory=list)     # FrameInfo.levels after every frame

    def lamp_rank(self, i: int) -> int | None:
        """Rank (0-based) of the first top-3 candidate whose bbox contains the true lamp centre."""
        lc = self.scr.lamp_center(self.times[i])
        for r, c in enumerate(self.top3[i]):
            if c.contains(*lc):
                return r
        return None

    def first_found(self, after: float = 0.0) -> float | None:
        """First time >= ``after`` at which the lamp is a top-3 candidate."""
        for i, t in enumerate(self.times):
            if t >= after and self.lamp_rank(i) is not None:
                return t
        return None

    def found_fraction(self, t0: float, t1: float) -> float:
        idx = [i for i, t in enumerate(self.times) if t0 <= t < t1]
        return sum(self.lamp_rank(i) is not None for i in idx) / max(1, len(idx))

    def lamp_candidate(self):
        """The current top-3 candidate containing the lamp (or None)."""
        lc = self.scr.lamp_center(self.times[-1])
        return next((c for c in self.top3[-1] if c.contains(*lc)), None)


def run_scene(spec: SceneSpec, duration: float, fps: float = FPS, seed: int = 0, **loc_kw) -> Run:
    scr = FakeScreen(spec)
    loc = LampLocator(**loc_kw)
    run = Run(scr, loc)
    # capture-like timing: jittered intervals, a few dropped frames and stalls
    for t, frame in scr.frames(fps=fps, duration=duration, jitter=0.15, drop=0.02, hitch=0.01, seed=seed):
        info = loc.update(t, frame)
        run.times.append(t)
        run.infos.append(info)
        run.top3.append(loc.candidates(3))
        run.levels.append(info.levels)
    return run


# --------------------------------------------------------------------------- #
# finding the lamp

@pytest.mark.parametrize("word,seed", [("shell", 0), ("bistro", 1), ("leaks", 2)])
def test_plain_scene(word, seed):
    spec = SceneSpec(width=1280, height=720, word=word, seed=seed, phase=None)
    run = run_scene(spec, 9.0, seed=seed)
    t = run.first_found()
    assert t is not None and t <= 6.0, f"lamp not in top 3 within 6 s (found at {t})"
    assert run.found_fraction(6.0, 9.0) == 1.0
    assert run.lamp_rank(len(run.times) - 1) == 0          # alone in the scene: it is the best candidate


@pytest.mark.parametrize("word,seed", [("sting", 3), ("break", 4), ("halls", 5), ("vector", 6)])
def test_scene_with_all_distractors(word, seed):
    spec = SceneSpec(width=1280, height=720, word=word, seed=seed, phase=None,
                     distractors=ALL_DISTRACTORS + ("panic",))
    run = run_scene(spec, 10.0, seed=seed)
    t = run.first_found()
    assert t is not None and t <= 6.0, f"lamp not in top 3 within 6 s (found at {t})"
    assert run.found_fraction(6.0, 10.0) >= 0.98
    # with ~10 s of blinking the Morse timing beats Simon Says' yellow button
    assert run.lamp_rank(len(run.times) - 1) == 0


@pytest.mark.parametrize("width,height,scale", [(2560, 1440, 1.0), (1920, 1080, 0.75)])
def test_small_unzoomed_lamp(width, height, scale):
    spec = SceneSpec(width=width, height=height, word="flick", seed=7, phase=None, module_scale=scale,
                     distractors=ALL_DISTRACTORS)
    run = run_scene(spec, 8.0, seed=7)
    assert run.scr.lamp_size() <= 15.0                       # the whole bomb is in view
    t = run.first_found()
    assert t is not None and t <= 6.0, f"small lamp not in top 3 within 6 s (found at {t})"
    c = run.lamp_candidate()
    assert c is not None
    x, y, w, h = c.bbox                                      # full-resolution frame pixels
    assert 0 <= x and 0 <= y and x + w <= width and y + h <= height
    assert w <= 12 * run.scr.lamp_size() and h <= 12 * run.scr.lamp_size()


def test_zoomed_lamp():
    spec = SceneSpec(width=1280, height=720, word="steak", seed=8, phase=None, zoomed=True,
                     distractors=ALL_DISTRACTORS)
    run = run_scene(spec, 8.0, seed=8)
    assert run.scr.lamp_size() >= 40
    t = run.first_found()
    assert t is not None and t <= 6.0
    rx, ry, rw, rh = run.lamp_candidate().region()
    assert rx >= 0 and ry >= 0 and rx + rw <= 1280 and ry + rh <= 720


def test_lamp_that_starts_late():
    """Blinking starts at 4 s (e.g. the locator was started first): found a few seconds later."""
    spec = SceneSpec(width=1280, height=720, word="sting", seed=0, phase=0.0, morse_start=4.0,
                     distractors=ALL_DISTRACTORS)
    run = run_scene(spec, 11.0, seed=1)
    t = run.first_found()
    assert t is not None and t - 4.0 <= 6.0
    assert run.found_fraction(t + 0.5, 11.0) >= 0.98


# --------------------------------------------------------------------------- #
# camera motion and full-screen changes

def test_refound_after_camera_pan():
    pan = Event("pan", 7.0, 1.0, dx=60, dy=-35)
    spec = SceneSpec(width=1280, height=720, word="shell", seed=9, phase=None,
                     distractors=ALL_DISTRACTORS, events=[pan])
    run = run_scene(spec, 14.0, seed=9)
    i_before = max(i for i, t in enumerate(run.times) if t < pan.start)
    lc = run.scr.lamp_center(run.times[i_before])
    before = next(c for c in run.top3[i_before] if c.contains(*lc))

    # the motion is reported (the app feeds those frames to decoders as None)
    during = [inf for inf, t in zip(run.infos, run.times) if pan.start + 0.1 <= t < pan.start + pan.length]
    assert during and sum(not inf.ok for inf in during) >= 0.9 * len(during)
    for levels, t in zip(run.levels, run.times):
        if pan.start + 0.1 <= t < pan.start + pan.length:
            assert all(v is None for v in levels.values())

    end = pan.start + pan.length
    t = run.first_found(after=end)
    assert t is not None and t - end <= 6.0, f"lamp not re-found within 6 s of the pan (at {t})"
    assert run.found_fraction(end + 1.0, 14.0) >= 0.98
    # a pure translation is recognised: same id, box moved with the picture, history kept
    shifts = [inf.shift for inf in run.infos if inf.shift]
    assert len(shifts) == 1
    dx, dy = shifts[0]
    assert abs(dx + 60) <= 2 * run.loc.factor and abs(dy - 35) <= 2 * run.loc.factor
    after = run.lamp_candidate()
    assert after.id == before.id
    hist = run.loc.history(after.id)
    assert hist[0][0] < pan.start                           # back-fill spans the pan


def test_refound_after_zoom_into_module():
    zoom = Event("zoom", 5.0, 1.0, dx=70, dy=-60, scale=2.5)
    spec = SceneSpec(width=1280, height=720, word="trick", seed=10, phase=None,
                     distractors=ALL_DISTRACTORS, events=[zoom])
    run = run_scene(spec, 13.0, seed=10)
    end = zoom.start + zoom.length
    assert any(inf.view_changed for inf, t in zip(run.infos, run.times) if t >= zoom.start)
    t = run.first_found(after=end + 0.5)
    assert t is not None and t - end <= 6.0
    assert run.found_fraction(t + 0.5, 13.0) >= 0.98


def test_pause_menu_and_capture_failure_are_flagged():
    pause = Event("pause", 5.0, 2.0)
    spec = SceneSpec(width=1280, height=720, word="halls", seed=11, phase=None,
                     distractors=ALL_DISTRACTORS, events=[pause])
    scr = FakeScreen(spec)
    loc = LampLocator()
    flagged_at_switch = []
    prev_paused = False
    for t, frame in scr.frames(fps=FPS, duration=12.0, jitter=0.1, seed=11):
        info = loc.update(t, frame)
        paused = pause.start <= t < pause.start + pause.length
        if paused != prev_paused:
            flagged_at_switch.append(not info.ok and info.moving)
        prev_paused = paused
        if 4.0 < t < 4.2:
            bad = loc.update(t + 0.001, None)                  # capture failure
            assert not bad.ok and bad.levels and all(v is None for v in bad.levels.values())
    assert flagged_at_switch == [True, True]                   # entering and leaving the menu
    lc = scr.lamp_center(t)
    assert any(c.contains(*lc) for c in loc.candidates(3))


# --------------------------------------------------------------------------- #
# room-light events

def test_emergency_lights():
    spec = SceneSpec(width=1280, height=720, word="boxes", seed=12, phase=None,
                     distractors=ALL_DISTRACTORS, events=[Event("emergency", 0.0, 60.0)])
    run = run_scene(spec, 10.0, seed=12)
    assert sum(inf.ok for inf in run.infos) >= 0.95 * len(run.infos)   # light pulses are not motion
    gains = [inf.gain for inf in run.infos if np.isfinite(inf.gain)]
    assert min(gains) < 0.8                                   # the pulses are seen as room light
    t = run.first_found()
    assert t is not None and t <= 6.0
    assert run.found_fraction(6.0, 10.0) >= 0.98


def test_lights_out():
    spec = SceneSpec(width=1280, height=720, word="beats", seed=13, phase=None,
                     distractors=ALL_DISTRACTORS, events=[Event("lights_out", 0.5, 9.0)])
    run = run_scene(spec, 10.0, seed=13)
    assert sum(inf.ok for inf in run.infos) >= 0.95 * len(run.infos)
    t = run.first_found()
    assert t is not None and t <= 6.0
    assert run.found_fraction(t + 0.5, 10.0) >= 0.98


# --------------------------------------------------------------------------- #
# no Morse lamp at all

@pytest.mark.parametrize("events", [[], [Event("emergency", 0.0, 60.0)]], ids=["normal", "emergency"])
def test_distractors_only(events):
    spec = SceneSpec(width=1280, height=720, word=None, seed=14,
                     distractors=ALL_DISTRACTORS + ("panic",), events=events)
    run = run_scene(spec, 10.0, seed=14)
    simon = run.scr.object_center("simon_yellow")
    statics = [run.scr.object_center(n) for n in STATIC_OBJECTS if run.scr.object_center(n)]
    assert len(statics) >= 5
    for t, top in zip(run.times, run.top3):
        for c in top:
            # never the timer digits, highlights, LEDs or the yellow wire
            assert not any(c.contains(*p) for p in statics), (t, c)
        if t > 2.0 and top:
            assert top[0].contains(*simon), (t, top[0])     # only Simon's yellow button may lead


# --------------------------------------------------------------------------- #
# levels

def test_history_levels_have_clear_on_off_contrast():
    spec = SceneSpec(width=1280, height=720, word="strobe", seed=15, phase=None,
                     distractors=ALL_DISTRACTORS)
    run = run_scene(spec, 9.0, seed=15)
    c = run.lamp_candidate()
    assert c is not None
    hist = run.loc.history(c.id)
    assert len(hist) >= 0.9 * 9.0 * FPS * 0.95              # the whole buffered epoch
    on = np.array([v for t, v in hist if v is not None and run.scr.lamp_on(t)])
    off = np.array([v for t, v in hist if v is not None and not run.scr.lamp_on(t)])
    assert len(on) > 20 and len(off) > 20
    assert np.percentile(on, 5) - np.percentile(off, 95) >= 60
    assert c.on_level - c.off_level >= 60

    # history and the per-frame levels describe the same numbers
    by_t = {round(t, 6): lv.get(c.id) for t, lv in zip(run.times, run.levels)}
    recent = [(t, v) for t, v in hist if t > run.times[-1] - 2.0]
    for t, v in recent:
        live = by_t[round(t, 6)]
        assert (v is None and live is None) or abs(v - live) < 1e-9

    # consistent with features.lamp_level on the full-resolution bbox
    frame = run.scr.render(run.times[-1])
    x, y, w, h = c.bbox
    full = lamp_level(frame[y:y + h, x:x + w])
    assert abs(run.levels[-1][c.id] - full) <= 0.15 * max(full, 20) + 3


def test_ids_are_stable():
    spec = SceneSpec(width=1280, height=720, word="brick", seed=16, phase=None,
                     distractors=ALL_DISTRACTORS)
    run = run_scene(spec, 12.0, seed=16)
    ids, versions = set(), set()
    t0 = run.first_found()
    for i, t in enumerate(run.times):
        if t > t0 + 1.0:
            lc = run.scr.lamp_center(t)
            c = next((c for c in run.top3[i] if c.contains(*lc)), None)
            assert c is not None
            ids.add(c.id)
            versions.add(c.version)
    assert len(ids) == 1
    assert len(versions) <= 2
    cid = ids.pop()
    assert run.loc.candidate(cid) is not None and run.loc.level(cid) == run.levels[-1][cid]


# --------------------------------------------------------------------------- #
# building blocks and speed

def test_phase_shift_estimates_translation():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (180, 320)).astype(np.float32)
    a = (a + np.roll(a, 1, 0) + np.roll(a, 1, 1)) / 3
    b = np.roll(np.roll(a, 7, 0), -12, 1) * 0.5                    # moved and darker
    dy, dx, peak = phase_shift(a, b)
    assert round(dy) == 7 and round(dx) == -12 and peak > 0.3
    _, _, peak2 = phase_shift(a, rng.integers(0, 255, (180, 320)).astype(np.float32))
    assert peak2 < 0.1


def test_speed_1440p():
    """Per-frame cost at 2560x1440 (generous bound: other processes may load the machine)."""
    spec = SceneSpec(width=2560, height=1440, word="shell", seed=0, distractors=ALL_DISTRACTORS)
    scr = FakeScreen(spec)
    frames = [(t, f) for t, f in scr.frames(fps=40, duration=3.0)]
    loc = LampLocator()
    dts = []
    for t, f in frames:
        c0 = time.perf_counter()
        loc.update(t, f)
        dts.append(time.perf_counter() - c0)
    assert loc.factor == 4 and loc.small_shape == (360, 640)
    assert np.median(dts[5:]) < 0.030
