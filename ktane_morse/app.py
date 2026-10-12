"""Main program: capture loop + answer display.

    python -m ktane_morse                 # find the Morse light automatically
    python -m ktane_morse --select        # drag a box around the light first
    python -m ktane_morse --region 1200,300,80,80
    python -m ktane_morse --no-gui        # console only
"""

from __future__ import annotations

import argparse
import queue
import sys
import threading
import time
from dataclasses import dataclass, field, replace

from .capture import GAME_TITLE, Grabber, Region, enable_dpi_awareness, find_window
from .decoder import MorseDecoder
from .decoding import Status
from .features import lamp_level
from .morse import WORDS, presses_from_start

SIGNAL_LOST_AFTER = 6.0  # seconds without blinking before an answer is considered "old"
AUTO_CANDIDATES = 3      # lights decoded in parallel during automatic search


@dataclass(frozen=True)
class Answer:
    word: str
    freq: str
    presses: int
    region: Region | None
    at: float

    @property
    def hint(self) -> str:
        if self.presses == 0:
            return "already at 3.505 - just press TX"
        return f"press > {self.presses} time{'s' if self.presses > 1 else ''} from 3.505, then TX"

    def line(self) -> str:
        return f"{self.word.upper()}  ->  {self.freq} MHz   ({self.hint})"


@dataclass(frozen=True)
class Snapshot:
    """What the UI shows. Replaced atomically by the engine thread."""

    mode: str = "starting"          # "auto" | "region" | "paused" | "starting"
    message: str = ""
    region: Region | None = None    # region currently being read
    status: Status | None = None    # decoder status of the most promising light
    answer: Answer | None = None    # latest committed answer
    answer_live: bool = False       # its light is still blinking
    fps: float = 0.0
    readers: int = 0                # lights being decoded right now (auto mode)


def answer_for(word: str, region: Region | None) -> Answer:
    return Answer(word, WORDS[word], presses_from_start(word), region, time.perf_counter())


class _Rate:
    def __init__(self) -> None:
        self.n = 0
        self.t0 = time.perf_counter()
        self.value = 0.0

    def tick(self) -> float:
        self.n += 1
        dt = time.perf_counter() - self.t0
        if dt >= 1.0:
            self.value, self.n, self.t0 = self.n / dt, 0, time.perf_counter()
        return self.value


class RegionReader:
    """Reads one fixed screen region at a high frame rate."""

    def __init__(self, region: Region, fps: float = 120.0) -> None:
        self.region = region
        self.period = 1.0 / fps
        self.decoder = MorseDecoder()
        self.next_t = time.perf_counter()
        self.announced: str | None = None
        self.last_signal = time.perf_counter()

    def step(self, grabber: Grabber) -> None:
        now = time.perf_counter()
        if now < self.next_t:
            time.sleep(self.next_t - now)
        self.next_t = max(self.next_t + self.period, time.perf_counter() - self.period)
        t = time.perf_counter()
        try:
            level: float | None = lamp_level(grabber.grab(self.region))
        except Exception:
            level = None
        self.decoder.feed(t, level)

    def status(self) -> Status:
        return self.decoder.status()


class Engine(threading.Thread):
    """Capture/decoding thread. Control it with command(); read .snapshot from any thread."""

    def __init__(self, region: Region | None = None, scan_area: Region | None = None,
                 on_answer=None, fps: float = 120.0, window_title: str = GAME_TITLE) -> None:
        super().__init__(daemon=True, name="engine")
        self.initial_region = region
        self.scan_area = scan_area
        self.window_title = window_title
        self.fps = fps
        self.on_answer = on_answer or (lambda a: None)
        self.snapshot = Snapshot()
        self._cmds: queue.Queue = queue.Queue()
        self._stop = threading.Event()

    # -- control (any thread) ------------------------------------------------
    def command(self, name: str, arg=None) -> None:
        self._cmds.put((name, arg))

    def stop(self) -> None:
        self._stop.set()
        self._cmds.put(("quit", None))

    # -- thread ----------------------------------------------------------------
    def run(self) -> None:
        enable_dpi_awareness()
        with Grabber() as grabber:
            nxt = ("region", self.initial_region) if self.initial_region else ("auto", None)
            while nxt[0] != "quit" and not self._stop.is_set():
                mode, arg = nxt
                if mode == "region":
                    nxt = self._run_region(grabber, arg)
                elif mode == "auto":
                    nxt = self._run_auto(grabber)
                else:  # paused; arg is the (mode, arg) to resume
                    self._set(mode="paused", message="Paused", status=None)
                    nxt = self._poll(nxt, block=True) or arg

    def _poll(self, current: tuple, block: bool = False) -> tuple | None:
        """Handle queued commands.

        Returns the (mode, arg) to switch to -- returning `current` itself
        restarts the current mode from scratch -- or None to carry on.
        """
        while True:
            try:
                name, arg = self._cmds.get(block=block, timeout=0.2 if block else None)
            except queue.Empty:
                if self._stop.is_set():
                    return "quit", None
                if block:
                    continue
                return None
            if name == "quit":
                return "quit", None
            if name == "region" and arg is not None:
                return "region", arg
            if name == "auto":
                return "auto", None
            if name == "pause":
                return ("paused", current) if current[0] != "paused" else current
            if name == "resume":
                return current[1] if current[0] == "paused" else current
            if name == "reset":
                self._set(answer=None, answer_live=False)
                return current

    def _set(self, **kw) -> None:
        self.snapshot = replace(self.snapshot, **kw)

    def _announce(self, word: str, region: Region | None) -> None:
        cur = self.snapshot.answer
        if cur is not None and cur.word == word and self.snapshot.answer_live:
            return  # the same light, re-read after a re-lock: nothing new to say
        ans = answer_for(word, region)
        self._set(answer=ans, answer_live=True)
        try:
            self.on_answer(ans)
        except Exception:
            pass

    # -- fixed region ------------------------------------------------------------
    def _run_region(self, grabber: Grabber, region: Region):
        reader = RegionReader(region, self.fps)
        rate = _Rate()
        self._set(mode="region", region=region, message=f"Reading the region {region}", readers=1)
        last_ui = 0.0
        while True:
            nxt = self._poll(("region", region))
            if nxt:
                return nxt
            reader.step(grabber)
            fps = rate.tick()
            now = time.perf_counter()
            if now - last_ui < 0.05:
                continue
            last_ui = now
            st = reader.status()
            if st.signal:
                reader.last_signal = now
            if st.answer and st.answer != reader.announced:
                reader.announced = st.answer
                self._announce(st.answer, region)
            live = self.snapshot.answer_live
            if reader.announced and now - reader.last_signal > SIGNAL_LOST_AFTER:
                # The light stopped (module solved, bomb over, or the view moved):
                # start fresh so the next Morse module in this spot is read from scratch.
                live = False
                reader = RegionReader(region, self.fps)
            self._set(status=st, fps=fps, answer_live=live and bool(self.snapshot.answer),
                      message=_progress_message(st, "the selected region"))

    # -- automatic search ----------------------------------------------------------
    def _scan_area(self, grabber: Grabber) -> Region:
        """The game window's client area if the game is running, else the given/primary monitor."""
        return find_window(self.window_title) or self.scan_area or grabber.primary()

    def _run_auto(self, grabber: Grabber):
        from .locator import LampLocator

        area = self._scan_area(grabber)
        locator = LampLocator()
        readers: dict[int, CandidateReader] = {}
        rate = _Rate()
        last_area_check = last_ui = time.perf_counter()
        self._set(mode="auto", region=None, status=None, readers=0,
                  message="Looking for the Morse light… (zoom in on the module and keep the bomb still)")
        while True:
            nxt = self._poll(("auto", None))
            if nxt:
                return nxt
            now = time.perf_counter()
            if now - last_area_check > 2.0:
                last_area_check = now
                new_area = self._scan_area(grabber)
                if new_area != area:  # game window moved, resized or appeared
                    area = new_area
                    locator.reset()
                    readers.clear()
            try:
                frame = grabber.grab(area)
            except Exception:
                frame = None
            t = time.perf_counter()
            info = locator.update(t, frame)

            # One decoder per promising light; every live one keeps being fed.
            fresh = set()
            for c in locator.candidates(AUTO_CANDIDATES):
                r = readers.get(c.id)
                if r is None or r.version != c.version:
                    # history already includes this frame
                    readers[c.id] = CandidateReader(c.id, c.version, locator.history(c.id))
                    fresh.add(c.id)
            for cid in list(readers):
                if locator.candidate(cid) is None:
                    del readers[cid]
                    continue
                if cid in fresh:
                    continue
                level = info.levels.get(cid) if info.ok else None
                readers[cid].decoder.feed(t, level)
            fps = rate.tick()

            if now - last_ui < 0.05:
                continue
            last_ui = now
            self._update_auto_ui(locator, readers, area, fps, now)

    def _update_auto_ui(self, locator, readers: dict, area: Region, fps: float, now: float) -> None:
        best: tuple | None = None
        for cid, r in readers.items():
            st = r.decoder.status()
            if st.signal:
                r.last_signal = now
            c = locator.candidate(cid)
            region = None
            if c is not None:
                x, y, w, h = c.bbox
                region = Region(area.left + x, area.top + y, w, h)
            if st.answer and st.answer != r.announced:
                r.announced = st.answer
                self._announce(st.answer, region)
            key = (bool(st.answer), st.signal, st.confidence, c.score if c else 0.0)
            if best is None or key > best[0]:
                best = (key, st, region)
        live = self.snapshot.answer_live and any(
            r.announced == (self.snapshot.answer.word if self.snapshot.answer else None)
            and now - r.last_signal < SIGNAL_LOST_AFTER for r in readers.values())
        if best is None:
            self._set(status=None, region=None, fps=fps, readers=0, answer_live=live,
                      message="Looking for the Morse light… (zoom in on the module and keep the bomb still)")
            return
        _, st, region = best
        self._set(status=st, region=region, fps=fps, readers=len(readers), answer_live=live,
                  message=_progress_message(st, "the candidate lights"))


class CandidateReader:
    """Decoder attached to one light found by the locator, back-filled with its buffered history."""

    def __init__(self, cid: int, version: int, history) -> None:
        self.cid = cid
        self.version = version
        self.decoder = MorseDecoder()
        self.announced: str | None = None
        self.last_signal = time.perf_counter()
        for t, level in history:
            self.decoder.feed(t, level)


def _progress_message(st: Status, where: str) -> str:
    if st.answer:
        return f"Decoded {st.answer.upper()} ({st.confidence:.0%})"
    if not st.signal:
        return f"No blinking light seen in {where} yet…"
    if st.ranking:
        w, p = st.ranking[0]
        return f"Reading… best guess {w.upper()} ({p:.0%})"
    return "Reading…"


# ---------------------------------------------------------------------------- #

def parse_region(text: str) -> Region:
    try:
        x, y, w, h = (int(v) for v in text.replace(" ", "").split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected X,Y,WIDTH,HEIGHT, e.g. 1200,300,80,80")
    if w < 2 or h < 2:
        raise argparse.ArgumentTypeError("region too small")
    return Region(x, y, w, h)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m ktane_morse",
        description="Reads the Morse Code module's light in Keep Talking and Nobody Explodes and tells you the frequency.",
    )
    ap.add_argument("--select", action="store_true", help="drag a box around the Morse light before starting")
    ap.add_argument("--region", type=parse_region, help="read this screen region: X,Y,WIDTH,HEIGHT (physical pixels)")
    ap.add_argument("--no-gui", action="store_true", help="console output only, no window")
    ap.add_argument("--speak", action="store_true", help="say the answer out loud")
    ap.add_argument("--check-log", action="store_true",
                    help="compare answers with the word the game wrote to its log file (for testing the decoder)")
    ap.add_argument("--game-dir", help="game folder, for --check-log if it is not in the default Steam location")
    ap.add_argument("--window", default=GAME_TITLE,
                    help="title of the window to scan in automatic mode (default: the game); "
                         "falls back to the primary monitor while that window is not open")
    ap.add_argument("--fps", type=float, default=120.0, help="capture rate for a selected region (default 120)")
    ap.add_argument("--hotkey-auto", default="F8", help="global hotkey: search automatically (default F8)")
    ap.add_argument("--hotkey-select", default="F9", help="global hotkey: select the light (default F9)")
    return ap


def main(argv=None) -> int:
    enable_dpi_awareness()  # before any window or capture object exists
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")  # never die on a console that can't show a character
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args(argv)

    region = args.region
    if args.select and region is None:
        from .select_region import pick_region

        region = pick_region()
        if region is None:
            print("Selection cancelled.")
            return 1

    log_watcher = None
    if args.check_log:
        from .gamelog import GameLogWatcher, find_log

        path = find_log(args.game_dir)
        if path is None:
            print("--check-log: could not find logs/ktane.log (use --game-dir).")
        else:
            log_watcher = GameLogWatcher(path)
            log_watcher.start()

    speak = {"on": args.speak}

    def on_answer(ans: Answer) -> None:
        check = ""
        if log_watcher is not None and log_watcher.words:
            check = "   [matches game log]" if ans.word in log_watcher.words else f"   [GAME LOG SAYS: {', '.join(log_watcher.words)}]"
        print(f"\n[{time.strftime('%H:%M:%S')}] {ans.line()}{check}", flush=True)
        if speak["on"]:
            from .speech import say, spoken_frequency

            say(f"{ans.word}. {spoken_frequency(ans.freq)}")

    engine = Engine(region=region, on_answer=on_answer, fps=args.fps, window_title=args.window)
    engine.start()
    try:
        if args.no_gui:
            _console_loop(engine)
        else:
            _gui_loop(engine, args, speak, log_watcher)
    except KeyboardInterrupt:
        pass
    finally:
        engine.stop()
        engine.join(2)
    return 0


def _console_loop(engine: Engine) -> None:
    print("Watching the screen. Ctrl+C to quit.", flush=True)
    last = ""
    while engine.is_alive():
        snap = engine.snapshot
        st = snap.status
        line = snap.message + (f"   {st.morse[-40:]}" if st and st.morse else "")
        if line != last:
            print("\r" + line.ljust(110)[:110], end="", flush=True)
            last = line
        time.sleep(0.2)


def _gui_loop(engine: Engine, args, speak: dict, log_watcher) -> None:
    import tkinter as tk

    from .hotkeys import HotkeyThread
    from .hud import AMBER, DIM, GREEN, RED, Hud
    from .select_region import pick_region

    root = tk.Tk()
    ui_cmds: queue.Queue = queue.Queue()
    keys = {"auto": args.hotkey_auto, "select": args.hotkey_select}
    hud = Hud(root, lambda c: ui_cmds.put(c), keys)
    hud.speak_var.set(speak["on"])
    hk = HotkeyThread(keys, lambda name: ui_cmds.put(name)).start_and_wait()
    if hk.failed:
        print(f"Could not register hotkeys {hk.failed} (another program uses them).")

    def select() -> None:
        engine.command("pause")
        root.withdraw()
        try:
            r = pick_region(root)
        finally:
            root.deiconify()
            root.attributes("-topmost", True)
        engine.command("region", r) if r else engine.command("resume")

    def poll() -> None:
        while True:
            try:
                c = ui_cmds.get_nowait()
            except queue.Empty:
                break
            if c == "quit":
                hk.stop()
                root.destroy()
                return
            if c == "select":
                select()
            elif c == "speak":
                speak["on"] = hud.speak_var.get()
            else:
                engine.command(c)
        snap = engine.snapshot
        if snap.answer:
            hint = snap.answer.hint
            if log_watcher is not None and log_watcher.words:
                hint += "   ✓ log" if snap.answer.word in log_watcher.words else f"   ✗ log: {', '.join(log_watcher.words)}"
            hud.show_answer(snap.answer.freq, snap.answer.word, hint, fresh=snap.answer_live)
        else:
            hud.show_answer(None, None, "", fresh=False)
        st = snap.status
        colour = GREEN if (st and st.answer) else (AMBER if st and st.signal else DIM)
        if snap.mode == "paused":
            colour = DIM
        hud.show_progress(st.morse if st else "", snap.message, colour)
        if not engine.is_alive():
            hud.show_progress("", "Capture stopped (see console).", RED)
        root.after(100, poll)

    root.after(100, poll)
    root.mainloop()
    hk.stop()
