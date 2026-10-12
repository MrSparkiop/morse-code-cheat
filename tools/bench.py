"""Score a Morse decoder against simulated captures.

    python -m tools.bench ktane_morse.decoders.baseline            # full public suite
    python -m tools.bench ktane_morse.decoders.baseline --quick    # ~1/4 size, for iterating
    python -m tools.bench ktane_morse.decoders.baseline --suite holdout
    python -m tools.bench MODULE --show-failures --json out.json

The decoder module must expose a class ``Decoder`` implementing
ktane_morse.decoding.Decoder (feed / status / reset). status() is polled every
50 ms of trace time, like a UI would.

Hard requirements (must be 0): WRONG (an answer that is not the flashed word)
and FP (any answer on a distractor that is not Morse at all). Then: as few
SILENT positives as possible, then low latency.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor

POLL = 0.05


def _load_suite(suite: str, quick: bool, seed: int):
    """Return (count, getter) where getter(i) builds scenario i."""
    if suite == "public":
        from ktane_morse.simulate import public_scenario

        sizes = (48, 48, 40) if quick else (160, 160, 120)
        return sum(sizes), lambda i: public_scenario(i, *sizes, seed=seed)
    if suite == "holdout":
        from tools.holdout import holdout_count, holdout_scenario

        n = holdout_count(quick)
        return n, lambda i: holdout_scenario(i, quick=quick, seed=seed)
    raise SystemExit(f"unknown suite {suite}")


def run_one(args) -> dict:
    module, suite, quick, seed, i = args
    _, get = _load_suite(suite, quick, seed)
    sc = get(i)
    Decoder = importlib.import_module(module).Decoder
    d = Decoder()
    first = first_t = last = None
    flips = 0
    next_poll = 0.0
    c0 = time.perf_counter()
    err = None
    try:
        for t, lvl in sc.trace:
            d.feed(t, lvl)
            if t >= next_poll:
                next_poll = t + POLL
                a = d.status().answer
                if a is not None:
                    if first is None:
                        first, first_t = a, t
                    elif a != last:
                        flips += 1
                    last = a
        st = d.status()
        if st.answer is not None and st.answer != last:
            if first is None:
                first, first_t = st.answer, sc.trace[-1][0]
            else:
                flips += 1
            last = st.answer
    except Exception as e:  # a crash counts as a failure, but keep benchmarking
        err = f"{type(e).__name__}: {e}"
    cpu = time.perf_counter() - c0
    return {
        "i": i, "name": sc.name, "word": sc.word, "must": sc.must_answer, "soft": sc.soft,
        "first": first, "t": first_t, "final": last, "flips": flips, "cpu": cpu,
        "samples": len(sc.trace), "notes": sc.notes, "error": err,
    }


def summarize(rows: list[dict]) -> dict:
    pos = [r for r in rows if r["word"] is not None and r["must"]]
    opt = [r for r in rows if r["word"] is not None and not r["must"]]
    neg = [r for r in rows if r["word"] is None and not r["soft"]]
    soft = [r for r in rows if r["soft"]]
    wrong = [r for r in pos + opt if r["first"] is not None and r["first"] != r["word"]]
    fp = [r for r in neg if r["first"] is not None]
    soft_fp = [r for r in soft if r["first"] is not None]
    silent = [r for r in pos if r["first"] is None]
    errors = [r for r in rows if r["error"]]
    lat = sorted(r["t"] for r in pos if r["first"] == r["word"])
    normal = sorted(r["t"] for r in pos if r["first"] == r["word"] and r["name"].startswith("normal"))

    def pct(xs, q):
        return round(xs[min(len(xs) - 1, int(q * len(xs)))], 2) if xs else None

    total_samples = sum(r["samples"] for r in rows)
    return {
        "scenarios": len(rows),
        "WRONG": len(wrong),
        "FP": len(fp),
        "ERRORS": len(errors),
        "silent": f"{len(silent)}/{len(pos)}",
        "silent_n": len(silent),
        "flips": sum(r["flips"] for r in rows),
        "latency_median": pct(lat, 0.5),
        "latency_p90": pct(lat, 0.9),
        "latency_max": pct(lat, 1.0),
        "normal_latency_median": pct(normal, 0.5),
        "normal_latency_p90": pct(normal, 0.9),
        "soft_fp": f"{len(soft_fp)}/{len(soft)}",
        "cpu_us_per_sample": round(1e6 * sum(r["cpu"] for r in rows) / max(1, total_samples), 1),
        "_wrong": wrong, "_fp": fp, "_silent": silent, "_errors": errors, "_soft_fp": soft_fp,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("module")
    ap.add_argument("--suite", default="public", choices=["public", "holdout"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--json")
    ap.add_argument("--show-failures", action="store_true")
    a = ap.parse_args(argv)

    n, _ = _load_suite(a.suite, a.quick, a.seed)
    t0 = time.time()
    jobs = [(a.module, a.suite, a.quick, a.seed, i) for i in range(n)]
    if a.jobs > 1:
        with ProcessPoolExecutor(a.jobs) as ex:
            rows = list(ex.map(run_one, jobs, chunksize=4))
    else:
        rows = [run_one(j) for j in jobs]
    s = summarize(rows)
    head = {k: v for k, v in s.items() if not k.startswith("_")}
    print(f"{a.module} [{a.suite}{' quick' if a.quick else ''} seed={a.seed}] in {time.time() - t0:.0f}s")
    print("  " + "  ".join(f"{k}={v}" for k, v in head.items()))
    if a.show_failures:
        for label in ("_errors", "_wrong", "_fp", "_soft_fp", "_silent"):
            for r in s[label][:25]:
                print(f"  {label[1:].upper():8} #{r['i']:<4} {r['name']:<28} said={r['first']} at={r['t']} {r['notes']} {r['error'] or ''}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump({"summary": head, "rows": rows}, f, indent=1)
    return 0 if s["WRONG"] == 0 and s["FP"] == 0 and s["ERRORS"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
