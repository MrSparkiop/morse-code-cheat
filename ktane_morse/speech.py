"""Say the answer out loud with the built-in Windows speech synthesizer (no extra packages)."""

from __future__ import annotations

import subprocess
import sys

_proc: subprocess.Popen | None = None


def say(text: str) -> None:
    """Speak text in the background, interrupting anything still being spoken."""
    global _proc
    if sys.platform != "win32":
        return
    if _proc is not None and _proc.poll() is None:
        _proc.kill()
    safe = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech; "
        "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$s.Rate = 1; $s.Speak('{safe}')"
    )
    try:
        _proc = subprocess.Popen(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except OSError:
        _proc = None


def spoken_frequency(freq: str) -> str:
    """'3.545' -> '3 point 5 4 5' so the digits are read one by one."""
    whole, _, frac = freq.partition(".")
    return f"{whole} point {' '.join(frac)}"
