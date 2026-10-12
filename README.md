# morse-code-cheat

Watches the screen while you play *Keep Talking and Nobody Explodes*, finds the
blinking light on the **Morse Code** module, decodes the word and tells you the
frequency to transmit — usually before the word has finished its first loop.

```
STROBE  ->  3.545 MHz   (press > 6 times from 3.505, then TX)
```

## Setup (once)

```bash
git clone https://github.com/MrSparkiop/morse-code-cheat.git
```

```bash
cd morse-code-cheat
```

```bash
py -3.14 -m venv .venv
```

```bash
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Only two packages are needed: `numpy` and `mss` (screen capture).

## Use

```bash
.venv\Scripts\python.exe -m ktane_morse
```

A small always-on-top window appears in the top-right corner (it is hidden
from screen captures, so it never reads itself). Then just play:

1. Start the program, then the game. The program looks for the light by itself
   — zoom in on the Morse module and keep the bomb still for a few seconds.
2. The decoded dots and dashes appear as they are read, with the best guess.
3. When it is sure, the frequency turns green (and is spoken if *Speak* is on).
   Press `>` the number of times shown, then **TX**.

If automatic search ever picks the wrong thing, press **F9** (works while the
game has focus) and drag a box around the Morse light. **F8** goes back to
automatic search. **Reset** forgets the current answer.

Useful options:

| option | what it does |
|---|---|
| `--select` | drag a box around the light at start-up |
| `--region X,Y,W,H` | read a fixed screen region (physical pixels) |
| `--speak` | say the answer out loud (Windows speech, no extra install) |
| `--no-gui` | console only |
| `--check-log` | also show whether the answer matches the word the game wrote to its own log (`logs/ktane.log`) — handy for checking the decoder; answers never come from the log |
| `--window "TITLE"` | scan this window in automatic mode (default: the game window; the primary monitor while it is closed) |
| `--hotkey-auto F8`, `--hotkey-select F9` | change the global hotkeys |

Tips:

- Run the game in its default *Fullscreen* (which is borderless) or *Windowed*
  mode. Both capture fine.
- Keep the mouse cursor and the bomb still while it reads. If you rotate the
  bomb it simply starts reading again.
- The table is for the English word list (also used by the Japanese, Korean and
  Chinese versions). Other game languages use different words.

## Try it without the game

```bash
.venv\Scripts\python.exe -m ktane_morse.fake_module --word strobe
```

opens a fake module that flashes a word with the game's exact timing. In a
second terminal run the reader on it:

```bash
.venv\Scripts\python.exe -m ktane_morse --window "Fake Morse Module"
```

## How it works

The full math, with formulas, is in [MATH.md](MATH.md). In short:


- **Timing** (read from the game itself): one unit is 0.27 s. A dot is 1 unit on
  and a dash 3. The gap inside a letter is 1 unit, between letters 4 units, and
  10 units before the word repeats.
- **Capture**: `mss` grabs the screen (about 360 fps for a small region on a
  high-refresh monitor). The process is per-monitor DPI aware, so coordinates
  are physical pixels on every monitor.
- **Signal**: each frame becomes one number, the "yellowness" (`min(R,G) - B`)
  of the most yellow pixels. The lit lamp is bright amber, while the unlit lamp,
  white highlights, grey metal and red emergency lighting all score low.
- **Decoding**: on/off run lengths are matched against all 16 words, at every
  point in their loop. An answer is committed only when exactly one word
  explains everything seen so far, with a safety margin.
- **Finding the light**: the screen is scanned for small regions that turn
  yellow on and off with Morse-like timing. Each candidate gets its own decoder,
  and the first one that decodes a real word wins.

## Development

```bash
.venv\Scripts\python.exe -m pytest -q
```

```bash
.venv\Scripts\python.exe -m tools.bench ktane_morse.decoder
```

`tools/bench.py` scores a decoder on simulated captures: every word at various
frame rates, noise levels and contrasts, plus lights-out, emergency lights,
camera motion, pauses, and distractors such as Simon Says, the timer and alarm
clocks. `--suite holdout` runs a second, independently written suite.
