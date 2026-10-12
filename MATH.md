# How the math works

This page explains every step from screen pixels to the frequency you transmit:

1. [The signal](#1-the-signal): timing of the Morse light.
2. [Pixels → one number](#2-pixels--one-number): how bright the lamp is.
3. [Finding the lamp](#3-finding-the-lamp): which spot on screen is blinking.
4. [Number → on/off runs](#4-number--onoff-runs): when the light switches.
5. [Runs → word](#5-runs--word): which of the 16 words explains the runs.
6. [Deciding when to answer](#6-deciding-when-to-answer): only when it is sure.
7. [Word → button presses](#7-word--button-presses): what to press.
8. [How well it works](#8-how-well-it-works): measured results.

---

## 1. The signal

The module flashes one word over and over. All lengths are multiples of one
unit $u$:

$$u = 0.27\ \text{s}$$

| part | light | length |
|---|---|---|
| dot | on | $1u = 0.27$ s |
| dash | on | $3u = 0.81$ s |
| gap inside a letter | off | $1u = 0.27$ s |
| gap between letters | off | $4u = 1.08$ s |
| gap before the word repeats | off | $10u = 2.70$ s |

The letter gap is 4 units and the word gap is 10 units. Textbook Morse uses
3 and 7.

**Length of one loop.** Take a word with $D$ dots, $H$ dashes and $L$ letters:

- the light is on for $D + 3H$ units;
- the gaps inside letters add $(D + H) - L$ units;
- the gaps between letters add $4(L-1)$ units;
- the word gap adds $10$ units.

Adding these up:

$$T(w) = u\,(2D + 4H + 3L + 6)$$

*Example*, **steak** = `... - . .- -.-` has $D = 6$, $H = 4$, $L = 5$:

$$T = 0.27 \cdot (12 + 16 + 15 + 6) = 0.27 \cdot 49 = 13.23\ \text{s}$$

The loops range from 13.2 s (*steak*) to 19.4 s.

**Frame rounding.** The game ends each wait on the first rendered frame after
its deadline. At $f$ frames per second, every on or off part is therefore a
little longer than nominal:

$$d = k\,u + \delta, \qquad \delta \sim \mathcal{U}\!\left(0, \tfrac{1}{f}\right)$$

At 60 fps that is 0 to 17 ms extra per part.

---

## 2. Pixels → one number

The lit lamp is bright amber-yellow, about $(251, 222, 71)$. The unlit lamp is
dull brown, about $(117, 62, 41)$. Every pixel gets a **yellowness** score:

$$y = \max\bigl(0,\ \min(R, G) - B\bigr)$$

| pixel | $(R,G,B)$ | $y$ |
|---|---|---|
| lit lamp | (251, 222, 71) | **151** |
| unlit lamp | (117, 62, 41) | 21 |
| white highlight | (240, 240, 240) | 0 |
| grey metal | (90, 90, 95) | 0 |
| red emergency light | (160, 40, 30) | 10 |

A yellow pixel needs both red and green high and blue low. White, grey and red
pixels all score low, so they can't be mistaken for the lit lamp.

The level of a region is the mean yellowness of its brightest pixels, the top
$p = 15\%$ (at least 4 pixels):

$$x = \frac{1}{k}\sum_{j=1}^{k} y_{(j)}, \qquad k = \max\bigl(4,\ \lfloor p \cdot N \rfloor\bigr)$$

where $y_{(1)} \ge y_{(2)} \ge \dots$ are the region's $N$ pixel scores sorted
from high to low. Using only the top pixels keeps the signal strong when the box
around the lamp is drawn loosely.

The lit lamp ignores room lighting, because the game draws it with an unlit
shader. During lights-out the "off" level drops, but the "on" level stays where
it was.

---

## 3. Finding the lamp

The whole game window is captured about 40 times per second.

**Downscale with max, not mean.** Each $f \times f$ block of the yellowness map
becomes one cell:

$$Y[i,j] = \max_{0 \le a,b < f} y[fi+a,\ fj+b]$$

A 10-pixel lamp survives this downscaling. An average would wash it out.

**Blink strength.** The last ~8 s of cells are kept. For each cell:

$$A[i,j] = \max_t Y_t[i,j] - \min_t Y_t[i,j]$$

Static things have $A \approx 0$, and the lamp has $A \approx 130$.

**Room light.** Lights-out and emergency lights change everything at once. The
room-light gain is estimated from the green channel $G$ of many sample points,
compared with a reference frame:

$$g = \operatorname{median}\left(\frac{G_\text{now}}{G_\text{ref}}\right)$$

Amplitude is also measured on $Y/g$. A cell only counts as blinking if it
blinks with the gain divided out too.

**Camera motion.** The program finds strong edges and checks whether each
edge's gradient sign matches the previous frame:

$$\text{agreement} = \frac{\#\{\text{strong gradients with the same sign}\}}{\#\{\text{strong gradients}\}}$$

Changing the light level never flips a gradient's sign, but moving the camera
does. Low agreement marks the frame as "moving", and the decoders treat it as
missing.

**Is it Morse?** Blinking cells are grouped into blobs. For each blob, the
on/off run lengths are checked against the allowed windows:

| run | allowed length (s) |
|---|---|
| on, dot | 0.16 – 0.42 |
| on, dash | 0.62 – 1.02 |
| off, inside a letter | 0.16 – 0.42 |
| off, between letters | 0.84 – 1.45 |
| off, word gap | 2.25 – 3.30 |

A blob's score is its blink strength × how plausible its blink rate is × the
fraction of runs that fit. The top 3 blobs each get their own decoder (section
5), filled with the buffered history so no seconds are lost. The first one to
decode a real word wins.

Things that blink but are not Morse fail this check:

- Simon Says flashes every 0.65 s.
- Timer digits change every 1 s.
- The alarm clock blinks every 0.25 s.
- Panic LEDs blink every 0.1 s.

---

## 4. Number → on/off runs

**Two levels.** In a short window the samples are split into two groups, "off"
with mean $\mu_0$ and "on" with mean $\mu_1$, with noise $\sigma$. Both levels
are then tracked slowly with an exponential average ($\tau = 0.5$ s):

$$\mu \leftarrow \mu + \left(1 - e^{-\Delta t/\tau}\right)(x - \mu)$$

**Evidence per sample.** How much more a sample $x$ looks like "on" than "off":

$$\ell = \log\frac{\mathcal{N}(x;\ \mu_1, \sigma)}{\mathcal{N}(x;\ \mu_0, \sigma)}
       = \frac{(x - \mu_0)^2 - (x - \mu_1)^2}{2\sigma^2}$$

It is clipped to $[-4, 4]$, so one garbage frame can't flip the state.

**Edges with CUSUM.** While the light is off, evidence for "on" accumulates:

$$S_t = \max\bigl(0,\ S_{t-1} + \ell_t\bigr)$$

An edge is accepted when $S_t \ge 10$ (and the new state lasts at least 30 ms).
It is the same in the other direction. The edge time is taken from where the
evidence started to build up, not from when it crossed the threshold.

The output is a list of run lengths: on, off, on, off, …

**Censored runs.** The first run after the program starts, or after a missing
stretch, was already in progress. Its true length is only known to be **at
least** what was seen. The same is true for the run happening right now.

---

## 5. Runs → word

**Each word as a loop of elements.** Each word is written as a cycle of $n$
elements, each with a unit count $k_i$:

- on: $k \in \{1, 3\}$
- off: $k \in \{1, 4, 10\}$

*Example*: **halls** starts `on 1, off 1, on 1, off 1, on 1, off 1, on 1, off 4, on 1, off 1, on 3, off 4, …`
and ends with `off 10`, then starts again.

**Duration model.** A run that really is element $i$ has a length close to
$k_i u$, plus the frame delay $b$:

$$p(d \mid k_i, u) = (1 - \varepsilon)\ \mathcal{N}\!\left(d;\ k_i u + b,\ \sigma_i^2\right) + \varepsilon\ p_\text{broad}(d)$$

$$\sigma_i^2 = (0.05\,k_i u)^2 + \sigma_0^2 + \sigma_\text{edge}^2$$

where:

- $\sigma_0 = 8$ ms covers game frames and screen refresh.
- $\sigma_\text{edge}$ comes from the time between captured frames.
- $\varepsilon = 2\%$ is the chance of an odd run, and $p_\text{broad}$ is a
  wide density used for those runs.

A censored run uses the probability of being *at least* that long:

$$P(D \ge d) = 1 - \Phi\!\left(\frac{d - k_i u - b}{\sigma_i}\right)$$

**Unknown starting point: the forward algorithm.** The program may start
watching in the middle of the loop. So it tracks every possible position $i$ in
the loop at once:

$$\alpha_t(i) = \log P(d_1, \dots, d_t,\ \text{run } t \text{ is element } i)$$

$$\alpha_t(i) = \log p(d_t \mid k_i, u) + \operatorname{logsumexp}_j \Bigl[\alpha_{t-1}(j) + \log P(j \to i)\Bigr]$$

The step from one run to the next, $P(j \to i)$, is almost always $i = j + 1$
(the next element of the loop). Small probabilities allow for mistakes:

| transition | meaning | probability |
|---|---|---|
| $j \to j+1$ | normal | ≈ 1 |
| $j \to j+3$ | a short gap was missed, so one run covers element + gap + element | 0.3% |
| three runs → one element | a short false blip split one element | 0.3% |
| $j \to$ anywhere | lost track after garbage, start over | 0.1% |

**Unknown speed.** $u$ is not assumed to be exactly 0.27 s. The calculation
runs on a grid of 50 values of $u$ from 0.17 to 0.43 s. The prior on $u$ is
flat between 0.21 and 0.345 s, with soft edges outside that range. So the game
can run up to about 25% slower or faster without fooling the decoder.

**Score of a word.** Sum over all positions and all speeds:

$$\mathcal{L}(w) = \log \sum_{u} \pi(u) \sum_{i=1}^{n} e^{\alpha_T(i,\,u)} - \log n$$

The $-\log n$ term means every starting point was equally likely.

**Missing stretches.** For example, the cursor passed over the lamp, or a
window covered it. If the gap lasted $\Delta$ seconds, the elements it hid must
add up to about $\Delta$:

$$\Delta \approx u \sum_{\text{hidden elements}} k_j$$

This is scored with a Gaussian on the summed length. It is mixed 80/20 with
"start a fresh alignment" for gaps such as the pause menu. Short dropouts
therefore cost almost nothing.

**Two "none of the above" models** compete with the 16 words:

- **random Morse**: correct dot and dash timing, but random letters;
- **blinker**: any light that keeps repeating its own on/off lengths, such as
  Simon Says, a clock or a timer.

**Probability of each word** (Bayes, with all 16 words equally likely):

$$P(w \mid \text{runs}) = \frac{e^{\mathcal{L}(w)}}{\sum_{w'} e^{\mathcal{L}(w')} + e^{\mathcal{L}(\text{random})} + e^{\mathcal{L}(\text{blinker})}}$$

### Why it can answer early

The loop is cyclic, so a fragment from the middle of a word is already useful.
A full loop isn't needed. For example, `....` (H) followed by a letter gap
matches two places:

- **halls**: the H at the start of the word;
- **shell**: the H after the S.

If a *word gap* came before the `....`, only *halls* fits. If `...` (S) came
before it, only *shell* fits. The forward algorithm keeps both possibilities
alive and lets the next runs decide.

The hardest words are **slick, trick, flick** and **brick**. They all end in
`-ick`, so they differ only in their first letters.

---

## 6. Deciding when to answer

A wrong answer costs a strike, so the program answers only when all of these
hold:

$$P(w^\ast \mid \text{runs}) \ge 0.995$$

$$\mathcal{L}(w^\ast) - \mathcal{L}(\text{2nd best word}) \ge 7 \text{ nats} \quad (\approx 1100{:}1)$$

$$\mathcal{L}(w^\ast) - \mathcal{L}(\text{random}),\ \ \mathcal{L}(w^\ast) - \mathcal{L}(\text{blinker}) \ge 6 \text{ nats} \quad (\approx 400{:}1)$$

These must hold on 2 updates in a row, and the best speed $u$ must be inside
the grid. Once given, an answer never changes.

---

## 7. Word → button presses

The display always starts at **3.505 MHz**. The arrows step through the 16
frequencies in sorted order and stop at the ends; they don't wrap around. So
the number of `>` presses is the word's position in the table:

| # | word | MHz | | # | word | MHz |
|---|---|---|---|---|---|---|
| 0 | shell | 3.505 | | 8 | flick | 3.555 |
| 1 | halls | 3.515 | | 9 | bombs | 3.565 |
| 2 | slick | 3.522 | | 10 | break | 3.572 |
| 3 | trick | 3.532 | | 11 | brick | 3.575 |
| 4 | boxes | 3.535 | | 12 | steak | 3.582 |
| 5 | leaks | 3.542 | | 13 | sting | 3.592 |
| 6 | strobe | 3.545 | | 14 | vector | 3.595 |
| 7 | bistro | 3.552 | | 15 | beats | 3.600 |

---

## 8. How well it works

The decoder is tested on simulated captures (`tools/bench.py`). They include:

- capture rates from 20 to 360 fps, with jitter, dropped frames and stalls;
- noise and low contrast;
- lights-out and emergency lights;
- camera movement and pauses;
- distractors: Simon Says, timer digits, alarm clock, panic LEDs, and random
  Morse.

| test set | cases | wrong answers | false alarms | no answer | median time | 90% within |
|---|---|---|---|---|---|---|
| public, seed 1 | 440 | **0** | **0** | 3 / 302 | 9.8 s | 17.4 s |
| public, seed 2 | 440 | **0** | **0** | 2 / 293 | 9.9 s | 15.5 s |
| holdout | 450 | **0** | **0** | 0 / 288 | 9.6 s | 15.4 s |

A median of about 9.6 s is well under one loop of the word, which takes 13 to
19 s.
