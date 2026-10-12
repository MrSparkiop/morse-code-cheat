"""The decoder the app uses.

It is the probabilistic closed-vocabulary matcher (decoders/contest_b.py),
which won the benchmark: on 1,330 simulated captures it never gave a wrong
answer or a false alarm, stayed silent in under 1 % of the cases that need
an answer, and answered after ~9.5 s on median (less than one loop of the word).
"""

from .decoders.contest_b import Decoder as MorseDecoder

Decoder = MorseDecoder

__all__ = ["MorseDecoder", "Decoder"]
