"""Morse alphabet and the KTANE "On the Subject of Morse Code" word table."""

from __future__ import annotations

MORSE = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".", "F": "..-.",
    "G": "--.", "H": "....", "I": "..", "J": ".---", "K": "-.-", "L": ".-..",
    "M": "--", "N": "-.", "O": "---", "P": ".--.", "Q": "--.-", "R": ".-.",
    "S": "...", "T": "-", "U": "..-", "V": "...-", "W": ".--", "X": "-..-",
    "Y": "-.--", "Z": "--..",
}
REVERSE_MORSE = {code: letter for letter, code in MORSE.items()}

# Word -> frequency (MHz), in the order the manual lists them (ascending frequency).
WORDS: dict[str, str] = {
    "shell": "3.505",
    "halls": "3.515",
    "slick": "3.522",
    "trick": "3.532",
    "boxes": "3.535",
    "leaks": "3.542",
    "strobe": "3.545",
    "bistro": "3.552",
    "flick": "3.555",
    "bombs": "3.565",
    "break": "3.572",
    "brick": "3.575",
    "steak": "3.582",
    "sting": "3.592",
    "vector": "3.595",
    "beats": "3.600",
}

FREQUENCIES = list(WORDS.values())


def encode(word: str) -> list[str]:
    """Return the Morse code of each letter, e.g. "sos" -> ["...", "---", "..."]."""
    return [MORSE[ch] for ch in word.upper()]


def presses_from_start(word: str) -> int:
    """Number of right-arrow presses from the lowest frequency (3.505 MHz) to this word's."""
    return FREQUENCIES.index(WORDS[word])
