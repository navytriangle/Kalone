"""Character vocabulary and action-type constants.

To support a new alphabet (e.g. Hangul), extend CHARS here. Index 0 is reserved for PAD.
"""
import string

CHARS = string.ascii_lowercase + " |0123456789"
PAD = 0
stoi = {c: i + 1 for i, c in enumerate(CHARS)}
itos = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(CHARS) + 1

# Action types emitted by the decoder at every step.
GEN, COPY, EOS = 0, 1, 2
BOS = 3  # only used as the "previous action" fed to the first decoder step


def encode(text: str) -> list:
    return [stoi[c] for c in text]
