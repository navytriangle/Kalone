"""Synthetic editing tasks + batching.

Every task produces an `Example`:
  inp    : the text the model reads (everything in it is copyable)
  acts   : the gold action sequence, ("copy", start, end_inclusive) or ("gen", char)
  target : the text those actions must produce (used by tests and evaluation)

Add your own task with @register_task("name"); see docs/DESIGN.md.
"""
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch

from .vocab import COPY, EOS, GEN, stoi

Action = tuple  # ("copy", s, e) or ("gen", ch)


@dataclass
class Example:
    inp: str
    acts: List[Action]
    target: str


TASKS: Dict[str, Callable[[random.Random], Example]] = {}


def register_task(name: str):
    def deco(fn):
        TASKS[name] = fn
        return fn

    return deco


def rand_word(rng: random.Random, lo: int, hi: int) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(lo, hi)))


def _sentence(rng: random.Random):
    words = [rand_word(rng, 3, 6) for _ in range(rng.randint(3, 6))]
    src = " ".join(words)
    starts, pos = [], 0
    for w in words:
        starts.append(pos)
        pos += len(w) + 1
    ends = [s + len(w) for s, w in zip(starts, words)]  # exclusive
    return words, src, starts, ends


def _cp(acts: List[Action], start: int, end_exclusive: int) -> None:
    if end_exclusive > start:
        acts.append(("copy", start, end_exclusive - 1))


@register_task("replace")
def replace_task(rng: random.Random) -> Example:
    """'a b c|r1|xyz' -> 'a xyz c'  (all COPY: the new word is copied from the instruction)"""
    words, src, starts, ends = _sentence(rng)
    k, new = rng.randrange(len(words)), rand_word(rng, 3, 5)
    inp = f"{src}|r{k}|{new}"
    ns, acts = len(src) + 4, []
    _cp(acts, 0, starts[k])
    _cp(acts, ns, ns + len(new))
    _cp(acts, ends[k], len(src))
    return Example(inp, acts, " ".join(words[:k] + [new] + words[k + 1:]))


@register_task("delete")
def delete_task(rng: random.Random) -> Example:
    """'a b c|d1' -> 'a c'"""
    words, src, starts, ends = _sentence(rng)
    k, acts = rng.randrange(len(words)), []
    if k == 0:
        _cp(acts, starts[1], len(src))
    else:
        _cp(acts, 0, ends[k - 1])
        _cp(acts, ends[k], len(src))
    return Example(f"{src}|d{k}", acts, " ".join(words[:k] + words[k + 1:]))


@register_task("append")
def append_task(rng: random.Random) -> Example:
    """'a b c|a' -> 'a b c done'  (COPY the sentence, GEN the literal ' done')"""
    _, src, _, _ = _sentence(rng)
    acts: List[Action] = []
    _cp(acts, 0, len(src))
    acts += [("gen", c) for c in " done"]
    return Example(f"{src}|a", acts, src + " done")


def make_example(rng: Optional[random.Random] = None, tasks: Optional[List[str]] = None) -> Example:
    rng = rng or random
    return TASKS[rng.choice(tasks or sorted(TASKS))](rng)


def run_actions(inp: str, acts: List[Action]) -> str:
    """Execute an action list: COPY pastes inp[s:e+1], GEN writes one character."""
    return "".join(inp[a[1]: a[2] + 1] if a[0] == "copy" else a[1] for a in acts)


def batchify(exs: List[Example]):
    B = len(exs)
    Lm = max(len(e.inp) for e in exs)
    T = max(len(e.acts) for e in exs) + 1  # +1 for EOS
    x = torch.zeros(B, Lm, dtype=torch.long)
    ty, ch, st, en = (torch.full((B, T), -100) for _ in range(4))
    for b, e in enumerate(exs):
        x[b, : len(e.inp)] = torch.tensor([stoi[c] for c in e.inp])
        for t, a in enumerate(e.acts):
            if a[0] == "gen":
                ty[b, t], ch[b, t] = GEN, stoi[a[1]]
            else:
                ty[b, t], st[b, t], en[b, t] = COPY, a[1], a[2]
        ty[b, len(e.acts)] = EOS
    return x, ty, ch, st, en
