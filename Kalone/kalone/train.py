"""Train and evaluate Kalone.

    python -m kalone.train --steps 2000 --bs 64
"""
import argparse
import os
import random

import torch
import torch.nn as nn

from .data import TASKS, batchify, make_example
from .model import CopyLM


def evaluate(model: CopyLM, n: int = 300, seed: int = 1234, tasks=None) -> dict:
    """Exact-match accuracy overall and per task, on a fixed held-out seed."""
    rng = random.Random(seed)
    names = tasks or sorted(TASKS)
    hits = {t: 0 for t in names}
    total = {t: 0 for t in names}
    for i in range(n):
        t = names[i % len(names)]
        ex = make_example(rng, [t])
        total[t] += 1
        hits[t] += model.generate(ex.inp)[0] == ex.target
    out = {t: hits[t] / total[t] for t in names if total[t]}
    out["overall"] = sum(hits.values()) / sum(total.values())
    return out


def train(steps=2000, bs=64, lr=2e-3, seed=0, emb=64, hidden=192, log_every=250,
          eval_every=500, out="checkpoints/kalone.pt", tasks=None):
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = CopyLM(emb, hidden)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(1, steps + 1):
        loss = model(*batchify([make_example(rng, tasks) for _ in range(bs)]))
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % log_every == 0:
            msg = f"step {step:5d} loss {loss.item():.3f}"
            if step % eval_every == 0:
                msg += f" | exact-match {evaluate(model, 100, tasks=tasks)['overall']:.2%}"
            print(msg, flush=True)
    if out:
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        torch.save({"state": model.state_dict(), "config": {"E": emb, "H": hidden}}, out)
    return model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=192)
    ap.add_argument("--out", default="checkpoints/kalone.pt")
    ap.add_argument("--tasks", nargs="*", default=None, help=f"subset of {sorted(TASKS)}")
    a = ap.parse_args()
    model = train(a.steps, a.bs, a.lr, a.seed, a.emb, a.hidden, out=a.out, tasks=a.tasks)
    res = evaluate(model, 300, tasks=a.tasks)
    print("\nfinal exact-match:", {k: f"{v:.2%}" for k, v in res.items()})


if __name__ == "__main__":
    main()
