"""Run a trained checkpoint.

    python -m kalone.demo --ckpt checkpoints/kalone.pt "abc def ghi|r1|xyz" "abc def|a"
"""
import argparse
import random

import torch

from .data import make_example
from .model import CopyLM


def load(path: str) -> CopyLM:
    ck = torch.load(path, map_location="cpu")
    cfg = ck.get("config", {})
    model = CopyLM(cfg.get("E", 64), cfg.get("H", 192))
    model.load_state_dict(ck["state"] if "state" in ck else ck)
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="checkpoints/kalone.pt")
    ap.add_argument("inputs", nargs="*", help="e.g. 'abc def ghi|r1|xyz' (r=replace, d=delete, a=append)")
    a = ap.parse_args()
    model = load(a.ckpt)
    inputs = a.inputs or [make_example(random.Random(i)).inp for i in range(5)]
    for inp in inputs:
        text, trace = model.generate(inp)
        print(f"input : {inp}\noutput: {text}\nsteps : {' '.join(trace)}\n")


if __name__ == "__main__":
    main()
