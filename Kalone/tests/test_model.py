import random

import torch

from kalone import CopyLM, batchify, make_example


def tiny():
    torch.manual_seed(0)
    return CopyLM(E=16, H=32)


def test_forward_is_finite():
    rng = random.Random(0)
    loss = tiny()(*batchify([make_example(rng) for _ in range(8)]))
    assert torch.isfinite(loss)


def test_can_overfit_a_fixed_batch():
    rng = random.Random(0)
    batch = batchify([make_example(rng) for _ in range(16)])
    model = tiny()
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = model(*batch).item()
    for _ in range(60):
        loss = model(*batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5


def test_generate_returns_text_and_trace():
    ex = make_example(random.Random(0))
    text, trace = tiny().generate(ex.inp)
    assert isinstance(text, str) and isinstance(trace, list)
