import random

from kalone import TASKS, batchify, make_example, run_actions
from kalone.vocab import stoi


def test_registered_tasks():
    assert {"replace", "delete", "append"} <= set(TASKS)


def test_gold_actions_reproduce_target():
    """Every task's gold action list must execute to exactly its target text."""
    rng = random.Random(0)
    for name in TASKS:
        for _ in range(300):
            ex = TASKS[name](rng)
            assert run_actions(ex.inp, ex.acts) == ex.target, (name, ex)


def test_copy_spans_are_valid():
    rng = random.Random(1)
    for _ in range(500):
        ex = make_example(rng)
        for a in ex.acts:
            if a[0] == "copy":
                assert 0 <= a[1] <= a[2] < len(ex.inp)


def test_inputs_use_known_characters():
    rng = random.Random(2)
    for _ in range(200):
        assert all(c in stoi for c in make_example(rng).inp)


def test_batchify_shapes():
    rng = random.Random(3)
    exs = [make_example(rng) for _ in range(8)]
    x, ty, ch, st, en = batchify(exs)
    assert x.shape[0] == ty.shape[0] == 8
    assert ty.shape == ch.shape == st.shape == en.shape
