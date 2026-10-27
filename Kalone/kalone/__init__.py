"""Kalone: a tiny research prototype of an LLM that can copy-paste instead of retyping."""

from .data import TASKS, Example, batchify, make_example, register_task, run_actions
from .model import CopyLM

__all__ = [
    "CopyLM",
    "Example",
    "TASKS",
    "batchify",
    "make_example",
    "register_task",
    "run_actions",
]
__version__ = "0.1.0"
