from __future__ import annotations

from typing import Any

__all__ = ["summarize_runs", "evaluate_by_condition"]


def __getattr__(name: str) -> Any:
    if name == "summarize_runs":
        from .summarize import summarize_runs

        return summarize_runs
    if name == "evaluate_by_condition":
        from .condition_eval import evaluate_by_condition

        return evaluate_by_condition
    raise AttributeError(name)
