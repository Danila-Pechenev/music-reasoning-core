"""Procedural generators for music reasoning tasks.

The package mirrors the small public API commonly used with Reasoning Core:

    task = get_task("pitch_interval_reasoning")
    example = task.generate_example(level=5)
    score = task.score_answer(answer, example)
"""

from __future__ import annotations

import json
from typing import Any

from reasoning_core.template import Problem

from music_reasoning_tasks.chord_roman_reasoning import ChordRomanReasoning
from music_reasoning_tasks.pitch_interval_reasoning import PitchIntervalReasoning

__version__ = "0.3.2"

TASK_CLASSES = {
    "pitch_interval_reasoning": PitchIntervalReasoning,
    "chord_roman_reasoning": ChordRomanReasoning,
}


def _normalize_task_name(name: str) -> str:
    return name.replace("-", "_").replace(" ", "_").lower()


def list_tasks() -> list[str]:
    """Return the available music reasoning task families."""
    return sorted(TASK_CLASSES)


def get_task(name: str, *args: Any, **kwargs: Any):
    """Instantiate a music reasoning task by name."""
    normalized = _normalize_task_name(name)
    if normalized not in TASK_CLASSES:
        known = ", ".join(list_tasks())
        raise KeyError(f"Unknown music reasoning task {name!r}. Available tasks: {known}.")
    return TASK_CLASSES[normalized](*args, **kwargs)


def _metadata_from_entry(entry: Any) -> dict[str, Any]:
    metadata = getattr(entry, "metadata", None)
    if metadata is None and isinstance(entry, dict):
        metadata = entry.get("metadata", {})
    if isinstance(metadata, str):
        metadata = json.loads(metadata)
    return dict(metadata or {})


def score_answer(answer: object, entry: Problem | dict[str, Any]) -> float:
    """Score an answer using the task family recorded in a generated example."""
    metadata = _metadata_from_entry(entry)
    task_name = metadata.get("_task") or metadata.get("task")
    if task_name is None and isinstance(entry, dict):
        task_name = entry.get("task")
    if task_name is None:
        raise KeyError("Cannot infer task name from entry metadata.")
    task = get_task(str(task_name))
    if isinstance(entry, dict):
        metadata = json.loads(entry["metadata"]) if isinstance(entry.get("metadata"), str) else entry.get("metadata", {})
        entry = Problem(metadata=metadata, answer=entry.get("answer"), cot=entry.get("cot"))
    return task.score_answer(answer, entry)


__all__ = [
    "ChordRomanReasoning",
    "PitchIntervalReasoning",
    "get_task",
    "list_tasks",
    "score_answer",
]
