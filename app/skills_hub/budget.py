"""How many equipped skills, and how many words in them, fit in a turn."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, TypeVar

__all__ = ["check_equipped", "fit_to_budget", "word_count"]

T = TypeVar("T")


def _field(skill: Any, name: str) -> Any:
    return skill.get(name) if isinstance(skill, Mapping) else getattr(skill, name)


def word_count(skill: Any) -> int:
    """Words in a skill's name, description and instructions (a model or a dict)."""
    text = " ".join([
        str(_field(skill, "name") or ""),
        str(_field(skill, "description") or ""),
        *[str(item) for item in _field(skill, "instructions") or []],
    ])
    return len(text.split())


def check_equipped(skills: Iterable[Any], max_count: int, max_words: int) -> None:
    """Raise ``ValueError`` when the skills exceed either limit."""
    skills = list(skills)
    if len(skills) > max_count:
        raise ValueError(f"At most {max_count} skills may be equipped")
    words = sum(word_count(skill) for skill in skills)
    if words > max_words:
        raise ValueError(
            f"Equipped skills may contain at most {max_words} words (requested {words})"
        )


def fit_to_budget(skills: Iterable[T], max_count: int, max_words: int) -> tuple[list[T], list[T]]:
    """``(kept, dropped)``: skills kept in order until the next would pass a
    limit; that one and every one after it are dropped."""
    kept: list[T] = []
    dropped: list[T] = []
    words = 0
    for skill in skills:
        if dropped:
            dropped.append(skill)
            continue
        size = word_count(skill)
        if len(kept) + 1 > max_count or words + size > max_words:
            dropped.append(skill)
            continue
        kept.append(skill)
        words += size
    return kept, dropped
