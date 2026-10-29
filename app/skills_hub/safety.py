"""The injection check every skill passes before it reaches a model.

Skill text is untrusted: a shared skill goes into other people's model
context. The skills API runs the same check on create and update
(``SkillSafety.IsSafe`` in API_AIChatting); keep the two patterns identical,
and their tests on the same example strings.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

__all__ = ["skill_text_is_safe"]

#: Injection phrasing, not single words: "show the system name" or "show
#: previous revisions" is everyday wording.
_UNSAFE_SKILL_TEXT = re.compile(
    # "ignore the previous instructions", "override your system prompt",
    # "disregard the instructions above"
    r"\b(?:ignore|disregard|forget|override|bypass)\b.{0,40}(?:"
    r"\b(?:previous|prior|above|earlier|system|developer|your)\b.{0,30}"
    r"\b(?:instructions?|prompts?|rules|messages?|guidelines)\b"
    r"|\b(?:instructions?|prompts?|rules|guidelines)\b.{0,20}\b(?:above|before|earlier|so\s+far)\b)"
    # "reveal the system prompt", "print your hidden instructions"
    r"|\b(?:reveal|show|print|repeat|output|leak|exfiltrat\w*|dump)\b.{0,40}"
    r"\b(?:system|developer|hidden|initial)\s+(?:prompts?|instructions?|messages?)\b"
    # "reveal the token", "send the credentials"
    r"|\b(?:reveal|show|print|output|leak|exfiltrat\w*|dump|send)\b.{0,60}"
    r"\b(?:secrets?|tokens?|credentials?|passwords?|api[ _-]?keys?|environment\s+variables?)\b"
    # "disable the safety checks", "bypass authentication"
    r"|\b(?:disable|bypass|override|turn\s+off)\b.{0,40}"
    r"\b(?:safety|guardrails?|authentication|validation|security)\b",
    re.IGNORECASE | re.DOTALL,
)


def skill_text_is_safe(name: str, description: str, instructions: Iterable[str]) -> bool:
    """Whether a skill's text is safe to place in model context."""
    text = "\n".join([name or "", description or "", *(instructions or [])])
    return _UNSAFE_SKILL_TEXT.search(text) is None
