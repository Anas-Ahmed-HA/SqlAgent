"""Skills as a chat turn uses them: the safety check, the equipped limits,
the prompt message, and the AI rewrite of a user's wording into a skill.

Copied from the turn side of AI_RevitMCP/app/skills_hub (budget, prompt,
rewrite, safety). Skills live in the cloud skills API; the client reads and
saves them there and sends the equipped ones with each turn. Keep the safety
pattern identical to ``SkillSafety.IsSafe`` in API_AIChatting.

Imports nothing from the app around it, only ``pydantic``.
"""

from __future__ import annotations

from skills_hub.budget import check_equipped, fit_to_budget, word_count
from skills_hub.prompt import build_equipped_skills_message
from skills_hub.rewrite import (
    SKILL_REWRITE_PROMPT,
    SkillProposal,
    UnclearProposal,
    parse_proposal,
    skill_rewrite_prompt,
)
from skills_hub.safety import skill_text_is_safe

__all__ = [
    "SKILL_REWRITE_PROMPT",
    "SkillProposal",
    "UnclearProposal",
    "build_equipped_skills_message",
    "check_equipped",
    "fit_to_budget",
    "parse_proposal",
    "skill_rewrite_prompt",
    "skill_text_is_safe",
    "word_count",
]
