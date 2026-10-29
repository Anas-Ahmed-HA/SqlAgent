"""Skills adapter: the AI rewrite of a user's wording into a skill.

The agent keeps no skills. The client saves them to the cloud skills API
and sends the equipped ones with each chat turn (``ChatRequest.skills``,
checked in ``agent.host.turns``). The rewrite stays here because the model
does; parsing lives in ``skills_hub.rewrite``.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from agent.api.auth import TokenClaims, require_token
from shared.api_contract import Routes
from shared.api_schemas import SkillProposalRequest, SkillProposalResponse
from agent.host.bootstrap import NotReadyError, ensure_ready
from agent.clients.aiapi_credentials import GatewayCredentialsError
from agent.prompts.templates import SKILL_APP_CONTEXT
from skills_hub.rewrite import parse_proposal, skill_rewrite_prompt

__all__ = ["router", "propose_skill"]

logger = logging.getLogger("sql_agent.api")

router = APIRouter()


@router.post(Routes.SKILL_PROPOSE, response_model=SkillProposalResponse)
async def propose_skill(
    payload: SkillProposalRequest,
    request: Request,
    claims: TokenClaims = Depends(require_token),
) -> SkillProposalResponse:
    del claims
    bearer_token = request.headers.get("Authorization", "")[7:].strip()
    try:
        await ensure_ready(request.app, bearer_token)
        azure = request.app.state.runtime.azure
        if azure is None:
            raise NotReadyError("Agent is not ready")
        # Told it is a data assistant, the rewrite reads "views" as database views.
        raw = await azure.complete_text(skill_rewrite_prompt(SKILL_APP_CONTEXT), payload.source_text)
        # Nothing is remembered: the client saves the proposal itself.
        proposal = parse_proposal(raw)
    except (NotReadyError, GatewayCredentialsError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - return a useful API error
        logger.exception("skill.proposal_failed")
        raise HTTPException(status_code=502, detail=f"Could not rewrite skill: {exc}") from exc
    return SkillProposalResponse(proposal=proposal.model_dump())
