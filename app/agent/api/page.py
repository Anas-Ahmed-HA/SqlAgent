"""The test chat page: one HTML file served at ``/`` for trying the agent.

It talks to the same routes any client uses, from the same origin, so the
service needs no CORS settings. The page holds no secret: the user pastes
their own login token into it.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

__all__ = ["router"]

router = APIRouter()

_PAGE = Path(__file__).with_name("chat.html")


@router.get("/", response_class=HTMLResponse, include_in_schema=False)
async def chat_page() -> HTMLResponse:
    # Read per request so an edit to the page shows on reload.
    return HTMLResponse(_PAGE.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store"})
