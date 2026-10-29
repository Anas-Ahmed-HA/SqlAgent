"""HR employee verification boundary.

Owns the outbound employee lookup used to confirm a bearer token belongs to
its HR code. HTTP and matching rules live here; authentication flow lives in
``agent.api.auth``.
"""

from __future__ import annotations

import logging

import httpx
from fastapi import HTTPException

from shared.identity import (
    employee_hr_matches,
    normalize_employee_records,
    token_fingerprint,
)

__all__ = ["verify_employee_hr"]

logger = logging.getLogger("sql_agent.employee")


async def verify_employee_hr(hr_code: str, token: str, employee_api_url: str) -> None:
    """Confirm the employee service knows ``hr_code`` for ``token``.

    Raises ``HTTPException`` 401 when verification fails and lets transport
    errors propagate for callers to map to 503.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            employee_api_url,
            params={"userhrcode": hr_code},
            headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        )
    if response.status_code != 200:
        logger.warning(
            "auth.employee_verification_failed",
            extra={"token": token_fingerprint(token), "status": response.status_code},
        )
        raise HTTPException(status_code=401, detail="Employee verification failed")
    records = normalize_employee_records(response.json())
    if not records:
        raise HTTPException(status_code=401, detail="Employee was not found")
    if not employee_hr_matches(records, hr_code):
        logger.warning(
            "auth.employee_hr_mismatch", extra={"token": token_fingerprint(token)}
        )
        raise HTTPException(status_code=401, detail="Employee HR code does not match token")
    logger.info(
        "token is verified",
        extra={"token": token_fingerprint(token), "hr_code": hr_code},
    )
