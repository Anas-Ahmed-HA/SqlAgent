"""
JWT authentication and per-tenant memory scoping.

Tokens from the gateway are **decoded**, not signature-verified: the claims
(`UserHRCode`, `Email`, `exp`/`Expiration`) are validated here, and the real
signature check happens at the gateway on the outbound credential fetch
(`/apps/models`), so a forged token cannot obtain Azure credentials; it can
only use the agent stack once a legitimate user has built it. **Keep this
service on localhost / the internal network** -- a well-formed forged token
is accepted at this boundary.

Both expiry styles are supported and genuinely checked: the standard `exp`
claim and the custom `Expiration` claim (epoch seconds or ISO-8601,
including .NET 7-digit fractions). `HTTPException` is never swallowed into a
500, and bearer tokens are logged as a short fingerprint unless
`LOG_RAW_AUTH_TOKEN=1` opts into raw logging for debugging.

Claim names and toggles come from `AuthSettings.from_env()`, read from the
current environment on every `verify_token`/`auth_log_fields` call (or passed
explicitly), so no module reload is needed to pick up new configuration.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Annotated, Any

import jwt
import httpx
from fastapi import Header, HTTPException
from jwt import InvalidTokenError
from pydantic import BaseModel, Field

from shared.settings import EMPLOYEE_API_URL

from shared.identity import (
    find_claim as _find_claim,
    parse_expiry as _parse_expiry,
    token_fingerprint,
)

__all__ = [
    "AuthSettings",
    "TokenClaims",
    "auth_log_fields",
    "require_token",
    "token_fingerprint",
    "verify_token",
]

logger = logging.getLogger("sql_agent.auth")

#: How long a successful employee check is reused for the same token, so a
#: client polling (upload progress, every 250 ms) does not ask the employee
#: service each time. Never past the token's own expiry; a failure is not kept.
EMPLOYEE_CHECK_TTL_SECONDS = 300.0
_MAX_VERIFIED = 1024
#: Digest of (employee service, HR code, token) -> when the check expires.
_verified: dict[str, float] = {}

#: No JWT_SECRET_KEY is read anywhere in this module: tokens are decoded
#: (base64) and their claims validated -- see the module docstring for the
#: trust model and the deployment caveat that goes with it.


@dataclass(frozen=True)
class AuthSettings:
    """Claim names and verification toggles for gateway-issued JWTs.

    Read from the environment per verification call via `from_env` (never at
    import time), so tests can use `monkeypatch.setenv` and production picks
    up the process environment on every request without a module reload.
    """

    hr_code_claim: str = "UserHRCode"
    """Claim carrying the identity memory is scoped to. Lookup is
    case-insensitive, so `userhrcode` and `USERHRCODE` both resolve."""

    email_claim: str = "Email"
    """Carried through for logging and display. Never used as a memory key --
    an email address can be reassigned, an HR code is the stable identity."""

    log_raw_token: bool = False
    employee_api_url: str = EMPLOYEE_API_URL
    """Opt-in raw-token logging for debugging. Off by default so bearer
    tokens are never written to logs; see `auth_log_fields`."""

    @classmethod
    def from_env(cls) -> AuthSettings:
        """Build settings from the current process environment."""
        hr_code_claim = os.environ.get(
            "JWT_HR_CODE_CLAIM",
            # Accepts the older variable name so an existing .env keeps working.
            os.environ.get("JWT_HT_CODE_CLAIM", "UserHRCode"),
        )
        email_claim = os.environ.get("JWT_EMAIL_CLAIM", "Email")
        log_raw_token = os.environ.get("LOG_RAW_AUTH_TOKEN", "0").lower() in {
            "1",
            "true",
            "yes",
        }
        return cls(
            hr_code_claim=hr_code_claim,
            email_claim=email_claim,
            log_raw_token=log_raw_token,
            employee_api_url=os.environ.get("EMPLOYEE_VERIFICATION_URL", cls.employee_api_url),
        )


class TokenClaims(BaseModel):
    """Validated claims, plus the memory scope derived from them."""

    hr_code: str
    """`UserHRCode`. The identity everything is keyed on."""

    email: str | None = None
    """`Email`. Available for logging and display, deliberately not part of
    the memory key -- see `memory_scope`."""

    subject: str | None = None
    expires_at: datetime | None = None
    """Whichever of `exp` / `Expiration` applies. See `_parse_expiry`."""

    raw: dict[str, Any] = Field(default_factory=dict)

    def memory_scope(self, client_session_id: str | None) -> str:
        """The key conversation history is stored under.

        **This is the part that matters for isolation.** `session_id` arrives
        from the client and is otherwise unauthenticated, so if it were used
        directly as the memory key, anyone holding any valid token could read
        another tenant's conversation by guessing or replaying their session
        id — and conversations contain query results.

        Namespacing under `hr_code` makes cross-user collision impossible
        while still letting one person run several parallel conversations.

        **Email is not used here on purpose.** An HR code identifies a person
        for the life of their employment; an email address can be changed,
        reassigned after someone leaves, or differ in case between issuers.
        Keying history on it would either split one person's history in two or
        hand it to whoever inherits the address.
        """
        from shared.identity import conversation_key

        return conversation_key(self.hr_code, client_session_id)


def auth_log_fields(
    token: str | None,
    claims: TokenClaims | None,
    settings: AuthSettings | None = None,
) -> dict[str, Any]:
    """Build authentication fields, with raw-token logging opt-in only."""
    active = settings if settings is not None else AuthSettings.from_env()
    fields: dict[str, Any] = {
        "token": token_fingerprint(token) if token else "-",
        "decoded_token": claims.raw if claims is not None else {},
    }
    if active.log_raw_token and token:
        fields["raw_token"] = token
    return fields


def verify_token(
    authorization: str | None, settings: AuthSettings | None = None
) -> TokenClaims:
    """Validates a `Bearer <jwt>` header and returns its claims.

    Raises `HTTPException(401)` for anything the caller can fix by presenting a
    better token. The signature is deliberately NOT verified -- see the module
    docstring for the trust model.

    `settings` defaults to `AuthSettings.from_env()`, read from the current
    environment on every call so no module reload is needed to pick up new
    configuration.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401, detail="Authorization token is missing or invalid"
        )

    # split(" ", 1) rather than split(" ")[1]: a header with stray whitespace
    # would otherwise silently truncate the token and surface as "invalid".
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(
            status_code=401, detail="Authorization token is missing or invalid"
        )

    handle = token_fingerprint(token)
    active = settings if settings is not None else AuthSettings.from_env()

    try:
        # Decode without verifying a signature: no key is held in this process
        # and the gateway is the boundary that actually checks signatures.
        # PyJWT still raises InvalidTokenError for structurally-broken tokens
        # (bad base64, wrong segment count), which is all we care about here.
        payload = jwt.decode(token, options={"verify_signature": False})
    except InvalidTokenError:
        # Broader than DecodeError: any structurally malformed token. A token
        # whose *signature* is wrong but whose claims are well-formed decodes
        # fine here -- that is the point of the trust model.
        logger.info("auth.invalid", extra={"token": handle})
        raise HTTPException(status_code=401, detail="Token is invalid") from None

    # PyJWT skips `exp` validation when the signature is not verified, so it
    # is checked explicitly here (the custom `Expiration` claim is below).
    raw_standard = payload.get("exp")
    standard_exp = _parse_expiry(raw_standard)
    if "exp" in payload and standard_exp is None:
        logger.info("auth.invalid_expiry", extra={"token": handle, "claim": "exp"})
        raise HTTPException(status_code=401, detail="Token expiration is invalid")
    if standard_exp is not None and standard_exp <= datetime.now(timezone.utc):
        logger.info("auth.expired", extra={"token": handle, "claim": "exp"})
        raise HTTPException(status_code=401, detail="Token has expired") from None
    raw_custom = _find_claim(payload, "Expiration")
    custom_present = any(key.lower() == "expiration" for key in payload)
    custom_exp = _parse_expiry(raw_custom)
    expires_at = standard_exp or custom_exp

    if custom_present and custom_exp is None:
        logger.info(
            "auth.invalid_expiry", extra={"token": handle, "claim": "Expiration"}
        )
        raise HTTPException(status_code=401, detail="Token expiration is invalid")

    # Token expiry is mandatory and cannot be disabled: an unexpiring bearer
    # token is a permanent credential.
    if expires_at is None:
        logger.info("auth.no_expiry", extra={"token": handle})
        raise HTTPException(
            status_code=401, detail="Token does not have an expiration date"
        )

    # With the signature unverified, every expiry check is ours: `exp` above,
    # the custom `Expiration` claim here.
    if custom_exp is not None and custom_exp <= datetime.now(timezone.utc):
        logger.info("auth.expired", extra={"token": handle, "claim": "Expiration"})
        raise HTTPException(status_code=401, detail="Token has expired")

    hr_code = _find_claim(payload, active.hr_code_claim)
    if hr_code is None or not str(hr_code).strip():
        # Naming the available claims (not their values) turns a mystery 401
        # into a two-minute fix when the issuer uses different casing.
        logger.warning(
            "auth.missing_hr_code",
            extra={"token": handle, "claims": ",".join(sorted(payload))},
        )
        raise HTTPException(
            status_code=401,
            detail=(
                f"Token does not carry a '{active.hr_code_claim}' claim. "
                f"Claims present: {', '.join(sorted(payload))}. "
                "Set JWT_HR_CODE_CLAIM if your issuer uses a different name."
            ),
        )

    email = _find_claim(payload, active.email_claim)
    if email is None or not str(email).strip():
        # Required, same reasoning as the HR code: it feeds the gateway request
        # and the audit trail, and a token without it is not one of ours.
        logger.warning(
            "auth.missing_email",
            extra={"token": handle, "claims": ",".join(sorted(payload))},
        )
        raise HTTPException(
            status_code=401,
            detail=(
                f"Token does not carry a '{active.email_claim}' claim. "
                f"Claims present: {', '.join(sorted(payload))}."
            ),
        )

    claims = TokenClaims(
        hr_code=str(hr_code).strip(),
        email=str(email).strip(),
        subject=str(payload.get("sub")) if payload.get("sub") is not None else None,
        expires_at=expires_at,
        raw=payload,
    )
    logger.info(
        "auth.ok",
        extra={
            **auth_log_fields(token, claims, active),
            "hr_code": claims.hr_code,
            "email": claims.email or "-",
        },
    )
    return claims


async def require_token(
    authorization: Annotated[str | None, Header()] = None,
) -> TokenClaims:
    """FastAPI dependency. Declaring `authorization` as a `Header` makes the
    requirement visible in `/docs` and in the OpenAPI schema, so callers can
    see the endpoint is protected without reading the source."""
    claims = verify_token(authorization)
    settings = AuthSettings.from_env()
    token = authorization.split(" ", 1)[1].strip() if authorization else ""
    try:
        return await verify_employee_identity(claims, token, settings)
    except HTTPException:
        raise
    except (httpx.HTTPError, ValueError, TypeError, KeyError):
        logger.exception("auth.employee_verification_error", extra={"token": token_fingerprint(token)})
        raise HTTPException(status_code=503, detail="Employee verification service unavailable") from None


async def verify_employee_identity(
    claims: TokenClaims, token: str, settings: AuthSettings | None = None
) -> TokenClaims:
    """Verify the bearer token against the HR employee service.

    A success is reused for :data:`EMPLOYEE_CHECK_TTL_SECONDS` (at most until
    the token expires).
    """
    from agent.clients.employees import verify_employee_hr

    active = settings or AuthSettings.from_env()
    key = hashlib.sha256(
        "\n".join([active.employee_api_url or "", claims.hr_code, token]).encode("utf-8")
    ).hexdigest()
    now = time.monotonic()
    if _verified.get(key, 0.0) > now:
        return claims
    await verify_employee_hr(claims.hr_code, token, active.employee_api_url)
    _remember_verified(key, claims, now)
    return claims


def _remember_verified(key: str, claims: TokenClaims, now: float) -> None:
    ttl = EMPLOYEE_CHECK_TTL_SECONDS
    if claims.expires_at is not None:
        left = (claims.expires_at - datetime.now(timezone.utc)).total_seconds()
        ttl = min(ttl, left)
    if ttl <= 0:
        return
    if len(_verified) >= _MAX_VERIFIED:
        for stale in [k for k, until in _verified.items() if until <= now]:
            del _verified[stale]
        while len(_verified) >= _MAX_VERIFIED:
            del _verified[next(iter(_verified))]
    _verified[key] = now + ttl
