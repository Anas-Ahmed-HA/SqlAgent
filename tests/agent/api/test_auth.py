"""
Authentication and tenant-isolation tests.

Two of these encode the bugs found in the original `verify_token`, so a future
"simplification" back to `payload.get("Expiration")` fails the suite rather
than production.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agent.api.auth import verify_employee_identity as real_employee_check

SECRET = "test-secret-key-at-least-32-bytes-long!!"


@pytest.fixture(autouse=True)
def _auth_env(monkeypatch):
    """Auth settings are read per verification call, so plain monkeypatching
    is enough -- no module reload.

    No JWT_SECRET_KEY is set: decode-only auth ignores signatures entirely,
    so the SECRET constant below is only a signer for token-shaped fixtures
    and any other string would work just as well."""
    monkeypatch.setenv("JWT_HR_CODE_CLAIM", "UserHRCode")
    async def verified(claims, token, settings=None):
        return claims
    monkeypatch.setattr("agent.api.auth.verify_employee_identity", verified)
    yield


def make_token(**claims) -> str:
    """Mirrors the real token shape: Email, Expiration, UserHRCode, exp, nbf."""
    payload = {
        "Email": "farouk@example.com",
        "UserHRCode": "HR-100",
        **claims,
    }
    payload.setdefault("exp", datetime.now(timezone.utc) + timedelta(hours=1))
    return jwt.encode(payload, SECRET, algorithm="HS256")


def bearer(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def build_client(azure, mcp):
    from agent.config import AgentConfig
    from agent.loop import AgentLoop
    from agent.memory.store import MemoryStore
    from agent.host.runtime import Runtime
    from agent.api.chat import router as chat_router
    from agent.api.skills import router as skills_router
    from agent.api.status import router as status_router

    memory = MemoryStore()

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = Runtime(
            mcp=mcp,
            azure=azure,
            memory=memory,
            agent=AgentLoop(mcp=mcp, azure=azure, memory=memory, config=AgentConfig()),
            creds_source="env",
        )
        yield

    application = FastAPI(lifespan=lifespan)
    application.include_router(status_router)
    application.include_router(skills_router)
    application.include_router(chat_router)
    return TestClient(application), memory


# -- verify_token unit tests ------------------------------------------------


def test_valid_standard_exp_token_is_accepted():
    """REGRESSION: the original checked `payload.get("Expiration")`, which is
    None for a standard `exp` token -- so it rejected every valid token."""
    from agent.api.auth import verify_token

    claims = verify_token(f"Bearer {make_token()}")
    assert claims.hr_code == "HR-100"
    assert claims.expires_at is not None


def test_past_custom_expiration_is_rejected():
    """REGRESSION: the original only checked the claim *existed*, so a token
    whose custom `Expiration` was thirty days ago was accepted."""
    from fastapi import HTTPException

    from agent.api.auth import verify_token

    stale = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    token = jwt.encode(
        {
            "Email": "farouk@example.com",
            "UserHRCode": "HR-100",
            "Expiration": stale,
        },
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        verify_token(f"Bearer {token}")
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail.lower()


def test_future_custom_expiration_is_accepted():
    from agent.api.auth import verify_token

    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    token = jwt.encode(
        {
            "Email": "farouk@example.com",
            "UserHRCode": "HR-100",
            "Expiration": future,
        },
        SECRET,
        algorithm="HS256",
    )
    assert verify_token(f"Bearer {token}").hr_code == "HR-100"


def test_missing_expiry_is_rejected():
    from fastapi import HTTPException

    from agent.api.auth import verify_token

    token = jwt.encode(
        {"Email": "farouk@example.com", "UserHRCode": "HR-100"},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        verify_token(f"Bearer {token}")
    assert exc.value.status_code == 401


def test_no_expiry_401_is_not_converted_to_500():
    """REGRESSION: `except Exception` sat below a `raise HTTPException` in the
    same try block, so the 401 became a 500 with the message embedded."""
    from fastapi import HTTPException

    from agent.api.auth import verify_token

    token = jwt.encode(
        {"Email": "farouk@example.com", "UserHRCode": "HR-100"},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        verify_token(f"Bearer {token}")
    assert exc.value.status_code == 401, "must not be swallowed into a 500"
    assert "Internal Server Error" not in str(exc.value.detail)


def test_signature_is_deliberately_not_verified():
    """Decode-only auth: a token signed with ANY key is accepted as long as
    the claims (UserHRCode, Email, exp) are valid. Signature checking is the
    gateway's job on the outbound credential fetch -- see agent/api/auth.py."""
    from agent.api.auth import verify_token

    token = jwt.encode(
        {
            "Email": "farouk@example.com",
            "UserHRCode": "HR-100",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        "a-completely-different-signing-key-value",
        algorithm="HS256",
    )
    assert verify_token(f"Bearer {token}").hr_code == "HR-100"


def test_missing_email_is_rejected():
    """Email is required alongside UserHRCode: it feeds the gateway request
    and the audit trail."""
    from fastapi import HTTPException

    from agent.api.auth import verify_token

    token = jwt.encode(
        {"UserHRCode": "HR-100", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        verify_token(f"Bearer {token}")
    assert exc.value.status_code == 401
    assert "Email" in exc.value.detail


@pytest.mark.parametrize(
    "header", [None, "", "Bearer", "Bearer ", "Basic abc", "abc", "bearer x"]
)
def test_malformed_headers_are_rejected(header):
    from fastapi import HTTPException

    from agent.api.auth import verify_token

    with pytest.raises(HTTPException) as exc:
        verify_token(header)
    assert exc.value.status_code == 401


def test_hr_code_claim_lookup_is_case_insensitive():
    from agent.api.auth import verify_token

    token = jwt.encode(
        {
            "Email": "farouk@example.com",
            "userhrcode": "HR-XYZ",
            "exp": datetime.now(timezone.utc) + timedelta(hours=1),
        },
        SECRET,
        algorithm="HS256",
    )
    assert verify_token(f"Bearer {token}").hr_code == "HR-XYZ"


def test_missing_hr_code_names_the_available_claims():
    from fastapi import HTTPException

    from agent.api.auth import verify_token

    token = jwt.encode(
        {"tenant": "HR-1", "exp": datetime.now(timezone.utc) + timedelta(hours=1)},
        SECRET,
        algorithm="HS256",
    )
    with pytest.raises(HTTPException) as exc:
        verify_token(f"Bearer {token}")
    assert "tenant" in exc.value.detail, "should say which claims ARE present"


def test_no_jwt_secret_key_is_needed_at_all(monkeypatch):
    """Decode-only auth: with JWT_SECRET_KEY unset (the shipped .env no longer
    has it), a token with valid claims still verifies. There is no 500 and no
    misconfiguration path -- there is no key left to misconfigure."""
    from agent.api.auth import verify_token

    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    claims = verify_token(f"Bearer {make_token()}")
    assert claims.hr_code == "HR-100"


# -- memory scoping ---------------------------------------------------------


def test_memory_scope_namespaces_under_hr_code():
    from agent.api.auth import TokenClaims

    a = TokenClaims(hr_code="HR-A")
    b = TokenClaims(hr_code="HR-B")
    assert a.memory_scope("shared") != b.memory_scope("shared")
    assert a.memory_scope("shared").startswith("HR-A|")


def test_memory_scope_handles_missing_session_id():
    from agent.api.auth import TokenClaims

    assert TokenClaims(hr_code="HR-A").memory_scope(None) == "HR-A|default"


def test_memory_scope_strips_separator_injection():
    """Without this, session_id "X|HR-B|y" could forge another tenant's key."""
    from agent.api.auth import TokenClaims

    scope = TokenClaims(hr_code="HR-A").memory_scope("X|HR-B|y")
    assert scope.count("|") == 1


# -- endpoint behaviour -----------------------------------------------------


def test_chat_requires_a_token():
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    client, _ = build_client(FakeAzure([text_turn("hi")]), FakeMCP())
    with client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 401


def test_chat_accepts_a_valid_token():
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    client, _ = build_client(FakeAzure([text_turn("42 laptops.")]), FakeMCP())
    with client:
        response = client.post(
            "/chat", json={"message": "count laptops"}, headers=bearer(make_token())
        )
    assert response.status_code == 200
    assert response.json()["reply"] == "42 laptops."


def test_expired_token_is_rejected_at_the_endpoint():
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    expired = jwt.encode(
        {"UserHRCode": "HR-1", "exp": datetime.now(timezone.utc) - timedelta(minutes=5)},
        SECRET,
        algorithm="HS256",
    )
    client, _ = build_client(FakeAzure([text_turn("hi")]), FakeMCP())
    with client:
        response = client.post(
            "/chat", json={"message": "hi"}, headers=bearer(expired)
        )
    assert response.status_code == 401
    assert "expired" in response.json()["detail"].lower()


def test_memory_is_stored_under_the_hr_code_scope():
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    client, memory = build_client(FakeAzure([text_turn("ok")]), FakeMCP())
    with client:
        client.post(
            "/chat",
            json={"message": "remember this", "session_id": "s1"},
            headers=bearer(make_token(UserHRCode="HR-777")),
        )
    assert "HR-777|s1" in memory._sessions
    assert "s1" not in memory._sessions


def test_two_tenants_sharing_a_session_id_stay_isolated():
    """The reason memory_scope exists: session_id is client-supplied and
    otherwise unauthenticated, so it must not be the memory key on its own."""
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    azure = FakeAzure([text_turn("a"), text_turn("b"), text_turn("c")])
    client, _ = build_client(azure, FakeMCP())
    with client:
        client.post(
            "/chat",
            json={"message": "TENANT A SECRET", "session_id": "same"},
            headers=bearer(make_token(UserHRCode="HR-A")),
        )
        client.post(
            "/chat",
            json={"message": "hello", "session_id": "same"},
            headers=bearer(make_token(UserHRCode="HR-B")),
        )

    tenant_b_prompt = azure.calls[1]["messages"]
    contents = [m.get("content") for m in tenant_b_prompt]
    assert "TENANT A SECRET" not in contents


def test_same_tenant_different_sessions_are_separate():
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    azure = FakeAzure([text_turn("a"), text_turn("b")])
    client, memory = build_client(azure, FakeMCP())
    token = make_token(UserHRCode="HR-SAME")
    with client:
        client.post(
            "/chat",
            json={"message": "one", "session_id": "conv1"},
            headers=bearer(token),
        )
        client.post(
            "/chat",
            json={"message": "two", "session_id": "conv2"},
            headers=bearer(token),
        )
    assert {"HR-SAME|conv1", "HR-SAME|conv2"} <= set(memory._sessions)


def test_json_endpoint_requires_a_token():
    """Both chat routes enforce the same authentication policy."""
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    client, _ = build_client(FakeAzure([text_turn("hi")]), FakeMCP())
    with client:
        assert client.post("/chat", json={"message": "hi"}).status_code == 401


def test_json_endpoint_rejects_a_bad_token():
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    client, _ = build_client(FakeAzure([text_turn("hi")]), FakeMCP())
    with client:
        response = client.post(
            "/chat", json={"message": "hi"}, headers=bearer("not-a-jwt")
        )
    assert response.status_code == 401


def test_health_stays_unauthenticated():
    from tests.agent.fakes import FakeAzure, FakeMCP

    client, _ = build_client(FakeAzure(), FakeMCP())
    with client:
        assert client.get("/health").status_code == 200


@pytest.mark.parametrize("claim", ["exp", "Expiration"])
@pytest.mark.parametrize(
    "invalid_value", [None, 10**1000, float("inf"), float("-inf"), float("nan")]
)
def test_invalid_expiry_values_return_401(claim, invalid_value):
    """Out-of-range and non-finite timestamps are invalid credentials, not 500s."""
    from tests.agent.fakes import FakeAzure, FakeMCP, text_turn

    future = datetime.now(timezone.utc) + timedelta(hours=1)
    payload = {
        "Email": "farouk@example.com",
        "UserHRCode": "HR-100",
        "exp": future.timestamp(),
        "Expiration": future.isoformat(),
        claim: invalid_value,
    }
    token = jwt.encode(payload, SECRET, algorithm="HS256")
    client, _ = build_client(FakeAzure([text_turn("hi")]), FakeMCP())

    with client:
        response = client.post(
            "/chat", json={"message": "hi"}, headers=bearer(token)
        )

    assert response.status_code == 401
    assert response.json()["detail"] == "Token expiration is invalid"


def _employee_service(monkeypatch, asked):
    from fastapi import HTTPException

    from agent.api import auth

    monkeypatch.setattr(auth, "_verified", {})

    async def check(hr_code, token, url):
        asked.append(token)
        if token == "bad":
            raise HTTPException(status_code=401, detail="Employee verification failed")

    monkeypatch.setattr("agent.clients.employees.verify_employee_hr", check)


def test_a_successful_employee_check_is_reused_but_a_failure_is_not(monkeypatch):
    """A client polling the service often (every 250 ms, say): the employee service
    must not be asked on each poll."""
    import asyncio

    from fastapi import HTTPException

    from agent.api.auth import TokenClaims

    asked: list[str] = []
    _employee_service(monkeypatch, asked)
    claims = TokenClaims(hr_code="HR-1", expires_at=datetime.now(timezone.utc) + timedelta(hours=1))

    async def scenario():
        for _ in range(5):
            await real_employee_check(claims, "good")
        for _ in range(2):
            with pytest.raises(HTTPException):
                await real_employee_check(claims, "bad")

    asyncio.run(scenario())
    assert asked == ["good", "bad", "bad"]


def test_an_employee_check_is_not_reused_past_the_tokens_expiry(monkeypatch):
    import asyncio

    from agent.api.auth import TokenClaims

    asked: list[str] = []
    _employee_service(monkeypatch, asked)
    expired = TokenClaims(hr_code="HR-1", expires_at=datetime.now(timezone.utc) - timedelta(seconds=1))

    async def scenario():
        await real_employee_check(expired, "t")
        await real_employee_check(expired, "t")

    asyncio.run(scenario())
    assert asked == ["t", "t"]
