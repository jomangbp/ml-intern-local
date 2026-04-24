"""Authentication routes for HF OAuth.

Handles the OAuth 2.0 authorization code flow with HF as provider.
After successful auth, sets an HttpOnly cookie with the access token.
"""

import os
import secrets
import time
from urllib.parse import urlencode

import httpx
from dependencies import AUTH_ENABLED, check_org_membership, get_current_user
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from litellm import acompletion

from agent.core.llm_params import _resolve_llm_params
from agent.tools.codex_tool import codex_auth_status, codex_login_handler
from session_manager import session_manager

router = APIRouter(prefix="/auth", tags=["auth"])

# OAuth configuration from environment
OAUTH_CLIENT_ID = os.environ.get("OAUTH_CLIENT_ID", "")
OAUTH_CLIENT_SECRET = os.environ.get("OAUTH_CLIENT_SECRET", "")
OPENID_PROVIDER_URL = os.environ.get("OPENID_PROVIDER_URL", "https://huggingface.co")

# In-memory OAuth state store with expiry (5 min TTL)
_OAUTH_STATE_TTL = 300
oauth_states: dict[str, dict] = {}


def _cleanup_expired_states() -> None:
    """Remove expired OAuth states to prevent memory growth."""
    now = time.time()
    expired = [k for k, v in oauth_states.items() if now > v.get("expires_at", 0)]
    for k in expired:
        del oauth_states[k]


def get_redirect_uri(request: Request) -> str:
    """Get the OAuth callback redirect URI."""
    # In HF Spaces, use the SPACE_HOST if available
    space_host = os.environ.get("SPACE_HOST")
    if space_host:
        return f"https://{space_host}/auth/callback"
    # Otherwise construct from request
    return str(request.url_for("oauth_callback"))


@router.get("/login")
async def oauth_login(request: Request) -> RedirectResponse:
    """Initiate OAuth login flow."""
    if not OAUTH_CLIENT_ID:
        raise HTTPException(
            status_code=500,
            detail="OAuth not configured. Set OAUTH_CLIENT_ID environment variable.",
        )

    # Clean up expired states to prevent memory growth
    _cleanup_expired_states()

    # Generate state for CSRF protection
    state = secrets.token_urlsafe(32)
    oauth_states[state] = {
        "redirect_uri": get_redirect_uri(request),
        "expires_at": time.time() + _OAUTH_STATE_TTL,
    }

    # Build authorization URL
    params = {
        "client_id": OAUTH_CLIENT_ID,
        "redirect_uri": get_redirect_uri(request),
        "scope": "openid profile read-repos write-repos contribute-repos manage-repos inference-api jobs write-discussions",
        "response_type": "code",
        "state": state,
        "orgIds": os.environ.get(
            "HF_OAUTH_ORG_ID", "698dbf55845d85df163175f1"
        ),  # ml-agent-explorers
    }
    auth_url = f"{OPENID_PROVIDER_URL}/oauth/authorize?{urlencode(params)}"

    return RedirectResponse(url=auth_url)


@router.get("/callback")
async def oauth_callback(
    request: Request, code: str = "", state: str = ""
) -> RedirectResponse:
    """Handle OAuth callback."""
    # Verify state
    if state not in oauth_states:
        raise HTTPException(status_code=400, detail="Invalid state parameter")

    stored_state = oauth_states.pop(state)
    redirect_uri = stored_state["redirect_uri"]

    if not code:
        raise HTTPException(status_code=400, detail="No authorization code provided")

    # Exchange code for token
    token_url = f"{OPENID_PROVIDER_URL}/oauth/token"
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(
                token_url,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                    "client_id": OAUTH_CLIENT_ID,
                    "client_secret": OAUTH_CLIENT_SECRET,
                },
            )
            response.raise_for_status()
            token_data = response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"Token exchange failed: {e}")

    # Get user info
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(
            status_code=500,
            detail="Token exchange succeeded but no access_token was returned.",
        )

    # Fetch user info (optional — failure is not fatal)
    async with httpx.AsyncClient() as client:
        try:
            userinfo_response = await client.get(
                f"{OPENID_PROVIDER_URL}/oauth/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            userinfo_response.raise_for_status()
        except httpx.HTTPError:
            pass  # user_info not required for auth flow

    # Set access token as HttpOnly cookie (not in URL — avoids leaks via
    # Referrer headers, browser history, and server logs)
    is_production = bool(os.environ.get("SPACE_HOST"))
    response = RedirectResponse(url="/", status_code=302)
    response.set_cookie(
        key="hf_access_token",
        value=access_token,
        httponly=True,
        secure=is_production,  # Secure flag only in production (HTTPS)
        samesite="lax",
        max_age=3600 * 24 * 7,  # 7 days
        path="/",
    )
    return response


@router.get("/logout")
async def logout() -> RedirectResponse:
    """Log out the user by clearing the auth cookie."""
    response = RedirectResponse(url="/")
    response.delete_cookie(key="hf_access_token", path="/")
    return response


@router.get("/status")
async def auth_status() -> dict:
    """Check if OAuth is enabled on this instance."""
    return {"auth_enabled": AUTH_ENABLED}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)) -> dict:
    """Get current user info. Returns the authenticated user or dev user.

    Uses the shared auth dependency which handles cookie + Bearer token.
    """
    return user


ORG_NAME = "ml-agent-explorers"


@router.get("/org-membership")
async def org_membership(
    request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Check if the authenticated user belongs to the ml-agent-explorers org."""
    if not AUTH_ENABLED:
        return {"is_member": True}
    token = request.cookies.get("hf_access_token") or ""
    if not token:
        return {"is_member": False}
    is_member = await check_org_membership(token, ORG_NAME)
    return {"is_member": is_member}


@router.get("/codex/status")
async def codex_status(user: dict = Depends(get_current_user)) -> dict:
    """Get Codex CLI OAuth status for the current machine."""
    return codex_auth_status()


@router.post("/codex/login")
async def codex_login(body: dict | None = None, user: dict = Depends(get_current_user)) -> dict:
    """Initiate/verify Codex OAuth device flow and return status + message."""
    force = bool((body or {}).get("force", False))
    message, ok = await codex_login_handler({"force": force})
    status = codex_auth_status()
    return {
        "ok": ok,
        "message": message,
        **status,
    }


def _mask_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 6:
        return "*" * len(value)
    return ("*" * (len(value) - 4)) + value[-4:]


@router.get("/providers/status")
async def provider_status(user: dict = Depends(get_current_user)) -> dict:
    """Return whether provider API tokens are configured for this user.

    Includes env fallback so users who already configured MINIMAX_API_KEY /
    ZAI_API_KEY see "configured" immediately (similar to Codex status).
    """
    user_keys = session_manager.get_user_provider_keys(user["user_id"])
    effective = session_manager.get_effective_provider_keys(user["user_id"])

    minimax = effective.get("minimax", "")
    zai = effective.get("zai", "")

    return {
        "workspace_env_file": session_manager.get_user_workspace_env_file(user["user_id"]),
        "minimax": {
            "configured": bool(minimax),
            "masked": _mask_secret(minimax),
            "source": "user" if user_keys.get("minimax") else ("env" if minimax else "none"),
        },
        "zai": {
            "configured": bool(zai),
            "masked": _mask_secret(zai),
            "source": "user" if user_keys.get("zai") else ("env" if zai else "none"),
        },
    }


@router.post("/providers/tokens")
async def set_provider_tokens(body: dict | None = None, user: dict = Depends(get_current_user)) -> dict:
    """Set/clear per-user provider tokens for MiniMax and ZAI.

    Body (partial updates supported):
      { "minimax_api_key": "...", "zai_api_key": "..." }
      { "clear_minimax": true }
      { "clear_zai": true }

    Empty strings are ignored (do not overwrite existing tokens).
    """
    payload = body or {}
    current = session_manager.get_user_provider_keys(user["user_id"])
    updated = current.copy()

    if payload.get("clear_minimax"):
        updated.pop("minimax", None)
    elif "minimax_api_key" in payload:
        minimax_key = str(payload.get("minimax_api_key") or "").strip()
        if minimax_key:
            updated["minimax"] = minimax_key

    if payload.get("clear_zai"):
        updated.pop("zai", None)
    elif "zai_api_key" in payload:
        zai_key = str(payload.get("zai_api_key") or "").strip()
        if zai_key:
            updated["zai"] = zai_key

    normalized = session_manager.set_user_provider_keys(
        user["user_id"],
        updated,
    )

    return {
        "ok": True,
        "workspace_env_file": session_manager.get_user_workspace_env_file(user["user_id"]),
        "minimax": {
            "configured": bool(normalized.get("minimax")),
            "masked": _mask_secret(normalized.get("minimax", "")),
            "source": "user" if normalized.get("minimax") else "none",
        },
        "zai": {
            "configured": bool(normalized.get("zai")),
            "masked": _mask_secret(normalized.get("zai", "")),
            "source": "user" if normalized.get("zai") else "none",
        },
    }


@router.post("/providers/test")
async def test_provider(body: dict | None = None, user: dict = Depends(get_current_user)) -> dict:
    """Run a tiny completion against MiniMax or ZAI to validate token/config."""
    payload = body or {}
    provider = str(payload.get("provider") or "").strip().lower()
    model_map = {
        "minimax": "MiniMaxAI/MiniMax-M2.7",
        "zai": "zai-org/GLM-5.1",
    }
    model = model_map.get(provider)
    if not model:
        raise HTTPException(status_code=400, detail="provider must be 'minimax' or 'zai'")

    provider_keys = session_manager.get_effective_provider_keys(user["user_id"])
    llm_params = _resolve_llm_params(model, provider_keys=provider_keys, reasoning_effort="low")

    # Sanitized diagnostics to help users debug endpoint/model issues.
    resolved_model = llm_params.get("model")
    resolved_api_base = llm_params.get("api_base")

    try:
        resp = await acompletion(
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=8,
            timeout=12,
            **llm_params,
        )
        text = (resp.choices[0].message.content or "").strip() if resp.choices else ""
        return {
            "ok": True,
            "provider": provider,
            "model": model,
            "resolved_model": resolved_model,
            "resolved_api_base": resolved_api_base,
            "preview": text[:80],
        }
    except Exception as e:
        return {
            "ok": False,
            "provider": provider,
            "model": model,
            "resolved_model": resolved_model,
            "resolved_api_base": resolved_api_base,
            "error": str(e)[:500],
        }
