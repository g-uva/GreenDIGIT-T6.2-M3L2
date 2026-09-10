from __future__ import annotations

import base64
import hashlib
import html
import hmac
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.orm import Session

from m3l2.app.config import get_settings
from m3l2.app.db import AuthUser, RegisteredSite, SessionLocal, utc_now
from m3l2.site_adapter.auth import SitePrincipal, create_site_jwt, current_principal

router = APIRouter(prefix="/auth", tags=["Auth"])

VALID_ROLES = {"reader", "publisher", "site_admin"}
DEFAULT_ALLOWED_EMAILS_PATH = Path("allowed_emails.txt")
EMAIL_RE = re.compile(r"^[^@\s,]+@[^@\s,]+\.[^@\s,]+$")


@dataclass(frozen=True)
class AllowedAccess:
    email: str
    site_id: str | None
    roles: frozenset[str]


class TokenRequest(BaseModel):
    email: str
    password: str
    site_id: str | None = None
    role: str = "reader"


def get_db() -> Session:
    with SessionLocal() as session:
        yield session


def _repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path.cwd() / candidate


def _normalise_email(email: str) -> str:
    return email.strip().lower()


def _parse_roles(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset({"reader"})
    roles = {part.strip().lower() for part in raw.replace(";", "|").split("|") if part.strip()}
    unknown = roles - VALID_ROLES
    if unknown:
        raise HTTPException(status_code=500, detail=f"Invalid role in allowed_emails.txt: {', '.join(sorted(unknown))}")
    return frozenset(roles or {"reader"})


def load_allowed_access(path: str | Path | None = None) -> dict[str, list[AllowedAccess]]:
    settings = get_settings()
    allowed_path = _repo_path(path or settings.allowed_emails_path)
    if not allowed_path.exists():
        return {}
    allowed: dict[str, list[AllowedAccess]] = {}
    for raw_line in allowed_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [part.strip() for part in line.split(",")]
        email = _normalise_email(parts[0])
        if not EMAIL_RE.fullmatch(email):
            continue
        site_id = parts[1] if len(parts) > 1 and parts[1] else None
        roles = _parse_roles(parts[2] if len(parts) > 2 else None)
        allowed.setdefault(email, []).append(AllowedAccess(email=email, site_id=site_id, roles=roles))
    return allowed


def _access_for(email: str) -> list[AllowedAccess]:
    return load_allowed_access().get(_normalise_email(email), [])


def _site_options(access: list[AllowedAccess]) -> list[str]:
    return sorted({entry.site_id for entry in access if entry.site_id})


def _access_payload(
    access: list[AllowedAccess],
    registered_sites: dict[str, RegisteredSite] | None = None,
) -> list[dict[str, Any]]:
    roles_by_site: dict[str | None, set[str]] = {}
    for entry in access:
        roles_by_site.setdefault(entry.site_id, set()).update(entry.roles)

    payload = []
    for site_id, roles in sorted(roles_by_site.items(), key=lambda item: item[0] or ""):
        item: dict[str, Any] = {"site_id": site_id, "roles": sorted(roles)}
        if site_id:
            registered = (registered_sites or {}).get(site_id)
            item["registered"] = registered is not None
            if registered is not None:
                item["site_name"] = registered.site_name
                item["ri_type"] = registered.ri_type
        payload.append(item)
    return payload


def _roles_for_site(access: list[AllowedAccess], site_id: str) -> set[str]:
    roles: set[str] = set()
    for entry in access:
        if entry.site_id is None:
            roles.add("reader")
        elif entry.site_id == site_id:
            roles.update(entry.roles)
    return roles


def _resolve_site_and_role(email: str, site_id: str | None, role: str | None) -> tuple[str, str, list[str]]:
    access = _access_for(email)
    if not access:
        raise HTTPException(status_code=403, detail="Email is not allowed for this service")

    requested_role = (role or "reader").strip().lower()
    if requested_role not in VALID_ROLES:
        raise HTTPException(status_code=400, detail=f"Invalid role: {requested_role}")

    requested_site = (site_id or "").strip()
    if not requested_site:
        options = _site_options(access)
        if len(options) == 1:
            requested_site = options[0]
        else:
            raise HTTPException(status_code=400, detail="site_id is required for this email")

    allowed_roles = _roles_for_site(access, requested_site)
    if requested_role not in allowed_roles:
        raise HTTPException(status_code=403, detail=f"Email is not allowed role {requested_role} for site {requested_site}")
    return requested_site, requested_role, sorted(allowed_roles)


def _hash_password(password: str, salt: bytes | None = None, iterations: int = 260_000) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${base64.b64encode(salt).decode('ascii')}${base64.b64encode(digest).decode('ascii')}"


def _verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iterations_raw, salt_raw, digest_raw = password_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iterations_raw)
        salt = base64.b64decode(salt_raw.encode("ascii"))
        expected = base64.b64decode(digest_raw.encode("ascii"))
    except Exception:
        return False
    actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return hmac.compare_digest(actual, expected)


def _issue_token(session: Session, request: TokenRequest) -> dict[str, Any]:
    settings = get_settings()
    if not settings.jwt_secret:
        raise HTTPException(status_code=503, detail="JWT_SECRET is not configured")
    email = _normalise_email(request.email)
    if not request.password:
        raise HTTPException(status_code=400, detail="password is required")

    site_id, role, allowed_roles = _resolve_site_and_role(email, request.site_id, request.role)
    user = session.execute(select(AuthUser).where(AuthUser.email == email)).scalar_one_or_none()
    if user is None:
        user = AuthUser(email=email, password_hash=_hash_password(request.password), created_at=utc_now(), enabled=True)
        session.add(user)
        session.commit()
        session.refresh(user)
    elif not user.enabled:
        raise HTTPException(status_code=403, detail="User is disabled")
    elif not _verify_password(request.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect password")

    user.last_login_at = utc_now()
    session.commit()
    token = create_site_jwt(email, site_id, role, settings.jwt_secret, ttl_hours=settings.jwt_token_ttl_hours)
    return {
        "access_token": token,
        "token_type": "bearer",
        "expires_in": settings.jwt_token_ttl_hours * 3600,
        "email": email,
        "site_id": site_id,
        "role": role,
        "allowed_roles": allowed_roles,
    }


def _token_result_html(payload: dict[str, Any]) -> HTMLResponse:
    token = html.escape(payload["access_token"])
    email = html.escape(payload["email"])
    site_id = html.escape(payload["site_id"])
    role = html.escape(payload["role"])
    return HTMLResponse(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M3L2 API Token Generated</title>
    <link rel="stylesheet" href="/static/auth.css">
</head>
<body>
    <main class="auth-shell">
        <section class="auth-panel auth-panel-wide">
            <img src="/static/cropped-GD_logo.png" alt="GreenDIGIT" class="auth-logo">
            <h1>API Token Generated</h1>
            <p class="muted">Use this token as <code>Authorization: Bearer &lt;token&gt;</code>. It expires in 24 hours.</p>
            <dl class="token-meta">
                <div><dt>Email</dt><dd>{email}</dd></div>
                <div><dt>Site</dt><dd>{site_id}</dd></div>
                <div><dt>Role</dt><dd>{role}</dd></div>
            </dl>
            <label class="token-label" for="access-token">Access Token</label>
            <textarea id="access-token" readonly>{token}</textarea>
            <button type="button" onclick="navigator.clipboard.writeText(document.getElementById('access-token').value)">Copy Token</button>
            <script>
                localStorage.setItem("m3l2_token", "{token}");
                localStorage.setItem("m3l2_principal", JSON.stringify({{"email": "{email}", "site_id": "{site_id}", "role": "{role}"}}));
            </script>
            <div class="button-row">
                <a class="button-link" href="/ops/config/ui">Open Config</a>
                <a class="button-link secondary" href="/docs">Open API Docs</a>
            </div>
            <footer class="grant-footer">
                <p>This work is funded from the European Union's Horizon Europe research and innovation programme through the <a href="https://greendigit-project.eu/" target="_blank" rel="noopener">GreenDIGIT project</a>, under Grant Agreement No. <a href="https://cordis.europa.eu/project/id/101131207" target="_blank" rel="noopener">101131207</a>.</p>
                <img src="/static/EN-Funded-by-the-EU-POS-2.png" alt="Funded by the European Union">
            </footer>
        </section>
    </main>
</body>
</html>"""
    )


def _role_options(selected_role: str) -> str:
    return "\n".join(
        f'<option value="{option}"{" selected" if option == selected_role else ""}>{option}</option>'
        for option in ("reader", "publisher", "site_admin")
    )


def _login_choice_html() -> HTMLResponse:
    return HTMLResponse(
        """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M3L2 Login and Token</title>
    <link rel="stylesheet" href="/static/auth.css">
</head>
<body>
    <main class="auth-shell">
        <section class="auth-panel">
            <img src="/static/cropped-GD_logo.png" alt="GreenDIGIT" class="auth-logo">
            <h1>GreenDIGIT M3L2 API</h1>
            <h2>Login/token</h2>
            <div class="button-row">
                <a class="button-link" href="/auth/login?next=/ops/config/ui&role=site_admin">Login to config</a>
                <a class="button-link secondary" href="/auth/login/token">Get token</a>
            </div>
            <div class="info">
                <p>First login sets your password if your email is registered. Tokens are valid for 24 hours.</p>
                <p>If your login does not work, please contact g.j.teixeiradepinhoferreira@uva.nl</p>
            </div>
            <footer class="grant-footer">
                <p>This work is funded from the European Union's Horizon Europe research and innovation programme through the <a href="https://greendigit-project.eu/" target="_blank" rel="noopener">GreenDIGIT project</a>, under Grant Agreement No. <a href="https://cordis.europa.eu/project/id/101131207" target="_blank" rel="noopener">101131207</a>.</p>
                <img src="/static/EN-Funded-by-the-EU-POS-2.png" alt="Funded by the European Union">
            </footer>
        </section>
    </main>
</body>
</html>"""
    )


def _config_login_html(next: str, role: str) -> HTMLResponse:
    safe_next = next if next.startswith("/") and not next.startswith("//") else "/ops/config/ui"
    selected_role = role if role in VALID_ROLES else "site_admin"
    role_options = _role_options(selected_role)
    page = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M3L2 API Token Login</title>
    <link rel="stylesheet" href="/static/auth.css">
</head>
<body>
    <main class="auth-shell">
        <section class="auth-panel">
            <img src="/static/cropped-GD_logo.png" alt="GreenDIGIT" class="auth-logo">
            <h1>GreenDIGIT M3L2 API</h1>
            <h2>Login to config</h2>
            <form id="token-form">
                <input id="email" name="email" type="email" placeholder="Email" autocomplete="email" required>
                <input id="password" name="password" type="password" placeholder="Password" autocomplete="current-password" required>
                <input id="site_id" name="site_id" type="text" placeholder="Site ID, e.g. UTH-IOT">
                <select id="role" name="role">
                    __ROLE_OPTIONS__
                </select>
                <button type="submit">Login to config</button>
            </form>
            <p id="error" class="error" hidden></p>
            <div class="info">
                <p>First login sets your password if your email is registered. Tokens are valid for 24 hours.</p>
                <p>If your login does not work, please contact g.j.teixeiradepinhoferreira@uva.nl</p>
            </div>
            <footer class="grant-footer">
                <p>This work is funded from the European Union's Horizon Europe research and innovation programme through the <a href="https://greendigit-project.eu/" target="_blank" rel="noopener">GreenDIGIT project</a>, under Grant Agreement No. <a href="https://cordis.europa.eu/project/id/101131207" target="_blank" rel="noopener">101131207</a>.</p>
                <img src="/static/EN-Funded-by-the-EU-POS-2.png" alt="Funded by the European Union">
            </footer>
        </section>
    </main>
    <script>
        const form = document.getElementById("token-form");
        const error = document.getElementById("error");
        const nextUrl = "__NEXT_URL__";
        form.addEventListener("submit", async (event) => {
            event.preventDefault();
            error.hidden = true;
            const payload = {
                email: document.getElementById("email").value.trim(),
                password: document.getElementById("password").value,
                site_id: document.getElementById("site_id").value.trim() || null,
                role: document.getElementById("role").value,
            };
            const response = await fetch("/auth/token", {
                method: "POST",
                headers: {"Content-Type": "application/json"},
                body: JSON.stringify(payload),
            });
            const bodyText = await response.text();
            if (!response.ok) {
                try {
                    const body = JSON.parse(bodyText);
                    error.textContent = body.detail || "Login failed";
                } catch {
                    error.textContent = "Login failed";
                }
                error.hidden = false;
                return;
            }
            const body = JSON.parse(bodyText);
            localStorage.setItem("m3l2_token", body.access_token);
            localStorage.setItem("m3l2_principal", JSON.stringify({email: body.email, site_id: body.site_id, role: body.role}));
            window.location.href = nextUrl;
        });
    </script>
</body>
</html>"""
    return HTMLResponse(page.replace("__ROLE_OPTIONS__", role_options).replace("__NEXT_URL__", html.escape(safe_next, quote=True)))


def _token_login_html(role: str = "reader") -> HTMLResponse:
    selected_role = role if role in VALID_ROLES else "reader"
    role_options = _role_options(selected_role)
    page = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>M3L2 API Token Login</title>
    <link rel="stylesheet" href="/static/auth.css">
</head>
<body>
    <main class="auth-shell">
        <section class="auth-panel">
            <img src="/static/cropped-GD_logo.png" alt="GreenDIGIT" class="auth-logo">
            <h1>GreenDIGIT M3L2 API</h1>
            <h2>Get token</h2>
            <form method="post" action="/auth/login/token">
                <input name="email" type="email" placeholder="Email" autocomplete="email" required>
                <input name="password" type="password" placeholder="Password" autocomplete="current-password" required>
                <input name="site_id" type="text" placeholder="Site ID, e.g. UTH-IOT">
                <select name="role">
                    __ROLE_OPTIONS__
                </select>
                <button type="submit">Get token</button>
            </form>
            <div class="button-row">
                <a class="button-link secondary" href="/auth/login?next=/ops/config/ui&role=site_admin">Login to config</a>
                <a class="button-link secondary" href="/docs">Open API Docs</a>
            </div>
            <div class="info">
                <p>Use the generated token as <code>Authorization: Bearer &lt;token&gt;</code>.</p>
            </div>
            <footer class="grant-footer">
                <p>This work is funded from the European Union's Horizon Europe research and innovation programme through the <a href="https://greendigit-project.eu/" target="_blank" rel="noopener">GreenDIGIT project</a>, under Grant Agreement No. <a href="https://cordis.europa.eu/project/id/101131207" target="_blank" rel="noopener">101131207</a>.</p>
                <img src="/static/EN-Funded-by-the-EU-POS-2.png" alt="Funded by the European Union">
            </footer>
        </section>
    </main>
</body>
</html>"""
    return HTMLResponse(page.replace("__ROLE_OPTIONS__", role_options))


@router.get("/login", response_class=HTMLResponse, summary="HTML login/token choice page")
def login_page(next: str | None = Query(None), role: str = Query("site_admin")) -> HTMLResponse:
    if next is None:
        return _login_choice_html()
    return _config_login_html(next, role)


@router.get("/login/token", response_class=HTMLResponse, summary="HTML login page that displays a 24-hour JWT")
def token_login_page(role: str = Query("reader")) -> HTMLResponse:
    return _token_login_html(role)


@router.post("/login/token", response_class=HTMLResponse, summary="Login and display a 24-hour JWT")
async def token_login(request: Request, session: Session = Depends(get_db)) -> HTMLResponse:
    form = await request.form()
    token_request = TokenRequest(
        email=str(form.get("email") or ""),
        password=str(form.get("password") or ""),
        site_id=str(form.get("site_id") or "") or None,
        role=str(form.get("role") or "reader"),
    )
    return _token_result_html(_issue_token(session, token_request))


@router.post("/login", response_class=HTMLResponse, summary="Login and display a 24-hour JWT")
def login(request: TokenRequest, session: Session = Depends(get_db)) -> HTMLResponse:
    return _token_result_html(_issue_token(session, request))


@router.post("/token", summary="Login and return a 24-hour JWT as JSON")
def token(request: TokenRequest, session: Session = Depends(get_db)) -> dict[str, Any]:
    return _issue_token(session, request)


@router.get("/token", summary="Login with query parameters and return a 24-hour JWT as JSON")
def token_query(
    email: str = Query(...),
    password: str = Query(...),
    site_id: str | None = Query(None),
    role: str = Query("reader"),
    session: Session = Depends(get_db),
) -> dict[str, Any]:
    return _issue_token(session, TokenRequest(email=email, password=password, site_id=site_id, role=role))


@router.get("/verify-token", summary="Validate a Bearer token")
def verify_token(principal: SitePrincipal = Depends(current_principal)) -> dict[str, Any]:
    return {"valid": True, "email": principal.email, "site_id": principal.site_id, "role": principal.role}


@router.get("/me", summary="Return the authenticated user, role, and allowed sites")
def me(principal: SitePrincipal = Depends(current_principal)) -> dict[str, Any]:
    access = _access_for(principal.email)
    roles_for_current_site = sorted(_roles_for_site(access, principal.site_id)) if access else [principal.role]
    site_ids = [entry.site_id for entry in access if entry.site_id]
    registered_sites: dict[str, RegisteredSite] = {}
    if site_ids:
        with SessionLocal() as session:
            rows = session.execute(select(RegisteredSite).where(RegisteredSite.site_id.in_(site_ids))).scalars()
            registered_sites = {row.site_id: row for row in rows}
    return {
        "email": principal.email,
        "site_id": principal.site_id,
        "role": principal.role,
        "roles_for_current_site": roles_for_current_site,
        "sites": (
            _access_payload(access, registered_sites)
            if access
            else [{"site_id": principal.site_id, "roles": [principal.role], "registered": False}]
        ),
    }
