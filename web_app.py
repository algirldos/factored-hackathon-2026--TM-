"""
Customer web chat: serves src/web/chat.html and connects it to the OTP-verified agent.

    python web_app.py                 # http://localhost:8000 (PORT to change it)

The customer arrives from the link emailed when a case is opened (fraud_flow, CASE_CHANNEL=web).
The page sends the link token to POST /api/sessions; the server checks it, emails a one-time
code and opens an agent session. Messages go to POST /api/messages. Everything about the case
stays on the server: the browser only holds a random session id in memory.
"""
import os
import re
import secrets
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import fraud_agent as fa
import fraud_flow as ff

CHAT_PAGE = Path(__file__).with_name("src") / "web" / "chat.html"
IDLE_TIMEOUT_S = int(os.environ.get("WEB_IDLE_TIMEOUT_S", "300"))   # same 5 minutes as the page
MAX_MESSAGES = int(os.environ.get("WEB_MAX_MESSAGES", "40"))        # per session
MAX_SESSIONS = int(os.environ.get("WEB_MAX_SESSIONS", "200"))
MAX_TEXT = 1000
CARD_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline' "
        "https://fonts.googleapis.com; font-src https://fonts.gstatic.com; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def mask_cards(text: str) -> str:
    """Full card numbers never reach the agent, even if the page's own mask is bypassed."""
    return CARD_RE.sub(lambda m: "•••• " + re.sub(r"\D", "", m.group())[-4:], text)


@dataclass
class WebSession:
    case_id: str
    agent: dict                       # fraud_flow.new_session(...)
    last_seen: float = field(default_factory=time.monotonic)
    messages: int = 0

    @property
    def state(self) -> dict:
        return self.agent["state"]

    def status(self) -> dict:
        escalation = self.state.get("escalation") or {}
        return {"verified": bool(self.state.get("verified")),
                "escalated": bool(escalation.get("escalated")),
                "agent_first_name": escalation.get("agent_first_name")}


class SessionStore:
    """Sessions in memory with an idle timeout. One agent turn at a time (shared DuckDB connection)."""

    def __init__(self, idle_timeout_s: int = IDLE_TIMEOUT_S, max_sessions: int = MAX_SESSIONS):
        self.idle_timeout_s, self.max_sessions = idle_timeout_s, max_sessions
        self.sessions: dict[str, WebSession] = {}
        self.lock = threading.Lock()

    def _expire(self) -> None:
        cutoff = time.monotonic() - self.idle_timeout_s
        for sid in [s for s, v in self.sessions.items() if v.last_seen < cutoff]:
            del self.sessions[sid]

    def add(self, session: WebSession) -> str:
        self._expire()
        if len(self.sessions) >= self.max_sessions:
            raise HTTPException(503, "El chat está ocupado. Intenta de nuevo en unos minutos.")
        # A new session for the same case replaces the previous one (one browser at a time)
        for sid in [s for s, v in self.sessions.items() if v.case_id == session.case_id]:
            del self.sessions[sid]
        sid = secrets.token_urlsafe(32)
        self.sessions[sid] = session
        return sid

    def get(self, sid: str | None) -> WebSession:
        self._expire()
        session = self.sessions.get(sid or "")
        if session is None:
            raise HTTPException(401, "Tu sesión terminó. Abre de nuevo el enlace del correo.")
        session.last_seen = time.monotonic()
        return session

    def remove(self, sid: str | None) -> None:
        self.sessions.pop(sid or "", None)


class StartRequest(BaseModel):
    token: str = Field(min_length=10, max_length=200)


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)


def create_app(store: SessionStore | None = None) -> FastAPI:
    app = FastAPI(title="LATAM Bank chat", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store = store or SessionStore()

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.get("/")
    def page():
        return FileResponse(CHAT_PAGE, media_type="text/html")

    @app.post("/api/sessions")
    def start(body: StartRequest):
        with store.lock:
            case = ff.find_case_by_link(body.token)
            if case is None:
                raise HTTPException(404, "El enlace no es válido o ya venció. Si necesitas ayuda, "
                                         "comunícate con la línea oficial del banco.")
            try:
                agent = ff.new_session(case, channel="web")
            except SystemExit as e:  # LLM not configured (missing API key)
                raise HTTPException(503, "El asistente no está disponible en este momento.") from e
            otp = ff.create_otp(case, agent["profile"])
            if not otp.get("sent"):
                raise HTTPException(429, "Se alcanzó el máximo de códigos para este caso. "
                                         "Comunícate con la línea oficial del banco.")
            session = WebSession(case["case_id"], agent)
            sid = store.add(session)
        return {"session_id": sid,
                "first_name": agent["profile"].get("first_name") or "cliente",
                "opening_message": agent["opener"],
                "otp": {"email": otp.get("email"), "expires_in_minutes": otp.get("expires_in_minutes")},
                "idle_timeout_s": store.idle_timeout_s,
                **session.status()}

    @app.post("/api/messages")
    def message(body: MessageRequest, x_session_id: str | None = Header(default=None)):
        with store.lock:
            session = store.get(x_session_id)
            if session.messages >= MAX_MESSAGES:
                raise HTTPException(429, "Llegaste al límite de mensajes de esta sesión. "
                                         "Pide hablar con un asesor o abre de nuevo el enlace.")
            session.messages += 1
            reply = ff.reply_to(session.agent, mask_cards(body.text.strip()))
            return {"reply": reply, **session.status()}

    @app.delete("/api/sessions")
    def end(x_session_id: str | None = Header(default=None)):
        with store.lock:
            store.remove(x_session_id)
        return {"ended": True}

    @app.exception_handler(HTTPException)
    async def http_error(request: Request, exc: HTTPException):
        return JSONResponse({"error": exc.detail}, status_code=exc.status_code)

    return app


def main() -> None:
    import uvicorn
    fa.init(fa.connect())
    ff.setup_tables()
    print(f"Chat de clientes en {ff.PUBLIC_WEB_URL} | {fa.provider_label()} | {fa.delivery_summary()}")
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")), log_level="info")


if __name__ == "__main__":
    sys.exit(main())
