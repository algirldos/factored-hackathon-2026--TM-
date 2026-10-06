"""
Customer web chat: serves src/web/chat.html and connects it to the OTP-verified agent.

    python web_app.py                 # http://localhost:8000 (PORT to change it)

With PUBLIC_DEMO=1, GET /demo opens a fresh case for a customer flagged by the latest scoring
run and shows the one-time code on screen: a demo for judges and visitors that contacts nobody.

The customer arrives from the link emailed when a case is opened (fraud_flow, CASE_CHANNEL=web).
The page sends the link token to POST /api/sessions; the server checks it, emails a one-time
code and opens an agent session. Messages go to POST /api/messages. Everything about the case
stays on the server: the browser only holds a random session id in memory.
"""
import os
import random
import re
import secrets
import sys
import threading
import time
from collections import defaultdict, deque
from datetime import date
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

import fraud_agent as fa
import fraud_flow as ff

CHAT_PAGE = Path(__file__).with_name("src") / "web" / "chat.html"
IDLE_TIMEOUT_S = int(os.environ.get("WEB_IDLE_TIMEOUT_S", "300"))   # same 5 minutes as the page
MAX_MESSAGES = int(os.environ.get("WEB_MAX_MESSAGES", "40"))        # per session
MAX_SESSIONS = int(os.environ.get("WEB_MAX_SESSIONS", "200"))
MAX_TEXT = 1000
DEMO_PER_IP_HOUR = int(os.environ.get("DEMO_PER_IP_HOUR", "5"))      # public demos per visitor
DEMO_DAILY_LIMIT = int(os.environ.get("DEMO_DAILY_LIMIT", "150"))    # caps the LLM spend
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
        status = {"verified": bool(self.state.get("verified")),
                  "escalated": bool(escalation.get("escalated")),
                  "agent_first_name": escalation.get("agent_first_name")}
        if self.state.get("demo_code"):
            status["demo_code"] = self.state["demo_code"]
        return status


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


class DemoLimiter:
    """Public demo quotas: per visitor per hour and per day, in memory (single instance)."""

    def __init__(self, per_ip_hour: int = DEMO_PER_IP_HOUR, daily: int = DEMO_DAILY_LIMIT):
        self.per_ip_hour, self.daily = per_ip_hour, daily
        self.by_ip: dict[str, deque] = defaultdict(deque)
        self.day, self.count = date.today(), 0

    def allow(self, ip: str) -> str | None:
        """None if allowed, otherwise the reason it is not."""
        if self.day != date.today():
            self.day, self.count = date.today(), 0
        if self.count >= self.daily:
            return "La demo alcanzó su límite diario. Vuelve a intentarlo mañana."
        recent = self.by_ip[ip]
        cutoff = time.monotonic() - 3600
        while recent and recent[0] < cutoff:
            recent.popleft()
        if len(recent) >= self.per_ip_hour:
            return "Abriste varias demos en la última hora. Intenta de nuevo más tarde."
        recent.append(time.monotonic())
        self.count += 1
        return None


def client_ip(request: Request) -> str:
    # Render (and most hosts) put the visitor's address first in X-Forwarded-For
    forwarded = request.headers.get("x-forwarded-for", "")
    return forwarded.split(",")[0].strip() or (request.client.host if request.client else "?")


class StartRequest(BaseModel):
    token: str = Field(min_length=10, max_length=200)


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=MAX_TEXT)


def create_app(store: SessionStore | None = None, limiter: DemoLimiter | None = None) -> FastAPI:
    app = FastAPI(title="LATAM Bank chat", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.store = store = store or SessionStore()
    app.state.limiter = limiter = limiter or DemoLimiter()

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    @app.get("/")
    def page():
        return FileResponse(CHAT_PAGE, media_type="text/html")

    @app.get("/health")
    def health():
        with store.lock:
            try:
                fa.con.execute("SELECT 1").fetchone()
                database = "ok"
            except Exception:
                database = "error"
        return JSONResponse({"status": "ok" if database == "ok" else "degraded",
                             "database": database, "public_demo": ff.PUBLIC_DEMO},
                            status_code=200 if database == "ok" else 503)

    @app.get("/api/config")
    def config():
        return {"public_demo": ff.PUBLIC_DEMO}

    @app.get("/demo")
    def demo(request: Request):
        if not ff.PUBLIC_DEMO:
            raise HTTPException(404, "La demo pública no está activa.")
        refused = limiter.allow(client_ip(request))
        if refused:
            raise HTTPException(429, refused)
        with store.lock:
            pool = ff.demo_pool()
            random.shuffle(pool)
            for customer_id in pool[:5]:          # a few tries: a customer may have nothing to show
                opened = ff.open_demo_case(customer_id)
                if opened:
                    return RedirectResponse(f"/#t={opened[1]}", status_code=303)
        raise HTTPException(503, "No hay casos de demostración disponibles en este momento.")

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
            demo = ff.PUBLIC_DEMO and case.get("source") == ff.SRC_DEMO
            otp = ff.create_otp(case, agent["profile"], reveal_code=demo)
            if demo and otp.get("sent"):
                agent["state"]["demo_code"] = otp.pop("demo_code")
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
                "demo": demo,
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
    print(f"Chat de clientes en {ff.PUBLIC_WEB_URL} | {fa.provider_label()} | {fa.delivery_summary()}"
          f"{' | demo pública en /demo' if ff.PUBLIC_DEMO else ''}")
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"),
                port=int(os.environ.get("PORT", "8000")), log_level="info")


if __name__ == "__main__":
    sys.exit(main())
