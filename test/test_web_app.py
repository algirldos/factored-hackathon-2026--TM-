"""
Customer web chat: case links, the session API, the OTP gate through the page, escalation
and the security limits. The LLM is scripted (web_fakes.FakeChat); everything else is real.
"""
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

import fraud_agent as fa
import fraud_flow as ff
import web_app
from web_fakes import PROFILE, last_code, open_web_case, patch_agent, setup_database


@pytest.fixture
def outbox(monkeypatch):
    mails: list = []
    con = setup_database()
    patch_agent(monkeypatch.setattr, mails)
    monkeypatch.setattr(ff, "CASE_CHANNEL", "web")
    yield mails
    con.close()
    fa.con = None


@pytest.fixture
def client(outbox):
    return TestClient(web_app.create_app(web_app.SessionStore()))


@pytest.fixture
def link(outbox):
    case = open_web_case()
    return case, ff.create_case_link(case["case_id"])


def start(client, token):
    return client.post("/api/sessions", json={"token": token})


def say(client, sid, text):
    return client.post("/api/messages", json={"text": text}, headers={"X-Session-Id": sid})


# --- the email link ------------------------------------------------------------------
def test_new_web_case_emails_a_link_and_no_code(outbox, monkeypatch):
    sent_whatsapp = []
    monkeypatch.setattr(fa, "send_whatsapp", lambda *a, **k: sent_whatsapp.append(a) or "sent")
    case = open_web_case()
    ff.notify_case(case, PROFILE)
    mail = outbox[-1]
    assert "Hemos detectado un comportamiento sospechoso en su actividad bancaria" in mail["body"]
    assert f"{ff.PUBLIC_WEB_URL}/#t=" in mail["body"]                 # token in the fragment
    assert "código de verificación es" not in mail["body"] and sent_whatsapp == []


def test_only_the_hash_of_the_token_is_stored(link):
    case, token = link
    stored = fa.con.execute("SELECT token_hash FROM fraud_ops.case_links").fetchall()
    assert stored == [(ff.hash_token(token),)] and token not in str(stored)


def test_a_new_link_revokes_the_previous_one(link):
    case, old = link
    new = ff.create_case_link(case["case_id"])
    assert ff.find_case_by_link(old) is None
    assert ff.find_case_by_link(new)["case_id"] == case["case_id"]


@pytest.mark.parametrize("change", ["expired", "closed", "unknown"])
def test_invalid_links_open_nothing(client, link, change):
    case, token = link
    if change == "expired":
        fa.con.execute("UPDATE fraud_ops.case_links SET expires_at = ?", [ff.now() - timedelta(minutes=1)])
    elif change == "closed":
        ff.update_case(case["case_id"], ff.ST_CLOSED)
    else:
        token = "x" * 43
    response = start(client, token)
    assert response.status_code == 404
    assert "no es válido" in response.json()["error"]


# --- the session ---------------------------------------------------------------------
def test_opening_a_link_starts_a_session_and_emails_the_code(client, link, outbox):
    _, token = link
    data = start(client, token).json()
    assert data["first_name"] == "Ana" and data["verified"] is False
    assert "comportamiento sospechoso" in data["opening_message"]
    assert data["otp"]["email"] == fa.mask_email(PROFILE["email"])
    assert last_code(outbox)
    assert token not in str(data)


def test_nothing_is_revealed_before_the_code(client, link):
    sid = start(client, link[1]).json()["session_id"]
    reply = say(client, sid, "¿Qué pasó con mi cuenta?").json()
    assert reply["verified"] is False and "verificar tu identidad" in reply["reply"]


def test_wrong_code_then_right_code(client, link, outbox):
    sid = start(client, link[1]).json()["session_id"]
    wrong = say(client, sid, "000000").json()
    assert wrong["verified"] is False
    right = say(client, sid, last_code(outbox)).json()
    assert right["verified"] is True
    assert "Hemos detectado un comportamiento sospechoso" in right["reply"]


def test_customer_can_escalate_to_an_advisor(client, link, outbox):
    sid = start(client, link[1]).json()["session_id"]
    say(client, sid, last_code(outbox))
    data = say(client, sid, "Quiero hablar con un asesor").json()
    assert data["escalated"] is True and data["agent_first_name"] == "Luz"
    staff_mail = outbox[-1]
    assert "fuera de lo habitual" in staff_mail["body"]                # internal detail for staff


def test_card_numbers_never_reach_the_agent(client, link):
    sid = start(client, link[1]).json()["session_id"]
    say(client, sid, "mi tarjeta es 4111 1111 1111 1234")
    session = next(iter(client.app.state.store.sessions.values()))
    assert session.agent["chat"].received[-1] == "mi tarjeta es •••• 1234"


def test_requests_without_a_valid_session_are_rejected(client):
    assert client.post("/api/messages", json={"text": "hola"}).status_code == 401
    assert say(client, "inventada", "hola").status_code == 401


def test_idle_sessions_expire(client, link):
    sid = start(client, link[1]).json()["session_id"]
    client.app.state.store.idle_timeout_s = -1
    assert say(client, sid, "hola").status_code == 401


def test_reopening_the_link_replaces_the_old_session(client, link):
    first = start(client, link[1]).json()["session_id"]
    second = start(client, link[1]).json()["session_id"]
    assert say(client, first, "hola").status_code == 401
    assert say(client, second, "hola").status_code == 200


def test_message_limits(client, link, monkeypatch):
    sid = start(client, link[1]).json()["session_id"]
    assert say(client, sid, "x" * 1001).status_code == 422
    monkeypatch.setattr(web_app, "MAX_MESSAGES", 1)
    assert say(client, sid, "hola").status_code == 200
    assert say(client, sid, "otra vez").status_code == 429


def test_code_sends_are_capped_per_case(client, link):
    for _ in range(ff.OTP_MAX_SENDS):
        assert start(client, link[1]).status_code == 200
    assert start(client, link[1]).status_code == 429


def test_logout_ends_the_session(client, link):
    sid = start(client, link[1]).json()["session_id"]
    client.delete("/api/sessions", headers={"X-Session-Id": sid})
    assert say(client, sid, "hola").status_code == 401


def test_assistant_unavailable_is_reported(client, link, monkeypatch):
    def missing_key(*args):
        raise SystemExit("Falta ANTHROPIC_API_KEY")
    monkeypatch.setattr(fa, "create_llm_session", missing_key)
    assert start(client, link[1]).status_code == 503


# --- the page --------------------------------------------------------------------------
def test_page_is_served_with_security_headers(client):
    response = client.get("/")
    assert response.status_code == 200 and "Soporte de Tarjetas" in response.text
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]
    assert response.headers["Referrer-Policy"] == "no-referrer"
    assert response.headers["Cache-Control"] == "no-store"


def test_api_docs_are_not_exposed(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
