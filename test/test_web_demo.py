"""
Public demo (web_app.py /demo) and the deployment endpoints. Cases are opened through the real
fraud_flow.open_case on an in-memory bronze database; only the LLM, routing and email are fakes.
"""
import re
from urllib.parse import urlsplit

import pytest
from fastapi.testclient import TestClient

import fraud_agent as fa
import fraud_flow as ff
import web_app
from factories import AS_OF, bronze_connection, load_bronze, raw_users, transactions
from src.features.transaction_behavior import profile_window
from src.models.batch_scoring import setup_tables as setup_scoring_tables
from web_fakes import patch_agent

POOL = ["CLI-0003", "CLI-0007", "CLI-0011"]


@pytest.fixture
def outbox(monkeypatch):
    for name in ("con", "TX_COLUMNS", "CUSTOMER_COLUMNS", "SCORE_SCALE"):
        monkeypatch.setattr(fa, name, getattr(fa, name))      # restored after the test
    users = raw_users(20)
    window = profile_window(AS_OF)
    con = bronze_connection()
    load_bronze(con, users, transactions(sorted(users["customer_id"].unique()), window.start,
                                         window.days, per_day=0.3))
    fa.init(con)
    ff.setup_tables()
    setup_scoring_tables(con)
    for cid in POOL:
        con.execute("INSERT INTO fraud_ops.customer_anomaly VALUES ('2026-06-17', 'clusters_20260617', "
                    "?, 0, 9.0, 5, 5, 'monto_maximo_usd')", [cid])
    mails: list = []
    patch_agent(monkeypatch.setattr, mails)
    monkeypatch.setattr(ff, "PUBLIC_DEMO", True)
    yield mails
    con.close()


@pytest.fixture
def client(outbox):
    return TestClient(web_app.create_app(), follow_redirects=False)


def open_demo(client, ip="203.0.113.7"):
    response = client.get("/demo", headers={"X-Forwarded-For": f"{ip}, 10.0.0.1"})
    assert response.status_code == 303, response.text
    location = response.headers["location"]
    assert urlsplit(location).path == "/" and location.startswith("/#t=")
    return location.split("#t=", 1)[1]


def demo_cases():
    return fa.con.execute("SELECT case_id, customer_id, status FROM fraud_ops.fraud_cases "
                          "WHERE source = 'public_demo'").fetchall()


# --- deployment endpoints ---------------------------------------------------------------
def test_health_and_config(client):
    health = client.get("/health")
    assert health.status_code == 200 and health.json()["database"] == "ok"
    assert client.get("/api/config").json() == {"public_demo": True}


def test_health_reports_a_broken_database(client, monkeypatch):
    class Broken:
        def execute(self, *args):
            raise RuntimeError("sin conexión")
    monkeypatch.setattr(fa, "con", Broken())
    assert client.get("/health").status_code == 503


def test_demo_is_off_unless_enabled(client, monkeypatch):
    monkeypatch.setattr(ff, "PUBLIC_DEMO", False)
    assert client.get("/demo").status_code == 404
    assert client.get("/api/config").json() == {"public_demo": False}


# --- the demo flow ------------------------------------------------------------------------
def test_demo_opens_a_flagged_customer_case_and_sends_nothing(client, outbox):
    open_demo(client)
    [(case_id, customer_id, status)] = demo_cases()
    assert customer_id in POOL and status == ff.ST_OTP_SENT
    assert outbox == []                                         # no link email, no code email


def test_demo_session_shows_the_code_instead_of_emailing_it(client, outbox):
    token = open_demo(client)
    data = client.post("/api/sessions", json={"token": token}).json()
    assert data["demo"] is True and re.fullmatch(r"\d{6}", data["demo_code"])
    assert "aviso de demostración" in data["opening_message"] and "correo" not in data["opening_message"]
    assert outbox == []
    reply = client.post("/api/messages", json={"text": data["demo_code"]},
                        headers={"X-Session-Id": data["session_id"]}).json()
    assert reply["verified"] is True


def test_demo_resend_shows_a_new_code(client, outbox):
    data = client.post("/api/sessions", json={"token": open_demo(client)}).json()
    reply = client.post("/api/messages", json={"text": "Envíame un código nuevo"},
                        headers={"X-Session-Id": data["session_id"]}).json()
    assert reply["demo_code"] != data["demo_code"] and outbox == []


def test_demo_escalation_emails_nobody(client, outbox):
    data = client.post("/api/sessions", json={"token": open_demo(client)}).json()
    headers = {"X-Session-Id": data["session_id"]}
    client.post("/api/messages", json={"text": data["demo_code"]}, headers=headers)
    reply = client.post("/api/messages", json={"text": "Quiero hablar con un asesor"}, headers=headers).json()
    assert reply["escalated"] is True and outbox == []


def test_a_demo_never_blocks_a_real_case(client):
    open_demo(client)
    [(_, customer_id, _)] = demo_cases()
    assert ff.existing_open_case(customer_id) is None


def test_a_new_demo_for_the_same_customer_closes_the_previous_one(client, monkeypatch):
    monkeypatch.setattr(ff, "demo_pool", lambda limit=50: ["CLI-0003"])
    open_demo(client)
    open_demo(client)
    statuses = sorted(status for _, _, status in demo_cases())
    assert statuses == sorted([ff.ST_CLOSED, ff.ST_OTP_SENT])


def test_real_cases_still_email_the_code_with_the_demo_on(client, outbox):
    case = ff.open_case("CLI-0005", source="anomaly_model", notify=False)
    token = ff.create_case_link(case["case_id"])
    data = client.post("/api/sessions", json={"token": token}).json()
    assert data["demo"] is False and "demo_code" not in data
    assert re.search(r"es: \d{6}", outbox[-1]["body"])


# --- limits ------------------------------------------------------------------------------
def test_demo_quota_per_visitor(outbox):
    client = TestClient(web_app.create_app(limiter=web_app.DemoLimiter(per_ip_hour=2, daily=100)),
                        follow_redirects=False)
    open_demo(client, ip="198.51.100.1")
    open_demo(client, ip="198.51.100.1")
    assert client.get("/demo", headers={"X-Forwarded-For": "198.51.100.1"}).status_code == 429
    open_demo(client, ip="198.51.100.2")                        # another visitor still can


def test_demo_daily_quota(outbox):
    client = TestClient(web_app.create_app(limiter=web_app.DemoLimiter(per_ip_hour=10, daily=1)),
                        follow_redirects=False)
    open_demo(client)
    response = client.get("/demo", headers={"X-Forwarded-For": "198.51.100.9"})
    assert response.status_code == 429 and "límite diario" in response.json()["error"]


def test_demo_without_flagged_customers(client, monkeypatch):
    monkeypatch.setattr(ff, "demo_pool", lambda limit=50: [])
    assert client.get("/demo").status_code == 503
