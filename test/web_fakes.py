"""
A fake LLM and an in-memory fraud_ops database for the web chat tests and the local demo.
The fake LLM calls the real agent tools (behind policy.py), so the OTP gate, the case
details and the escalation are the real ones; only the model's wording is scripted.
"""
import re

import duckdb

import fraud_agent as fa
import fraud_flow as ff
from src.models.batch_scoring import setup_tables as setup_scoring_tables

PROFILE = {"customer_id": "CLI-WEB", "first_name": "Ana", "full_name": "Ana Web",
           "email": "ana@example.com", "phone": "+573001112233", "country": "Colombia"}
AGENT = {"agent_id": "AGT-1", "first_name": "Luz", "last_name": "Rojas", "email": "luz@banco.test",
         "specialty": "Fraudes", "routing_score": 100.0, "routing_reasons": ["habla español"]}


class _Reply:
    def __init__(self, text):
        self.text = text


class FakeChat:
    """Scripted assistant: verifies 6-digit codes, escalates on request, otherwise reviews the case."""

    def __init__(self, tools):
        self.tools = {t.__name__: t for t in tools}
        self.received: list[str] = []

    def get_history(self):
        return []

    def send_message(self, text):
        self.received.append(text)
        lowered = text.lower()
        if re.fullmatch(r"\s*\d{6}\s*", text):
            result = self.tools["verify_otp"](text.strip())
            if not result.get("verified"):
                return _Reply("El código no coincide. Revisa tu correo e inténtalo de nuevo.")
            details = self.tools["get_case_details"]()
            alert = (details.get("behavior_alert") or {}).get("message", {}).get("es", "")
            return _Reply(f"Gracias, tu identidad quedó verificada. {alert}\n\n"
                          "¿Reconoces tus movimientos recientes?")
        if "código nuevo" in lowered:
            result = self.tools["resend_otp"]()
            return _Reply("Te envié un código nuevo a tu correo." if result.get("sent")
                          else "No pude enviar otro código.")
        if "asesor" in lowered or "no reconozco" in lowered:
            result = self.tools["escalate_to_agent"]("es", "El cliente pide un asesor", "Resumen del caso")
            if isinstance(result, dict) and result.get("escalated"):
                return _Reply(f"Te comuniqué con {result['agent_first_name']}, asesora del banco.")
            return _Reply("Primero necesito verificar tu identidad con el código de tu correo.")
        result = self.tools["get_case_details"]()
        if result.get("error"):
            return _Reply("Primero necesito verificar tu identidad con el código de tu correo.")
        return _Reply("Te recomiendo bloquear la tarjeta desde la app si no reconoces algún movimiento.")


def fake_llm_session(system_instruction, tools):
    chat = FakeChat(tools)
    return {"client": None, "config": None, "model": "fake", "chat": chat}


def setup_database(con=None):
    """fraud_ops tables in DuckDB with one flagged customer; points the agent modules to it."""
    con = con or duckdb.connect(":memory:")
    fa.con = con
    ff.setup_tables()
    setup_scoring_tables(con)
    con.execute("""INSERT INTO fraud_ops.customer_anomaly VALUES
        ('2026-06-17', 'clusters_20260617', 'CLI-WEB', 1, 9.0, 5, 5,
         'num_transacciones_monthly,monto_total_usd_monthly,monto_promedio_usd,monto_mediano_usd,monto_maximo_usd')""")
    return con


def open_web_case(customer_id: str = "CLI-WEB") -> dict:
    """Insert a case like fraud_flow.open_case does and return it."""
    case = {"case_id": ff.new_id("CASE"), "customer_id": customer_id, "source": "anomaly_model",
            "transaction_ids": "", "contact_email": PROFILE["email"], "contact_whatsapp": ""}
    fa.con.execute(
        f"""INSERT INTO {ff.table('fraud_cases')} (case_id, customer_id, created_at, updated_at,
                source, transaction_ids, max_score, reasons, contact_email, contact_whatsapp,
                demo_mode, status, anomaly_score)
            VALUES (?, ?, ?, ?, 'anomaly_model', '', 0, 'demo', ?, '', TRUE, ?, 1.0)""",
        [case["case_id"], customer_id, ff.now(), ff.now(), PROFILE["email"], ff.ST_OTP_SENT])
    return case


def patch_agent(monkeypatch_setattr, outbox: list):
    """Fake LLM, profile, transactions, routing and email. `monkeypatch_setattr(obj, name, value)`."""
    monkeypatch_setattr(fa, "create_llm_session", fake_llm_session)
    monkeypatch_setattr(fa, "get_customer_profile", lambda cid: dict(PROFILE, customer_id=cid))
    monkeypatch_setattr(ff, "case_transactions", lambda case: [])
    monkeypatch_setattr(ff, "rank_agents", lambda language, profile: [AGENT])
    monkeypatch_setattr(fa, "send_email",
                        lambda to, subject, body: outbox.append({"to": to, "subject": subject,
                                                                 "body": body}) or "sent")


def last_code(outbox: list) -> str:
    """The OTP of the latest verification email."""
    for mail in reversed(outbox):
        found = re.search(r"es: (\d{6})", mail["body"])
        if found:
            return found.group(1)
    raise AssertionError("no se envió ningún código")
