#!/usr/bin/env python3
"""
OTP-verified fraud conversation flow - LATAM Bank (Factored AI & Data Hackathon 2026).

Flow:
  1. An anomaly is detected (your anomaly model, or fraud_agent's risk engine) and a fraud
     case is opened in latam_bank.fraud_ops.fraud_cases.
  2. A 6-digit OTP is generated, stored HASHED in latam_bank.fraud_ops.otp_codes and emailed.
  3. A WhatsApp conversation starts. The Gemini agent asks for the OTP and verifies it against
     the table. Until then, every tool that touches customer data is blocked in code.
  4. After verification, the agent explains the unusual activity and records the answer.
  5. If the customer wants a human, the case is routed to the best service agent: same
     language as the conversation (hard rule), fraud specialty, accent, country, shift,
     CSAT and workload. The escalation is stored in latam_bank.fraud_ops.escalations.

Usage:
  python fraud_flow.py --setup                               Create the fraud_ops tables
  python fraud_flow.py --trigger CUSTOMER_ID                 Open a case, send the OTP, chat in console
  python fraud_flow.py --trigger ID1 ID2 ID3                 Anomalous customers from your model (batch)
  python fraud_flow.py --from-file anomalies.csv             Same, reading a CSV/TXT with customer_id
  python fraud_flow.py --process-queue                       Same, reading fraud_ops.anomaly_queue
  python fraud_flow.py --trigger CUSTOMER_ID --tx TX1 TX2    One customer, with specific transactions
  python fraud_flow.py --chat CASE_ID                        Resume a case in the console
  python fraud_flow.py --route CUSTOMER_ID --language es     Preview the agent routing
  python fraud_flow.py --whatsapp-server                     Twilio webhook for real WhatsApp

Requires a MotherDuck token with WRITE access (the tables are created in latam_bank).
"""
import argparse
import base64
import hashlib
import hmac
import os
import secrets
import sys
import threading
import unicodedata
from datetime import datetime
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import fraud_agent as fa  # reuses connection, data access, risk engine and delivery

# ===========================================================================
# 1. Configuration
# ===========================================================================
OPS_SCHEMA = os.environ.get("FRAUD_OPS_SCHEMA", "fraud_ops")
AGENTS_TABLE = os.environ.get("AGENTS_TABLE", "bronze.service_agents")
OTP_TTL_MIN = int(os.environ.get("OTP_TTL_MIN", "10"))   # minutes a code stays valid
OTP_MAX_ATTEMPTS = 3                                      # wrong tries before the code is locked
OTP_MAX_SENDS = 3                                         # codes per case (initial + 2 resends)
OTP_SECRET = os.environ.get("OTP_SECRET", "")             # server-side pepper for the hash
MODEL_FALLBACK_TX = 3      # model-flagged customer with no rule hits: show the top N transactions
DEDUP_STATUSES = ("otp_sent", "verified", "customer_reported_fraud", "escalated")  # still open
SERVER_PORT = int(os.environ.get("WHATSAPP_SERVER_PORT", "8080"))
PUBLIC_URL = os.environ.get("PUBLIC_URL", "")             # e.g. https://xxxx.ngrok-free.app
# Print the OTP in the console (testing only). Defaults to on in demo mode, never in real mode.
SHOW_OTP_IN_CONSOLE = fa.DEMO_MODE and os.environ.get("SHOW_OTP_IN_CONSOLE", "1") != "0"

# Case statuses
ST_OTP_SENT, ST_VERIFIED = "otp_sent", "verified"
ST_LEGIT, ST_FRAUD = "customer_confirmed_legit", "customer_reported_fraud"
ST_ESCALATED, ST_CLOSED = "escalated", "closed"

# Routing weights (soft criteria; language is a hard filter)
W_FRAUD_SPECIALTY = 40
W_ACCENT = 20
W_COUNTRY = 15
W_ON_SHIFT = 15
W_CHAT_TYPE = 10
W_EXPERIENCE = {"specialist": 10, "senior": 7, "mid-senior": 4, "junior": 0}
CHAT_AGENT_TYPES = {"digital", "hybrid"}       # best suited for WhatsApp
EXCLUDED_AGENT_TYPES = {"in-person"}           # cannot take a chat
SHIFT_HOURS = {"morning": (6, 14), "afternoon": (14, 22), "night": (22, 6)}
LANGUAGE_KEYWORDS = {"es": "espanol", "pt": "portugues", "en": "ingles"}
LANGUAGE_NAMES = {"es": "español", "pt": "portugués", "en": "inglés"}


def table(name: str) -> str:
    return f"{OPS_SCHEMA}.{name}"


def new_id(prefix: str) -> str:
    return f"{prefix}-{secrets.token_hex(6).upper()}"


def now() -> datetime:
    return datetime.now().replace(microsecond=0)


def normalize(text: str | None) -> str:
    """Lowercase and strip accents: 'Español' -> 'espanol'."""
    if not text:
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower().strip()


def normalize_phone(phone: str | None) -> str:
    """'whatsapp:+57 300 111' -> '+57300111'."""
    if not phone:
        return ""
    phone = phone.replace("whatsapp:", "")
    return "+" + "".join(c for c in phone if c.isdigit()) if phone.strip().startswith("+") else \
        "".join(c for c in phone if c.isdigit())


# ===========================================================================
# 2. Operational tables in latam_bank
# ===========================================================================
def setup_tables() -> None:
    """Create the fraud_ops schema and tables (idempotent)."""
    con = fa.con
    con.execute(f"CREATE SCHEMA IF NOT EXISTS {OPS_SCHEMA}")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table('fraud_cases')} (
            case_id           VARCHAR NOT NULL,
            customer_id       VARCHAR NOT NULL,
            created_at        TIMESTAMP NOT NULL,
            updated_at        TIMESTAMP NOT NULL,
            source            VARCHAR NOT NULL,   -- anomaly_model | risk_engine
            transaction_ids   VARCHAR,            -- comma-separated
            max_score         DOUBLE,
            reasons           VARCHAR,
            contact_email     VARCHAR,
            contact_whatsapp  VARCHAR,
            demo_mode         BOOLEAN,
            status            VARCHAR NOT NULL,   -- otp_sent | verified | customer_confirmed_legit
                                                  -- | customer_reported_fraud | escalated | closed
            customer_comment  VARCHAR
        )""")
    # Migration for tables created by an earlier version
    con.execute(f"ALTER TABLE {table('fraud_cases')} ADD COLUMN IF NOT EXISTS anomaly_score DOUBLE")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table('anomaly_queue')} (
            customer_id    VARCHAR NOT NULL,      -- written by the anomaly detection model
            detected_at    TIMESTAMP NOT NULL,
            model_name     VARCHAR,
            anomaly_score  DOUBLE,
            processed_at   TIMESTAMP,             -- NULL = pending
            case_id        VARCHAR,               -- case opened for this row
            result         VARCHAR                -- case_opened | duplicate | not_found
        )""")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table('otp_codes')} (
            otp_id        VARCHAR NOT NULL,
            case_id       VARCHAR NOT NULL,
            customer_id   VARCHAR NOT NULL,
            code_hash     VARCHAR NOT NULL,       -- HMAC-SHA256, never the plain code
            channel       VARCHAR NOT NULL,       -- email
            created_at    TIMESTAMP NOT NULL,
            expires_at    TIMESTAMP NOT NULL,
            attempts      INTEGER NOT NULL,
            max_attempts  INTEGER NOT NULL,
            status        VARCHAR NOT NULL,       -- pending | verified | expired | locked | superseded
            verified_at   TIMESTAMP
        )""")
    con.execute(f"""
        CREATE TABLE IF NOT EXISTS {table('escalations')} (
            escalation_id   VARCHAR NOT NULL,
            case_id         VARCHAR NOT NULL,
            customer_id     VARCHAR NOT NULL,
            agent_id        VARCHAR,              -- NULL if no agent speaks the language
            agent_name      VARCHAR,
            language        VARCHAR NOT NULL,
            reason          VARCHAR,
            summary         VARCHAR,
            routing_score   DOUBLE,
            routing_reasons VARCHAR,
            created_at      TIMESTAMP NOT NULL,
            status          VARCHAR NOT NULL      -- open | pending_no_agent
        )""")


def update_case(case_id: str, status: str, comment: str | None = None) -> None:
    fa.con.execute(
        f"UPDATE {table('fraud_cases')} SET status = ?, updated_at = ?, "
        f"customer_comment = COALESCE(?, customer_comment) WHERE case_id = ?",
        [status, now(), comment, case_id])


def load_case(case_id: str) -> dict | None:
    cur = fa.con.execute(f"SELECT * FROM {table('fraud_cases')} WHERE case_id = ?", [case_id])
    row = cur.fetchone()
    return dict(zip([d[0] for d in cur.description], row)) if row else None


def find_open_case_by_phone(phone: str) -> dict | None:
    """Most recent non-closed case whose WhatsApp contact matches the sender."""
    cur = fa.con.execute(
        f"SELECT * FROM {table('fraud_cases')} WHERE contact_whatsapp = ? AND status <> ? "
        f"ORDER BY created_at DESC LIMIT 1", [normalize_phone(phone), ST_CLOSED])
    row = cur.fetchone()
    return dict(zip([d[0] for d in cur.description], row)) if row else None


# ===========================================================================
# 3. OTP: create, store hashed, email, verify
# ===========================================================================
def hash_code(otp_id: str, code: str) -> str:
    """HMAC-SHA256 of the code, salted with the OTP id and peppered with OTP_SECRET."""
    return hmac.new(OTP_SECRET.encode(), f"{otp_id}:{code}".encode(), hashlib.sha256).hexdigest()


def create_otp(case: dict, profile: dict) -> dict:
    """Generate a new code for the case, store its hash and email it."""
    con = fa.con
    sent = con.execute(f"SELECT COUNT(*) FROM {table('otp_codes')} WHERE case_id = ?",
                       [case["case_id"]]).fetchone()[0]
    if sent >= OTP_MAX_SENDS:
        return {"sent": False, "error": f"Maximum of {OTP_MAX_SENDS} codes per case reached."}

    # Only one active code per case
    con.execute(f"UPDATE {table('otp_codes')} SET status = 'superseded' "
                f"WHERE case_id = ? AND status = 'pending'", [case["case_id"]])

    code = f"{secrets.randbelow(10 ** 6):06d}"  # cryptographically secure, 6 digits
    otp_id = new_id("OTP")
    created = now()
    con.execute(f"INSERT INTO {table('otp_codes')} VALUES (?, ?, ?, ?, 'email', ?, ?, 0, ?, 'pending', NULL)",
                [otp_id, case["case_id"], case["customer_id"], hash_code(otp_id, code),
                 created, created + timedelta(minutes=OTP_TTL_MIN), OTP_MAX_ATTEMPTS])

    first = profile.get("first_name") or "cliente"
    body = (f"Hola {first},\n\nTu código de verificación de LATAM Bank es: {code}\n\n"
            f"Vence en {OTP_TTL_MIN} minutos. Escríbelo únicamente en la conversación de WhatsApp "
            f"que inició LATAM Bank. Ningún asesor te lo pedirá por llamada.\n\n"
            f"Si no reconoces esta solicitud, ignora este mensaje.")
    result = fa.send_email(case["contact_email"], "LATAM Bank: tu código de verificación", body)
    # Always report the delivery; optionally show the code so testing never gets stuck
    print(f"  [otp] correo a {fa.mask_email(case['contact_email'])}: {result}", flush=True)
    if SHOW_OTP_IN_CONSOLE:
        print(f"  [demo] código OTP = {code}  (SHOW_OTP_IN_CONSOLE=1)", flush=True)
    return {"sent": True, "delivery": result, "expires_in_minutes": OTP_TTL_MIN,
            "email": fa.mask_email(case["contact_email"])}


def verify_otp_code(case_id: str, code: str) -> dict:
    """Check a code against the active OTP of the case, enforcing expiry and attempt limits."""
    con = fa.con
    digits = "".join(c for c in str(code) if c.isdigit())
    row = con.execute(
        f"SELECT otp_id, code_hash, expires_at, attempts, max_attempts, status FROM {table('otp_codes')} "
        f"WHERE case_id = ? AND status IN ('pending', 'verified') ORDER BY created_at DESC LIMIT 1",
        [case_id]).fetchone()
    if not row:
        return {"verified": False, "error": "No active code. Offer to send a new one (resend_otp)."}
    otp_id, code_hash, expires_at, attempts, max_attempts, status = row

    # Idempotency: the same correct code sent again (e.g. a retried request) stays verified
    if status == "verified":
        if now() <= expires_at and hmac.compare_digest(hash_code(otp_id, digits), code_hash):
            return {"verified": True, "note": "Already verified."}
        return {"verified": False, "error": "No active code. Offer to send a new one (resend_otp)."}

    if now() > expires_at:
        con.execute(f"UPDATE {table('otp_codes')} SET status = 'expired' WHERE otp_id = ?", [otp_id])
        return {"verified": False, "error": "The code expired. Offer to send a new one (resend_otp)."}

    if len(digits) == 6 and hmac.compare_digest(hash_code(otp_id, digits), code_hash):
        con.execute(f"UPDATE {table('otp_codes')} SET status = 'verified', verified_at = ? "
                    f"WHERE otp_id = ?", [now(), otp_id])
        update_case(case_id, ST_VERIFIED)
        return {"verified": True}

    attempts += 1
    status = "locked" if attempts >= max_attempts else "pending"
    con.execute(f"UPDATE {table('otp_codes')} SET attempts = ?, status = ? WHERE otp_id = ?",
                [attempts, status, otp_id])
    if status == "locked":
        return {"verified": False, "error": "Too many wrong attempts; the code was locked. "
                                            "Offer to send a new one (resend_otp)."}
    return {"verified": False, "error": "Wrong code.", "attempts_left": max_attempts - attempts}


# ===========================================================================
# 4. Fraud cases
# ===========================================================================
def case_transactions(case: dict) -> list[dict]:
    """Transactions of the case, scored against the customer's history, with card labels."""
    ids = [i for i in (case.get("transaction_ids") or "").split(",") if i]
    history = fa.load_transactions([case["customer_id"]]).get(case["customer_id"], [])
    scored = {t["id"]: t for t in fa.score_transactions(history)}
    txs = [scored[i] for i in ids if i in scored]
    products = fa.load_products([t["product"] for t in txs])
    for t in txs:
        t["card"] = fa.describe_product(products.get(t["product"]))
        if case.get("source") == "anomaly_model":
            t["reasons"] = ["Marcada por el modelo de detección de anomalías del banco"] + t["reasons"]
    return txs


def existing_open_case(customer_id: str) -> str | None:
    """Case still in progress for this customer (avoids sending several OTPs for one anomaly)."""
    row = fa.con.execute(
        f"SELECT case_id FROM {table('fraud_cases')} WHERE customer_id = ? AND status IN "
        f"({', '.join('?' * len(DEDUP_STATUSES))}) ORDER BY created_at DESC LIMIT 1",
        [customer_id, *DEDUP_STATUSES]).fetchone()
    return row[0] if row else None


def open_case(customer_id: str, tx_ids: list[str] | None = None, days: int = fa.DEFAULT_DAYS,
              source: str = "risk_engine", anomaly_score: float | None = None,
              email: str | None = None, whatsapp: str | None = None) -> dict | None:
    """
    Open a fraud case, send the OTP by email and the opening WhatsApp message.
    Returns the case, or None when it was skipped (unknown customer, duplicate, nothing to show).

    source="anomaly_model": the customer was flagged by the external model, so a case is
      ALWAYS opened. The transactions shown are tx_ids if given; otherwise the risk engine's
      suspicious ones; otherwise the top MODEL_FALLBACK_TX by score in the window.
    source="risk_engine": a case is opened only if the rules find suspicious transactions.
    email / whatsapp: explicit recipients for this case (they override the demo/real defaults).
    """
    profile = fa.get_customer_profile(customer_id)
    if profile is None:
        print(f"  {customer_id}: no existe en {fa.CUSTOMERS_TABLE}; se omite.")
        return None
    duplicate = existing_open_case(customer_id)
    if duplicate:
        print(f"  {customer_id}: ya tiene el caso abierto {duplicate}; no se envía otro OTP.")
        return {"case_id": duplicate, "customer_id": customer_id, "duplicate": True}
    history = fa.load_transactions([customer_id]).get(customer_id, [])

    if tx_ids:
        source = "anomaly_model"
        scored = {t["id"]: t for t in fa.score_transactions(history)}
        flagged = [scored[i] for i in tx_ids if i in scored]
    else:
        window, _ = fa.analyze_window(history, days)
        flagged = [t for t in window if t["score"] >= fa.SUSPICIOUS_THRESHOLD]
        if not flagged and source == "anomaly_model":
            # The model saw something the rules did not: show the most unusual recent activity
            flagged = sorted(window, key=lambda t: t["score"], reverse=True)[:MODEL_FALLBACK_TX]
    if not flagged:
        print(f"  {customer_id}: sin transacciones para mostrar en {days} días; no se abre caso.")
        return None

    reasons = sorted({r for t in flagged for r in t["reasons"]})
    if source == "anomaly_model":
        reasons.insert(0, "Cliente marcado por el modelo de detección de anomalías")
    email, whatsapp = fa.resolve_recipients(profile, email, whatsapp)
    case = {
        "case_id": new_id("CASE"), "customer_id": customer_id, "source": source,
        "transaction_ids": ",".join(t["id"] for t in flagged),
        "max_score": max(t["score"] for t in flagged),
        "reasons": " | ".join(reasons)[:2000],
        "contact_email": email, "contact_whatsapp": normalize_phone(whatsapp),
    }
    fa.con.execute(
        f"""INSERT INTO {table('fraud_cases')} (case_id, customer_id, created_at, updated_at, source,
                transaction_ids, max_score, reasons, contact_email, contact_whatsapp, demo_mode,
                status, anomaly_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [case["case_id"], customer_id, now(), now(), source, case["transaction_ids"],
         case["max_score"], case["reasons"], email, case["contact_whatsapp"], fa.DEMO_MODE,
         ST_OTP_SENT, anomaly_score])

    otp = create_otp(case, profile)
    whatsapp_result = fa.send_whatsapp(
        whatsapp, opening_message(profile, email),
        template_vars={"first_name": profile.get("first_name") or "cliente",
                       "email": fa.mask_email(email) or "", "case_id": case["case_id"],
                       "bank": "LATAM Bank"})
    print(f"  {customer_id}: caso {case['case_id']} ({source}) | {len(flagged)} transacciones | "
          f"puntaje máx. {case['max_score']:.0f} | OTP {otp.get('delivery')} | "
          f"WhatsApp {whatsapp_result}")
    return case


def open_cases_for_customers(customer_ids: list[str], days: int = fa.DEFAULT_DAYS,
                             scores: dict[str, float] | None = None) -> list[dict]:
    """Batch entry point for the anomaly model: one case per flagged customer."""
    unique = list(dict.fromkeys(c.strip() for c in customer_ids if c and c.strip()))
    print(f"Procesando {len(unique)} clientes marcados por el modelo...")
    cases = []
    for cid in unique:
        case = open_case(cid, days=days, source="anomaly_model",
                         anomaly_score=(scores or {}).get(cid))
        if case:
            cases.append(case)
    opened = [c for c in cases if not c.get("duplicate")]
    print(f"\nResumen: {len(opened)} casos nuevos, {len(cases) - len(opened)} ya abiertos, "
          f"{len(unique) - len(cases)} omitidos.")
    phones = {c.get("contact_whatsapp") for c in opened}
    if fa.DEMO_MODE and len(opened) > len(phones):
        print(f"Modo demo: {len(opened)} casos comparten {len(phones)} número(s) de WhatsApp; el "
              f"servidor responde al caso más reciente de cada número. Para un caso específico usa "
              f"--chat CASE_ID, o asigna un número por cliente en {fa.DEMO_RECIPIENTS_FILE}.")
    return cases


def read_customer_file(path: str) -> tuple[list[str], dict[str, float]]:
    """CSV with a customer_id column (optional anomaly_score), or a TXT with one ID per line."""
    import csv
    with open(path, encoding="utf-8-sig", newline="") as f:
        text = f.read()
    first_line = text.splitlines()[0] if text.strip() else ""
    if "customer_id" in first_line.lower():
        rows = list(csv.DictReader(text.splitlines()))
        key = next(k for k in rows[0] if k.strip().lower() == "customer_id")
        score_key = next((k for k in rows[0] if k.strip().lower() == "anomaly_score"), None)
        ids = [r[key] for r in rows]
        scores = {r[key]: float(r[score_key]) for r in rows if score_key and r.get(score_key)}
        return ids, scores
    return [line.split(",")[0] for line in text.splitlines()], {}


def process_queue(days: int = fa.DEFAULT_DAYS, limit: int = 50) -> list[dict]:
    """Open cases for pending rows of fraud_ops.anomaly_queue and mark them as processed."""
    rows = fa.con.execute(
        f"SELECT DISTINCT customer_id, MAX(anomaly_score) FROM {table('anomaly_queue')} "
        f"WHERE processed_at IS NULL GROUP BY customer_id LIMIT {int(limit)}").fetchall()
    if not rows:
        print("No hay clientes pendientes en la cola.")
        return []
    scores = {cid: score for cid, score in rows if score is not None}
    cases = open_cases_for_customers([cid for cid, _ in rows], days, scores)
    by_customer = {c["customer_id"]: c for c in cases}
    for cid, _ in rows:
        case = by_customer.get(cid)
        result = ("duplicate" if case and case.get("duplicate")
                  else "case_opened" if case else "not_found")
        fa.con.execute(
            f"UPDATE {table('anomaly_queue')} SET processed_at = ?, case_id = ?, result = ? "
            f"WHERE customer_id = ? AND processed_at IS NULL",
            [now(), case["case_id"] if case else None, result, cid])
    return cases


def opening_message(profile: dict, email: str | None) -> str:
    """First WhatsApp message. It reveals nothing about the account (anti-phishing)."""
    first = profile.get("first_name") or "cliente"
    return (f"Hola {first}, te escribe el equipo de seguridad de LATAM Bank. Detectamos actividad "
            f"inusual en tu cuenta. Para proteger tu información, primero verifica tu identidad: "
            f"escribe aquí el código de 6 dígitos que enviamos a tu correo {fa.mask_email(email)}. "
            f"Nunca te pediremos contraseñas, PIN ni CVV.")


# ===========================================================================
# 5. Agent routing (service_agents)
# ===========================================================================
def language_code(value: str) -> str | None:
    """Map 'es', 'spanish', 'Español', 'pt', 'portugués', 'en', 'inglés'... to es / pt / en."""
    v = normalize(value)
    if v.startswith(("es", "sp")):
        return "es"
    if v.startswith(("pt", "po")):
        return "pt"
    if v.startswith(("en", "in")):
        return "en"
    return None


def on_shift(shift: str | None, hour: int) -> bool:
    s = normalize(shift)
    if s == "rotating":
        return True
    if s not in SHIFT_HOURS:
        return False
    start, end = SHIFT_HOURS[s]
    return start <= hour < end if start < end else hour >= start or hour < end


def load_agents() -> list[dict]:
    """Active agents, deduplicated by agent_id (the bronze layer has ~2% duplicates)."""
    cur = fa.con.execute(f"""
        SELECT agent_id, first_name, last_name, email, native_accent, country_of_origin,
               agent_type, experience_level, languages, specialty, avg_csat,
               total_monthly_interactions, work_shift
        FROM {AGENTS_TABLE}
        WHERE agent_status = 'Active'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY agent_id ORDER BY agent_id) = 1
    """)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r)) for r in cur.fetchall()]


def rank_agents(language: str, profile: dict, hour: int | None = None, top: int = 3) -> list[dict]:
    """
    Hard filters: speaks the conversation language, active, can take a chat.
    Soft score: fraud specialty, accent, country, on shift, digital/hybrid, experience,
    CSAT, and lower workload.
    """
    lang = language_code(language)
    if lang is None:
        return []
    keyword = LANGUAGE_KEYWORDS[lang]
    hour = datetime.now().hour if hour is None else hour
    accent = normalize(profile.get("accent"))
    country = normalize(profile.get("country"))
    ranked = []
    for a in load_agents():
        if keyword not in normalize(a["languages"]):
            continue  # HARD RULE: never route a conversation to an agent who doesn't speak it
        if normalize(a["agent_type"]) in EXCLUDED_AGENT_TYPES:
            continue
        score, why = 0.0, [f"habla {LANGUAGE_NAMES[lang]}"]
        if "fraude" in normalize(a["specialty"]):
            score += W_FRAUD_SPECIALTY
            why.append("especialista en fraudes")
        if accent and accent != "neutral" and normalize(a["native_accent"]) == accent:
            score += W_ACCENT
            why.append(f"mismo acento ({a['native_accent']})")
        if country and normalize(a["country_of_origin"]) == country:
            score += W_COUNTRY
            why.append(f"mismo país ({a['country_of_origin']})")
        if on_shift(a["work_shift"], hour):
            score += W_ON_SHIFT
            why.append(f"en turno ({a['work_shift']})")
        if normalize(a["agent_type"]) in CHAT_AGENT_TYPES:
            score += W_CHAT_TYPE
            why.append(f"atiende chat ({a['agent_type']})")
        score += W_EXPERIENCE.get(normalize(a["experience_level"]), 0)
        csat = float(a["avg_csat"]) if a["avg_csat"] is not None else 3.5
        score += csat * 3
        score -= (a["total_monthly_interactions"] or 0) / 100  # balance workload
        ranked.append({**a, "routing_score": round(score, 1), "routing_reasons": why})
    ranked.sort(key=lambda a: a["routing_score"], reverse=True)
    return ranked[:top]


def escalate(case: dict, profile: dict, language: str, reason: str, summary: str) -> dict:
    """Pick the best agent, store the escalation and notify the agent."""
    lang = language_code(language) or "es"
    candidates = rank_agents(lang, profile)
    escalation_id = new_id("ESC")
    best = candidates[0] if candidates else None
    fa.con.execute(
        f"INSERT INTO {table('escalations')} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [escalation_id, case["case_id"], case["customer_id"],
         best["agent_id"] if best else None,
         f"{best['first_name']} {best['last_name']}" if best else None,
         lang, reason, summary,
         best["routing_score"] if best else None,
         "; ".join(best["routing_reasons"]) if best else None,
         now(), "open" if best else "pending_no_agent"])
    if not best:
        return {"escalated": False, "escalation_id": escalation_id,
                "error": f"No active agent speaks {LANGUAGE_NAMES.get(lang, lang)}. "
                         f"The request was queued; a human will contact the customer later."}

    update_case(case["case_id"], ST_ESCALATED)
    # Notify the agent (in demo mode, to the test inbox instead of the agent's address)
    to = os.environ.get("TEST_ALERT_EMAIL") if fa.DEMO_MODE else best["email"]
    note = (f"Nuevo caso escalado: {case['case_id']} | cliente {case['customer_id']}\n"
            f"Idioma: {LANGUAGE_NAMES[lang]} | Motivo: {reason}\n\nResumen:\n{summary}\n\n"
            f"Por qué te asignamos: {'; '.join(best['routing_reasons'])}")
    delivery = fa.send_email(to, f"LATAM Bank: caso {case['case_id']} asignado", note)
    print(f"  [escalamiento] {escalation_id} -> {best['agent_id']} {best['first_name']} "
          f"{best['last_name']} ({best['routing_score']}) | aviso: {delivery}", flush=True)
    return {"escalated": True, "escalation_id": escalation_id,
            "agent_first_name": best["first_name"], "agent_specialty": best["specialty"],
            "language": LANGUAGE_NAMES[lang]}


# ===========================================================================
# 6. Gemini tools for the WhatsApp conversation
# ===========================================================================
def build_flow_tools(case: dict, profile: dict) -> list:
    """Tools bound to ONE case. Data tools refuse to run until the OTP is verified."""
    # Session state. Gemini calls can be retried (e.g. after a 503), which re-runs the tools,
    # so every tool with side effects must be idempotent within the session.
    state = {"verified": False, "last_resend": None, "escalation": None}
    not_verified = {"error": "Identity not verified yet. Ask for the 6-digit code sent by email "
                             "and call verify_otp. Do not share any account information."}

    def verify_otp(code: str) -> dict:
        """Verifies the 6-digit one-time code the customer received by email.

        Args:
            code: The code exactly as the customer wrote it.
        """
        print("  [tool] verify_otp(******)", flush=True)  # never log the code
        if state["verified"]:
            return {"verified": True, "note": "Identity already verified in this conversation."}
        result = verify_otp_code(case["case_id"], code)
        if result.get("verified"):
            state["verified"] = True  # never downgraded by a later (retried) call
        return result

    def resend_otp() -> dict:
        """Sends a new one-time code to the customer's email (when it expired or was locked)."""
        print("  [tool] resend_otp()", flush=True)
        if state["verified"]:
            return {"sent": False, "note": "The customer is already verified; no code is needed."}
        if state["last_resend"] and (now() - state["last_resend"]).total_seconds() < 60:
            return {"sent": True, "note": "A code was sent less than a minute ago; ask the "
                                          "customer to check their email (and spam folder)."}
        result = create_otp(case, profile)
        if result.get("sent"):
            state["last_resend"] = now()
        return result

    def get_case_details() -> dict:
        """Returns the unusual transactions of this case with their risk score and reasons. Requires verify_otp first."""
        print("  [tool] get_case_details()", flush=True)
        if not state["verified"]:
            return not_verified
        txs = case_transactions(case)
        return {"case_id": case["case_id"],
                "flagged_by": "anomaly detection model" if case.get("source") == "anomaly_model"
                              else "rule-based risk engine",
                "anomaly_score": case.get("anomaly_score"),
                "transactions": [fa.to_json(t) for t in txs]}

    def list_recent_transactions(days: int, limit: int) -> dict:
        """Lists the customer's recent transactions, newest first. Requires verify_otp first.

        Args:
            days: Days to look back from the last recorded transaction (use 180 by default).
            limit: Maximum number of transactions (for example 10).
        """
        print(f"  [tool] list_recent_transactions(days={days}, limit={limit})", flush=True)
        if not state["verified"]:
            return not_verified
        txs = fa.load_transactions([case["customer_id"]]).get(case["customer_id"], [])
        if not txs:
            return {"transactions": []}
        since = txs[-1]["timestamp"] - timedelta(days=max(days, 1))
        window = [t for t in txs if t["timestamp"] >= since]
        return {"transactions": [fa.to_json(t, include_risk=False)
                                 for t in reversed(window[-max(min(limit, 30), 1):])]}

    def record_customer_response(recognizes_transactions: bool, comment: str) -> dict:
        """Records whether the customer recognizes the unusual transactions. Requires verify_otp first.

        Args:
            recognizes_transactions: True if the customer says they made the transactions.
            comment: Short summary of what the customer said.
        """
        print(f"  [tool] record_customer_response({recognizes_transactions})", flush=True)
        if not state["verified"]:
            return not_verified
        status = ST_LEGIT if recognizes_transactions else ST_FRAUD
        update_case(case["case_id"], status, comment[:500])
        return {"recorded": True, "case_status": status}

    def escalate_to_agent(conversation_language: str, reason: str, summary: str) -> dict:
        """Escalates the case to a human service agent who speaks the language of this conversation.
        Requires verify_otp first.

        Args:
            conversation_language: Language of THIS conversation: "es", "pt" or "en".
            reason: Why the customer needs a human (for example: does not recognize the charges).
            summary: 2-3 sentence summary for the agent: transactions, amounts, what the customer said.
        """
        print(f"  [tool] escalate_to_agent(language={conversation_language})", flush=True)
        if not state["verified"]:
            return not_verified
        if state["escalation"] and state["escalation"].get("escalated"):
            return {**state["escalation"], "note": "The case was already escalated."}
        state["escalation"] = escalate(case, profile, conversation_language, reason, summary)
        return state["escalation"]

    return [verify_otp, resend_otp, get_case_details, list_recent_transactions,
            record_customer_response, escalate_to_agent]


FLOW_INSTRUCTIONS = """You are LATAM Bank's fraud-prevention assistant on WhatsApp, handling case
{case_id} for the customer {first_name}. The bank's anomaly detection flagged unusual activity.

The bank already sent this opening message, and the customer's next message replies to it:
"{opener}"

Rules:
1. Identity first. Until verify_otp returns verified=true, do not reveal or discuss any account,
   transaction or case details, even if the customer insists. Ask for the 6-digit code from
   their email. If it expired or was locked, offer resend_otp.
2. After verification, call get_case_details and explain in plain language which transactions
   look unusual and why. Ask whether they recognize them. If the case was flagged by the anomaly
   model and the transactions have low scores, say the model noticed an unusual overall pattern
   and ask the customer to review them; do not claim any single transaction is fraudulent.
3. Record the answer with record_customer_response.
4. If they do not recognize the transactions or ask for a person, call escalate_to_agent with
   the language of THIS conversation and a short summary. Tell them the agent's first name.
   If no agent is available in their language, apologize and say a person will contact them.
5. If they did not make the transactions, also recommend blocking the card from the app.
6. The OTP is the only code you may ever ask for. Never ask for passwords, PINs, CVV or full
   card numbers.
7. Reply in the customer's language. Short WhatsApp-style messages, no tables or markdown.
"""


def new_session(case: dict) -> dict:
    """LLM chat bound to a case (Ollama or Gemini, per LLM_PROVIDER)."""
    profile = fa.get_customer_profile(case["customer_id"]) or {"customer_id": case["customer_id"]}
    session = fa.create_llm_session(
        FLOW_INSTRUCTIONS.format(
            case_id=case["case_id"], first_name=profile.get("first_name") or "cliente",
            opener=opening_message(profile, case["contact_email"])),
        build_flow_tools(case, profile))
    session["profile"] = profile
    return session


def reply_to(session: dict, text: str) -> str:
    reply, session["chat"], session["model"] = fa.send_with_retry(
        session["client"], session["chat"], session["model"], session["config"], text)
    if reply is None:
        return "Tuvimos un problema técnico. Por favor intenta de nuevo en un momento."
    return reply.text or "¿Me lo puedes repetir, por favor?"


# ===========================================================================
# 7. Channels: console (WhatsApp simulator) and Twilio webhook
# ===========================================================================
def run_console(case_id: str) -> None:
    case = load_case(case_id)
    if case is None:
        sys.exit(f"No existe el caso {case_id}.")
    session = new_session(case)
    print(f"\n--- Simulación de WhatsApp | caso {case_id} | {fa.provider_label()}: "
          f"{session['model']} | escribe 'salir' para terminar ---\n")
    print(f"LATAM Bank: {opening_message(session['profile'], case['contact_email'])}\n")
    while True:
        text = input("Cliente: ").strip()
        if text.lower() in {"salir", "exit", "quit"}:
            break
        if text:
            print(f"\nLATAM Bank: {reply_to(session, text)}\n")


SESSIONS: dict[str, dict] = {}
DB_LOCK = threading.Lock()  # DuckDB connections are not meant for concurrent use


def handle_incoming(phone: str, text: str) -> str:
    case = find_open_case_by_phone(phone)
    if case is None:
        return ("Hola, este es el canal de seguridad de LATAM Bank. "
                "No tienes casos abiertos en este momento.")
    session = SESSIONS.get(case["case_id"])
    if session is None:
        session = SESSIONS[case["case_id"]] = new_session(case)
    return reply_to(session, text)


def valid_twilio_signature(url: str, params: dict, signature: str | None) -> bool:
    """Twilio request validation: HMAC-SHA1 of URL + sorted params, keyed with the auth token."""
    token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    data = url + "".join(k + params[k] for k in sorted(params))
    expected = base64.b64encode(hmac.new(token.encode(), data.encode(), hashlib.sha1).digest()).decode()
    return hmac.compare_digest(expected, signature or "")


class WhatsAppWebhook(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path.split("?")[0] != "/whatsapp":
            self.send_response(404)
            self.end_headers()
            return
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        params = {k: v[0] for k, v in parse_qs(raw).items()}
        if PUBLIC_URL and not valid_twilio_signature(PUBLIC_URL.rstrip("/") + self.path, params,
                                                     self.headers.get("X-Twilio-Signature")):
            self.send_response(403)
            self.end_headers()
            return
        # Answer Twilio immediately (it times out after ~15 s) and reply asynchronously
        self.send_response(200)
        self.send_header("Content-Type", "text/xml")
        self.end_headers()
        self.wfile.write(b"<Response></Response>")
        phone, text = params.get("From", ""), params.get("Body", "").strip()
        if phone and text:
            threading.Thread(target=self._process, args=(phone, text), daemon=True).start()

    @staticmethod
    def _process(phone: str, text: str) -> None:
        print(f"[whatsapp] <- {fa.mask_phone(phone)}", flush=True)
        with DB_LOCK:
            answer = handle_incoming(phone, text)
        print(f"[whatsapp] -> {fa.mask_phone(phone)}: {fa.send_whatsapp(phone, answer)}", flush=True)

    def log_message(self, *args):  # keep the console clean
        pass


def run_server() -> None:
    if not PUBLIC_URL:
        print("[aviso] PUBLIC_URL no está definida: no se valida la firma de Twilio.")
    print(f"Webhook de WhatsApp en http://0.0.0.0:{SERVER_PORT}/whatsapp (Ctrl+C para detener)")
    ThreadingHTTPServer(("0.0.0.0", SERVER_PORT), WhatsAppWebhook).serve_forever()


# ===========================================================================
# 8. CLI
# ===========================================================================
def run_route_preview(customer_id: str, language: str) -> None:
    profile = fa.get_customer_profile(customer_id)
    if profile is None:
        sys.exit(f"No encontré al cliente {customer_id}.")
    print(f"Cliente {customer_id}: país {profile.get('country')}, acento {profile.get('accent')}, "
          f"idioma de la conversación: {language}\n")
    ranked = rank_agents(language, profile, top=5)
    if not ranked:
        print("Ningún asesor activo habla ese idioma.")
    for a in ranked:
        print(f"{a['routing_score']:>6}  {a['agent_id']}  {a['first_name']} {a['last_name']} | "
              f"{a['languages']} | {a['specialty']} | {a['agent_type']} | {a['work_shift']}")
        print(f"        {'; '.join(a['routing_reasons'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description="OTP-verified fraud conversation flow")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--setup", action="store_true", help="create the fraud_ops tables")
    mode.add_argument("--trigger", nargs="+", metavar="CUSTOMER_ID",
                      help="open cases for these customers (one ID also opens the console chat)")
    mode.add_argument("--from-file", metavar="PATH", help="CSV/TXT with anomalous customer_ids")
    mode.add_argument("--process-queue", action="store_true", help="process fraud_ops.anomaly_queue")
    mode.add_argument("--chat", metavar="CASE_ID", help="resume a case in the console")
    mode.add_argument("--route", metavar="CUSTOMER_ID", help="preview the agent routing")
    mode.add_argument("--whatsapp-server", action="store_true", help="run the Twilio webhook")
    mode.add_argument("--test-whatsapp", metavar="NUMBER",
                      help="send a test opener to this number (+57...) and show Twilio's answer")
    parser.add_argument("--tx", nargs="+", metavar="TX_ID", help="transactions flagged by your model")
    parser.add_argument("--language", default="es", help="conversation language for --route")
    parser.add_argument("--days", type=int, default=fa.DEFAULT_DAYS)
    parser.add_argument("--no-chat", action="store_true", help="with --trigger: do not open the console chat")
    parser.add_argument("--email", help="with --trigger: send the OTP to this email")
    parser.add_argument("--whatsapp", help="with --trigger: start the WhatsApp chat on this number (+57...)")
    args = parser.parse_args()

    if args.email and not fa.is_valid_email(args.email):
        sys.exit(f"Correo inválido: {args.email}")
    if args.whatsapp and not fa.is_valid_phone(args.whatsapp):
        sys.exit(f"WhatsApp inválido: {args.whatsapp}. Usa formato internacional, ej. +573001234567")
    if (args.email or args.whatsapp) and not (args.trigger and len(args.trigger) == 1):
        sys.exit("--email y --whatsapp solo se usan con --trigger y un solo cliente.")
    if args.test_whatsapp:
        # No database needed: only checks the Twilio configuration
        if not fa.is_valid_phone(args.test_whatsapp):
            sys.exit("Número inválido. Usa formato internacional, ej. +573001234567")
        print(fa.delivery_summary())
        print(f"Plantilla: {os.environ.get('TWILIO_CONTENT_SID') or 'no (texto libre)'}")
        test_vars = {"first_name": "Prueba", "email": "p***@gmail.com",
                     "case_id": "CASE-TEST", "bank": "LATAM Bank"}
        text = opening_message({"first_name": "Prueba"}, "prueba@gmail.com")
        print("Resultado:", fa.send_whatsapp(args.test_whatsapp, text, template_vars=test_vars))
        return
    if not OTP_SECRET:
        print("[aviso] OTP_SECRET no está definido en el .env: los hashes de OTP usan una clave vacía.")
    fa.init(fa.connect())
    setup_tables()  # idempotent: safe to run every time
    print(fa.delivery_summary())

    if args.setup:
        print(f"Tablas listas en {fa.DATABASE}.{OPS_SCHEMA}: "
              f"anomaly_queue, fraud_cases, otp_codes, escalations")
    elif args.trigger and (len(args.trigger) == 1 or args.tx):
        if args.tx and len(args.trigger) > 1:
            sys.exit("--tx solo se puede usar con un cliente.")
        case = open_case(args.trigger[0], args.tx, args.days, source="anomaly_model",
                         email=args.email, whatsapp=args.whatsapp)
        if case and not args.no_chat:
            run_console(case["case_id"])
    elif args.trigger:
        open_cases_for_customers(args.trigger, args.days)
    elif args.from_file:
        ids, scores = read_customer_file(args.from_file)
        open_cases_for_customers(ids, args.days, scores)
    elif args.process_queue:
        process_queue(args.days)
    elif args.chat:
        run_console(args.chat)
    elif args.route:
        run_route_preview(args.route, args.language)
    else:
        run_server()


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nHasta luego.")
