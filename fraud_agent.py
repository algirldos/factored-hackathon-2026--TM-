#!/usr/bin/env python3
"""
Fraud detection and alerting agent - LATAM Bank (Factored AI & Data Hackathon 2026).

Gemini talks to the authenticated customer, queries their transactions in MotherDuck,
scores fraud risk with explainable rules plus the bank's fraud_score, and alerts the
customer by email and/or WhatsApp (with human confirmation and a demo mode).

Usage:
  python fraud_agent.py --columns                 Show how table columns were mapped
  python fraud_agent.py --demo-customers          Suggest interesting customers for a demo
  python fraud_agent.py --customer CUST-000123    Chat with the agent for that customer
  python fraud_agent.py --monitor ID1 ID2 ...     Proactive review and alerts, no chat
  python fraud_agent.py --evaluate 40             Measure the rules against is_fraud

See README.md for setup and the full list of environment variables.
"""
import argparse
import base64
import json
import math
import os
import re
import smtplib
import statistics
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta
from decimal import Decimal
from email.message import EmailMessage
from pathlib import Path

import duckdb

if __name__ == "__main__":
    sys.modules.setdefault("fraud_agent", sys.modules["__main__"])


# ===========================================================================
# 0. Environment variables (.env next to this script; never overrides existing vars)
# ===========================================================================
def load_dotenv_file() -> None:
    """Minimal .env loader so the script also works outside VS Code."""
    path = Path(__file__).with_name(".env")
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv_file()

# ===========================================================================
# 1. Configuration
# ===========================================================================
DATABASE = os.environ.get("MOTHERDUCK_DB", "latam_bank")
TRANSACTIONS_TABLE = os.environ.get("TRANSACTIONS_TABLE", "bronze.transactions")
CUSTOMERS_TABLE = os.environ.get("CUSTOMERS_TABLE", "bronze.customers")
PRODUCTS_TABLE = os.environ.get("PRODUCTS_TABLE", "bronze.products")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.8-flash")
# Fallback chain, comma-separated. Each model has its own free-tier quota.
GEMINI_FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_FALLBACK_MODELS", os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash")
).split(",") if m.strip()]
# LLM provider: "anthropic" (Claude API), "ollama" (local models) or "gemini" (Google API)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5")
ANTHROPIC_FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "ANTHROPIC_FALLBACK_MODELS", "claude-haiku-4-5-20251001").split(",") if m.strip()]
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1024"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_FALLBACK_MODELS = [m.strip() for m in os.environ.get(
    "OLLAMA_FALLBACK_MODELS", "gemma4:e4b").split(",") if m.strip()]
OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))       # context window
OLLAMA_NUM_PREDICT = int(os.environ.get("OLLAMA_NUM_PREDICT", "1024"))  # max tokens per answer
OLLAMA_THINK = os.environ.get("OLLAMA_THINK", "0") == "1"            # reasoning: slower
MAX_TOOL_CALLS = 5                 # tool calls per customer message, for every provider
RETRYABLE_CODES = {500, 503, 504, 529}  # temporary overload: worth retrying the same model
QUOTA_CODE = 429                   # quota exhausted: retrying wastes time, switch model
MAX_RETRIES = 3

DEMO_MODE = os.environ.get("DEMO_MODE", "1") != "0"          # alerts go only to demo recipients
CONFIRM_SENDS = os.environ.get("CONFIRM_SENDS", "1") != "0"  # a human approves every alert

# Delivery switches: turn a channel off without touching its credentials
EMAIL_ENABLED = os.environ.get("EMAIL_ENABLED", "1") != "0"
WHATSAPP_ENABLED = os.environ.get("WHATSAPP_ENABLED", "1") != "0"
# Optional CSV (customer_id,email,whatsapp) with a demo recipient per customer
DEMO_RECIPIENTS_FILE = os.environ.get("DEMO_RECIPIENTS_FILE", "demo_recipients.csv")

SUSPICIOUS_THRESHOLD = 30  # score from which a transaction is considered suspicious
HIGH_RISK_THRESHOLD = 60   # high-risk score (triggers an alert in --monitor)
MIN_HISTORY = 5            # minimum prior transactions needed to compare against habits
BURST_WINDOW_MIN = 10      # minutes used to detect bursts of transactions
AMOUNT_ZSCORE = 3.0        # standard deviations for an amount to be considered atypical
DEFAULT_DAYS = 180         # ~33 transactions per customer in 3 years: 30 days is usually 1 tx
MAX_TRAVEL_KMH = 900       # faster than a commercial flight = "impossible travel"
MIN_TRAVEL_KM = 300        # ignore short distances (GPS noise, nearby cities)
IN_PERSON_CHANNELS = {"atm", "branch", "pos"}  # channels where the card is physically present

# Possible column names per logical field (the first one that exists wins).
# If your table uses a different name, add it at the START of the matching list.
TX_COLUMN_CANDIDATES = {
    "id": ["transaction_id", "id"],
    "customer": ["customer_id", "client_id"],
    "timestamp": ["transaction_timestamp", "transaction_datetime", "transaction_date",
                  "timestamp", "created_at", "process_date"],
    "amount": ["amount", "transaction_amount", "amount_local", "value"],
    "amount_usd": ["amount_usd"],  # used for statistics so currencies are comparable
    "currency": ["currency", "currency_code"],
    "product": ["product_id", "card_id", "account_id"],
    "merchant": ["merchant_name", "merchant", "merchant_id"],
    "category": ["merchant_category", "category", "mcc"],
    "city": ["transaction_city", "city", "merchant_city"],
    "country": ["transaction_country", "country", "merchant_country", "country_code"],
    "channel": ["channel", "transaction_channel"],
    "type": ["transaction_type", "type"],
    "status": ["transaction_status", "status"],
    "response_code": ["response_code"],
    "model_score": ["fraud_score", "risk_score"],
    "fraud_label": ["is_fraud", "fraud_flag", "is_fraudulent"],
    "lat": ["latitude", "lat"],
    "lon": ["longitude", "lon", "lng"],
}
CUSTOMER_COLUMN_CANDIDATES = {
    "id": ["customer_id", "client_id", "id"],
    "full_name": ["full_name", "customer_name", "name"],
    "first_name": ["first_name", "given_name"],
    "last_name": ["last_name", "surname", "family_name"],
    "email": ["email", "email_address"],
    "phone": ["mobile_phone", "phone_number", "phone", "mobile"],
    "country": ["country", "country_code"],
    "language": ["preferred_language", "language"],
    "segment": ["segment"],
    "status": ["customer_status", "status"],
    "accent": ["detected_accent", "accent"],
    "city": ["city"],
    "updated_at": ["last_updated", "updated_at"],
}
REQUIRED_TX_FIELDS = ["id", "customer", "timestamp", "amount"]
DECLINED_VALUES = {"declined", "rejected", "denied", "failed", "rechazada", "rechazado"}
TRUE_VALUES = {"true", "1", "yes", "y", "t", "si", "sí"}

AUDIT_LOG = Path(__file__).with_name("sent_alerts.jsonl")

# Module-level state, filled in by init()
con: duckdb.DuckDBPyConnection | None = None
TX_COLUMNS: dict = {}        # logical field -> real column name in the transactions table
CUSTOMER_COLUMNS: dict = {}  # logical field -> real column name in the customers table
SCORE_SCALE = 1.0            # multiplier that brings fraud_score to a 0-100 scale


# ===========================================================================
# 2. Connection and column mapping
# ===========================================================================
def quote_ident(column: str) -> str:
    """Quote a SQL identifier."""
    return '"' + column.replace('"', '""') + '"'


def table_columns(table: str) -> list[str]:
    """Real column names of `schema.table` in the current database."""
    schema, name = table.split(".", 1) if "." in table else ("main", table)
    rows = con.execute(
        """SELECT column_name FROM information_schema.columns
           WHERE table_catalog = current_database() AND table_schema = ? AND table_name = ?
           ORDER BY ordinal_position""",
        [schema, name],
    ).fetchall()
    return [r[0] for r in rows]


def resolve_columns(columns: list[str], candidates: dict) -> dict:
    """Map each logical field to the first candidate present in `columns` (case-insensitive)."""
    by_lower = {c.lower(): c for c in columns}
    return {field: next((by_lower[c] for c in options if c in by_lower), None)
            for field, options in candidates.items()}


def init(connection: duckdb.DuckDBPyConnection) -> None:
    """Store the connection, detect columns and the scale of fraud_score."""
    global con, TX_COLUMNS, CUSTOMER_COLUMNS, SCORE_SCALE
    con = connection
    tx_cols = table_columns(TRANSACTIONS_TABLE)
    if not tx_cols:
        sys.exit(f"No encontré la tabla {TRANSACTIONS_TABLE}. Revisa TRANSACTIONS_TABLE y MOTHERDUCK_DB.")
    TX_COLUMNS = resolve_columns(tx_cols, TX_COLUMN_CANDIDATES)
    missing = [f for f in REQUIRED_TX_FIELDS if not TX_COLUMNS[f]]
    if missing:
        sys.exit(f"Faltan columnas obligatorias en {TRANSACTIONS_TABLE}: {missing}\n"
                 f"Columnas reales: {tx_cols}\nAgrega el nombre correcto en TX_COLUMN_CANDIDATES.")
    CUSTOMER_COLUMNS = resolve_columns(table_columns(CUSTOMERS_TABLE), CUSTOMER_COLUMN_CANDIDATES)

    # fraud_score may come as a probability (0-1) or as 0-100; normalize to 0-100
    if TX_COLUMNS["model_score"]:
        max_score = con.execute(
            f"SELECT MAX(TRY_CAST({quote_ident(TX_COLUMNS['model_score'])} AS DOUBLE)) "
            f"FROM {TRANSACTIONS_TABLE}"
        ).fetchone()[0]
        SCORE_SCALE = 100.0 if max_score is not None and max_score <= 1.0 else 1.0


def connect() -> duckdb.DuckDBPyConnection:
    token = os.environ.get("MOTHERDUCK_TOKEN")
    if not token:
        sys.exit("Falta MOTHERDUCK_TOKEN en el .env")
    return duckdb.connect(f"md:{DATABASE}?motherduck_token={token}")


# ===========================================================================
# 3. Data access
# ===========================================================================
def _tx_select_list() -> str:
    """SELECT list that exposes every logical field (NULL when the column does not exist)."""
    sql_types = {"id": "VARCHAR", "customer": "VARCHAR", "timestamp": "TIMESTAMP",
                 "amount": "DOUBLE", "amount_usd": "DOUBLE", "model_score": "DOUBLE",
                 "lat": "DOUBLE", "lon": "DOUBLE"}
    parts = []
    for field in TX_COLUMN_CANDIDATES:
        column = TX_COLUMNS.get(field)
        if not column:
            parts.append(f"NULL AS {field}")
        else:
            parts.append(f"TRY_CAST({quote_ident(column)} AS {sql_types.get(field, 'VARCHAR')}) AS {field}")
    return ", ".join(parts)


def load_transactions(customer_ids: list[str]) -> dict[str, list[dict]]:
    """Deduplicated transactions for one or more customers, sorted by timestamp."""
    cols = TX_COLUMNS
    if len(customer_ids) == 1:
        # Plain equality lets DuckDB push the filter down (cheaper on MotherDuck)
        where, params = f"{quote_ident(cols['customer'])} = ?", [customer_ids[0]]
    else:
        where = f"CAST({quote_ident(cols['customer'])} AS VARCHAR) IN (SELECT UNNEST(?::VARCHAR[]))"
        params = [list(customer_ids)]
    # The bronze layer has ~2% duplicates: keep one row per transaction_id
    sql = f"""
        SELECT {_tx_select_list()} FROM {TRANSACTIONS_TABLE}
        WHERE {where}
        QUALIFY ROW_NUMBER() OVER (PARTITION BY {quote_ident(cols['id'])}
                                   ORDER BY {quote_ident(cols['timestamp'])} DESC) = 1
    """
    cur = con.execute(sql, params)
    names = [d[0] for d in cur.description]
    by_customer: dict[str, list[dict]] = defaultdict(list)
    for row in cur.fetchall():
        tx = dict(zip(names, row))
        if tx["timestamp"] is not None:
            by_customer[tx["customer"]].append(tx)
    for txs in by_customer.values():
        txs.sort(key=lambda t: t["timestamp"])
    return by_customer


def get_customer_profile(customer_id: str) -> dict | None:
    """Most recent customer record, or None if the customer does not exist."""
    cols = CUSTOMER_COLUMNS
    if not cols.get("id"):
        return {"customer_id": customer_id}
    fields = [f"CAST({quote_ident(c)} AS VARCHAR) AS {k}" for k, c in cols.items() if c]
    order = f"ORDER BY {quote_ident(cols['updated_at'])} DESC" if cols.get("updated_at") else ""
    cur = con.execute(
        f"SELECT {', '.join(fields)} FROM {CUSTOMERS_TABLE} "
        f"WHERE {quote_ident(cols['id'])} = ? {order} LIMIT 1",
        [customer_id],
    )
    row = cur.fetchone()
    if not row:
        return None
    profile = dict(zip([d[0] for d in cur.description], row))
    if not profile.get("full_name"):
        profile["full_name"] = " ".join(
            x for x in [profile.get("first_name"), profile.get("last_name")] if x) or None
    profile["customer_id"] = customer_id
    return profile


def load_products(product_ids: list[str]) -> dict[str, dict]:
    """Card/account info for display: type, last 4 digits and status (never the full number)."""
    ids = sorted({p for p in product_ids if p})
    if not ids:
        return {}
    try:
        rows = con.execute(f"""
            SELECT CAST(product_id AS VARCHAR), product_type,
                   right(CAST(product_number AS VARCHAR), 4), product_status
            FROM {PRODUCTS_TABLE}
            WHERE CAST(product_id AS VARCHAR) IN (SELECT UNNEST(?::VARCHAR[]))
            QUALIFY ROW_NUMBER() OVER (PARTITION BY product_id ORDER BY last_updated DESC) = 1
        """, [ids]).fetchall()
    except duckdb.Error:
        return {}  # products table missing or with another schema: degrade gracefully
    return {pid: {"product_type": ptype, "last4": last4, "product_status": status}
            for pid, ptype, last4, status in rows}


def describe_product(info: dict | None) -> str | None:
    """Human-readable card label, e.g. 'Credit Card ****1234'."""
    if not info:
        return None
    return " ".join(x for x in [info.get("product_type"),
                                f"****{info['last4']}" if info.get("last4") else None] if x)


# ===========================================================================
# 4. Risk engine (explainable rules + the bank's fraud_score)
# ===========================================================================
def is_true(value) -> bool:
    return value is not None and str(value).strip().lower() in TRUE_VALUES


def risk_level(score: float) -> str:
    if score >= HIGH_RISK_THRESHOLD:
        return "high"
    return "medium" if score >= SUSPICIOUS_THRESHOLD else "low"


def comparable_amount(tx: dict) -> float | None:
    """Amount used for statistics: USD when the column exists, otherwise the local amount."""
    return tx["amount_usd"] if TX_COLUMNS.get("amount_usd") else tx["amount"]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two coordinates, in kilometers."""
    lat1, lon1, lat2, lon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    a = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


def is_in_person(tx: dict) -> bool:
    """True for channels where the card is physically present (or when channel is unknown)."""
    return not tx["channel"] or tx["channel"].strip().lower() in IN_PERSON_CHANNELS


def score_transactions(txs: list[dict], since: datetime | None = None) -> list[dict]:
    """
    Score every transaction (0-100) comparing it ONLY with the customer's earlier history.
    Returns the transactions with timestamp >= since, enriched with scores and reasons.
    is_fraud is never used here: it is the evaluation label, not a signal.
    """
    has_time = any(t["timestamp"].hour or t["timestamp"].minute for t in txs)

    # Running history, updated after each transaction is scored
    amounts: list[float] = []
    countries, cities, categories = set(), set(), set()
    night_count = 0
    last_located = None  # last earlier in-person transaction with coordinates
    usd = bool(TX_COLUMNS.get("amount_usd"))
    scored = []

    for i, tx in enumerate(txs):
        if since is None or tx["timestamp"] >= since:
            reasons, rule_score = [], 0
            prior_count = i
            amount = comparable_amount(tx)

            # Rule 1: amount far above the customer's usual spending
            if amount is not None and len(amounts) >= MIN_HISTORY:
                mean, std = statistics.fmean(amounts), statistics.pstdev(amounts)
                if std > 0 and (amount - mean) / std >= AMOUNT_ZSCORE and amount > 2 * mean:
                    rule_score += 40
                    unit = " USD" if usd else ""
                    reasons.append(f"Monto {amount / mean:.1f} veces mayor a su promedio "
                                   f"({mean:,.0f}{unit})")

            if prior_count >= MIN_HISTORY:
                # Rule 2: new country (or, failing that, new city)
                if tx["country"] and tx["country"] not in countries:
                    rule_score += 30
                    reasons.append(f"Primera transacción registrada en {tx['country']}")
                elif tx["city"] and tx["city"] not in cities:
                    rule_score += 15
                    reasons.append(f"Primera transacción registrada en {tx['city']}")
                # Rule 3: new merchant category with an above-median amount
                if tx["category"] and tx["category"] not in categories and amount and amounts \
                        and amount > statistics.median(amounts):
                    rule_score += 10
                    reasons.append(f"Categoría nueva para el cliente: {tx['category']}")
                # Rule 4: late-night transaction for a customer who almost never does that
                if has_time and tx["timestamp"].hour < 5 and night_count / prior_count < 0.05:
                    rule_score += 15
                    reasons.append(f"Hora inusual ({tx['timestamp']:%H:%M}) para sus hábitos")

            if has_time:
                # Transactions in the preceding burst window
                window_start = tx["timestamp"] - timedelta(minutes=BURST_WINDOW_MIN)
                recent = []
                j = i - 1
                while j >= 0 and txs[j]["timestamp"] >= window_start:
                    recent.append(txs[j])
                    j -= 1
                # Rule 5: burst of transactions
                if len(recent) >= 2:
                    rule_score += 25
                    reasons.append(f"{len(recent) + 1} transacciones en {BURST_WINDOW_MIN} minutos")
                # Rule 6: several declines right before (typical card-testing pattern)
                declines = [p for p in recent
                            if p["status"] and p["status"].strip().lower() in DECLINED_VALUES]
                if len(declines) >= 2:
                    rule_score += 20
                    reasons.append(f"{len(declines)} intentos rechazados justo antes")

            # Rule 7: impossible travel between two in-person transactions
            if (last_located and is_in_person(tx)
                    and tx["lat"] is not None and tx["lon"] is not None):
                km = haversine_km(last_located["lat"], last_located["lon"], tx["lat"], tx["lon"])
                hours = (tx["timestamp"] - last_located["timestamp"]).total_seconds() / 3600
                if km >= MIN_TRAVEL_KM and (hours <= 0 or km / hours > MAX_TRAVEL_KMH):
                    rule_score += 35
                    reasons.append(f"Viaje imposible: {km:,.0f} km desde la compra presencial "
                                   f"anterior en {max(hours, 0) * 60:.0f} minutos")

            rule_score = min(rule_score, 100)
            model_score = None
            if tx["model_score"] is not None:
                model_score = round(min(tx["model_score"] * SCORE_SCALE, 100), 1)
                if model_score >= SUSPICIOUS_THRESHOLD:
                    reasons.append(f"El modelo antifraude del banco le asignó {model_score:.0f}/100")
            # Final score: the strongest of the two signals
            score = max(rule_score, model_score or 0)
            scored.append({**tx, "rule_score": rule_score, "model_score": model_score,
                           "score": score, "level": risk_level(score), "reasons": reasons})

        # Add the current transaction to the running history
        if comparable_amount(tx) is not None:
            amounts.append(comparable_amount(tx))
        if tx["lat"] is not None and tx["lon"] is not None and is_in_person(tx):
            last_located = tx
        if tx["country"]:
            countries.add(tx["country"])
        if tx["city"]:
            cities.add(tx["city"])
        if tx["category"]:
            categories.add(tx["category"])
        if has_time and tx["timestamp"].hour < 5:
            night_count += 1
    return scored


RISK_FIELDS = {"score", "rule_score", "model_score", "level", "reasons"}


def to_json(tx: dict, include_risk: bool = True) -> dict:
    """JSON-safe copy of a transaction for the LLM (never includes the is_fraud label)."""
    out = {}
    for key, value in tx.items():
        if key in {"fraud_label", "customer", "lat", "lon"} or value is None or value == []:
            continue
        if not include_risk and key in RISK_FIELDS:
            continue
        if isinstance(value, datetime):
            value = value.isoformat(sep=" ", timespec="minutes")
        elif isinstance(value, Decimal):
            value = float(value)
        out[key] = value
    return out


def analyze_window(txs: list[dict], days: int) -> tuple[list[dict], datetime | None]:
    """Score the `days` window that ends at the customer's last transaction.
    The dataset is historical, so 'recent' is relative to the last transaction, not today."""
    if not txs:
        return [], None
    last = txs[-1]["timestamp"]
    return score_transactions(txs, since=last - timedelta(days=days)), last


# ===========================================================================
# 5. Alert delivery (SMTP email and WhatsApp via Twilio)
# ===========================================================================
def mask_email(email: str | None) -> str | None:
    if not email or "@" not in email:
        return email
    user, domain = email.split("@", 1)
    return f"{user[:1]}***@{domain}"


def mask_phone(phone: str | None) -> str | None:
    if not phone:
        return phone
    digits = "".join(c for c in phone if c.isdigit())
    return f"***{digits[-4:]}" if len(digits) >= 4 else "***"


def is_valid_email(email: str | None) -> bool:
    return bool(email) and "@" in email and "." in email.split("@")[-1] and " " not in email


def is_valid_phone(phone: str | None) -> bool:
    """E.164 format: '+' followed by 8 to 15 digits (e.g. +573001234567)."""
    digits = (phone or "").replace("whatsapp:", "").replace(" ", "")
    return digits.startswith("+") and digits[1:].isdigit() and 8 <= len(digits) - 1 <= 15


_demo_recipients_cache: dict | None = None


def load_demo_recipients() -> dict[str, dict]:
    """customer_id -> {"email", "whatsapp"} from DEMO_RECIPIENTS_FILE (if the file exists)."""
    global _demo_recipients_cache
    if _demo_recipients_cache is not None:
        return _demo_recipients_cache
    import csv
    path = Path(DEMO_RECIPIENTS_FILE)
    if not path.is_absolute():
        path = Path(__file__).with_name(DEMO_RECIPIENTS_FILE)
    mapping = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                row = {k.strip().lower(): (v or "").strip() for k, v in row.items() if k}
                if row.get("customer_id"):
                    mapping[row["customer_id"]] = {"email": row.get("email") or None,
                                                   "whatsapp": row.get("whatsapp") or None}
    _demo_recipients_cache = mapping
    return mapping


def resolve_recipients(profile: dict, email: str | None = None,
                       whatsapp: str | None = None) -> tuple[str | None, str | None]:
    """
    (email, whatsapp) recipients, by priority:
      1. explicit values (e.g. --email / --whatsapp on the command line)
      2. demo mode: the customer's row in DEMO_RECIPIENTS_FILE, then TEST_ALERT_EMAIL / TEST_ALERT_WHATSAPP
      3. real mode: the customer's own email and mobile phone
    """
    if DEMO_MODE:
        mapped = load_demo_recipients().get(profile.get("customer_id"), {})
        return (email or mapped.get("email") or os.environ.get("TEST_ALERT_EMAIL"),
                whatsapp or mapped.get("whatsapp") or os.environ.get("TEST_ALERT_WHATSAPP"))
    phone = (profile.get("phone") or "").replace(" ", "")
    return email or profile.get("email"), whatsapp or (phone if phone.startswith("+") else None)


def delivery_summary() -> str:
    """One-line description of how alerts will be delivered, printed at startup."""
    def state(enabled: bool, configured: bool) -> str:
        if not enabled:
            return "desactivado"
        return "activo" if configured else "simulado (sin credenciales)"
    smtp_ok = all(os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))
    twilio_ok = all(os.environ.get(k) for k in
                    ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"))
    mode = "DEMO" if DEMO_MODE else "REAL"
    extra = f" | {len(load_demo_recipients())} destinos por cliente en {DEMO_RECIPIENTS_FILE}" \
        if DEMO_MODE and load_demo_recipients() else ""
    return (f"Modo {mode} | correo: {state(EMAIL_ENABLED, smtp_ok)} | "
            f"WhatsApp: {state(WHATSAPP_ENABLED, twilio_ok)}{extra}")


def send_email(to: str | None, subject: str, body: str) -> str:
    if not EMAIL_ENABLED:
        return "disabled (EMAIL_ENABLED=0)"
    host, user, password = (os.environ.get(k) for k in ("SMTP_HOST", "SMTP_USER", "SMTP_PASSWORD"))
    if not (host and user and password and to):
        return "simulated (missing SMTP settings or recipient)"
    msg = EmailMessage()
    msg["Subject"], msg["From"], msg["To"] = subject, os.environ.get("SMTP_FROM", user), to
    msg.set_content(body)
    port = int(os.environ.get("SMTP_PORT", "587"))
    try:
        if port == 465:
            # Implicit TLS: the connection is encrypted from the first byte
            with smtplib.SMTP_SSL(host, port, timeout=20) as smtp:
                smtp.login(user, password)
                smtp.send_message(msg)
        else:
            # STARTTLS (port 587): plain connection upgraded to TLS
            with smtplib.SMTP(host, port, timeout=20) as smtp:
                smtp.starttls()
                smtp.login(user, password)
                smtp.send_message(msg)
        return "sent"
    except smtplib.SMTPAuthenticationError:
        return "error: SMTP rejected the credentials (for Gmail use an App Password)"
    except Exception as e:
        return f"error: {e} (check SMTP_HOST and SMTP_PORT: 587 or 465)"


class _KeepMissing(dict):
    """format_map helper: unknown placeholders are left as they are instead of failing."""
    def __missing__(self, key):
        return "{" + key + "}"


def twilio_template_fields(template_vars: dict) -> dict:
    """
    ContentSid / ContentVariables for a Twilio Content Template (needed by senders that only
    accept approved templates, Twilio error 21654).
    TWILIO_CONTENT_VARIABLES maps template slots to our fields, e.g.
      {"1": "{first_name}", "2": "{email}"}
    Available fields: first_name, email, case_id, bank.
    """
    fields = {"ContentSid": os.environ["TWILIO_CONTENT_SID"].strip()}
    spec = os.environ.get("TWILIO_CONTENT_VARIABLES", "").strip()
    if spec:
        mapping = json.loads(spec)
        rendered = {str(slot): str(value).format_map(_KeepMissing(template_vars))
                    for slot, value in mapping.items()}
        fields["ContentVariables"] = json.dumps(rendered, ensure_ascii=False)
    return fields


def send_whatsapp(to: str | None, body: str, template_vars: dict | None = None) -> str:
    """
    Send a WhatsApp message through Twilio.
    template_vars: pass it for business-initiated messages (the case opener). If
    TWILIO_CONTENT_SID is set, the approved template is sent instead of the free text `body`.
    Replies inside an open conversation are always sent as free text.
    """
    if not WHATSAPP_ENABLED:
        return "disabled (WHATSAPP_ENABLED=0)"
    sid, token, sender = (os.environ.get(k) for k in
                          ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_WHATSAPP_FROM"))
    if not (sid and token and sender and to):
        return "simulated (missing Twilio settings or recipient)"

    def wa(number: str) -> str:
        return number if number.startswith("whatsapp:") else f"whatsapp:{number}"

    fields = {"From": wa(sender), "To": wa(to)}
    if template_vars is not None and os.environ.get("TWILIO_CONTENT_SID", "").strip():
        try:
            fields.update(twilio_template_fields(template_vars))
        except json.JSONDecodeError:
            return "error: TWILIO_CONTENT_VARIABLES is not valid JSON"
        kind = "template"
    else:
        fields["Body"] = body[:1500]  # WhatsApp bodies are limited to ~1600 characters
        kind = "text"

    # Twilio REST API (no SDK needed)
    req = urllib.request.Request(
        f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json",
        data=urllib.parse.urlencode(fields).encode())
    req.add_header("Authorization", "Basic " + base64.b64encode(f"{sid}:{token}".encode()).decode())
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            json.load(resp)
        return "sent" if kind == "text" else "sent (template)"
    except urllib.error.HTTPError as e:
        if e.code == 401:
            return "error: Twilio 401 - check TWILIO_ACCOUNT_SID (starts with AC) and TWILIO_AUTH_TOKEN"
        detail = e.read().decode(errors="ignore")[:300]
        hint = (" -> this sender needs an approved template: set TWILIO_CONTENT_SID "
                "(and TWILIO_CONTENT_VARIABLES)") if '"code":21654' in detail.replace(" ", "") \
            or '"code":63016' in detail.replace(" ", "") else ""
        return f"error: Twilio {e.code} {detail}{hint}"
    except Exception as e:
        return f"error: {e}"


GREETING = re.compile(r"^\s*(hola|ol[áa]|buen[oa]s(\s+\w+)?|estimad[oa])\b[^,:.!\n]*[,:.!]?\s*",
                      re.IGNORECASE)


def compose_alert(profile: dict, suspicious: list[dict], message: str) -> str:
    """Customer-facing alert text (Spanish), including an anti-phishing reminder."""
    first = profile.get("first_name") or (profile.get("full_name") or "").split(" ")[0] or "cliente"
    body = GREETING.sub("", message.strip(), count=1).strip() or message.strip()
    lines = [f"Hola {first},", "", body[:1].upper() + body[1:], "", "Transacciones en revisión:"]
    for tx in suspicious:
        place = " ".join(x for x in [tx.get("merchant") or tx.get("category"), tx.get("city")] if x)
        card = f" | {tx['card']}" if tx.get("card") else ""
        lines.append(f"- {tx['timestamp']:%Y-%m-%d %H:%M} | {tx['amount']:,.2f} "
                     f"{tx.get('currency') or ''} | {place}{card}")
    lines += ["", "Si no reconoces estas transacciones, bloquea tu tarjeta desde la app o "
                  "comunícate con la línea oficial de LATAM Bank.",
              "LATAM Bank nunca te pedirá contraseñas, PIN, CVV ni códigos por este medio."]
    return "\n".join(lines)


def send_alert(profile: dict, suspicious: list[dict], channel: str, message: str,
               source: str) -> dict:
    """Show the alert, ask for human confirmation, send it and write the audit log."""
    channel = channel.strip().lower()
    if channel not in {"email", "whatsapp", "both"}:
        return {"sent": False, "error": "channel must be 'email', 'whatsapp' or 'both'"}
    body = compose_alert(profile, suspicious, message)
    email_to, whatsapp_to = resolve_recipients(profile)

    # Human-in-the-loop: nothing leaves the system without explicit approval
    if CONFIRM_SENDS:
        print("\n" + "-" * 60)
        print(f"ALERTA PROPUESTA ({'MODO DEMO' if DEMO_MODE else 'REAL'}) por {channel}:")
        print(f"  email -> {mask_email(email_to)} | whatsapp -> {mask_phone(whatsapp_to)}")
        print("-" * 60 + "\n" + body + "\n" + "-" * 60)
        if not input("¿Confirmas el envío? (s/n): ").strip().lower().startswith("s"):
            return {"sent": False, "reason": "The human operator did not approve the alert."}

    results = {}
    if channel in {"email", "both"}:
        results["email"] = send_email(email_to, "LATAM Bank: actividad inusual en tu cuenta", body)
    if channel in {"whatsapp", "both"}:
        results["whatsapp"] = send_whatsapp(whatsapp_to, body)

    # Append-only audit trail (one JSON object per line)
    record = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "source": source,
        "customer_id": profile.get("customer_id"), "demo_mode": DEMO_MODE,
        "transactions": [t["id"] for t in suspicious],
        "scores": [t["score"] for t in suspicious], "results": results,
    }
    with AUDIT_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {"sent": any(r.startswith(("sent", "simulated")) for r in results.values()),
            "results": results, "demo_mode": DEMO_MODE}


# ===========================================================================
# 6. Gemini tools (scoped to the authenticated customer)
# ===========================================================================
def build_tools(customer_id: str, profile: dict) -> list:
    """The tools take no customer_id: they can only see the session's customer."""
    session: dict = {"suspicious": {}}  # last risk assessment, keyed by transaction id

    def customer_transactions() -> list[dict]:
        return load_transactions([customer_id]).get(customer_id, [])

    def view_customer_profile() -> dict:
        """Returns the authenticated customer's profile: name, country, language and contact channels (masked)."""
        print("  [tool] view_customer_profile()", flush=True)
        return {"customer_id": customer_id, "name": profile.get("full_name"),
                "country": profile.get("country"), "language": profile.get("language"),
                "segment": profile.get("segment"), "status": profile.get("status"),
                "email": mask_email(profile.get("email")), "phone": mask_phone(profile.get("phone"))}

    def list_transactions(days: int, limit: int) -> dict:
        """Lists the authenticated customer's most recent transactions, newest first.

        Args:
            days: Days to look back, counted from the last recorded transaction (use 180 by default).
            limit: Maximum number of transactions to return (for example 20).
        """
        print(f"  [tool] list_transactions(days={days}, limit={limit})", flush=True)
        txs = customer_transactions()
        if not txs:
            return {"transactions": [], "note": "The customer has no recorded transactions."}
        since = txs[-1]["timestamp"] - timedelta(days=max(days, 1))
        window = [t for t in txs if t["timestamp"] >= since]
        return {"total_in_window": len(window),
                "last_transaction": txs[-1]["timestamp"].isoformat(sep=" ", timespec="minutes"),
                "transactions": [to_json(t, include_risk=False)
                                 for t in reversed(window[-max(min(limit, 50), 1):])]}

    def assess_fraud_risk(days: int) -> dict:
        """Assesses the fraud risk of the authenticated customer's recent transactions.

        Uses explainable rules (atypical amount, new country or city, bursts, consecutive
        declines, unusual hour) plus the bank's anti-fraud model score. Returns the suspicious
        transactions with a 0-100 score, a level (low/medium/high) and the reasons.

        Args:
            days: Days to analyze, counted from the last recorded transaction (use 180 by default).
        """
        print(f"  [tool] assess_fraud_risk(days={days})", flush=True)
        scored, last = analyze_window(customer_transactions(), max(days, 1))
        suspicious = sorted((t for t in scored if t["score"] >= SUSPICIOUS_THRESHOLD),
                            key=lambda t: t["score"], reverse=True)
        products = load_products([t["product"] for t in suspicious])
        for t in suspicious:
            t["card"] = describe_product(products.get(t["product"]))
        session["suspicious"] = {t["id"]: t for t in suspicious}
        return {"transactions_analyzed": len(scored),
                "suspicious_count": len(suspicious),
                "max_level": suspicious[0]["level"] if suspicious else "low",
                "until": last.isoformat(sep=" ", timespec="minutes") if last else None,
                "details": [to_json(t) for t in suspicious[:10]]}

    def send_fraud_alert(transaction_ids: list[str], channel: str, message: str) -> dict:
        """Sends the customer a possible-fraud alert by email, WhatsApp or both.

        Only accepts IDs that assess_fraud_risk flagged as suspicious in this conversation.
        A human operator must approve the alert before it is sent.

        Args:
            transaction_ids: IDs of the suspicious transactions to include in the alert.
            channel: Delivery channel: "email", "whatsapp" or "both".
            message: Body of the alert only, 1-2 sentences in the customer's language. No greeting
                (the template already says "Hola <name>"), no signature, no sensitive data.
        """
        print(f"  [tool] send_fraud_alert({transaction_ids}, channel={channel})", flush=True)
        if not session["suspicious"]:
            return {"sent": False, "error": "Run assess_fraud_risk first."}
        invalid = [i for i in transaction_ids if str(i) not in session["suspicious"]]
        if invalid or not transaction_ids:
            return {"sent": False,
                    "error": f"IDs not flagged as suspicious in the last assessment: {invalid}"}
        txs = [session["suspicious"][str(i)] for i in transaction_ids]
        return send_alert(profile, txs, channel, message, source="agent")

    def escalate_to_agent(conversation_language: str, reason: str, summary: str,
                          transaction_ids: list[str]) -> dict:
        """Escalates the conversation to a human service agent who speaks the customer's language.
        Use it when the customer does not recognize transactions or asks for a person, after
        they agree.

        Args:
            conversation_language: Language of THIS conversation: "es", "pt" or "en".
            reason: Why the customer needs a human (for example: does not recognize a transfer).
            summary: 2-3 sentence summary for the agent: transactions, amounts, what the customer said.
            transaction_ids: IDs of the transactions the customer disputes (can be empty).
        """
        print(f"  [tool] escalate_to_agent(language={conversation_language})", flush=True)
        if session.get("escalation", {}).get("escalated"):
            return {**session["escalation"], "note": "The case was already escalated."}
        import fraud_flow  # lazy: fraud_flow imports this module
        fraud_flow.setup_tables()
        case = fraud_flow.open_chat_case(customer_id, [str(i) for i in transaction_ids], reason)
        session["escalation"] = fraud_flow.escalate(case, profile, conversation_language,
                                                    reason, summary)
        return session["escalation"]

    return [view_customer_profile, list_transactions, assess_fraud_risk, send_fraud_alert,
            escalate_to_agent]


SYSTEM_INSTRUCTIONS = """You are the LATAM Bank security assistant. You are serving the
authenticated customer {name} (ID {customer_id}). Your tools can only see this customer's data.

Rules:
- To talk about risk or fraud, ALWAYS use assess_fraud_risk. Never compute risk yourself.
- Explain the reasons in plain language. A high score is a suspicion, not a confirmed fraud.
- The data is historical: "recent" means relative to the last recorded transaction.
- Before calling send_fraud_alert, tell the customer which transactions you will include and
  through which channel, and wait for their agreement. Only use IDs flagged by the assessment.
- If the customer does not recognize a transaction, recommend blocking the card from the app
  and offer to escalate to a human agent. If they agree, call escalate_to_agent and tell them
  the agent's first name.
- Only offer actions your tools can perform. You cannot block cards: the customer does that in
  the app or through the bank's official line.
- Never ask for or reveal passwords, PINs, CVVs or full card numbers.
- If a question is not about their transactions or security, say so and do not invent data.
- Customers transact about once or twice a month, so use days=180 by default.
- Be efficient: call each tool at most once per customer message. If assess_fraud_risk finds
  nothing suspicious, say so clearly and OFFER to review a longer period (365 days); do not
  widen the window on your own.
- Always reply in the customer's language (Spanish or Portuguese), briefly, in plain text
  (no tables or markdown: the same answers are sent by WhatsApp).
"""


# ===========================================================================
# 7. Run modes
# ===========================================================================
# ===========================================================================
# 6b. LLM providers: Gemini (Google API) or Ollama (local models)
#
# Both expose the same small interface, so the rest of the code does not care which one
# is running: client.chats.create(model, config, history) -> chat, chat.send_message(text)
# -> reply with .text, and chat.get_history().
# ===========================================================================
class LLMError(Exception):
    """Provider error with an HTTP-like `code`, understood by send_with_retry."""
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


class _Reply:
    def __init__(self, text: str):
        self.text = text


def _json_type(annotation) -> dict:
    """Python type hint -> JSON schema type (enough for our tools)."""
    import typing
    origin = typing.get_origin(annotation)
    if origin in (list, tuple):
        args = typing.get_args(annotation)
        return {"type": "array", "items": _json_type(args[0]) if args else {"type": "string"}}
    return {str: {"type": "string"}, int: {"type": "integer"}, float: {"type": "number"},
            bool: {"type": "boolean"}}.get(annotation, {"type": "string"})


def _parse_docstring(doc: str) -> tuple[str, dict]:
    """Google-style docstring -> (description, {arg: description})."""
    description, args, current, in_args = [], {}, None, False
    for line in doc.splitlines():
        stripped = line.strip()
        if stripped == "Args:":
            in_args = True
            continue
        if not in_args:
            description.append(stripped)
        elif ":" in stripped and not line.startswith(" " * 8):
            current, text = stripped.split(":", 1)
            current = current.strip()
            args[current] = text.strip()
        elif current and stripped:
            args[current] += " " + stripped
    return " ".join(x for x in description if x), args


def tool_schema(fn) -> dict:
    """Build the Ollama/OpenAI tool schema from a Python function (hints + docstring)."""
    import inspect
    import typing
    hints = typing.get_type_hints(fn)
    description, arg_docs = _parse_docstring(inspect.getdoc(fn) or "")
    properties, required = {}, []
    for name, param in inspect.signature(fn).parameters.items():
        properties[name] = {**_json_type(hints.get(name, str)), "description": arg_docs.get(name, "")}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "function", "function": {
        "name": fn.__name__, "description": description,
        "parameters": {"type": "object", "properties": properties, "required": required}}}


def _coerce(value, annotation):
    """Small local models sometimes send '30' for an int or 'true' for a bool: fix that."""
    import typing
    try:
        if annotation is int and not isinstance(value, int):
            return int(float(str(value)))
        if annotation is bool and not isinstance(value, bool):
            return str(value).strip().lower() in TRUE_VALUES
        if typing.get_origin(annotation) is list and isinstance(value, str):
            return [v.strip() for v in value.strip("[]").replace('"', "").split(",") if v.strip()]
    except (TypeError, ValueError):
        pass
    return value


class OllamaConfig:
    def __init__(self, system_instruction: str, tools: list):
        import typing
        self.system_instruction = system_instruction
        self.tools = {fn.__name__: fn for fn in tools}
        self.schemas = [tool_schema(fn) for fn in tools]
        self.hints = {fn.__name__: typing.get_type_hints(fn) for fn in tools}

    def run_tool(self, name: str, arguments: dict) -> str:
        fn = self.tools.get(name)
        if fn is None:
            return json.dumps({"error": f"Unknown tool {name}"})
        args = {k: _coerce(v, self.hints[name].get(k, str)) for k, v in (arguments or {}).items()
                if k in self.hints[name]}
        try:
            result = fn(**args)
        except TypeError as e:
            result = {"error": f"Wrong arguments for {name}: {e}"}
        return json.dumps(result, ensure_ascii=False, default=str)


class OllamaChat:
    """Chat session with a local model. Runs the tool-calling loop itself."""
    def __init__(self, model: str, config: OllamaConfig, history: list | None = None):
        self.model, self.config = model, config
        self.messages = list(history) if history else [
            {"role": "system", "content": config.system_instruction}]
        self.think = OLLAMA_THINK

    def get_history(self) -> list:
        return list(self.messages)

    def _call(self, messages: list):
        import ollama
        kwargs = dict(model=self.model, messages=messages, tools=self.config.schemas,
                      options={"num_ctx": OLLAMA_NUM_CTX, "num_predict": OLLAMA_NUM_PREDICT})
        try:
            return ollama.chat(**kwargs, think=self.think)
        except ollama.ResponseError as e:
            if "think" in str(e).lower():  # model without reasoning support: retry without it
                return ollama.chat(**kwargs)
            raise

    def send_message(self, text: str) -> _Reply:
        import ollama
        pending = self.messages + [{"role": "user", "content": text}]
        for _ in range(MAX_TOOL_CALLS + 1):
            try:
                response = self._call(pending)
            except ollama.ResponseError as e:
                raise LLMError(getattr(e, "status_code", None), str(e)) from e
            except (ConnectionError, OSError) as e:
                raise LLMError(None, "No hay conexión con Ollama. Ábrelo o revisa OLLAMA_HOST.") from e
            message = response.message
            pending.append(message)
            if not message.tool_calls:
                self.messages = pending  # commit only on success, like the Gemini SDK
                return _Reply(message.content or "")
            for call in message.tool_calls:
                result = self.config.run_tool(call.function.name, call.function.arguments)
                pending.append({"role": "tool", "content": result, "tool_name": call.function.name})
        self.messages = pending
        return _Reply("Disculpa, no pude completar la consulta. ¿Me lo repites de otra forma?")


class _OllamaChats:
    def create(self, model: str, config: OllamaConfig, history: list | None = None) -> OllamaChat:
        return OllamaChat(model, config, history)


class OllamaClient:
    chats = _OllamaChats()


class AnthropicConfig:
    """System prompt + tools in the Claude API format (reuses the same Python functions)."""
    def __init__(self, system_instruction: str, tools: list):
        self.system_instruction = system_instruction
        self.ollama = OllamaConfig(system_instruction, tools)  # schemas, coercion, execution
        self.schemas = [{"name": t["function"]["name"],
                         "description": t["function"]["description"],
                         "input_schema": t["function"]["parameters"]}
                        for t in self.ollama.schemas]

    def run_tool(self, name: str, arguments: dict) -> str:
        return self.ollama.run_tool(name, arguments)


class AnthropicChat:
    """Chat session with Claude. Runs the tool-use loop (stop_reason == "tool_use")."""
    def __init__(self, client, model: str, config: AnthropicConfig, history: list | None = None):
        self.client, self.model, self.config = client, model, config
        self.messages = list(history) if history else []

    def get_history(self) -> list:
        return list(self.messages)

    def send_message(self, text: str) -> _Reply:
        import anthropic
        pending = self.messages + [{"role": "user", "content": text}]
        for _ in range(MAX_TOOL_CALLS + 1):
            try:
                response = self.client.messages.create(
                    model=self.model, max_tokens=ANTHROPIC_MAX_TOKENS,
                    system=self.config.system_instruction,
                    tools=self.config.schemas, messages=pending)
            except anthropic.APIStatusError as e:
                code = 503 if e.status_code == 529 else e.status_code  # 529 = overloaded
                raise LLMError(code, str(e)) from e
            except anthropic.APIConnectionError as e:
                raise LLMError(None, f"No hay conexión con la API de Anthropic: {e}") from e
            pending.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                self.messages = pending  # commit only on success
                return _Reply("".join(b.text for b in response.content if b.type == "text"))
            results = [{"type": "tool_result", "tool_use_id": block.id,
                        "content": self.config.run_tool(block.name, block.input)}
                       for block in response.content if block.type == "tool_use"]
            pending.append({"role": "user", "content": results})
        self.messages = pending
        return _Reply("Disculpa, no pude completar la consulta. ¿Me lo repites de otra forma?")


class _AnthropicChats:
    def __init__(self, sdk_client):
        self.sdk_client = sdk_client

    def create(self, model: str, config: AnthropicConfig, history: list | None = None) -> AnthropicChat:
        return AnthropicChat(self.sdk_client, model, config, history)


class AnthropicClient:
    def __init__(self):
        import anthropic
        self.chats = _AnthropicChats(anthropic.Anthropic(max_retries=2))  # reads ANTHROPIC_API_KEY


def provider_label() -> str:
    return {"anthropic": "Claude", "ollama": "Ollama local"}.get(LLM_PROVIDER, "Gemini")


def create_llm_session(system_instruction: str, tools: list) -> dict:
    """
    Session for the configured provider: {"client", "config", "model", "chat"}.
    The client is kept inside the session on purpose (a garbage-collected Gemini client
    closes its HTTP connection: "Cannot send a request, as the client has been closed").
    """
    if LLM_PROVIDER == "anthropic":
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Falta ANTHROPIC_API_KEY en el .env")
        client = AnthropicClient()
        config, model = AnthropicConfig(system_instruction, tools), ANTHROPIC_MODEL
    elif LLM_PROVIDER == "ollama":
        client, config, model = OllamaClient(), OllamaConfig(system_instruction, tools), OLLAMA_MODEL
    elif LLM_PROVIDER == "gemini":
        from google import genai
        from google.genai import types
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            sys.exit("Falta GEMINI_API_KEY en el .env")
        client = genai.Client()
        config = types.GenerateContentConfig(
            system_instruction=system_instruction, tools=tools,
            # Cap on tool calls per customer message (avoids loops and wasted compute)
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                maximum_remote_calls=MAX_TOOL_CALLS))
        model = GEMINI_MODEL
    else:
        sys.exit(f"LLM_PROVIDER desconocido: {LLM_PROVIDER}. Usa 'anthropic', 'ollama' o 'gemini'.")
    return {"client": client, "config": config, "model": model,
            "chat": client.chats.create(model=model, config=config)}


def run_chat(customer_id: str) -> None:
    profile = get_customer_profile(customer_id)
    if profile is None:
        sys.exit(f"No encontré al cliente {customer_id} en {CUSTOMERS_TABLE}.")
    session = create_llm_session(
        SYSTEM_INSTRUCTIONS.format(name=profile.get("full_name") or "customer",
                                   customer_id=customer_id),
        build_tools(customer_id, profile))
    client, config, model, chat = (session[k] for k in ("client", "config", "model", "chat"))
    print(f"Agente antifraude ({provider_label()}: {model}) | cliente {customer_id} "
          f"({profile.get('full_name') or 'sin nombre'}) | modo {'DEMO' if DEMO_MODE else 'REAL'}")
    print("Escribe 'salir' para terminar.\n")
    while True:
        question = input("Cliente: ").strip()
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if not question:
            continue
        reply, chat, model = send_with_retry(client, chat, model, config, question)
        if reply is not None:
            print(f"\nAgente: {reply.text or '(sin texto; intenta reformular)'}\n")


def _is_daily_quota(error: Exception) -> bool:
    """True when a 429 is a per-day quota (retrying in seconds will not help)."""
    text = str(error)
    return "PerDay" in text or "per day" in text.lower()


def send_with_retry(client, chat, model: str, config, question: str):
    """
    Send a message, resilient to Gemini errors:
      - 500/503/504 (overload): retry the same model with backoff (2, 4, 8 s).
      - 429 (quota exhausted): do not retry; switch to the next model right away.
      - then walk the GEMINI_FALLBACK_MODELS chain, keeping the conversation history.
    Returns (reply or None, chat, model in use).
    """
    fallbacks = {"anthropic": ANTHROPIC_FALLBACK_MODELS, "ollama": OLLAMA_FALLBACK_MODELS}.get(
        LLM_PROVIDER, GEMINI_FALLBACK_MODELS)
    chain = [model] + [m for m in fallbacks if m != model]
    for index, current in enumerate(chain):
        if index > 0:
            print(f"  [llm] cambiando a {current} (se conserva la conversación)", flush=True)
            # The failed message is not in the history, so it can be resent safely
            chat = client.chats.create(model=current, config=config, history=chat.get_history())
        for attempt in range(MAX_RETRIES):
            try:
                return chat.send_message(question), chat, current
            except Exception as e:
                code = getattr(e, "code", None)
                if code == QUOTA_CODE:
                    scope = "cuota diaria agotada" if _is_daily_quota(e) else "límite de uso"
                    print(f"  [llm] {current}: {scope} (429)", flush=True)
                    break  # next model
                if code == 404:
                    hint = f" (descárgalo con: ollama pull {current})" if LLM_PROVIDER == "ollama" else ""
                    print(f"  [llm] {current}: modelo no disponible (404){hint}", flush=True)
                    break  # next model
                if code not in RETRYABLE_CODES:
                    print(f"\nError del modelo: {e}\n")
                    return None, chat, current
                wait = 2 ** (attempt + 1)
                print(f"  [llm] {current} saturado ({code}); reintento en {wait}s...", flush=True)
                time.sleep(wait)
    print("\n[llm] Ningún modelo respondió. Con Gemini: revisa la cuota (429). Con Ollama: "
          "revisa que esté abierto y que los modelos estén descargados.\n")
    return None, chat, chain[-1]


def run_monitor(customer_ids: list[str], days: int, channel: str) -> None:
    """Proactive review: on high risk, propose an alert (no LLM, fixed message)."""
    data = load_transactions(customer_ids)  # one query for all customers
    for cid in customer_ids:
        scored, _ = analyze_window(data.get(cid, []), days)
        high = sorted((t for t in scored if t["score"] >= HIGH_RISK_THRESHOLD),
                      key=lambda t: t["score"], reverse=True)[:5]
        print(f"\n{cid}: {len(scored)} transacciones en {days} días, {len(high)} de riesgo alto")
        for tx in high:
            print(f"   {tx['id']} | {tx['timestamp']:%Y-%m-%d %H:%M} | {tx['amount']:,.2f} | "
                  f"{tx['score']:.0f} | {'; '.join(tx['reasons'])}")
        if high:
            products = load_products([t["product"] for t in high])
            for t in high:
                t["card"] = describe_product(products.get(t["product"]))
            profile = get_customer_profile(cid) or {"customer_id": cid}
            result = send_alert(
                profile, high, channel,
                "Detectamos actividad inusual en tu cuenta y queremos confirmar que fuiste tú.",
                source="monitor")
            print(f"   alerta: {result}")


def run_demo_customers(days: int, limit: int) -> None:
    """Customers with the most suspicious activity in the dataset's last `days` days."""
    cols = TX_COLUMNS
    max_score_sql = (f"MAX(TRY_CAST({quote_ident(cols['model_score'])} AS DOUBLE)) * {SCORE_SCALE}"
                     if cols["model_score"] else "NULL")
    label_count_sql = (f"SUM(CASE WHEN lower(CAST({quote_ident(cols['fraud_label'])} AS VARCHAR)) IN "
                       f"('true','1','yes','y','t') THEN 1 ELSE 0 END)" if cols["fraud_label"] else "NULL")
    ts = f"TRY_CAST({quote_ident(cols['timestamp'])} AS TIMESTAMP)"
    sql = f"""
        WITH last_tx AS (SELECT MAX({ts}) AS ts FROM {TRANSACTIONS_TABLE})
        SELECT CAST({quote_ident(cols['customer'])} AS VARCHAR) AS customer, COUNT(*) AS tx_count,
               {max_score_sql} AS max_model_score, {label_count_sql} AS labeled_fraud
        FROM {TRANSACTIONS_TABLE}, last_tx
        WHERE {ts} >= last_tx.ts - INTERVAL {int(days)} DAY
        GROUP BY 1
        ORDER BY labeled_fraud DESC NULLS LAST, max_model_score DESC NULLS LAST
        LIMIT {int(limit)}
    """
    print(f"Clientes con más señales en los últimos {days} días del dataset:\n")
    print(f"{'customer':<20}{'tx':>6}{'max score':>12}{'is_fraud':>10}")
    for customer, n, score, labeled in con.execute(sql).fetchall():
        print(f"{customer:<20}{n:>6}{'' if score is None else f'{score:.0f}':>12}"
              f"{'' if labeled is None else labeled:>10}")
    print("\nPrueba el chat con: python fraud_agent.py --customer <customer>")


def run_evaluation(n_customers: int, min_tx: int = 10) -> None:
    """Precision and recall of the rules, the fraud_score and their combination vs is_fraud."""
    cols = TX_COLUMNS
    if not cols["fraud_label"]:
        sys.exit("La tabla no tiene columna de etiqueta (is_fraud); no se puede evaluar.")
    label_sql = f"lower(CAST({quote_ident(cols['fraud_label'])} AS VARCHAR)) IN ('true','1','yes','y','t')"
    half = max(n_customers // 2, 1)
    # Balanced sample: half customers with at least one fraud, half without
    rows = con.execute(f"""
        WITH c AS (
            SELECT CAST({quote_ident(cols['customer'])} AS VARCHAR) AS cid, COUNT(*) AS n,
                   MAX(CASE WHEN {label_sql} THEN 1 ELSE 0 END) AS has_fraud
            FROM {TRANSACTIONS_TABLE} GROUP BY 1 HAVING COUNT(*) >= {int(min_tx)})
        (SELECT cid FROM c WHERE has_fraud = 1 ORDER BY random() LIMIT {half})
        UNION ALL
        (SELECT cid FROM c WHERE has_fraud = 0 ORDER BY random() LIMIT {half})
    """).fetchall()
    customer_ids = [r[0] for r in rows]
    if not customer_ids:
        sys.exit("No encontré clientes para evaluar.")
    data = load_transactions(customer_ids)
    all_scored = [t for cid in customer_ids for t in score_transactions(data.get(cid, []))]
    positives = sum(is_true(t["fraud_label"]) for t in all_scored)
    print(f"Evaluación sobre {len(customer_ids)} clientes, {len(all_scored)} transacciones, "
          f"{positives} etiquetadas como fraude ({positives / max(len(all_scored), 1):.2%}).\n")
    print(f"{'signal':<12}{'threshold':>10}{'alerts':>8}{'precision':>11}{'recall':>9}")
    for signal, key in [("rules", "rule_score"), ("model", "model_score"), ("combined", "score")]:
        if key == "model_score" and not cols["model_score"]:
            continue
        for threshold in (SUSPICIOUS_THRESHOLD, HIGH_RISK_THRESHOLD):
            alerts = [t for t in all_scored if (t[key] or 0) >= threshold]
            true_pos = sum(is_true(t["fraud_label"]) for t in alerts)
            precision = true_pos / len(alerts) if alerts else 0
            recall = true_pos / positives if positives else 0
            print(f"{signal:<12}{threshold:>10}{len(alerts):>8}{precision:>11.1%}{recall:>9.1%}")
    print("\nMuestra balanceada (mitad clientes con fraude): la precisión real será menor.")


def run_show_columns() -> None:
    print(f"Base: {DATABASE}\n\nTransacciones ({TRANSACTIONS_TABLE}):")
    for field, column in TX_COLUMNS.items():
        mark = " (obligatoria)" if field in REQUIRED_TX_FIELDS else ""
        print(f"  {field:<14} -> {column or '— no encontrada'}{mark}")
    print(f"\nClientes ({CUSTOMERS_TABLE}):")
    for field, column in CUSTOMER_COLUMNS.items():
        print(f"  {field:<14} -> {column or '— no encontrada'}")
    if TX_COLUMNS.get("model_score"):
        print(f"\nfraud_score se multiplica por {SCORE_SCALE:g} para llevarlo a 0-100.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fraud detection and alerting agent")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--customer", help="authenticated customer ID (opens the chat)")
    mode.add_argument("--monitor", nargs="+", metavar="ID", help="proactive review without chat")
    mode.add_argument("--demo-customers", action="store_true", help="suggest customers for a demo")
    mode.add_argument("--evaluate", type=int, metavar="N", help="evaluate on N customers")
    mode.add_argument("--columns", action="store_true", help="show the column mapping")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS,
                        help=f"analysis window in days (default {DEFAULT_DAYS})")
    parser.add_argument("--channel", default="both", choices=["email", "whatsapp", "both"])
    args = parser.parse_args()

    init(connect())
    if args.customer or args.monitor:
        print(delivery_summary())
    if args.columns:
        run_show_columns()
    elif args.demo_customers:
        run_demo_customers(args.days, 15)
    elif args.evaluate:
        run_evaluation(args.evaluate)
    elif args.monitor:
        run_monitor(args.monitor, args.days, args.channel)
    else:
        run_chat(args.customer)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, EOFError):
        print("\nHasta luego.")
