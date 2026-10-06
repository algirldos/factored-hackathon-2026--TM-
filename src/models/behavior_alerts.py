"""
Behavior alerts for the agent, built from the batch scoring results in fraud_ops.

Customers must not learn that they are profiled or grouped by spending pattern. So the agent
(and the LLM behind it) only ever receives a neutral signal with a fixed message, built from an
allow-list of fields: clusters, metric names, medians, thresholds and scores never leave the
server. Bank staff get the internal detail through `internal_note`, outside the LLM.
"""
import logging

from src.models.batch_scoring import DEFAULT_MIN_SUSPICIOUS, OPS_SCHEMA

log = logging.getLogger(__name__)

CUSTOMER_MESSAGES = {
    "es": "Hemos detectado un comportamiento sospechoso en su actividad bancaria.",
    "pt": "Detectamos um comportamento suspeito na sua atividade bancária.",
}
AGENT_INSTRUCTION = (
    "Share only this message, in the customer's language, and ask them to review their recent "
    "activity. Do not explain how it was detected and do not mention segments, groups, profiles, "
    "comparisons with other customers, metrics, thresholds or scores.")

METRIC_LABELS = {  # for bank staff only
    "num_transacciones_monthly": "número de transacciones del mes",
    "monto_total_usd_monthly": "monto total del mes",
    "monto_promedio_usd": "monto promedio por transacción",
    "monto_mediano_usd": "monto mediano por transacción",
    "monto_maximo_usd": "monto de la transacción más alta",
}


def latest_behavior_anomaly(con, customer_id: str, schema: str = OPS_SCHEMA) -> dict | None:
    """
    Internal result of the latest scoring run for this customer, or None if the customer
    was not scored or the scoring tables do not exist yet. Never send this to the LLM.
    """
    try:
        row = con.execute(f"""
            SELECT as_of, model_version, n_features, n_suspicious, suspicious_features
            FROM {schema}.customer_anomaly WHERE customer_id = ?
            ORDER BY as_of DESC, model_version DESC LIMIT 1""", [customer_id]).fetchone()
    except Exception as e:  # tables not created yet, or no access: the agent works without it
        log.info("Sin resultados de comportamiento para %s: %s", customer_id, e)
        return None
    if row is None:
        return None
    as_of, version, n_features, n_suspicious, features = row
    return {"as_of": as_of, "model_version": version, "n_features": int(n_features),
            "n_suspicious": int(n_suspicious),
            "suspicious_features": [f for f in (features or "").split(",") if f]}


def is_suspicious(anomaly: dict | None, min_suspicious: int = DEFAULT_MIN_SUSPICIOUS) -> bool:
    return bool(anomaly) and anomaly["n_suspicious"] >= min_suspicious


def customer_facing_alert(anomaly: dict | None,
                          min_suspicious: int = DEFAULT_MIN_SUSPICIOUS) -> dict | None:
    """
    The only behavior information the agent may see: None, or a neutral alert built from an
    allow-list. Nothing from `anomaly` is copied except the date of the reviewed period.
    """
    if not is_suspicious(anomaly, min_suspicious):
        return None
    return {"unusual_activity_detected": True,
            "message": dict(CUSTOMER_MESSAGES),
            "reviewed_period_end": str(anomaly["as_of"]),
            "instruction": AGENT_INSTRUCTION}


def internal_note(anomaly: dict | None, min_suspicious: int = DEFAULT_MIN_SUSPICIOUS) -> str:
    """Detail for bank staff (escalation email). Never send it to the customer or the LLM."""
    if not is_suspicious(anomaly, min_suspicious):
        return ""
    labels = ", ".join(METRIC_LABELS.get(f, f) for f in anomaly["suspicious_features"])
    return (f"Alerta de comportamiento ({anomaly['model_version']}, 30 días hasta "
            f"{anomaly['as_of']}): {anomaly['n_suspicious']} de {anomaly['n_features']} métricas "
            f"fuera de lo habitual para su segmento: {labels}. Uso interno: no compartir con el "
            f"cliente que se le perfila por segmento.")
