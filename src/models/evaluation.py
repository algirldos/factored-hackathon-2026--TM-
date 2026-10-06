"""
Held-out evaluation of the clustering model against the bank's fraud_score, the agent's
explainable rules and a random baseline.

Protocol (no leakage):
- The model is trained with data up to `train_as_of`.
- It is evaluated on 30-day windows that end after `train_as_of`, which it never saw.
- The label is customer-level: at least one `is_fraud` transaction in the window. It is used
  only here, never to train or to score.
- Every method is compared on the same customers: those with activity in the window that the
  clustering model can score.
"""
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.features.transaction_behavior import Window, behavior_windows
from src.models.batch_scoring import score_as_of
from src.models.behavior_scoring import SUSPICIOUS_FEATURES
from src.models.training import TrainedArtifacts

log = logging.getLogger(__name__)

THRESHOLDS = range(1, len(SUSPICIOUS_FEATURES) + 1)
RULE_THRESHOLDS = (30, 60)   # SUSPICIOUS_THRESHOLD and HIGH_RISK_THRESHOLD of fraud_agent


# ---------------------------------------------------------------------------
# Labels and baseline scores from the database
# ---------------------------------------------------------------------------
def window_labels(con, window: Window) -> pd.DataFrame:
    """Per customer with activity in the window: fraud label and the bank's max fraud_score."""
    return con.execute("""
        SELECT customer_id,
               BOOL_OR(COALESCE(is_fraud, FALSE)) AS has_fraud,
               MAX(fraud_score) AS max_fraud_score
        FROM (SELECT * FROM bronze.transactions
              WHERE transaction_date >= ? AND transaction_date < ?
              QUALIFY ROW_NUMBER() OVER (PARTITION BY transaction_id
                                         ORDER BY transaction_date DESC) = 1)
        GROUP BY customer_id""",
        [window.start.to_pydatetime(), window.end.to_pydatetime()]).df()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def ranking_metrics(y: pd.Series, score: pd.Series, weight: pd.Series | None = None) -> dict:
    """Average precision and ROC-AUC; None when the window has a single class."""
    y = y.astype(int)
    if y.nunique() < 2:
        return {"average_precision": None, "roc_auc": None}
    score = score.fillna(score.min() - 1 if score.notna().any() else 0)
    return {"average_precision": round(float(average_precision_score(y, score, sample_weight=weight)), 4),
            "roc_auc": round(float(roc_auc_score(y, score, sample_weight=weight)), 4)}


def flag_metrics(y: pd.Series, flagged: pd.Series, weight: pd.Series | None = None) -> dict:
    """Precision, recall, F1 and lift of a set of flagged customers (optionally weighted)."""
    w = weight if weight is not None else pd.Series(1.0, index=y.index)
    y, flagged = y.astype(bool), flagged.astype(bool)
    positives, n = float(w[y].sum()), float(w.sum())
    flagged_n, tp = float(w[flagged].sum()), float(w[flagged & y].sum())
    precision = tp / flagged_n if flagged_n else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    base_rate = positives / n if n else 0.0
    return {"flagged": int(round(flagged_n)), "true_positives": int(round(tp)),
            "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "lift": round(precision / base_rate, 2) if base_rate else None}


def precision_at_k(y: pd.Series, score: pd.Series, k: int) -> float | None:
    """Precision of the k customers with the highest score (ties broken by order)."""
    if k <= 0:
        return None
    top = score.fillna(-np.inf).sort_values(ascending=False, kind="stable").index[:k]
    return round(float(y.loc[top].astype(bool).mean()), 4)


def cluster_rank_score(summary: pd.DataFrame) -> pd.Series:
    """Rank by number of suspicious metrics, then by mean deviation (bounded tie-breaker)."""
    tie = summary["anomaly_score"].fillna(0).clip(upper=99) / 100
    return summary["n_suspicious"] + tie


# ---------------------------------------------------------------------------
# Rules baseline (fraud_agent's explainable rules) on a weighted sample
# ---------------------------------------------------------------------------
def rules_scores(customer_ids: list[str], window: Window) -> pd.Series:
    """Max rule score of each customer's transactions in the window (history before it counts).
    Needs fraud_agent initialized on the same database (fa.init)."""
    import fraud_agent as fa
    data = fa.load_transactions(customer_ids)
    scores = {}
    for cid in customer_ids:
        scored = fa.score_transactions(data.get(cid, []), since=window.start.to_pydatetime())
        in_window = [t["rule_score"] for t in scored if t["timestamp"] < window.end.to_pydatetime()]
        scores[cid] = max(in_window) if in_window else 0
    return pd.Series(scores, dtype="float64")


def rules_baseline(data: pd.DataFrame, window: Window, negatives_sample: int, seed: int) -> dict:
    """
    The rules are slow in Python, so they run on every positive customer and a random sample of
    negatives; negatives are weighted by (all negatives / sampled negatives) so the metrics
    estimate the full population.
    """
    positives = data[data["has_fraud"]]
    negatives = data[~data["has_fraud"]]
    sampled = negatives.sample(min(negatives_sample, len(negatives)), random_state=seed)
    sample = pd.concat([positives, sampled])
    weight = pd.Series(1.0, index=sample.index)
    if len(sampled):
        weight.loc[sampled.index] = len(negatives) / len(sampled)
    score = rules_scores(sample["customer_id"].tolist(), window).reindex(sample["customer_id"]).to_numpy()
    score = pd.Series(score, index=sample.index)
    return {"sample_customers": int(len(sample)), "sampled_negatives": int(len(sampled)),
            **ranking_metrics(sample["has_fraud"], score, weight),
            "thresholds": {str(t): flag_metrics(sample["has_fraud"], score >= t, weight)
                           for t in RULE_THRESHOLDS}}


# ---------------------------------------------------------------------------
# One window and the whole evaluation
# ---------------------------------------------------------------------------
@dataclass
class WindowEvaluation:
    as_of: str
    report: dict
    data: pd.DataFrame          # one row per evaluated customer (for inspection)


def evaluate_window(con, artifacts: TrainedArtifacts, as_of, rules_negatives: int = 0,
                    seed: int = 42) -> WindowEvaluation:
    as_of = pd.Timestamp(as_of).normalize()
    trained_on = pd.Timestamp(artifacts.metadata["as_of"])
    if as_of <= trained_on:
        raise ValueError(f"La ventana termina el {as_of.date()}, que no es posterior al "
                         f"entrenamiento ({trained_on.date()}): habría fuga de datos.")
    _, window = behavior_windows(as_of)
    result = score_as_of(con, artifacts, as_of)
    labels = window_labels(con, window)
    data = result.summary.merge(labels, on="customer_id", how="inner").reset_index(drop=True)
    data["has_fraud"] = data["has_fraud"].astype(bool)
    y = data["has_fraud"]
    cluster_score = cluster_rank_score(data)
    bank_score = data["max_fraud_score"].astype("float64")
    rng = np.random.default_rng(seed)
    random_score = pd.Series(rng.random(len(data)), index=data.index)

    thresholds = {}
    for k in THRESHOLDS:
        flagged = data["n_suspicious"] >= k
        n_flagged = int(flagged.sum())
        thresholds[str(k)] = {
            **flag_metrics(y, flagged),
            # Same number of alerts for every method: a fair comparison of the rankings
            "fraud_score_precision_at_same_k": precision_at_k(y, bank_score, n_flagged),
            "random_precision_at_same_k": precision_at_k(y, random_score, n_flagged),
        }
    report = {
        "window": {"start": window.start.date().isoformat(),
                   "end_inclusive": (window.end - pd.Timedelta(days=1)).date().isoformat()},
        "customers": int(len(data)), "positives": int(y.sum()),
        "base_rate": round(float(y.mean()), 5) if len(data) else None,
        "ranking": {"cluster_model": ranking_metrics(y, cluster_score),
                    "fraud_score": ranking_metrics(y, bank_score),
                    "random": ranking_metrics(y, random_score)},
        "cluster_thresholds": thresholds,
    }
    if rules_negatives:
        report["rules"] = rules_baseline(data, window, rules_negatives, seed)
    log.info("Ventana %s: %d clientes, %d con fraude", as_of.date(), len(data), int(y.sum()))
    return WindowEvaluation(as_of.date().isoformat(), report, data)


def evaluate(con, artifacts: TrainedArtifacts, eval_as_of: list, rules_negatives: int = 0,
             seed: int = 42) -> dict:
    windows = [evaluate_window(con, artifacts, d, rules_negatives, seed) for d in eval_as_of]
    pooled = pd.concat([w.data for w in windows], ignore_index=True)
    y = pooled["has_fraud"]
    return {
        "model_version": artifacts.metadata["model_version"],
        "train_as_of": artifacts.metadata["as_of"],
        "features": list(SUSPICIOUS_FEATURES),
        "label": "cliente con al menos una transacción is_fraud en la ventana de 30 días",
        "windows": {w.as_of: w.report for w in windows},
        "pooled": {"customer_windows": int(len(pooled)), "positives": int(y.sum()),
                   "ranking": {"cluster_model": ranking_metrics(y, cluster_rank_score(pooled)),
                               "fraud_score": ranking_metrics(y, pooled["max_fraud_score"].astype("float64"))},
                   "cluster_thresholds": {str(k): flag_metrics(y, pooled["n_suspicious"] >= k)
                                          for k in THRESHOLDS}},
    }


def to_markdown(report: dict) -> str:
    """Human-readable summary of an evaluation report."""
    def pct(v):
        return "–" if v is None else f"{v:.1%}"

    def num(v):
        return "–" if v is None else f"{v:.3f}"

    lines = [f"# Evaluación del modelo de clústeres ({report['model_version']})", "",
             f"- Entrenado con datos hasta **{report['train_as_of']}**; evaluado en ventanas de 30 "
             f"días posteriores (sin fuga).",
             f"- Etiqueta: {report['label']} (solo para evaluar).",
             f"- Métricas evaluadas: {', '.join(report['features'])}.", ""]
    for as_of, w in report["windows"].items():
        lines += [f"## Ventana {w['window']['start']} a {w['window']['end_inclusive']}", "",
                  f"{w['customers']:,} clientes con actividad, {w['positives']} con fraude "
                  f"(tasa base {pct(w['base_rate'])}).", "",
                  "| Método | Precisión promedio (AP) | ROC-AUC |", "|---|---|---|"]
        for name, label in [("cluster_model", "Modelo de clústeres"), ("fraud_score", "fraud_score del banco"),
                            ("random", "Aleatorio")]:
            r = w["ranking"][name]
            lines.append(f"| {label} | {num(r['average_precision'])} | {num(r['roc_auc'])} |")
        if "rules" in w:
            r = w["rules"]
            lines.append(f"| Reglas del agente (muestra ponderada de {r['sample_customers']:,}) | "
                         f"{num(r['average_precision'])} | {num(r['roc_auc'])} |")
        lines += ["", "| Métricas sospechosas ≥ | Marcados | Precisión | Recall | F1 | Lift | "
                      "fraud_score, misma cantidad | Aleatorio, misma cantidad |",
                  "|---|---|---|---|---|---|---|---|"]
        for k, t in w["cluster_thresholds"].items():
            lines.append(f"| {k} | {t['flagged']:,} | {pct(t['precision'])} | {pct(t['recall'])} | "
                         f"{num(t['f1'])} | {t['lift'] if t['lift'] is not None else '–'} | "
                         f"{pct(t['fraud_score_precision_at_same_k'])} | {pct(t['random_precision_at_same_k'])} |")
        if "rules" in w:
            lines += ["", "| Reglas del agente: puntaje ≥ | Marcados (estimado) | Precisión | Recall |",
                      "|---|---|---|---|"]
            for k, t in w["rules"]["thresholds"].items():
                lines.append(f"| {k} | {t['flagged']:,} | {pct(t['precision'])} | {pct(t['recall'])} |")
        lines.append("")
    pooled = report["pooled"]
    lines += ["## Todas las ventanas juntas", "",
              f"{pooled['customer_windows']:,} cliente-ventanas, {pooled['positives']} con fraude.", "",
              "| Método | AP | ROC-AUC |", "|---|---|---|"]
    for name, label in [("cluster_model", "Modelo de clústeres"), ("fraud_score", "fraud_score del banco")]:
        r = pooled["ranking"][name]
        lines.append(f"| {label} | {num(r['average_precision'])} | {num(r['roc_auc'])} |")
    return "\n".join(lines) + "\n"
