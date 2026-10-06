"""
Held-out evaluation of the clustering model (read-only).

    python evaluate_clusters.py --train-as-of 2026-03-18 \
        --eval-as-of 2026-04-17 2026-05-17 2026-06-17 --rules-negatives 3000

Trains in memory with data up to --train-as-of (nothing is saved), scores each 30-day window
that ends on an --eval-as-of date, and compares the model with the bank's fraud_score, the
agent's rules (weighted sample, optional) and a random ranking. Writes
reports/evaluation_<train>.json and .md.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from src.models.evaluation import evaluate, to_markdown
from src.models.training import train


def connect():
    from src.database.connections import get_motherduck_connection  # needs MOTHERDUCK_TOKEN
    return get_motherduck_connection("latam_bank", read_only=True)


def main(argv=None, connect_fn=connect) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--train-as-of", required=True, help="Último día de datos de entrenamiento")
    parser.add_argument("--eval-as-of", required=True, nargs="+",
                        help="Último día de cada ventana de evaluación (posteriores al entrenamiento)")
    parser.add_argument("--rules-negatives", type=int, default=0,
                        help="Clientes sin fraude muestreados para evaluar las reglas (0 = no evaluarlas)")
    parser.add_argument("--out", default="reports", help="Carpeta del reporte")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    con = connect_fn()
    try:
        if args.rules_negatives:
            import fraud_agent as fa
            fa.init(con)   # the rules read transactions through fraud_agent
        artifacts = train(con, args.train_as_of)
        report = evaluate(con, artifacts, args.eval_as_of, args.rules_negatives)
    finally:
        con.close()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"evaluation_{artifacts.metadata['as_of'].replace('-', '')}"
    (out / f"{stem}.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / f"{stem}.md").write_text(to_markdown(report), encoding="utf-8")
    print(to_markdown(report))
    print(f"Reporte en {out / stem}.md y .json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
