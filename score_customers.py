"""
Score customers' last 30 days against their cluster profile and write the results to fraud_ops.

    python score_customers.py --as-of 2026-06-17 --dry-run     # compute and print, write nothing
    python score_customers.py --as-of 2026-06-17               # write fraud_ops tables + queue
    python score_customers.py --as-of 2026-06-17 --min-suspicious 3 --model-dir models/clusters_20260617

Writing needs a MotherDuck token with write access (MOTHERDUCK_TOKEN in .env).
Enqueued customers become cases only when `python fraud_flow.py --process-queue` runs.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

from src.models.batch_scoring import (DEFAULT_MIN_SUSPICIOUS, run_summary, score_as_of,
                                      write_results)
from src.models.training import load_artifacts


def connect(read_only: bool):
    from src.database.connections import get_motherduck_connection  # needs MOTHERDUCK_TOKEN
    return get_motherduck_connection("latam_bank", read_only=read_only)


def latest_model_dir(models_dir: str = "models") -> Path:
    folders = sorted(p for p in Path(models_dir).glob("clusters_*") if p.is_dir())
    if not folders:
        raise FileNotFoundError(f"No hay modelos en {models_dir}/clusters_*; entrena con train_clusters.py")
    return folders[-1]


def main(argv=None, connect_fn=connect) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--as-of", required=True, help="Último día de la ventana de 30 días (AAAA-MM-DD)")
    parser.add_argument("--model-dir", help="Carpeta del modelo (por defecto, el más reciente en models/)")
    parser.add_argument("--min-suspicious", type=int, default=DEFAULT_MIN_SUSPICIOUS,
                        help=f"Métricas sospechosas para encolar un cliente (por defecto {DEFAULT_MIN_SUSPICIOUS})")
    parser.add_argument("--dry-run", action="store_true", help="Calcula y muestra el resumen sin escribir")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    artifacts = load_artifacts(args.model_dir or latest_model_dir())
    con = connect_fn(read_only=args.dry_run)
    try:
        result = score_as_of(con, artifacts, args.as_of)
        summary = run_summary(result, args.min_suspicious)
        if not args.dry_run:
            run = write_results(con, result, args.min_suspicious)
            summary.update(run_id=run["run_id"], n_enqueued=run["n_enqueued"])
    finally:
        con.close()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Simulación: no se escribió nada." if args.dry_run else
          "Resultados escritos en fraud_ops. Procesa la cola con: python fraud_flow.py --process-queue")
    return 0


if __name__ == "__main__":
    sys.exit(main())
