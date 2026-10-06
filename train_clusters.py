"""
Train the customer clustering model and the cluster behavior profiles.

    python train_clusters.py --as-of 2026-05-31
    python train_clusters.py --as-of 2026-05-31 --models-dir models

Reads MotherDuck with a read-only connection (MOTHERDUCK_TOKEN in .env) and writes
models/clusters_<YYYYMMDD>/ with the two .joblib files and metadata.json.
"""
import argparse
import json
import logging
import sys

from src.models.training import save_artifacts, train


def connect():
    from src.database.connections import get_motherduck_connection  # needs MOTHERDUCK_TOKEN
    return get_motherduck_connection("latam_bank", read_only=True)


def main(argv=None, connect_fn=connect) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--as-of", required=True,
                        help="Fecha de corte (AAAA-MM-DD): solo se usan datos hasta ese día")
    parser.add_argument("--models-dir", default="models", help="Carpeta de salida (por defecto models)")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    con = connect_fn()
    try:
        artifacts = train(con, args.as_of)
    finally:
        con.close()
    folder = save_artifacts(artifacts, args.models_dir)
    summary = {k: artifacts.metadata[k] for k in
               ("model_version", "as_of", "n_customers", "n_customers_with_activity",
                "cluster_sizes", "pca_explained_variance")}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Artefactos guardados en {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
