"""
Customer clustering: preprocessor -> PCA -> KMeans
(notebooks/03_clustering_refactored.ipynb, sections 2-10 and 21-23).
"""
import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.contracts import CUSTOMER_FEATURES, validate
from src.preprocessing.pipelines import CustomerPreprocessor

NUMERICAL_COLUMNS = ["Edad"]
LOG_COLUMNS = ["credit_score", "income_usd", "Deuda_usd", "Activos_financieros_usd"]
CATEGORICAL_COLUMNS = ["country", "education_level"]
REQUIRED_INPUT_COLUMNS = ["customer_id", *NUMERICAL_COLUMNS, *LOG_COLUMNS, *CATEGORICAL_COLUMNS]

N_PCA_COMPONENTS = 6
N_CLUSTERS = 3
RANDOM_STATE = 42
BUNDLE_KEYS = {"preprocessor", "pca", "kmeans", "required_input_columns"}


def fit_customer_clustering(features: pd.DataFrame, n_components: int = N_PCA_COMPONENTS,
                            n_clusters: int = N_CLUSTERS,
                            random_state: int = RANDOM_STATE) -> dict:
    """Fit the full model and return it as the bundle saved in customer_cluster_model.joblib."""
    missing = [c for c in REQUIRED_INPUT_COLUMNS if c not in features.columns]
    if missing:
        raise KeyError(f"Faltan columnas de cliente: {missing}")
    validate(features, CUSTOMER_FEATURES)
    preprocessor = CustomerPreprocessor(log_columns=LOG_COLUMNS,
                                        numerical_columns=NUMERICAL_COLUMNS,
                                        categorical_columns=CATEGORICAL_COLUMNS)
    pca = PCA(n_components=n_components, random_state=random_state)
    x_pca = pca.fit_transform(preprocessor.fit_transform(features))
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(x_pca)
    return {
        "preprocessor": preprocessor, "pca": pca, "kmeans": kmeans,
        "feature_config": {"log_columns": LOG_COLUMNS, "numerical_columns": NUMERICAL_COLUMNS,
                           "categorical_columns": CATEGORICAL_COLUMNS},
        "required_input_columns": REQUIRED_INPUT_COLUMNS,
        "pca_columns": [f"pca_{i}" for i in range(1, n_components + 1)],
        "n_pca_components": n_components, "n_clusters": n_clusters,
        "random_state": random_state,
    }


def validate_bundle(bundle: dict) -> None:
    missing = BUNDLE_KEYS - set(bundle)
    if missing:
        raise KeyError(f"El modelo de clústeres no tiene {sorted(missing)}")


def load_bundle(path) -> dict:
    bundle = joblib.load(path)
    validate_bundle(bundle)
    return bundle


def assign_clusters(bundle: dict, features: pd.DataFrame) -> pd.DataFrame:
    """customer_id -> cluster for every row of `features`."""
    validate_bundle(bundle)
    missing = [c for c in bundle["required_input_columns"] if c not in features.columns]
    if missing:
        raise KeyError(f"Faltan columnas de cliente: {missing}")
    if features.empty:
        return pd.DataFrame({"customer_id": pd.Series(dtype="object"),
                             "cluster": pd.Series(dtype="int64")})
    validate(features, CUSTOMER_FEATURES)
    x_pca = bundle["pca"].transform(bundle["preprocessor"].transform(features))
    return pd.DataFrame({"customer_id": features["customer_id"].to_numpy(),
                         "cluster": bundle["kmeans"].predict(x_pca).astype("int64")})
