from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

# Load .env from project root
load_dotenv(BASE_DIR / ".env")



# ============================================================
# Configuración MotherDuck
# ============================================================

class MotherDuckConfig:

    TOKEN = os.getenv("MOTHERDUCK_TOKEN")

    @classmethod
    def validate(cls):
        """
        Valida que exista un token de MotherDuck
        y que tenga una estructura básica de JWT.
        """

        if not cls.TOKEN:
            raise ValueError(
                f"No se encontró MOTHERDUCK_TOKEN en el archivo .env en {BASE_DIR}"
            )

        if cls.TOKEN.count(".") != 2:
            raise ValueError(
                "MOTHERDUCK_TOKEN no tiene una estructura JWT válida."
            )


class PathsConfig:

    BASE_DIR = BASE_DIR

    _clean_data_path = os.getenv("CLEAN_DATA_PATH")

    if _clean_data_path:
        CLEAN_DATA_PATH = BASE_DIR / Path(_clean_data_path)
    else:
        CLEAN_DATA_PATH = (
            BASE_DIR
            / "src"
            / "data"
            / "processed"
            / "reduced_users_df.csv"
        )

    @classmethod
    def ensure_dirs(cls):
        cls.CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
# ============================================================
# Configuración del proyecto
# ============================================================

class ProjectConfig:

    ENV = os.getenv("ENV", "local")