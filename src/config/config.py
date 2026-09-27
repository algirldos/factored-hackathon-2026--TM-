from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# Rutas
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR / "config" / ".env"


# ============================================================
# Cargar variables de entorno
# ============================================================

load_dotenv(ENV_PATH)


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
                f"No se encontró MOTHERDUCK_TOKEN en:\n{ENV_PATH}"
            )

        if cls.TOKEN.count(".") != 2:
            raise ValueError(
                "MOTHERDUCK_TOKEN no tiene una estructura JWT válida."
            )


# ============================================================
# Configuración del proyecto
# ============================================================

class ProjectConfig:

    ENV = os.getenv("ENV", "local")