import duckdb

from config.config import MotherDuckConfig


def get_motherduck_connection(
    database: str | None = None,
    read_only: bool = True
):
    """
    Crea una conexión a MotherDuck.
    """

    MotherDuckConfig.validate()

    if database:
        connection_string = (
            f"md:{database}"
            f"?motherduck_token={MotherDuckConfig.TOKEN}"
            f"&read_only={str(read_only).lower()}"
        )
    else:
        connection_string = (
            f"md:?motherduck_token={MotherDuckConfig.TOKEN}"
        )

    return duckdb.connect(connection_string)