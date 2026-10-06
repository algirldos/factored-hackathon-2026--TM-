"""
Currency conversion to USD.

Rates come from bronze.daily_exchange_rates for the scoring date (see
src/features/sources.py). The static file src/config/currency_config.json is kept
only for the country -> currency mapping and as an explicit fallback for exploration.
"""
import json
import unicodedata
from pathlib import Path

CURRENCY_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "currency_config.json"


def normalize_key(value) -> str:
    """'México', 'Mexico ' and 'MEXICO' map to the same key."""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode()
    return text.strip().lower()


def load_currency_config(path: Path = CURRENCY_CONFIG_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def country_currency_map(config: dict | None = None) -> dict[str, str]:
    """Country (normalized) -> currency code of the customer's income."""
    config = config or load_currency_config()
    return {normalize_key(k): v for k, v in config["country_currency"].items()}


def static_usd_rates(config: dict | None = None) -> dict[str, float]:
    """Fixed rates from currency_config.json. Only for exploration: they do not change over time."""
    config = config or load_currency_config()
    return dict(config["currency_to_usd"])


def usd_rates_from_rows(rows) -> dict[str, float]:
    """
    Build {currency: USD per 1 unit} from exchange-rate rows.

    Each row is (source_currency, target_currency, exchange_rate), read as
    "1 source_currency = exchange_rate target_currency". Rows quoted to USD are used
    directly; rows quoted from USD are inverted. A direct quote wins over an inverted one.
    """
    rates = {"USD": 1.0}
    inverted = {}
    for source, target, rate in rows:
        if rate is None or rate <= 0:
            continue
        if target == "USD" and source != "USD":
            rates[source] = float(rate)
        elif source == "USD" and target != "USD":
            inverted[target] = 1.0 / float(rate)
    for currency, rate in inverted.items():
        rates.setdefault(currency, rate)
    return rates
