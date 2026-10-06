"""USD conversion: rate direction, precedence and the country -> currency mapping."""
import pytest

from src.features.currency import (country_currency_map, normalize_key, static_usd_rates,
                                   usd_rates_from_rows)


@pytest.mark.parametrize("raw", ["México", "Mexico", " MEXICO "])
def test_country_names_are_normalized(raw):
    assert normalize_key(raw) == "mexico"


def test_direct_quotes_to_usd_are_used_as_is():
    assert usd_rates_from_rows([("COP", "USD", 0.00025)])["COP"] == 0.00025


def test_quotes_from_usd_are_inverted():
    assert usd_rates_from_rows([("USD", "MXN", 20.0)])["MXN"] == pytest.approx(0.05)


def test_direct_quote_wins_over_inverted_quote():
    rates = usd_rates_from_rows([("USD", "ARS", 1000.0), ("ARS", "USD", 0.0011)])
    assert rates["ARS"] == 0.0011


def test_usd_is_always_one_and_invalid_rates_are_ignored():
    rates = usd_rates_from_rows([("COP", "USD", 0), ("MXN", "USD", None), ("COP", "EUR", 0.0002)])
    assert rates == {"USD": 1.0}


def test_static_config_covers_every_country_currency():
    countries = country_currency_map()
    rates = static_usd_rates()
    assert {"colombia", "mexico", "argentina"} <= set(countries)
    assert all(currency in rates for currency in countries.values())
