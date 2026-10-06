"""Contact data shown to the LLM is masked, and risk levels follow the documented thresholds."""
import pytest

import fraud_agent as fa


@pytest.mark.parametrize("email, expected", [
    ("ana.perez@gmail.com", "a***@gmail.com"),
    (None, None),
    ("sin-arroba", "sin-arroba"),
])
def test_mask_email(email, expected):
    assert fa.mask_email(email) == expected


@pytest.mark.parametrize("phone, expected", [
    ("+57 300 123 4567", "***4567"),
    ("12", "***"),
    (None, None),
])
def test_mask_phone(phone, expected):
    assert fa.mask_phone(phone) == expected


def test_profile_tool_never_exposes_full_contact_data(offline):
    tools = {t.__name__: t for t in fa.build_tools(
        "CLI-TEST", {"full_name": "Ana", "email": "ana.perez@gmail.com", "phone": "+573001234567"})}
    profile = tools["view_customer_profile"]()
    assert profile["email"] == "a***@gmail.com"
    assert profile["phone"] == "***4567"


@pytest.mark.parametrize("score, level", [(0, "low"), (29.9, "low"), (30, "medium"),
                                          (59.9, "medium"), (60, "high"), (100, "high")])
def test_risk_levels(score, level):
    assert fa.risk_level(score) == level
