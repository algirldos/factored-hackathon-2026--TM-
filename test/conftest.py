"""
Shared fixtures. The tests never touch MotherDuck, the LLM APIs, email or WhatsApp:
everything that would leave the machine is replaced with an in-memory fake.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root: fraud_agent, policy...

import fraud_agent as fa  # noqa: E402
import fraud_flow as ff  # noqa: E402

DEMO_OTP = "123456"


@pytest.fixture
def offline(monkeypatch):
    """Replace data access and OTP checks with fakes, so no test needs credentials."""
    monkeypatch.setattr(ff, "verify_otp_code", lambda case_id, code: {"verified": code == DEMO_OTP})
    monkeypatch.setattr(fa, "load_transactions", lambda ids: {})
    monkeypatch.setattr(ff, "case_transactions", lambda case: [])
    monkeypatch.setattr(ff, "update_case", lambda *args, **kwargs: None)


@pytest.fixture
def flow_tools(offline):
    """Tools of the OTP-verified flow for a fake case, keyed by name."""
    tools = ff.build_flow_tools({"case_id": "CASE-TEST", "customer_id": "CLI-TEST"},
                                {"first_name": "Ana"})
    return {t.__name__: t for t in tools}


@pytest.fixture
def agent_tools(offline):
    """Tools of the operator console chat (fraud_agent.py), keyed by name."""
    tools = fa.build_tools("CLI-TEST", {"full_name": "Ana Test"})
    return {t.__name__: t for t in tools}
