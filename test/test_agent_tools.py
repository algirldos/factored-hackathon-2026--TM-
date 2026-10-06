"""
Integration tests: the agent's real tools behind policy.py, as the LLM providers see them.
Data access is faked (see conftest.offline), so no credentials are needed.
"""
import json

import pytest

import fraud_agent as fa
import policy
from conftest import DEMO_OTP

DATA_TOOLS = ["get_case_details", "list_recent_transactions",
              "record_customer_response", "escalate_to_agent"]
CALLS = {
    "get_case_details": lambda t: t(),
    "list_recent_transactions": lambda t: t(30, 5),
    "record_customer_response": lambda t: t(False, "no fui yo"),
    "escalate_to_agent": lambda t: t("es", "no reconoce cobros", "resumen"),
}


def test_every_tool_is_registered(flow_tools, agent_tools):
    for name in [*flow_tools, *agent_tools]:
        assert name in policy.TOOL_POLICY, name


@pytest.mark.parametrize("name", DATA_TOOLS)
def test_data_tools_blocked_without_otp(flow_tools, name):
    assert CALLS[name](flow_tools[name]) == policy.NOT_VERIFIED


def test_wrong_code_keeps_tools_blocked(flow_tools):
    assert flow_tools["verify_otp"]("000000") == {"verified": False}
    assert flow_tools["list_recent_transactions"](30, 5) == policy.NOT_VERIFIED


def test_right_code_unlocks_tools(flow_tools):
    assert flow_tools["verify_otp"](DEMO_OTP) == {"verified": True}
    assert flow_tools["list_recent_transactions"](30, 5) == {"transactions": []}
    assert flow_tools["get_case_details"]()["case_id"] == "CASE-TEST"


def test_verification_is_not_undone_by_a_retried_call(flow_tools):
    flow_tools["verify_otp"](DEMO_OTP)
    flow_tools["verify_otp"]("000000")  # an LLM retry must not downgrade the session
    assert flow_tools["list_recent_transactions"](30, 5) == {"transactions": []}


def test_policy_applies_through_the_llm_tool_loop(flow_tools):
    """The Claude/Ollama path runs tools by name with JSON arguments: the gate still holds."""
    config = fa.AnthropicConfig("system", list(flow_tools.values()))
    blocked = json.loads(config.run_tool("list_recent_transactions", {"days": "30", "limit": 5}))
    assert blocked == policy.NOT_VERIFIED

    config.run_tool("verify_otp", {"code": DEMO_OTP})
    allowed = json.loads(config.run_tool("list_recent_transactions", {"days": "30", "limit": 5}))
    assert allowed == {"transactions": []}


def test_tool_schemas_are_unchanged_by_the_policy(flow_tools):
    schema = fa.tool_schema(flow_tools["list_recent_transactions"])["function"]
    assert schema["name"] == "list_recent_transactions"
    assert list(schema["parameters"]["properties"]) == ["days", "limit"]
    assert schema["parameters"]["required"] == ["days", "limit"]
    assert schema["parameters"]["properties"]["days"]["type"] == "integer"


def test_gemini_accepts_the_guarded_tools(flow_tools):
    types = pytest.importorskip("google.genai.types")
    types.GenerateContentConfig(system_instruction="system", tools=list(flow_tools.values()))


def test_console_chat_tools_are_scoped_and_available(agent_tools):
    assert list(agent_tools) == ["view_customer_profile", "list_transactions", "assess_fraud_risk",
                                 "send_fraud_alert", "escalate_to_agent"]
    profile = agent_tools["view_customer_profile"]()
    assert profile["customer_id"] == "CLI-TEST"


def test_alert_requires_a_previous_assessment(agent_tools):
    result = agent_tools["send_fraud_alert"](["TRX-1"], "email", "mensaje")
    assert result["sent"] is False
