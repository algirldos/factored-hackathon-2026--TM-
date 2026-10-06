"""
Action permissions for the agent's tools - LATAM Bank (Factored AI & Data Hackathon 2026).

Permissions are enforced here, in code, not in the model's prompt: whatever the LLM tries,
a tool only runs if this table allows it. A tool that is not in the table never runs
(deny by default), so a new tool must be registered here before the agent can use it.
"""
import functools
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class ToolPolicy:
    requires_verification: bool   # identity must be verified (OTP) in this session
    requires_human_approval: bool  # a person approves the action before it takes effect
    side_effect: bool              # changes state outside the conversation (sends, writes)


# Human approval is enforced by the tool itself (send_alert asks the operator when
# CONFIRM_SENDS=1); it is recorded here so the full permission model lives in one place.
TOOL_POLICY: dict[str, ToolPolicy] = {
    # Identity
    "verify_otp":               ToolPolicy(False, False, True),
    "resend_otp":               ToolPolicy(False, False, True),
    # Read-only data
    "view_customer_profile":    ToolPolicy(True, False, False),
    "list_transactions":        ToolPolicy(True, False, False),
    "list_recent_transactions": ToolPolicy(True, False, False),
    "assess_fraud_risk":        ToolPolicy(True, False, False),
    "get_case_details":         ToolPolicy(True, False, False),
    # Actions
    "record_customer_response": ToolPolicy(True, False, True),
    "send_fraud_alert":         ToolPolicy(True, True, True),
    "escalate_to_agent":        ToolPolicy(True, False, True),
}

NOT_VERIFIED = {"error": "Identity not verified yet. Ask for the 6-digit code sent by email "
                         "and call verify_otp. Do not share any account information."}


def enforce(tools: list[Callable], is_verified: Callable[[], bool]) -> list[Callable]:
    """
    Wrap each tool so it only runs when its policy allows it.
    The wrappers keep the name, signature and docstring, so tool schemas do not change.
    """
    unregistered = [t.__name__ for t in tools if t.__name__ not in TOOL_POLICY]
    if unregistered:
        raise ValueError(f"Tools without a policy in policy.TOOL_POLICY: {unregistered}")

    def guard(tool: Callable) -> Callable:
        policy = TOOL_POLICY[tool.__name__]

        @functools.wraps(tool)
        def guarded(*args, **kwargs):
            if policy.requires_verification and not is_verified():
                print(f"  [policy] {tool.__name__} bloqueada: identidad sin verificar", flush=True)
                return NOT_VERIFIED
            return tool(*args, **kwargs)
        return guarded

    return [guard(t) for t in tools]
