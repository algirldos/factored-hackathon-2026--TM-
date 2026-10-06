"""Unit tests for policy.py: the permission table and its enforcement."""
import inspect

import pytest

import policy


def tool_a(days: int) -> dict:
    """Reads something.

    Args:
        days: Days to look back.
    """
    return {"ok": True, "days": days}


def test_every_policy_entry_is_well_formed():
    for name, rule in policy.TOOL_POLICY.items():
        assert isinstance(rule, policy.ToolPolicy), name
        # A tool that needs human approval always has a side effect
        assert not rule.requires_human_approval or rule.side_effect, name


def test_identity_tools_do_not_require_verification():
    # Otherwise the customer could never verify
    assert not policy.TOOL_POLICY["verify_otp"].requires_verification
    assert not policy.TOOL_POLICY["resend_otp"].requires_verification


def test_unregistered_tool_is_rejected():
    with pytest.raises(ValueError, match="tool_a"):
        policy.enforce([tool_a], is_verified=lambda: True)


def test_blocks_until_verified(monkeypatch):
    monkeypatch.setitem(policy.TOOL_POLICY, "tool_a", policy.ToolPolicy(True, False, False))
    verified = {"value": False}
    [guarded] = policy.enforce([tool_a], is_verified=lambda: verified["value"])

    assert guarded(7) == policy.NOT_VERIFIED
    verified["value"] = True  # checked on every call, not once at build time
    assert guarded(7) == {"ok": True, "days": 7}


def test_tool_without_verification_runs_unverified(monkeypatch):
    monkeypatch.setitem(policy.TOOL_POLICY, "tool_a", policy.ToolPolicy(False, False, False))
    [guarded] = policy.enforce([tool_a], is_verified=lambda: False)
    assert guarded(days=3) == {"ok": True, "days": 3}


def test_wrapper_keeps_name_signature_and_docstring(monkeypatch):
    monkeypatch.setitem(policy.TOOL_POLICY, "tool_a", policy.ToolPolicy(True, False, False))
    [guarded] = policy.enforce([tool_a], is_verified=lambda: True)
    assert guarded.__name__ == "tool_a"
    assert inspect.signature(guarded) == inspect.signature(tool_a)
    assert inspect.getdoc(guarded) == inspect.getdoc(tool_a)
