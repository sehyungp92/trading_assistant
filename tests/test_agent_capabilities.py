"""Tests for agent capability schemas."""
from schemas.agent_capabilities import AgentCapability, AgentType, CapabilityCheckResult


class TestAgentTypeEnum:
    def test_has_7_members(self):
        assert len(AgentType) == 7

    def test_values_are_lowercase(self):
        for member in AgentType:
            assert member.value == member.value.lower()


class TestAgentCapabilitySchema:
    def test_construction_with_defaults(self):
        cap = AgentCapability(agent_type=AgentType.DAILY_ANALYSIS)
        assert cap.can_execute_shell is False
        assert cap.max_concurrent_tasks == 1
        assert cap.allowed_actions == []
        assert cap.forbidden_actions == []

    def test_construction_with_custom_values(self):
        cap = AgentCapability(
            agent_type=AgentType.BUG_TRIAGE,
            allowed_actions=["read_source", "open_github_issue"],
            forbidden_actions=["merge_pr"],
            can_execute_shell=False,
            max_concurrent_tasks=2,
        )
        assert cap.agent_type == AgentType.BUG_TRIAGE
        assert "read_source" in cap.allowed_actions
        assert cap.max_concurrent_tasks == 2


class TestCapabilityCheckResult:
    def test_result_construction(self):
        result = CapabilityCheckResult(
            allowed=True,
            agent_type="daily_analysis",
            action="generate_report",
            reason="Allowed",
        )
        assert result.allowed is True
        assert result.agent_type == "daily_analysis"
