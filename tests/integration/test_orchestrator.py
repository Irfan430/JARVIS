"""
Integration tests for the Orchestrator (full pipeline).
"""

import pytest
from src.orchestrator.core import Orchestrator, OrchestratorConfig, Session
from src.planner.agent import PlannerConfig


class TestSession:
    """Tests for Session."""

    def test_creation(self):
        session = Session(user_id=12345)
        assert session.user_id == 12345
        assert session.message_count == 0
        assert session.is_expired is False

    def test_touch(self):
        session = Session(user_id=12345)
        session.touch()
        assert session.message_count == 1

    def test_unique_ids(self):
        s1 = Session(user_id=1)
        s2 = Session(user_id=2)
        assert s1.id != s2.id


class TestOrchestrator:
    """Integration tests for the Orchestrator."""

    @pytest.fixture
    def orchestrator(self):
        config = OrchestratorConfig(
            enable_memory=True,
            enable_security=True,
            planner_config=PlannerConfig(max_steps=5),
        )
        return Orchestrator(config)

    def test_init(self, orchestrator):
        assert orchestrator.tools is not None
        assert orchestrator.memory is not None
        assert orchestrator.security is not None
        assert orchestrator.planner is not None

    def test_tools_registered(self, orchestrator):
        tool_names = orchestrator.tools.list_tools()
        assert "calculator" in tool_names
        assert "shell" in tool_names
        assert "search" in tool_names
        assert "file" in tool_names
        assert "python" in tool_names
        assert "browser" in tool_names
        assert "api" in tool_names

    def test_create_session(self, orchestrator):
        session = orchestrator.create_session(user_id=12345)
        assert session.user_id == 12345
        assert orchestrator.get_active_sessions() == 1

    def test_reuse_session(self, orchestrator):
        s1 = orchestrator.create_session(user_id=12345)
        s2 = orchestrator.create_session(user_id=12345)
        assert s1.id == s2.id

    @pytest.mark.asyncio
    async def test_process_calculation(self, orchestrator):
        result = await orchestrator.process_message(
            user_id=12345,
            message="Calculate 2 + 2",
        )
        assert result["success"] is True
        assert "4" in str(result["response"])
        assert result["steps_completed"] >= 1

    @pytest.mark.asyncio
    async def test_process_shell_command(self, orchestrator):
        result = await orchestrator.process_message(
            user_id=12345,
            message="Run command echo hello",
        )
        assert result["success"] is True
        assert "hello" in result["response"].lower()

    @pytest.mark.asyncio
    async def test_process_code_execution(self, orchestrator):
        result = await orchestrator.process_message(
            user_id=12345,
            message="Write a Python function that prints 42",
        )
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_process_general_query(self, orchestrator):
        result = await orchestrator.process_message(
            user_id=12345,
            message="Hello, how are you?",
        )
        assert "response" in result

    @pytest.mark.asyncio
    async def test_blocked_user(self, orchestrator):
        orchestrator.security.block_user(99999)
        result = await orchestrator.process_message(
            user_id=99999,
            message="Hello",
        )
        assert result["success"] is False
        assert "denied" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_rate_limit(self, orchestrator):
        from src.security.guard import RateLimiter
        orchestrator.security.rate_limiter = RateLimiter(max_requests=2, window_seconds=60)

        for _ in range(2):
            await orchestrator.process_message(user_id=12345, message="Hi")

        result = await orchestrator.process_message(user_id=12345, message="Hi again")
        assert result["success"] is False
        assert "rate" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_empty_input(self, orchestrator):
        result = await orchestrator.process_message(
            user_id=12345,
            message="",
        )
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_memory_stored(self, orchestrator):
        await orchestrator.process_message(
            user_id=12345,
            message="Calculate 5 + 5",
        )
        stats = orchestrator.get_memory_stats()
        assert stats["total_entries"] >= 2  # user message + assistant response

    def test_tools_info(self, orchestrator):
        info = orchestrator.get_tools_info()
        assert len(info) >= 7
        names = [t["name"] for t in info]
        assert "calculator" in names

    @pytest.mark.asyncio
    async def test_multiple_users(self, orchestrator):
        r1 = await orchestrator.process_message(user_id=100, message="Calculate 1 + 1")
        r2 = await orchestrator.process_message(user_id=200, message="Calculate 2 + 2")
        assert orchestrator.get_active_sessions() == 2
        assert r1["success"] is True
        assert r2["success"] is True
