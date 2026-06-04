"""
Tests for the Planner agent.
"""

import pytest
from src.planner.agent import (
    Planner, PlannerConfig, Plan, PlanStep,
    PlanStatus, StepStatus,
)


class TestPlanStep:
    """Tests for PlanStep dataclass."""

    def test_creation_defaults(self):
        step = PlanStep()
        assert step.id is not None
        assert step.status == StepStatus.PENDING
        assert step.result is None
        assert step.error is None
        assert step.depends_on == []
        assert step.args == {}

    def test_creation_with_values(self):
        step = PlanStep(
            description="Search the web",
            tool="search",
            args={"query": "test"},
        )
        assert step.description == "Search the web"
        assert step.tool == "search"
        assert step.args == {"query": "test"}

    def test_to_dict(self):
        step = PlanStep(description="Test step", tool="calculator")
        d = step.to_dict()
        assert d["description"] == "Test step"
        assert d["tool"] == "calculator"
        assert d["status"] == "pending"
        assert "id" in d

    def test_unique_ids(self):
        step1 = PlanStep()
        step2 = PlanStep()
        assert step1.id != step2.id


class TestPlan:
    """Tests for Plan dataclass."""

    def test_creation_defaults(self):
        plan = Plan()
        assert plan.id is not None
        assert plan.status == PlanStatus.PENDING
        assert plan.steps == []
        assert plan.is_complete is False

    def test_is_complete_when_completed(self):
        plan = Plan(status=PlanStatus.COMPLETED)
        assert plan.is_complete is True

    def test_is_complete_when_failed(self):
        plan = Plan(status=PlanStatus.FAILED)
        assert plan.is_complete is True

    def test_is_complete_when_pending(self):
        plan = Plan(status=PlanStatus.PENDING)
        assert plan.is_complete is False

    def test_progress_empty(self):
        plan = Plan()
        assert plan.progress == 0.0

    def test_progress_partial(self):
        plan = Plan(steps=[
            PlanStep(status=StepStatus.COMPLETED),
            PlanStep(status=StepStatus.RUNNING),
            PlanStep(status=StepStatus.PENDING),
        ])
        assert plan.progress == pytest.approx(1 / 3)

    def test_progress_all_complete(self):
        plan = Plan(steps=[
            PlanStep(status=StepStatus.COMPLETED),
            PlanStep(status=StepStatus.SKIPPED),
        ])
        assert plan.progress == 1.0

    def test_to_dict(self):
        plan = Plan(query="test", steps=[PlanStep(description="step1")])
        d = plan.to_dict()
        assert d["query"] == "test"
        assert len(d["steps"]) == 1
        assert d["status"] == "pending"


class TestPlanner:
    """Tests for the Planner agent."""

    def test_init_default(self):
        planner = Planner()
        assert planner.config.max_steps == 10
        assert planner.config.temperature == 0.3

    def test_init_custom_config(self):
        config = PlannerConfig(max_steps=5, temperature=0.5)
        planner = Planner(config)
        assert planner.config.max_steps == 5

    @pytest.mark.asyncio
    async def test_create_plan_search(self, planner_config):
        planner = Planner(planner_config)
        plan = await planner.create_plan("Search for Python tutorials")
        assert plan.status == PlanStatus.PENDING
        assert len(plan.steps) >= 1
        assert plan.steps[0].tool == "search"

    @pytest.mark.asyncio
    async def test_create_plan_calculator(self, planner_config):
        planner = Planner(planner_config)
        plan = await planner.create_plan("Calculate 2 + 2")
        assert len(plan.steps) >= 1
        assert any(s.tool == "calculator" for s in plan.steps)

    @pytest.mark.asyncio
    async def test_create_plan_shell(self, planner_config):
        planner = Planner(planner_config)
        plan = await planner.create_plan("Run command ls -la")
        assert any(s.tool == "shell" for s in plan.steps)

    @pytest.mark.asyncio
    async def test_create_plan_code(self, planner_config):
        planner = Planner(planner_config)
        plan = await planner.create_plan("Write a Python function")
        assert any(s.tool == "python" for s in plan.steps)

    @pytest.mark.asyncio
    async def test_create_plan_file(self, planner_config):
        planner = Planner(planner_config)
        plan = await planner.create_plan("Read file /tmp/test.txt")
        assert any(s.tool == "file" for s in plan.steps)

    @pytest.mark.asyncio
    async def test_create_plan_browser(self, planner_config):
        planner = Planner(planner_config)
        plan = await planner.create_plan("Open https://example.com")
        assert any(s.tool == "browser" for s in plan.steps)

    @pytest.mark.asyncio
    async def test_create_plan_default(self, planner_config):
        planner = Planner(planner_config)
        plan = await planner.create_plan("Hello, how are you?")
        assert len(plan.steps) >= 1
        assert plan.steps[0].tool == "ai_chat"

    @pytest.mark.asyncio
    async def test_plan_max_steps(self):
        config = PlannerConfig(max_steps=2)
        planner = Planner(config)
        # Query that would normally produce many steps
        plan = await planner.create_plan("Search and calculate and run command and write code")
        assert len(plan.steps) <= config.max_steps

    def test_plan_caching(self):
        planner = Planner()
        plan = Plan(query="test", status=PlanStatus.PENDING)
        planner._plan_cache["test"] = plan

        cached = planner.get_cached_plan("test")
        assert cached is plan

    def test_plan_cache_miss(self):
        planner = Planner()
        assert planner.get_cached_plan("nonexistent") is None

    def test_clear_cache(self):
        planner = Planner()
        planner._plan_cache["test"] = Plan()
        planner.clear_cache()
        assert planner.get_cached_plan("test") is None
