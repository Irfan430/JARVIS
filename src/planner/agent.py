"""
Planner Agent — LLM-powered query decomposition into executable plans.

The planner receives a user query plus session context, calls an LLM to
produce a structured Plan of ordered steps, each targeting a specific tool
with its parameters.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PlanStatus(str, Enum):
    DRAFT = "draft"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class PlanStep:
    """A single step in a plan — corresponds to one tool invocation."""

    step_id: str
    tool_name: str
    params: dict[str, Any] = field(default_factory=dict)
    description: str = ""
    depends_on: list[str] = field(default_factory=list)
    status: StepStatus = StepStatus.PENDING
    result: Any = None
    error: str | None = None
    retry_count: int = 0
    max_retries: int = 2
    timeout: float = 60.0

    def can_run(self, completed_steps: set[str]) -> bool:
        """Check if all dependencies are satisfied."""
        return self.status == StepStatus.PENDING and all(
            dep in completed_steps for dep in self.depends_on
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "params": self.params,
            "description": self.description,
            "depends_on": self.depends_on,
            "status": self.status.value,
        }


@dataclass
class Plan:
    """A complete execution plan returned by the planner."""

    plan_id: str
    query: str
    steps: list[PlanStep] = field(default_factory=list)
    status: PlanStatus = PlanStatus.DRAFT
    reasoning: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @property
    def pending_steps(self) -> list[PlanStep]:
        return [s for s in self.steps if s.status == StepStatus.PENDING]

    @property
    def completed_step_ids(self) -> set[str]:
        return {s.step_id for s in self.steps if s.status == StepStatus.COMPLETED}

    @property
    def all_done(self) -> bool:
        return all(s.status in (StepStatus.COMPLETED, StepStatus.SKIPPED) for s in self.steps)

    @property
    def has_failures(self) -> bool:
        return any(s.status == StepStatus.FAILED for s in self.steps)

    def get_step(self, step_id: str) -> PlanStep | None:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "query": self.query,
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "reasoning": self.reasoning,
        }


# ---------------------------------------------------------------------------
# LLM Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface any LLM backend must satisfy."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> str:
        """Return raw text completion."""
        ...


@runtime_checkable
class LLMProviderWithTools(Protocol):
    """Extended interface for providers supporting tool-use / function calling."""

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        response_format: dict | None = None,
    ) -> str:
        ...

    async def generate_with_tools(
        self,
        messages: list[dict[str, str]],
        tools: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict[str, Any]:
        """Return a structured tool-use response."""
        ...


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PlannerConfig:
    """Knobs for the planner."""

    model: str = "gpt-4o"
    temperature: float = 0.1
    max_tokens: int = 2048
    max_steps: int = 10
    max_retries: int = 2
    planning_timeout: float = 30.0


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

PLANNING_SYSTEM_PROMPT = """\
You are JARVIS planning agent. Given a user query and available context, \
decompose the query into an ordered sequence of tool invocations.

You MUST respond with valid JSON matching this schema:

{
  "reasoning": "<brief explanation of your plan>",
  "steps": [
    {
      "tool": "<tool_name>",
      "params": { ... },
      "description": "<what this step does>",
      "depends_on": ["<step_id>"]   // optional: step_ids this depends on
    }
  ]
}

Rules:
- Each step MUST reference a valid tool name from the available tools.
- Use "depends_on" for steps that require output from earlier steps.
- Keep the plan minimal — do not add unnecessary steps.
- If the query is conversational and needs no tools, return steps=[].
- Respond ONLY with valid JSON. No markdown fences, no commentary.
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


class Planner:
    """
    LLM-based planner that decomposes user queries into executable plans.

    Works with any LLM provider that exposes an async `generate` method.
    Falls back to simple rule-based planning if the LLM output is unparseable.
    """

    def __init__(self, llm: LLMProvider, config: PlannerConfig | None = None) -> None:
        self.llm = llm
        self.config = config or PlannerConfig()
        self._available_tools: list[dict[str, Any]] = []
        logger.info("Planner initialised (model=%s)", self.config.model)

    def set_available_tools(self, tool_schemas: list[dict[str, Any]]) -> None:
        """Inject tool schemas so the planner knows what's available."""
        self._available_tools = tool_schemas
        logger.info("Planner registered %d tool schemas", len(tool_schemas))

    # -- Public API ---------------------------------------------------------

    async def create_plan(
        self,
        query: str,
        *,
        context: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> Plan:
        """
        Decompose *query* into a Plan of steps.

        Parameters
        ----------
        query : str
            The raw user query.
        context : dict, optional
            Extra session context (recent results, user prefs, etc.).
        history : list[dict], optional
            Prior conversation turns for multi-turn planning.
        """
        plan_id = f"plan-{uuid.uuid4().hex[:12]}"
        logger.info("Creating plan '%s' for query: %.80s", plan_id, query)

        messages = self._build_messages(query, context, history)

        try:
            raw = await self._call_llm(messages)
            plan = self._parse_plan(plan_id, query, raw)
        except Exception as exc:
            logger.warning("LLM planning failed (%s), using fallback", exc)
            plan = self._fallback_plan(plan_id, query)

        # Enforce max steps
        if len(plan.steps) > self.config.max_steps:
            logger.warning(
                "Plan has %d steps, truncating to %d",
                len(plan.steps),
                self.config.max_steps,
            )
            plan.steps = plan.steps[: self.config.max_steps]

        plan.status = PlanStatus.IN_PROGRESS
        logger.info(
            "Plan '%s' created with %d steps", plan_id, len(plan.steps)
        )
        return plan

    # -- Internals ----------------------------------------------------------

    def _build_messages(
        self,
        query: str,
        context: dict[str, Any] | None,
        history: list[dict[str, str]] | None,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []

        # System prompt with tool inventory
        tool_names = [t.get("function", {}).get("name", "?") for t in self._available_tools]
        system = PLANNING_SYSTEM_PROMPT + "\n\nAvailable tools: " + ", ".join(tool_names)
        messages.append({"role": "system", "content": system})

        # Conversation history
        if history:
            messages.extend(history)

        # User context
        user_content = query
        if context:
            ctx_str = json.dumps(context, default=str, indent=2)
            user_content += f"\n\n<Context>\n{ctx_str}\n</Context>"
        messages.append({"role": "user", "content": user_content})

        return messages

    async def _call_llm(self, messages: list[dict[str, str]]) -> str:
        """Call the LLM with timeout."""
        raw = await asyncio.wait_for(
            self.llm.generate(
                messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            ),
            timeout=self.config.planning_timeout,
        )
        return raw

    def _parse_plan(self, plan_id: str, query: str, raw: str) -> Plan:
        """Parse LLM JSON output into a Plan."""
        # Strip markdown fences if present
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)

        data = json.loads(cleaned)
        reasoning = data.get("reasoning", "")
        steps: list[PlanStep] = []

        for i, step_data in enumerate(data.get("steps", [])):
            step_id = f"step-{i}"
            steps.append(
                PlanStep(
                    step_id=step_id,
                    tool_name=step_data["tool"],
                    params=step_data.get("params", {}),
                    description=step_data.get("description", ""),
                    depends_on=step_data.get("depends_on", []),
                    max_retries=self.config.max_retries,
                )
            )

        return Plan(
            plan_id=plan_id,
            query=query,
            steps=steps,
            reasoning=reasoning,
        )

    def _fallback_plan(self, plan_id: str, query: str) -> Plan:
        """
        Rule-based fallback when the LLM fails.  Tries to route common
        patterns without LLM assistance.
        """
        q = query.lower()
        steps: list[PlanStep] = []

        if any(w in q for w in ("search", "find", "look up", "google")):
            steps.append(
                PlanStep(
                    step_id="step-0",
                    tool_name="web_search",
                    params={"query": query},
                    description="Search the web",
                )
            )
        elif any(w in q for w in ("calculate", "math", "compute")):
            steps.append(
                PlanStep(
                    step_id="step-0",
                    tool_name="calculator",
                    params={"expression": query},
                    description="Calculate the result",
                )
            )
        elif any(w in q for w in ("weather", "forecast", "temperature")):
            steps.append(
                PlanStep(
                    step_id="step-0",
                    tool_name="weather",
                    params={"query": query},
                    description="Get weather information",
                )
            )
        else:
            # Generic: pass to chat model
            steps.append(
                PlanStep(
                    step_id="step-0",
                    tool_name="chat",
                    params={"message": query},
                    description="Respond to the user query",
                )
            )

        return Plan(
            plan_id=plan_id,
            query=query,
            steps=steps,
            reasoning="Fallback rule-based plan (LLM unavailable)",
            metadata={"fallback": True},
        )



