"""
Core orchestrator that coordinates planner, executor, memory, and tools.
"""

import uuid
import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
from src.planner.agent import Planner, PlannerConfig, Plan, PlanStatus, StepStatus
from src.memory.store import MemoryStore, MemoryType
from src.tools.base import ToolRegistry, ToolResult
from src.security.guard import SecurityGuard, RateLimiter, InputValidator


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    max_concurrent_sessions: int = 100
    session_timeout: int = 3600
    enable_memory: bool = True
    enable_security: bool = True
    planner_config: Optional[PlannerConfig] = None


@dataclass
class Session:
    """A user session."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_id: int = 0
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    message_count: int = 0
    context: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_expired(self) -> bool:
        elapsed = (datetime.now() - self.last_active).total_seconds()
        return elapsed > 3600  # default timeout

    def touch(self) -> None:
        self.last_active = datetime.now()
        self.message_count += 1


class Orchestrator:
    """
    Core orchestrator that coordinates all JARVIS components.
    """

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or OrchestratorConfig()
        self.planner = Planner(self.config.planner_config)
        self.tools = ToolRegistry()
        self.memory = MemoryStore()
        self.security = SecurityGuard()
        self._sessions: Dict[str, Session] = {}
        self._active_plans: Dict[str, Plan] = {}

        # Register default tools
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Register all default tools."""
        from src.tools.calculator_tool import CalculatorTool
        from src.tools.shell_tool import ShellTool
        from src.tools.search_tool import SearchTool
        from src.tools.file_tool import FileTool
        from src.tools.python_tool import PythonTool
        from src.tools.browser_tool import BrowserTool
        from src.tools.api_tool import APITool

        for tool_cls in [CalculatorTool, ShellTool, SearchTool, FileTool, PythonTool, BrowserTool, APITool]:
            self.tools.register(tool_cls())

    def create_session(self, user_id: int) -> Session:
        """Create or retrieve a user session."""
        # Check if user has an existing active session
        for session in self._sessions.values():
            if session.user_id == user_id and not session.is_expired:
                session.touch()
                return session

        # Create new session
        session = Session(user_id=user_id)
        self._sessions[session.id] = session
        return session

    async def process_message(
        self,
        user_id: int,
        message: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Process a user message through the full pipeline.
        """
        # Security checks
        if not self.security.is_user_allowed(user_id):
            return {"success": False, "error": "Access denied"}

        user_str = str(user_id)
        if not self.security.check_rate_limit(user_str):
            return {"success": False, "error": "Rate limit exceeded. Please wait."}

        is_valid, error = self.security.validate_input(message)
        if not is_valid:
            return {"success": False, "error": error}

        # Get or create session
        session = self.create_session(user_id)
        if session_id and session_id in self._sessions:
            session = self._sessions[session_id]

        # Store user message in memory
        if self.config.enable_memory:
            self.memory.add(
                content=message,
                memory_type=MemoryType.CONVERSATION,
                metadata={"role": "user", "user_id": user_id},
            )

        # Create execution plan
        plan = await self.planner.create_plan(message)
        self._active_plans[plan.id] = plan

        # Execute plan steps
        results = []
        plan.status = PlanStatus.RUNNING

        for step in plan.steps:
            step.status = StepStatus.RUNNING
            try:
                result = await self.tools.execute(step.tool, **step.args)
                step.result = result.data if result.success else None
                step.error = result.error if not result.success else None
                step.status = StepStatus.COMPLETED if result.success else StepStatus.FAILED
                results.append(result)
            except Exception as e:
                step.error = str(e)
                step.status = StepStatus.FAILED
                results.append(ToolResult(success=False, error=str(e), tool_name=step.tool))

            # Stop if a step fails (or continue based on configuration)
            if step.status == StepStatus.FAILED:
                break

        # Determine overall plan status
        if all(s.status == StepStatus.COMPLETED for s in plan.steps):
            plan.status = PlanStatus.COMPLETED
        elif any(s.status == StepStatus.FAILED for s in plan.steps):
            plan.status = PlanStatus.FAILED
        else:
            plan.status = PlanStatus.COMPLETED

        # Format response
        response_data = self._format_response(plan, results)

        # Store assistant response in memory
        if self.config.enable_memory:
            self.memory.add(
                content=response_data.get("response", ""),
                memory_type=MemoryType.CONVERSATION,
                metadata={"role": "assistant", "user_id": user_id},
            )

        session.touch()
        return response_data

    def _format_response(self, plan: Plan, results: List[ToolResult]) -> Dict[str, Any]:
        """Format plan execution results into a response."""
        response_parts = []

        for step in plan.steps:
            if step.status == StepStatus.COMPLETED and step.result:
                if isinstance(step.result, dict):
                    # Try to extract meaningful content
                    if "content" in step.result:
                        response_parts.append(step.result["content"])
                    elif "result" in step.result:
                        response_parts.append(str(step.result["result"]))
                    elif "results" in step.result:
                        for r in step.result["results"][:3]:
                            if isinstance(r, dict):
                                response_parts.append(
                                    f"**{r.get('title', 'Result')}**\n{r.get('snippet', r.get('url', ''))}"
                                )
                    else:
                        response_parts.append(str(step.result))
                else:
                    response_parts.append(str(step.result))
            elif step.status == StepStatus.FAILED:
                response_parts.append(f"⚠️ {step.description}: {step.error}")

        response_text = "\n\n".join(response_parts) if response_parts else "I wasn't able to process your request."

        return {
            "success": plan.status == PlanStatus.COMPLETED,
            "response": response_text,
            "plan_id": plan.id,
            "steps_completed": sum(1 for s in plan.steps if s.status == StepStatus.COMPLETED),
            "steps_total": len(plan.steps),
            "progress": plan.progress,
        }

    def get_session(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def get_active_sessions(self) -> int:
        return sum(1 for s in self._sessions.values() if not s.is_expired)

    def get_tools_info(self) -> List[Dict[str, Any]]:
        return self.tools.get_definitions()

    def get_memory_stats(self) -> Dict[str, Any]:
        return {"total_entries": self.memory.size}
