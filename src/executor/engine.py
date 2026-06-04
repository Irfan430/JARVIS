"""
Execution Engine — Drives plan steps through the tool router.

Handles:
  - Sequential and parallel step execution respecting dependencies
  - Retries on transient failures
  - Step-level result collection
  - Graceful error handling per step
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .tool_router import ToolRouter, ToolOutcome, ToolResult, ToolError

# Lazy import to avoid circular dependency
def _step_status():
    from ..planner.agent import StepStatus
    return StepStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    """Result of executing a single plan step."""

    step_id: str
    tool_name: str
    outcome: ToolOutcome
    elapsed_ms: float = 0.0
    retries_used: int = 0
    skipped: bool = False

    @property
    def success(self) -> bool:
        return self.outcome.success

    @property
    def output(self) -> Any:
        if isinstance(self.outcome, ToolResult):
            return self.outcome.output
        return None

    @property
    def error(self) -> str | None:
        if isinstance(self.outcome, ToolError):
            return self.outcome.error
        return None


@dataclass
class ExecutionResult:
    """Aggregated result after running an entire plan."""

    plan_id: str
    step_results: list[StepResult] = field(default_factory=list)
    total_elapsed_ms: float = 0.0
    all_succeeded: bool = True
    failed_step_ids: list[str] = field(default_factory=list)

    @property
    def outputs(self) -> dict[str, Any]:
        """Map step_id → output for all successful steps."""
        return {
            r.step_id: r.output
            for r in self.step_results
            if r.success
        }

    @property
    def errors(self) -> dict[str, str]:
        """Map step_id → error message for all failed steps."""
        return {
            r.step_id: r.error
            for r in self.step_results
            if not r.success and r.error
        }


# ---------------------------------------------------------------------------
# Execution Engine
# ---------------------------------------------------------------------------

class ExecutionEngine:
    """
    Executes a Plan by routing each step through the ToolRouter.

    Supports:
      - Dependency-aware execution (steps with depends_on run after deps)
      - Independent steps run concurrently
      - Per-step retries with configurable backoff
      - Timeout per step
    """

    def __init__(
        self,
        router: ToolRouter,
        *,
        default_timeout: float = 60.0,
        retry_base_delay: float = 1.0,
        max_concurrent_steps: int = 5,
    ) -> None:
        self.router = router
        self.default_timeout = default_timeout
        self.retry_base_delay = retry_base_delay
        self._step_semaphore = asyncio.Semaphore(max_concurrent_steps)
        logger.info(
            "ExecutionEngine created (default_timeout=%.1fs, max_concurrent=%d)",
            default_timeout,
            max_concurrent_steps,
        )

    async def execute_plan(self, plan: Any) -> ExecutionResult:
        """
        Execute all steps in a plan, respecting dependencies.

        Parameters
        ----------
        plan : Plan
            A Plan object from the planner module.
        """
        SS = _step_status()
        start = time.monotonic()
        result = ExecutionResult(plan_id=plan.plan_id)
        completed_ids: set[str] = set()

        logger.info(
            "Starting execution of plan '%s' (%d steps)",
            plan.plan_id,
            len(plan.steps),
        )

        while not plan.all_done:
            # Find steps that can run now
            runnable = [
                s for s in plan.pending_steps if s.can_run(completed_ids)
            ]

            if not runnable:
                # No runnable steps but plan not done → stuck
                remaining = plan.pending_steps
                if remaining:
                    logger.error(
                        "Plan '%s' has %d unreachable steps (missing deps?)",
                        plan.plan_id,
                        len(remaining),
                    )
                    for s in remaining:
                        s.status = SS.SKIPPED
                        result.step_results.append(
                            StepResult(
                                step_id=s.step_id,
                                tool_name=s.tool_name,
                                outcome=ToolError(
                                    tool_name=s.tool_name,
                                    error="Step skipped: unsatisfied dependencies",
                                    error_type="DependencyError",
                                ),
                                skipped=True,
                            )
                        )
                        completed_ids.add(s.step_id)
                break

            # Execute runnable steps concurrently
            logger.info(
                "Executing %d step(s) in parallel: %s",
                len(runnable),
                [s.step_id for s in runnable],
            )
            tasks = [self._execute_step(s, completed_ids) for s in runnable]
            step_results = await asyncio.gather(*tasks, return_exceptions=True)

            for sr in step_results:
                if isinstance(sr, Exception):
                    logger.error("Unexpected error from gather: %s", sr)
                    continue
                result.step_results.append(sr)
                if sr.success:
                    completed_ids.add(sr.step_id)
                else:
                    result.failed_step_ids.append(sr.step_id)
                    result.all_succeeded = False

        result.total_elapsed_ms = (time.monotonic() - start) * 1000
        logger.info(
            "Plan '%s' finished: %d/%d steps succeeded in %.0fms",
            plan.plan_id,
            len(result.step_results) - len(result.failed_step_ids),
            len(plan.steps),
            result.total_elapsed_ms,
        )
        return result

    async def execute_steps_sequential(
        self, steps: list[Any]
    ) -> list[StepResult]:
        """Execute a flat list of steps sequentially (no dependency check)."""
        completed_ids: set[str] = set()
        results: list[StepResult] = []

        for step in steps:
            sr = await self._execute_step(step, completed_ids)
            results.append(sr)
            if sr.success:
                completed_ids.add(sr.step_id)

        return results

    async def execute_step(self, step: Any) -> StepResult:
        """Execute a single step directly."""
        return await self._execute_step(step, set())

    # -- Internal -----------------------------------------------------------

    async def _execute_step(self, step: Any, completed_ids: set[str]) -> StepResult:
        """Execute a single step with retries."""
        SS = _step_status()
        timeout = getattr(step, "timeout", self.default_timeout) or self.default_timeout
        max_retries = getattr(step, "max_retries", 2)
        tool_name = step.tool_name
        params = dict(step.params)  # copy

        # Inject dependency results into params as ${step_id}
        for dep_id in step.depends_on:
            if dep_id in completed_ids:
                params[f"_{dep_id}_result"] = completed_ids  # marker

        step.status = SS.RUNNING
        last_outcome: ToolOutcome | None = None

        for attempt in range(1 + max_retries):
            async with self._step_semaphore:
                logger.debug(
                    "Step '%s' attempt %d/%d (tool=%s)",
                    step.step_id,
                    attempt + 1,
                    1 + max_retries,
                    tool_name,
                )

                outcome = await self.router.dispatch(
                    tool_name, params, timeout=timeout
                )

                if outcome.success:
                    step.status = SS.COMPLETED
                    step.result = outcome.output if isinstance(outcome, ToolResult) else None
                    elapsed = outcome.elapsed_ms if isinstance(outcome, ToolResult) else 0
                    logger.info(
                        "Step '%s' completed (tool=%s, %.1fms)",
                        step.step_id,
                        tool_name,
                        elapsed,
                    )
                    return StepResult(
                        step_id=step.step_id,
                        tool_name=tool_name,
                        outcome=outcome,
                        elapsed_ms=elapsed,
                        retries_used=attempt,
                    )

                last_outcome = outcome
                step.error = outcome.error if isinstance(outcome, ToolError) else str(outcome)

                # Only retry if retryable
                is_retryable = (
                    isinstance(outcome, ToolError) and outcome.retryable
                )
                if not is_retryable:
                    break

                if attempt < max_retries:
                    delay = self.retry_base_delay * (2 ** attempt)
                    logger.warning(
                        "Step '%s' failed (retryable), retrying in %.1fs: %s",
                        step.step_id,
                        delay,
                        outcome.error,
                    )
                    await asyncio.sleep(delay)

        # All retries exhausted
        step.status = SS.FAILED
        elapsed = (
            last_outcome.elapsed_ms
            if isinstance(last_outcome, ToolResult)
            else (last_outcome.elapsed_ms if isinstance(last_outcome, ToolError) else 0)
        )
        logger.error(
            "Step '%s' FAILED after %d attempt(s): %s",
            step.step_id,
            1 + max_retries,
            last_outcome.error if isinstance(last_outcome, ToolError) else "unknown",
        )
        return StepResult(
            step_id=step.step_id,
            tool_name=tool_name,
            outcome=last_outcome or ToolError(
                tool_name=tool_name,
                error="Unknown failure",
                error_type="UnknownError",
            ),
            elapsed_ms=elapsed,
            retries_used=max_retries,
        )
