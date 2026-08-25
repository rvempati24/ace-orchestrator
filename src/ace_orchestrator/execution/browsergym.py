"""BrowserGym-backed task environments with one isolated worker process per episode."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from ace_orchestrator.benchmarks.base import BenchmarkTask, BrowserGymTask
from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.execution.environment import (
    EnvironmentFactory,
    EnvironmentObservation,
    EnvironmentSession,
)

_WORKER_ENVIRONMENTS: dict[str, Any] = {}


def _browsergym_worker(session_id: str, operation: str, payload: dict[str, Any]) -> Any:
    """Own a synchronous Gym environment inside a dedicated child process."""

    if operation == "reset":
        import gymnasium as gym

        environment_id = str(payload["environment_id"])
        if environment_id.startswith("browsergym/miniwob."):
            import browsergym.miniwob  # noqa: F401

        kwargs: dict[str, Any] = {
            "headless": bool(payload["headless"]),
            "pre_observation_delay": float(payload["pre_observation_delay"]),
        }
        if environment_id.startswith("browsergym/miniwob.") and payload.get("miniwob_url"):
            kwargs["task_kwargs"] = {"base_url": str(payload["miniwob_url"])}
        environment = gym.make(environment_id, **kwargs)
        try:
            observation, info = environment.reset(seed=int(payload["seed"]))
        except BaseException:
            environment.close()
            raise
        _WORKER_ENVIRONMENTS[session_id] = environment
        return {"observation": observation, "info": info, "worker_pid": os.getpid()}

    environment = _WORKER_ENVIRONMENTS.get(session_id)
    if operation == "close":
        if environment is not None:
            try:
                environment.close()
            finally:
                _WORKER_ENVIRONMENTS.pop(session_id, None)
        return None
    if environment is None:
        raise RuntimeError(f"BrowserGym worker has no environment for session {session_id}")
    if operation == "step":
        observation, reward, terminated, truncated, info = environment.step(payload["action"])
        return {
            "observation": observation,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": info,
        }
    raise ValueError(f"unknown BrowserGym worker operation: {operation}")


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return str(value)


def _extract_action_refs(value: Any) -> tuple[str, ...]:
    refs: set[str] = set()

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            ref = item.get("browsergym_id")
            if ref is not None:
                refs.add(str(ref))
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return tuple(
        sorted(refs, key=lambda ref: (not ref.isdigit(), int(ref) if ref.isdigit() else ref))
    )


def _encode_screenshot(screenshot: Any) -> str | None:
    if screenshot is None:
        return None
    if isinstance(screenshot, bytes):
        return base64.b64encode(screenshot).decode("ascii")
    try:
        from PIL import Image

        image = Image.fromarray(screenshot)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")
    except (ImportError, TypeError, ValueError):
        return None


def computer_action_to_browsergym(action: ComputerAction) -> str:
    """Translate the harness action vocabulary into BrowserGym high-level calls."""

    kind = action.kind.strip().lower().replace("-", "_")

    def ref() -> str:
        if not action.target_ref:
            raise ValueError(f"{kind} requires target_ref")
        return action.target_ref

    if kind == "click":
        return f"click({ref()!r})"
    if kind in {"double_click", "dblclick"}:
        return f"dblclick({ref()!r})"
    if kind == "hover":
        return f"hover({ref()!r})"
    if kind == "focus":
        return f"focus({ref()!r})"
    if kind == "clear":
        return f"clear({ref()!r})"
    if kind in {"type", "fill"}:
        if action.value is None:
            raise ValueError(f"{kind} requires value")
        return f"fill({ref()!r}, {action.value!r})"
    if kind in {"select", "select_option"}:
        if action.value is None:
            raise ValueError(f"{kind} requires value")
        return f"select_option({ref()!r}, {action.value!r})"
    if kind == "press":
        if action.value is None:
            raise ValueError("press requires a key combination in value")
        return f"press({ref()!r}, {action.value!r})"
    if kind in {"drag", "drag_and_drop"}:
        if not action.target:
            raise ValueError("drag_and_drop requires the destination ref in target")
        return f"drag_and_drop({ref()!r}, {action.target!r})"
    if kind == "scroll":
        delta_x = float(action.metadata.get("delta_x", 0))
        delta_y = float(action.metadata.get("delta_y", action.value or 0))
        return f"scroll({delta_x!r}, {delta_y!r})"
    if kind in {"navigate", "goto"}:
        url = action.target or action.value
        if not url or urlparse(url).scheme not in {"http", "https"}:
            raise ValueError("navigate requires an http(s) URL")
        return f"goto({url!r})"
    if kind in {"back", "go_back"}:
        return "go_back()"
    if kind in {"forward", "go_forward"}:
        return "go_forward()"
    if kind == "noop":
        wait_ms = float(action.metadata.get("wait_ms", 1000))
        return f"noop({wait_ms!r})"
    raise ValueError(f"unsupported BrowserGym action kind: {action.kind}")


@dataclass(frozen=True)
class BrowserGymRuntimeConfig:
    headless: bool = True
    pre_observation_delay: float = 0.0
    miniwob_url: str | None = None


class BrowserGymEnvironmentSession(EnvironmentSession):
    """A task-owned BrowserGym episode executing in its own process."""

    def __init__(self, task: BrowserGymTask, config: BrowserGymRuntimeConfig) -> None:
        super().__init__(task)
        self.browsergym_task = task
        self.config = config
        self._executor = ProcessPoolExecutor(max_workers=1)
        self._started = False
        self._raw_observation: dict[str, Any] | None = None
        self._reset_info: dict[str, Any] = {}
        self._step_info: dict[str, Any] = {}
        self._observation_count = 0
        self._worker_pid: int | None = None
        self._reset_latency_s = 0.0
        self._step_latencies_s: list[float] = []
        self._last_reward = 0.0
        self._cumulative_reward = 0.0
        self._terminated = False
        self._truncated = False
        self._last_action_metrics: dict[str, Any] = {}

    async def _rpc(self, operation: str, payload: dict[str, Any]) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, _browsergym_worker, self.session_id, operation, payload
        )

    async def _start(self) -> None:
        if self._started:
            return
        started = perf_counter()
        result = await self._rpc(
            "reset",
            {
                "environment_id": self.browsergym_task.environment_id,
                "seed": self.browsergym_task.seed,
                "headless": self.config.headless,
                "pre_observation_delay": self.config.pre_observation_delay,
                "miniwob_url": self.config.miniwob_url,
            },
        )
        self._reset_latency_s = perf_counter() - started
        self._raw_observation = result["observation"]
        self._reset_info = _json_safe(result["info"])
        self._worker_pid = int(result["worker_pid"])
        self._started = True

    def _map_observation(self) -> EnvironmentObservation:
        if self._raw_observation is None:
            raise RuntimeError("BrowserGym environment has no observation")
        raw = self._raw_observation
        self._observation_count += 1
        semantic_state = _json_safe(raw.get("axtree_object", {}))
        return EnvironmentObservation(
            observation_id=f"{self.session_id}:observation:{self._observation_count}",
            values={
                "goal": _json_safe(raw.get("goal")),
                "url": _json_safe(raw.get("url")),
                "semantic_state": semantic_state,
                "dom_state": _json_safe(raw.get("dom_object", {})),
                "focused_ref": _json_safe(raw.get("focused_element_bid")),
                "last_action": _json_safe(raw.get("last_action")),
                "last_action_error": _json_safe(raw.get("last_action_error")),
                "open_pages_urls": _json_safe(raw.get("open_pages_urls", [])),
                "active_page_index": _json_safe(raw.get("active_page_index")),
            },
            screenshot_base64=_encode_screenshot(raw.get("screenshot")),
            available_action_refs=_extract_action_refs(raw.get("axtree_object", {})),
            metadata={
                "environment_id": self.browsergym_task.environment_id,
                "seed": self.browsergym_task.seed,
                "reward": self._last_reward,
                "cumulative_reward": self._cumulative_reward,
                "terminated": self._terminated,
                "truncated": self._truncated,
                "info": self._step_info or self._reset_info,
                "worker_pid": self._worker_pid,
            },
        )

    async def _observe(self) -> EnvironmentObservation:
        await self._start()
        return self._map_observation()

    def _validate_action_refs(self, action: ComputerAction) -> None:
        if self._raw_observation is None:
            return
        available = set(_extract_action_refs(self._raw_observation.get("axtree_object", {})))
        kind = action.kind.strip().lower().replace("-", "_")
        for candidate in (
            action.target_ref,
            action.target if kind in {"drag", "drag_and_drop"} else None,
        ):
            if candidate is not None and candidate not in available:
                raise ValueError(
                    f"unknown action ref {candidate!r}; observation refs are stale or invalid"
                )

    async def _apply(self, action: ComputerAction) -> EnvironmentObservation:
        await self._start()
        self._last_action_metrics = {}
        if self._terminated or self._truncated:
            raise RuntimeError("BrowserGym episode already finished")
        self._validate_action_refs(action)
        browsergym_action = computer_action_to_browsergym(action)
        started = perf_counter()
        result = await self._rpc("step", {"action": browsergym_action})
        latency = perf_counter() - started
        self._step_latencies_s.append(latency)
        self._raw_observation = result["observation"]
        self._last_reward = float(result["reward"])
        self._cumulative_reward += self._last_reward
        self._terminated = bool(result["terminated"])
        self._truncated = bool(result["truncated"])
        self._step_info = _json_safe(result["info"])
        self._last_action_metrics = {
            "browsergym_action": browsergym_action,
            "environment_step_latency_s": latency,
            "reward": self._last_reward,
            "cumulative_reward": self._cumulative_reward,
            "terminated": self._terminated,
            "truncated": self._truncated,
        }
        error = str(self._raw_observation.get("last_action_error") or "")
        if error:
            raise RuntimeError(error)
        return self._map_observation()

    async def _close(self) -> None:
        try:
            if self._started:
                await self._rpc("close", {})
        finally:
            await asyncio.to_thread(self._executor.shutdown, wait=True, cancel_futures=True)

    def _action_record_metadata(self) -> dict[str, Any]:
        return {"environment_session_id": self.session_id, **self._last_action_metrics}

    def metrics_snapshot(self) -> dict[str, Any]:
        return {
            "backend": "browsergym_process",
            "environment_id": self.browsergym_task.environment_id,
            "seed": self.browsergym_task.seed,
            "worker_pid": self._worker_pid,
            "reset_latency_s": self._reset_latency_s,
            "step_latencies_s": list(self._step_latencies_s),
            "step_count": len(self._step_latencies_s),
            "last_reward": self._last_reward,
            "cumulative_reward": self._cumulative_reward,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "closed": self.closed,
        }


class BrowserGymEnvironmentFactory(EnvironmentFactory):
    """Provision a fresh process, browser, context, and task for every episode."""

    def __init__(
        self,
        *,
        headless: bool = True,
        pre_observation_delay: float = 0.0,
        miniwob_url: str | None = None,
    ) -> None:
        self.config = BrowserGymRuntimeConfig(
            headless=headless,
            pre_observation_delay=pre_observation_delay,
            miniwob_url=miniwob_url or os.getenv("MINIWOB_URL"),
        )
        self.sessions: list[BrowserGymEnvironmentSession] = []

    async def create(self, task: BenchmarkTask) -> BrowserGymEnvironmentSession:
        if not isinstance(task, BrowserGymTask):
            environment_id = task.metadata.get("browsergym_environment_id")
            if not environment_id:
                raise TypeError("BrowserGymEnvironmentFactory requires a BrowserGymTask")
            task = BrowserGymTask(
                task.task_id,
                task.instruction,
                task.domain,
                task.metadata,
                str(environment_id),
                int(task.metadata.get("seed", 0)),
            )
        if task.environment_id.startswith("browsergym/miniwob.") and not self.config.miniwob_url:
            raise ValueError(
                "MiniWoB tasks require miniwob_url or the MINIWOB_URL environment variable"
            )
        session = BrowserGymEnvironmentSession(task, self.config)
        self.sessions.append(session)
        return session
