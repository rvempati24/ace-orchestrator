from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import Mapping
from time import perf_counter
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ace_orchestrator.core.actions import ComputerAction
from ace_orchestrator.core.models import Subgoal, Usage
from ace_orchestrator.execution.environment import EnvironmentObservation
from ace_orchestrator.inference.base import CUABackend, CUAProposal
from ace_orchestrator.inference.observation import compact_observation
from ace_orchestrator.policies.base import Policy

ACTION_PROTOCOL = """Return exactly one JSON object with this shape:
{"actions":[{"kind":"click","target_ref":"12"}],"done":false,"summary":"..."}
Use only observed refs. Supported action kinds are click, double_click, hover, focus, clear,
fill, type, select_option, select, press, drag_and_drop, scroll, navigate, back, forward, noop,
and finish. Put text, selected options, or key combinations in value. For drag_and_drop, put
the destination ref in target. For scroll, use numeric delta_x/delta_y fields. Set done=true
only when the subgoal is complete. Return at most three environment actions per response."""


class OpenAICompatibleCUA(CUABackend):
    """Optional real-inference client; Modal's vLLM server uses this protocol."""

    allowed_actions = {
        "back",
        "clear",
        "click",
        "double_click",
        "drag_and_drop",
        "fill",
        "finish",
        "focus",
        "forward",
        "hover",
        "navigate",
        "noop",
        "press",
        "scroll",
        "select",
        "select_option",
        "type",
    }
    grounded_actions = {
        "clear",
        "click",
        "double_click",
        "drag_and_drop",
        "fill",
        "focus",
        "hover",
        "press",
        "select",
        "select_option",
        "type",
    }
    max_actions_per_proposal = 3

    def __init__(self, base_url: str, model_id: str, api_key: str | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_id = model_id
        self.api_key = api_key

    async def propose(
        self,
        *,
        system_prompt: str,
        subgoal: Subgoal,
        observation: EnvironmentObservation,
        policy: Policy,
    ) -> CUAProposal:
        started = perf_counter()
        payload = await asyncio.to_thread(
            self._request,
            system_prompt,
            subgoal,
            observation,
            policy,
        )
        elapsed = perf_counter() - started
        choice = (payload.get("choices") or [{}])[0]
        content = (choice.get("message") or {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError("CUA endpoint returned no text content")
        decoded = self._extract_json(content)
        raw_actions = decoded.get("actions", [])
        if not isinstance(raw_actions, list):
            raise ValueError("CUA actions must be a JSON list")
        if len(raw_actions) > self.max_actions_per_proposal:
            raise ValueError(
                f"CUA proposed {len(raw_actions)} actions; maximum is "
                f"{self.max_actions_per_proposal}"
            )
        actions: list[ComputerAction] = []
        finish_seen = False
        for row in raw_actions:
            if not isinstance(row, Mapping):
                raise ValueError("each CUA action must be a JSON object")
            kind = str(row.get("kind", "")).strip().lower().replace("-", "_")
            if kind not in self.allowed_actions:
                raise ValueError(f"unsupported CUA action: {kind}")
            if kind == "finish":
                finish_seen = True
                continue
            target_ref = str(row["target_ref"]) if row.get("target_ref") is not None else None
            target = str(row["target"]) if row.get("target") is not None else None
            if kind in self.grounded_actions and not target_ref:
                raise ValueError(f"CUA action {kind} requires target_ref")
            if (
                target_ref
                and observation.available_action_refs
                and target_ref not in observation.available_action_refs
            ):
                raise ValueError(f"CUA proposed unknown target_ref: {target_ref}")
            if kind == "drag_and_drop":
                if not target:
                    raise ValueError(
                        "CUA action drag_and_drop requires a destination ref in target"
                    )
                if (
                    observation.available_action_refs
                    and target not in observation.available_action_refs
                ):
                    raise ValueError(f"CUA proposed unknown drag destination ref: {target}")
            actions.append(
                ComputerAction(
                    kind=kind,
                    target_ref=target_ref,
                    target=target,
                    value=str(row["value"]) if row.get("value") is not None else None,
                    metadata={
                        key: value
                        for key, value in row.items()
                        if key not in {"kind", "target_ref", "target", "value"}
                    },
                )
            )
        done = decoded.get("done", False)
        if not isinstance(done, bool):
            raise ValueError("CUA done field must be a boolean")
        token_usage = payload.get("usage") or {}
        usage = Usage(
            wall_clock_latency_s=elapsed,
            model_latency_s=elapsed,
            input_tokens=int(token_usage.get("prompt_tokens", 0)),
            output_tokens=int(token_usage.get("completion_tokens", 0)),
            model_calls=1,
        )
        return CUAProposal(
            tuple(actions),
            done or finish_seen,
            str(decoded.get("summary", "")),
            usage,
        )

    def _request(
        self,
        system_prompt: str,
        subgoal: Subgoal,
        observation: EnvironmentObservation,
        policy: Policy,
    ) -> dict:
        body = self._build_request_body(system_prompt, subgoal, observation, policy)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310
                return json.load(response)
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"CUA endpoint request failed: {error}") from error

    def _build_request_body(
        self,
        system_prompt: str,
        subgoal: Subgoal,
        observation: EnvironmentObservation,
        policy: Policy,
    ) -> dict:
        user_content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "subgoal": subgoal.description,
                        "observation": compact_observation(observation),
                        "contract": {
                            "reasoning_budget": policy.config.reasoning_budget,
                            "max_actions_this_response": self.max_actions_per_proposal,
                        },
                    },
                    ensure_ascii=False,
                ),
            }
        ]
        if observation.screenshot_base64:
            if len(observation.screenshot_base64) > 20_000_000:
                raise ValueError("screenshot exceeds the 20 MB encoded-size guard")
            base64.b64decode(observation.screenshot_base64, validate=True)
            mime_type = observation.screenshot_mime_type or "image/png"
            if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
                raise ValueError(f"unsupported screenshot MIME type: {mime_type}")
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{observation.screenshot_base64}"
                    },
                }
            )
        return {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": f"{system_prompt.strip()}\n\n{ACTION_PROTOCOL}"},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0,
            "max_tokens": policy.config.max_output_tokens,
        }

    @staticmethod
    def _extract_json(raw: str) -> dict:
        candidate = raw.strip()
        if candidate.startswith("```"):
            candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            start, end = candidate.find("{"), candidate.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("CUA response was not valid JSON") from None
            value = json.loads(candidate[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("CUA response must be a JSON object")
        return value
