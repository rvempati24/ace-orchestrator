from __future__ import annotations

import asyncio
import base64
import json
from time import perf_counter
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ace_orchestrator.core.models import ActionRecord, Subgoal, Usage
from ace_orchestrator.execution.environment import EnvironmentObservation
from ace_orchestrator.inference.base import CUABackend, CUAProposal
from ace_orchestrator.policies.base import Policy


class OpenAICompatibleCUA(CUABackend):
    """Optional real-inference client; Modal's vLLM server uses this protocol."""

    allowed_actions = {"click", "type", "select", "navigate", "scroll", "read", "finish"}

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
        actions: list[ActionRecord] = []
        for row in raw_actions:
            kind = str(row.get("kind", ""))
            if kind not in self.allowed_actions:
                raise ValueError(f"unsupported CUA action: {kind}")
            if kind == "finish":
                continue
            actions.append(
                ActionRecord(
                    kind,
                    str(row.get("target", "")),
                    metadata={
                        key: value for key, value in row.items() if key not in {"kind", "target"}
                    },
                )
            )
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
            bool(decoded.get("done", False)),
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
        user_content: list[dict] = [
            {
                "type": "text",
                "text": json.dumps(
                    {
                        "subgoal": subgoal.description,
                        "observation": observation.values,
                        "contract": {
                            "reasoning_budget": policy.config.reasoning_budget,
                            "return": "JSON with actions, done, and summary",
                        },
                    }
                ),
            }
        ]
        if observation.screenshot_base64:
            if len(observation.screenshot_base64) > 20_000_000:
                raise ValueError("screenshot exceeds the 20 MB encoded-size guard")
            base64.b64decode(observation.screenshot_base64, validate=True)
            user_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{observation.screenshot_base64}"},
                }
            )
        body = json.dumps(
            {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0,
                "max_tokens": policy.config.max_output_tokens,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(
            f"{self.base_url}/chat/completions", data=body, headers=headers, method="POST"
        )
        try:
            with urlopen(request, timeout=120) as response:  # noqa: S310
                return json.load(response)
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"CUA endpoint request failed: {error}") from error

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
