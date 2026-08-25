from ace_orchestrator.policies.base import Policy, PolicyConfig


class FastPolicy(Policy):
    def __init__(self, model_id: str = "mock-model") -> None:
        super().__init__(
            "fast",
            PolicyConfig("low", 256, 4_000, model_id, 3, -0.10, 1.0, 0.001),
        )


class MediumPolicy(Policy):
    def __init__(self, model_id: str = "mock-model") -> None:
        super().__init__(
            "medium",
            PolicyConfig("medium", 768, 16_000, model_id, 6, 0.0, 2.0, 0.003),
        )


class DeepPolicy(Policy):
    def __init__(self, model_id: str = "mock-model") -> None:
        super().__init__(
            "deep",
            PolicyConfig("high", 1_500, 32_000, model_id, 10, 0.08, 4.0, 0.008),
        )


def default_policies(model_id: str = "mock-model") -> dict[str, Policy]:
    policies: list[Policy] = [FastPolicy(model_id), MediumPolicy(model_id), DeepPolicy(model_id)]
    return {policy.policy_id: policy for policy in policies}
