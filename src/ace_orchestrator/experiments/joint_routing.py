"""Stub types for expert/compute/autonomy Pareto experiments; no learned router yet."""

from __future__ import annotations

from dataclasses import dataclass

from ace_orchestrator.orchestration.joint_router import RoutingDimensions


@dataclass(frozen=True)
class JointRoutingArm:
    name: str
    dimensions: RoutingDimensions


DEFAULT_ARMS = (
    JointRoutingArm("expert_only", RoutingDimensions(route_expert=True)),
    JointRoutingArm("compute_only", RoutingDimensions(route_expert=False, route_compute=True)),
    JointRoutingArm("expert_compute", RoutingDimensions(route_expert=True, route_compute=True)),
    JointRoutingArm(
        "expert_compute_autonomy",
        RoutingDimensions(route_expert=True, route_compute=True, route_autonomy=True),
    ),
)
