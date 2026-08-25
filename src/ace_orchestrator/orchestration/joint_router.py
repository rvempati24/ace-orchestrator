from dataclasses import dataclass


@dataclass(frozen=True)
class RoutingDimensions:
    """Experiment switchboard for future joint-routing/Pareto studies."""

    route_expert: bool = True
    route_compute: bool = False
    route_autonomy: bool = False
    route_modality: bool = False
