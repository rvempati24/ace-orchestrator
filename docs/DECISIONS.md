# Architecture decisions

## ADR-001: Start from a clean Python package

**Decision:** Build `ace-orchestrator` independently and leave Attune read-only.

**Why:** Attune's sound browser and model-serving ideas are coupled to a Next.js UI,
Prisma, and personalization. A research harness needs replaceable, dependency-light
components and deterministic local tests.

## ADR-002: Use dataclasses and the standard library in the core

**Decision:** Public state is represented by typed dataclasses and string enums. JSON
serialization is explicit.

**Why:** The V0 schema is small, and avoiding a runtime framework dependency keeps mock
experiments cheap. Runtime validation is performed at boundaries. Pydantic can be added
later if schema evolution makes it worthwhile.

## ADR-003: Keep expert, compute policy, autonomy, and modality orthogonal

**Decision:** These fields meet only in `ExecutionContract`.

**Why:** Joint-routing experiments require changing one allocation dimension without
rewriting experts. Policies describe compute configuration; experts describe capability.

## ADR-004: Log one complete run per JSONL line

**Decision:** A trajectory includes the plan, candidates, selections, every attempt,
states, actions, usage, verification, recovery, and final outcome.

**Why:** Complete records are streamable and directly usable for offline routing datasets.
`schema_version` is mandatory so later migrations are explicit.

## ADR-005: Mock execution is the default

**Decision:** Local seeded experts and environments are the only default runtime.
External calls require explicitly constructing a real backend.

**Why:** Tests and early routing studies should be reproducible and unable to spend money.
Every experiment has model-call and estimated-dollar guards.

## ADR-006: Modal serves inference, not orchestration

**Decision:** The orchestrator runs locally; an optional authenticated Modal Server hosts
Holo behind an OpenAI-compatible vLLM endpoint.

**Why:** This keeps control logic inspectable and cheap while allowing GPU inference to
scale to zero. The deployment file is never imported by package initialization or tests.
The proposed Holo model and GPU are initial experiment settings, not architectural choices.

## ADR-007: Verification consumes expert results but remains replaceable

**Decision:** The V0 verifier checks reported success, completed actions, and absence of an
execution error. Real environments must supply task-specific verifiers later.

**Why:** Treating the mock signal as ground truth is adequate only for testing the harness;
keeping the interface first-class prevents it from becoming the research reward by accident.

## ADR-008: Recovery is bounded and explicit

**Decision:** Retry once with a stronger policy, reroute once, then log return to the global
planner. V0 stops that subgoal after returning control rather than silently inventing a plan.

**Why:** The requested behavior is observable without embedding a brittle recursive planner.
