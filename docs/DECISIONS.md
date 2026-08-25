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

## ADR-009: Environments belong to task episodes, not experts

**Decision:** `EnvironmentFactory` creates one isolated `EnvironmentSession` per orchestration
run. The executor passes that session into stateless experts. Retries and reroutes within the
run preserve the same session; independent and specialization-matrix trials receive fresh
sessions.

**Why:** Mutable browser/application state must survive expert handoffs without leaking across
parallel tasks. This also permits one expert configuration to serve many sessions concurrently
and gives the orchestrator centralized lifecycle and cleanup control.

## ADR-010: Commands and action records are different types

**Decision:** A model or expert emits `ComputerAction`. Only an `EnvironmentSession` may create
an `ActionRecord`, after serializing the mutation and binding it to before/after observation
IDs and timestamps.

**Why:** Proposed intent is not evidence of execution. Separating the types prevents experts
from self-assigning success and makes stale-observation and trajectory checks enforceable.

## ADR-011: Isolate BrowserGym episodes in worker processes

**Decision:** Every BrowserGym session owns a single-worker process containing its Gymnasium
environment, Playwright driver, Chromium browser, and context.

**Why:** BrowserGym's API is synchronous and its Playwright instance is process-global. A process
per task preserves driver affinity, prevents browser-state leakage, and permits real parallel
episodes. The async orchestrator communicates over process IPC and logs the resulting overhead.

## ADR-012: Benchmark outcomes verify real browser tasks

**Decision:** `BrowserGymVerifier` requires positive task reward, termination without truncation,
and successful action execution. Expert self-reports are necessary but not sufficient.

**Why:** A CUA should not grade its own work. MiniWoB supplies deterministic reward and terminal
signals that provide a clean acceptance boundary for the adapter and future model experiments.
