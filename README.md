# Ace Orchestrator

Ace Orchestrator is an early research harness for hierarchical orchestration of
heterogeneous computer-use agents (CUAs). It asks:

> Can hierarchical orchestration over heterogeneous computer-use experts improve the
> success–latency–cost frontier by dynamically allocating capability, compute, and autonomy?

This repository does not claim novelty, implement RL, or provide a production CUA. V0 is a
small, typed, fully local experiment framework with structured trajectories and optional
Modal-hosted inference.

## Architecture

```text
user task
   │
   ▼
planner ──► dependency-ordered semantic subgoals
   │
   ▼
expert router ──► web / spreadsheet / CRM / generalist
   │
   ▼
policy router ──► fast / medium / deep compute
   │
   ▼
ExecutionContract(expert, policy, autonomy, modality)
   │
   ▼
executor + environment ──► actions and state transition
   │
   ▼
verifier ── success ──► next subgoal
   │
   └── failure ──► stronger policy ──► another expert ──► planner

Every edge and outcome ──► versioned JSONL trajectory
```

The core has no runtime dependencies beyond Python 3.11+. Mock experts are seeded and make
no network calls.

## Terminology

- **Orchestrator:** owns global state and advances the task through verified subgoals.
- **Expert:** a capability implementation—initially the same CUA can be specialized by prompt.
- **Policy:** a compute allocation such as model ID, reasoning budget, token limit, and action
  allowance. It is not an expert.
- **Execution contract:** the explicit assignment of one subgoal to an expert, policy,
  autonomy horizon, and modality.
- **Autonomy horizon:** how many actions an expert may take, or eventually a semantic
  `until_subgoal_complete` condition. Semantic horizons still have a safety cap.
- **Verifier:** independently scores the state transition and returns structured checks.

## Local setup and mock demo

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest
ace-orchestrator demo
```

The demo writes one complete record to `trajectories/demo.jsonl` and prints it. It runs the
same call shape intended for real agents:

```python
result = await orchestrator.run(
    "Research these companies and then enter the results into the spreadsheet."
)
```

## Add an expert

Subclass `experts.base.Expert`, implement async `execute`, and register the instance:

```python
registry.register(MyExpert("web", "Web specialist", ("web",)))
```

The orchestrator has no expert IDs hard-coded. For prompt specialization, construct
`PromptedCUAExpert` instances with the same backend and different prompt files.

## Add a policy

Create a `Policy` with a `PolicyConfig`, add it to the policy mapping, then teach a
`PolicyRouter` when to select it. Policy configuration holds compute—not domain capability.

## Specialization matrix

```bash
ace-orchestrator specialization --trials 5
```

This evaluates every mock expert in every domain, prints the matrix and diagonal advantage,
and exports `results/specialization.json` and `.csv`. Replace mock experts with prompted real
CUAs only after setting a conservative `ExperimentBudget`.

`experiments.routing_baselines` compares monolithic generalist, static, LLM, and offline
oracle orchestrators built by the caller. `experiments.joint_routing` defines future arms for
expert-only, compute-only, joint expert/compute, and expert/compute/autonomy studies.

## Optional Modal inference

`deploy/modal_inference.py` exposes `Hcompany/Holo-3.1-35B-A3B` through an authenticated,
scale-to-zero vLLM server on an H200. This follows Modal's current
[vLLM example](https://modal.com/docs/examples/vllm_inference),
[Server guidance](https://modal.com/docs/guide/functions), and
[Image API](https://modal.com/docs/guide/images). Holo is an Apache-2.0 multimodal computer-use
model; see its [model card](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B).

```bash
python -m pip install -e ".[modal]"
modal setup
modal deploy deploy/modal_inference.py
modal run deploy/modal_inference.py
```

The last command prints the endpoint without sending an inference request. Configure a Modal
proxy token, then explicitly construct `OpenAICompatibleCUA(base_url, model_id, api_key)`.
No test, import, demo, or experiment deploys Modal or allocates a GPU.

Modal solves only model inference. A real run still needs a `ComputerEnvironment` that
captures screenshots/semantic state, applies grounded browser or desktop actions, and
provides task-specific verification. Those are deliberately not faked as “real integration.”

## Trajectories and budgets

Each JSONL line includes the planner decision, candidate sets, selected contract, before/after
state, actions, token/cost/latency usage, verification, retries, reroutes, and final outcome.
See [docs/TRAJECTORY_SCHEMA.md](docs/TRAJECTORY_SCHEMA.md).

Mock mode is the default. External calls require explicit construction. `ExperimentBudget`
limits model calls and estimated dollars; experiment runners stop when the guard is reached.

## Current limitations

- The included verifier validates mock/expert result signals, not real-world task outcomes.
- No concrete Playwright or native-desktop environment is included yet.
- LLM planner/router classes need a caller-supplied structured model adapter.
- Recovery logs return to the global planner after retry/reroute exhaustion; recursive
  replanning is not implemented in V0.
- Simulated probabilities illustrate the harness and are not evidence about CUA performance.
- Estimated cost requires provider pricing metadata; the generic vLLM client reports tokens
  but leaves dollar cost at zero.

## Roadmap

1. Build a grounded Playwright environment and deterministic benchmark verifier.
2. Run the real prompted-expert specialization matrix with fixed model/compute.
3. Measure generalist, static, LLM, and offline-oracle routing on identical trajectories.
4. Test joint expert/compute/autonomy allocations and their Pareto frontier.
5. Improve verification and collect broad offline coverage before considering learned routing.

Architectural rationale is recorded in [docs/DECISIONS.md](docs/DECISIONS.md); the future RL
insertion point is described in [docs/RL_FUTURE.md](docs/RL_FUTURE.md). The source audit that
motivated the clean implementation is in [ACE_AUDIT.md](ACE_AUDIT.md).
