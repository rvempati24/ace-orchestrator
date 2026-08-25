# Phase 1: task-owned execution environments

Phase 1 establishes the ownership and isolation boundary required before BrowserGym or a real
CUA is connected.

## Acceptance criteria

- [x] `ComputerAction` is an unexecuted command; `ActionRecord` is created by the environment.
- [x] Experts do not store a task environment.
- [x] Every orchestration run receives a fresh `EnvironmentSession` from a factory.
- [x] Parallel runs receive different session and benchmark-task identities.
- [x] Dependent subgoals in one task hand the same session between specialized experts.
- [x] Retry and rerouting preserve the same session and log the state strategy.
- [x] Specialization candidates receive identical benchmark task IDs in independent sessions.
- [x] Environment mutations are serialized and stale observations are rejected.
- [x] Sessions close after success, failure, cancellation, or expert exceptions.
- [x] The mock demo and specialization experiment remain runnable without network or GPU calls.
- [x] Trajectory schema 1.0 records session lineage and command execution evidence.

## Verification commands

```bash
.venv/bin/pytest
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/ace-orchestrator demo --output /tmp/ace-phase1-demo.jsonl
.venv/bin/ace-orchestrator specialization --trials 5 \
  --json /tmp/ace-phase1-specialization.json \
  --csv /tmp/ace-phase1-specialization.csv
```

The BrowserGym adapter is Phase 2. Phase 1 deliberately leaves the default environment as a
deterministic local mock.
