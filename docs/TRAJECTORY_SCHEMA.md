# Trajectory schema 1.0.0

Each JSONL line is one complete orchestration run.

Required top-level fields:

- `schema_version`, `task_id`, `user_goal`, `created_at`, `finished_at`;
- `environment` with the task-owned session ID, factory, and benchmark task;
- `planner_decision` with structured subgoals;
- `available_experts` with descriptions and capabilities;
- `subgoals`, each with domain, success, retry/reroute counts, and attempts;
- `escalations` in chronological order;
- `usage` with wall/model latency, tokens, calls, and estimated dollars;
- `final_verification` and `final_task_success`.

Each attempt contains:

- the complete `execution_contract`;
- the `environment_session_id` used by the attempt;
- expert and policy candidates, selections, and reasoning;
- explicit autonomy horizon and modality;
- start/end state snapshots;
- action records, each containing the proposed `ComputerAction`, timestamps, before/after
  observation IDs, environment-assigned success, and any execution error;
- wall/model latency, tokens, cost, and call count;
- structured verification checks and any error.

Compatibility rule: additive fields may appear within a minor schema version. Renaming,
removing, or changing the meaning of a field requires a new major schema version and a
migration script.

Version 1.0.0 separates unexecuted commands from environment-produced records and introduces
task-owned environment-session identity. It is intentionally incompatible with V0's flat
action record.
