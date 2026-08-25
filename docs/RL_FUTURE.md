# Where reinforcement learning could plug in later

RL is intentionally absent from V0. The future policy can consume the task, current state,
available contracts, and prior trajectory, then jointly choose:

```text
(next subgoal, expert, compute policy, autonomy horizon, modality)
```

The natural insertion point is a learned replacement for the planner and routing interfaces;
experts, executors, verifiers, and the trajectory schema need not change. Logged candidate
sets are important: without them, a selected action cannot be compared with alternatives.

Candidate rewards may combine verified task/subgoal success with penalties for latency,
estimated inference cost, execution errors, retries, and human intervention. Verification
quality is the main prerequisite—training against self-reported expert success would teach
the router to exploit the verifier.

Before RL, establish three things empirically:

1. specialized experts produce a real diagonal advantage;
2. an offline oracle materially beats the generalist and static routers;
3. logged outcomes have enough coverage to estimate counterfactual contracts.

Only then add offline policy learning or contextual bandits before considering full RL.
