# Phase 3: prompted CUA experiments

Phase 3 connects model inference to the real BrowserGym loop while preserving the task-owned
environment boundary from Phases 1 and 2. Browser processes remain local isolated workers;
Modal is only an optional OpenAI-compatible multimodal inference service.

## Started and verified

- [x] Phase 2 is published on `main` at commit `04dc82b`.
- [x] Raw BrowserGym DOM/AX payloads are compacted into grounded element records before inference.
- [x] PNG screenshots retain their correct MIME type in multimodal requests.
- [x] The model receives a strict JSON action protocol and the currently valid grounded refs.
- [x] Unsupported actions, unknown refs, invalid completion flags, and batches over three actions
  fail before environment mutation.
- [x] Autonomy horizons count executed environment actions, not model turns.
- [x] Endpoint/parser failures become bounded expert failures so the orchestrator can recover.
- [x] A prompted expert completes a real two-turn `enter-text` MiniWoB episode in acceptance tests.
- [x] A runnable endpoint-backed smoke example constructs the full orchestration path.
- [x] The Modal definition is configured to serve revision-pinned Holo 3.1 through vLLM 0.21
  on one H200.
- [x] The deployment definition loads successfully with Modal SDK 1.5.4 and an authenticated
  local profile; no cloud resources were created.

## Remaining completion criteria

- [ ] Deploy the inference app with explicit user approval for the cost-bearing live step.
- [ ] Create a Modal proxy token scoped to the deployment environment.
- [ ] Pass endpoint health, model-list, text JSON, and screenshot JSON smoke checks.
- [ ] Run the five Phase 2 MiniWoB tasks with Holo using fixed model, policy, prompt, and seeds.
- [ ] Add at least one browser-specialist prompt and compare it with the generalist under identical
  compute and environment conditions.
- [ ] Export success, model latency, browser latency, tokens, estimated cost, invalid-action rate,
  and cold-start measurements.
- [ ] Confirm all environment cleanup and budget guards under endpoint errors and timeouts.
- [ ] Commit and publish only after the live acceptance report is reproducible.

## Local protocol verification

```bash
.venv-browsergym/bin/pytest -m "not browsergym"
MINIWOB_URL="file://$PWD/.miniwob-plusplus/miniwob/html/miniwob/" \
  .venv-browsergym/bin/pytest -m browsergym -vv
.venv-browsergym/bin/ruff check .
.venv-browsergym/bin/ruff format --check .
```

The browser suite includes a model-shaped deterministic backend. It exercises the same
`PromptedCUAExpert` and environment interfaces as a remote endpoint without spending GPU money.

## Live smoke command

Install Modal only when a live deployment is intended:

```bash
python -m pip install -e ".[modal,browsergym]"
modal setup
modal deploy deploy/modal_inference.py
```

Create a Modal proxy token and combine its ID and secret as `wk-....ws-...`. Then run:

```bash
export ACE_CUA_BASE_URL="https://YOUR-SERVER.modal.direct/v1"
export ACE_CUA_API_KEY="wk-TOKEN_ID.ws-TOKEN_SECRET"
export ACE_CUA_MODEL_ID="Hcompany/Holo-3.1-35B-A3B"
export MINIWOB_URL="file://$PWD/.miniwob-plusplus/miniwob/html/miniwob/"

.venv-browsergym/bin/python examples/browsergym_prompted.py \
  --task click-test --seed 0 --output trajectories/phase3-smoke.jsonl
```

Modal Servers require proxy authentication by default. The combined proxy token works as a
Bearer API key for OpenAI-compatible clients. Do not use a Modal API token (`ak-`/`as-`) here.

No Modal deployment or GPU allocation is triggered by imports, tests, or the example unless the
example is explicitly run with a live endpoint.
