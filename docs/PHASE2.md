# Phase 2: real BrowserGym execution

Phase 2 connects the Phase 1 ownership boundary to BrowserGym 0.14.3 and the pinned MiniWoB++
benchmark. Each task episode owns a dedicated worker process, Gymnasium environment, Chromium
browser, and browser context. Experts remain stateless; retry and rerouting transfer the same
live session, while parallel tasks cannot share browser state.

## Acceptance criteria

- [x] BrowserGym is an optional, exactly pinned dependency; the core remains dependency-free.
- [x] `BrowserGymTask` preserves environment ID, seed, benchmark metadata, and task identity.
- [x] A factory creates a fresh process/browser/context for every benchmark episode.
- [x] Two tasks execute concurrently in different worker processes and session IDs.
- [x] BrowserGym observations map screenshots, accessibility state, DOM state, URLs, and refs
  into `EnvironmentObservation`.
- [x] Click, fill/type, select, press, scroll, navigation, hover, focus, clear, drag, and history
  commands map to BrowserGym's high-level action API.
- [x] Unknown refs, stale observations, unsupported commands, and non-HTTP navigation fail at
  the environment boundary.
- [x] BrowserGym reward and termination—not the expert's assertion—determine verification.
- [x] Retry and expert rerouting preserve the same live browser episode.
- [x] Browser processes close after success, expert exceptions, action failures, and cancellation.
- [x] Five deterministic MiniWoB tasks pass: `click-test`, `click-button`, `enter-text`,
  `choose-list`, and `copy-paste`.
- [x] Trajectory schema 1.1 records environment reset/step latency, reward, termination, seed,
  environment ID, and worker identity.

## Reproducible setup

BrowserGym 0.14.3 pins Playwright 1.44 and greenlet 3.0.3. That combination does not build on
Python 3.13, so the tested browser environment uses Python 3.12. The dependency-free core still
supports Python 3.11+.

```bash
python3.12 -m venv .venv-browsergym
source .venv-browsergym/bin/activate
python -m pip install -e ".[dev,browsergym]"
playwright install chromium

git clone https://github.com/Farama-Foundation/miniwob-plusplus.git .miniwob-plusplus
git -C .miniwob-plusplus switch --detach 7fd85d71a4b60325c6585396ec4f48377d049838
export MINIWOB_URL="file://$PWD/.miniwob-plusplus/miniwob/html/miniwob/"
```

The MiniWoB checkout and virtual environment are ignored by Git. The frozen MiniWoB commit is
the one recommended by BrowserGym's package metadata.

## Verification commands

```bash
.venv-browsergym/bin/pytest -m "not browsergym"
MINIWOB_URL="file://$PWD/.miniwob-plusplus/miniwob/html/miniwob/" \
  .venv-browsergym/bin/pytest -m browsergym -vv
.venv-browsergym/bin/ruff check .
.venv-browsergym/bin/ruff format --check .
```

The first command is safe without BrowserGym, Chromium, MiniWoB assets, a model endpoint, or a
GPU. The real suite launches headless Chromium and currently contains 10 acceptance cases.

## Runtime boundary

BrowserGym is synchronous and keeps a process-global Playwright driver. Running each episode in
its own single-worker process preserves Playwright thread affinity and allows actual parallel
browser execution. IPC returns observations and action outcomes to the async orchestrator. It
adds some serialization overhead, which is now visible in reset and step timing fields and can
be measured before considering a lower-latency custom harness.

The included `ScriptedMiniWoBExpert` is test instrumentation, not a production agent. The next
phase can connect prompted experts to a CUA backend (including optional Modal inference) while
reusing the same environment, action, verification, isolation, and trajectory contracts.
