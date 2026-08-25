# Ace / Attune audit

The accessible repository is `rvempati/Attune`, whose README describes **Ace** as the
computer-use agent inside the Attune product. No separate nearby `Ace` checkout was found.
The worktree was audited read-only because it contains substantial uncommitted work.

## 1. High-level architecture

Attune is a Next.js/TypeScript application with five interleaved layers:

1. API/UI task lifecycle and live preview;
2. an asynchronous CUA loop driven by a Holo vision-language model;
3. a Playwright browser executor with screenshots, accessible page state, and
   observation-scoped element references;
4. personalization, privacy, approval, correction, and training-data capture;
5. Prisma/SQLite persistence plus Modal scripts for LoRA training and vLLM inference.

The execution path is model-driven: observe a real browser, ask Holo for grounded action
candidates, apply personalization and risk checks, execute through Playwright, and retain
the live session when approval is required.

## 2. Useful reusable components

- The discriminated browser-action schema and strict validation boundary.
- Observation-scoped references that bind model actions to elements actually observed.
- The Playwright lifecycle (`start`, `observe`, `perform`, `screenshot`, `stop`).
- Async execution, explicit step results, and fail-closed handling for malformed or
  ungrounded model actions.
- The OpenAI-compatible multimodal model wrapper and cold-start handling for Modal.
- Privacy-aware screenshot/accessibility-state handling and structured demonstrations.
- Run/outcome metrics and versioned training-example concepts.
- The authenticated, scale-to-zero Modal/vLLM deployment pattern.

These are reusable **designs**, but most implementations import Next.js aliases, Prisma,
UI preview state, or personalization types and are not clean Python library components.

## 3. Tightly coupled to personalization

- Candidate ranking combines rules, learned preferences, demonstrations, and per-user LoRAs.
- Corrections, memory retrieval, adapter registries, retention policy, and training capture
  are shaped around learning one user's choices.
- The Prisma schema, API routes, and metrics assume a local product with users, task runs,
  decisions, corrections, and UI approvals.
- Live sessions publish Attune/Ace-specific preview messages and use process-global storage.

These concerns should not enter the orchestration research core.

## 4. Technical debt and confusing abstractions

- Browser execution, CUA policy, risk gating, personalization, persistence, and UI lifecycle
  cross module boundaries, making isolated reuse difficult.
- `BrowserExecutor` is nominally an interface but fixes its mode to Playwright.
- Model transport behavior depends on endpoint-string detection and provider-specific
  exceptions.
- The action space is browser-only and mixes semantic targeting, selectors, and refs.
- Session state is process-local and global; it is unsuitable for reproducible experiments.
- Evaluation focuses on personalization/intervention, not contract-level routing outcomes.
- Telemetry is distributed across database records rather than one stable experiment record.
- The tracked repository has no tracked tests; one current untracked test covers training
  capture. Type-checking passes, while the test command could not start its local IPC socket
  in the audit sandbox.
- Comments and documentation disagree in places about whether screenshots are part of the
  primary observation path.
- The Modal worker is useful but hard-coded to one model, GPU, cache layout, and adapter
  convention.

## 5. Recommendation

Do not convert Attune into the research repository. Port the following ideas behind new,
small interfaces:

- strict action/result schemas;
- observation-scoped grounding when the real browser environment is added;
- async environment lifecycle;
- OpenAI-compatible inference served by authenticated Modal/vLLM;
- fail-closed model parsing;
- versioned trajectory records and explicit cost/latency accounting.

Do not port the UI, Prisma schema, personalization engine, per-user training pipeline,
global live-session store, or product-specific approval flow. The V0 implementation in this
repository therefore reuses no Attune source code; it selectively carries forward those
architectural lessons.
