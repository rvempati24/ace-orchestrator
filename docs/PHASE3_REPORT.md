# Phase 3 live acceptance report

Phase 3 passes its live acceptance target: the browser-specialist expert completed all five fixed
MiniWoB tasks through real isolated Chromium workers and revision-pinned Holo inference on Modal.
The generalist completed three of five under the same model, GPU, policy, seed, environment, and
action budget. The machine-readable aggregate is in `results/phase3_live.json`.

## Fixed experiment

- Date: 2026-08-25
- Model: `Hcompany/Holo-3.1-35B-A3B`
- Revision: `2bdb92851a8cd9d72cdd891fdf38cfcc7fefae2c`
- Server: vLLM 0.21.0, one Modal H200, bfloat16, 16,384-token context
- Benchmark: BrowserGym 0.14.3 with pinned MiniWoB++, seed 1
- Tasks: `click-test`, `click-button`, `enter-text`, `choose-list`, `copy-paste`
- Policy: `fast`, 256 output tokens, three-action autonomy horizon
- Variable under comparison: expert prompt only

## Results

| Expert | Success | Model calls | Model latency | Browser step latency | Tokens in/out | Invalid actions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Generalist | 3/5 | 10 | 53.590 s | 3.008 s | 8,517 / 392 | 1/10 (10%) |
| Browser specialist | 5/5 | 7 | 28.658 s | 2.034 s | 7,178 / 279 | 0/8 (0%) |

The specialist reduced model calls by 30%, model latency by 46.5%, input tokens by 15.7%, and
environment actions by 20% on this small fixed suite. These five tasks establish an end-to-end
acceptance result, not a statistically broad benchmark claim.

The generalist repeatedly clicked the `choose-list` combobox instead of selecting the named
option. On `copy-paste`, it attempted clipboard-style `press` actions and produced one failed
browser action. The specialist uses accessibility roles and observed values: it selected the
combobox value directly and filled the destination with the visible source value.

## Bugs found and corrected

- Holo used a `text` field for a fill action. The client now normalizes common value aliases and
  rejects truly missing action values before environment mutation.
- Experts did not receive their own recent actions, allowing repeated unchanged clicks. The last
  ten action outcomes are now included in each subsequent compact observation.
- The BrowserGym verifier claimed benchmark reward was authoritative but required the model's
  `done` bit. It now uses successful actions, positive reward, termination, and non-truncation;
  prompted experts also stop immediately when the environment terminates.
- vLLM selected FlashInfer JIT paths that did not finish within a useful startup interval. The
  deployed server now pins Triton for GDN prefill and unquantized MoE and skips multimodal memory
  profiling with an explicit image budget.

## Cold start and reliability

Three cached cold starts reached readiness in 286–370 seconds (median 353 seconds). Normal cached
checkpoint loading was 43–60 seconds; vLLM profile and multimodal warmup dominated the rest.
The private endpoint returned HTTP 503 while starting. Transient 503s were also observed around
scale transitions, including a brief overlapping-container event. Warm inference was stable
enough for the specialist to pass 5/5 consecutively.

This makes scale-to-zero appropriate for low-frequency experiments, but not yet for a latency SLO.
Before production use, add a bounded readiness gate/retry policy and decide whether a one-container
warm pool is worth the idle cost.

## Cost and credentials

The Modal rate query on the run date reported H200 at $4.54/hour. With eight CPUs and 64 GiB RAM,
the configured container rate was approximately $5.4304/hour before persistent volume storage.
Final matrix active wall time corresponds to about $0.15 of compute. Modal's app-scoped billing
report measured $10.13 for the complete work session, including exploratory deploys, failed JIT
starts, repeated cold starts, overlap, and idle windows; $8.39 was H200 usage.

The generic vLLM client reports token cost as zero because inference is self-hosted; Modal resource
time is the relevant cost. Retaining roughly 65.4 GiB of model weights at the reported
$0.09/GiB-month volume rate is approximately $5.89/month, excluding smaller cache overhead.

Workspace RBAC was disabled, so an environment-scoped proxy token was unavailable. A temporary
workspace-wide token was created only after explicit approval, stored in a mode-600 temporary file,
used for the live suite, revoked, and deleted. No credential or raw token is present in Git.

References: [Holo model card](https://huggingface.co/Hcompany/Holo-3.1-35B-A3B),
[Modal proxy authentication](https://modal.com/docs/guide/webhook-proxy-auth), and
[Modal vLLM example](https://modal.com/docs/examples/vllm_inference).
