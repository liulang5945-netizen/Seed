# Seed — runtime for the Taiji Native Cognitive Architecture

**Taiji** is the native cognitive architecture and model. **Seed** is the project, product and runtime that trains, evaluates, deploys and hosts Taiji. Taiji is being redesigned around learned perception, world state, workspace, memory, goals, reasoning, planning and generation while reusing mature algorithms where they fit the project—not rebuilding intelligence from primitive one-hot mechanisms.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

> Current status: the Taiji Native Architecture v1 target is specified; the executable code is still the lower-level Taiji Substrate Kernel v8 (TSK-v8), not a completed cognitive architecture or AGI claim.

## Naming

| Name | Meaning |
|---|---|
| **Seed** | project, product, distribution and runtime; package `seed` |
| **Taiji** | complete native cognitive architecture and model; target package `taiji/` |
| **TSK-v8** | current executable byte/fabric/memory/motor research kernel and compatibility line |
| **Legacy NeuroPlex** | the frozen Transformer baseline in `neuroplex/`; the replaced bottom layer is `neuroplex/layers.py::TransformerBlock` |
| `taiji.*` in `scripts/archive/` | a historical import alias for `neuroplex`, not the current architecture |

## Architecture direction

Taiji is capability-first, not anti-Transformer by ideology. It may reuse learned embeddings, attention-like routing, state-space or graph operators, optimizers, reinforcement learning, retrieval infrastructure and CUDA kernels. The invariant is that Taiji owns the cognitive state and decision path; no Transformer, Legacy runtime or teacher model may remain the runtime cognitive subject.

The target path is:

```text
multimodal observations
  → learned perception and variable-duration abstractions
  → predictive world/self state + workspace
  ↔ working / episodic / semantic / procedural memory
  → goals / reasoning / imagination / planning
  → language / tool / body action
  → real outcome and continued learning
```

See the [core project requirements](plans/active/TAIJI_CORE_REQUIREMENTS.md), [Taiji Native Architecture v1](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md), and the [current roadmap](plans/active/SEED_DEVELOPMENT_ROADMAP_2026_08.md).

## Current TSK-v8 kernel baseline

The current executable kernel does not wrap a Transformer. Its narrower path is:

```text
raw byte receptors
  → hierarchical reciprocal prediction errors
  → persistent recurrent region states
  ↔ distributed episodic field and cortical readback
  → balanced sparse cortical receptor bank
  → one byte motor organ
  → emitted action returns as the next sensation
```

| Transformer responsibility | Current TSK-v8 experiment |
|---|---|
| tokenizer + learned embedding | 256 raw-byte receptors + boundary receptor |
| positional encoding | causal ticks and persistent state |
| self-attention | sparse reciprocal prediction and recurrent transitions |
| residual/FFN state | membrane integration, inhibition, adaptive thresholds, traces |
| KV cache / external retrieval | bounded dynamic state plus distributed associative engrams; no event K/V slots |
| global backpropagation | existing-edge local prediction/state/motor/memory deltas |
| LM head | all-state sparse receptor bank + one motor population |
| autoregressive decode | motor byte fed back through the same sensor |

The implementation imports neither `transformers` nor the legacy `neuroplex` runtime. PyTorch is used only as a tensor execution engine.

## Algorithm

For region `r`, the previous local trace predicts both the current lower-level activity and the region's own next activity:

```math
\hat y_t^{r-1}=D^r q_{t-1}^r, \qquad e_t^{r-1}=y_t^{r-1}-\hat y_t^{r-1}
```

```math
\hat a_t^r=T^r q_{t-1}^r
```

The region integrates bottom-up error, recurrent prediction, and delayed top-down context:

```math
u_t^r=Bound(\lambda_u u_{t-1}^r+\alpha_g(D^r)^Te_t^{r-1}+\alpha_T\hat a_t^r+\alpha_c c_t^r)
```

Activity is formed through an adaptive threshold and a local inhibitory pool. Learning is online and local:

```math
\Delta D^r=\eta_D e_t^{r-1}(q_{t-1}^r)^T
```

```math
\Delta T^r=\eta_T(a_t^r-T^rq_{t-1}^r)(q_{t-1}^r)^T
```

```math
\Delta M=\eta_M(onehot(b_t)-p_{t-1})c_{t-1}^T
```

The motor does not discard a random cortical subset. It concatenates every region's fast activity and slow trace into `s_t`, then a fixed balanced single-fan-out receptor map `H` folds every coordinate into `K` shared evidence channels:

```math
\tilde c_{t,k}=|G_k|^{-1/2}\sum_{j\in G_k}\sigma_j s_{t,j},
\qquad
c_t=\gamma_c\frac{\tilde c_t}{\lVert\tilde c_t\rVert_2+\epsilon}
```

Every cortical coordinate reaches exactly one receptor, and every action competes on the same `K` channels.

An active transition is stored only after the full causal tuple is available. A cortical cue `s`, executed action `a`, reward `r`, resulting sensation `o`, causal tick, episode signature and provenance excite one overlapping engram population `h`. Existing recurrent edges learn cue-to-event completion:

```math
h^{event}=\phi(Qs+\gamma_e(Aa+Oo+r\rho+Tt+Ee+Pp)),
\qquad
\Delta W^{mem}=\eta_m g(h^{event}-W^{mem}h^{cue})(h^{cue})^T
```

`g` is a novelty/reward write gate. Recurrent resonance gates all recalled action, outcome, value, time, episode and provenance evidence; recalled cortical state is injected into the next fabric tick. Events share the same fixed population and edge topology—writing an event does not append a row, key or value.

Every update is restricted to stored fixed-fan-in edges. There is no dense structural mask, attention matrix, context window, optimizer, `backward()`, teacher model, or distillation path.

The kernel tensor shapes, update order, state contract, complexity, and code mapping are preserved in the [archived TSK-v8 implementation specification](plans/archive/implementation/TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md). These equations are candidate substrate mechanisms, not the complete Taiji architecture.

## Quick start

```bash
python -m pip install -e ".[dev]"
python scripts/training/verify_taiji_native_v7.py
python scripts/training/verify_taiji_n7_context.py
python scripts/training/verify_taiji_n8_delayed_trace.py
python scripts/training/verify_taiji_n9_long_free_run.py
python scripts/training/verify_taiji_n10_sparse_migration.py
python scripts/training/verify_taiji_n11_active_environment.py
python scripts/training/verify_taiji_m5_episodic_field.py
python -m pytest tests/taiji_native -q
```

Seed runtime compatibility API:

```python
from seed import Seed

model = Seed()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)

print(model.score_bytes(b"abcdabcdabcdabcd"))
print(model.generate(b"a", length=8))

checkpoint = model.checkpoint()
restored = Seed.from_checkpoint(checkpoint)
```

Researchers may still import the current `Taiji` class directly for TSK-v8 kernel falsification experiments. During P1, Seed remains the product-facing runtime while cognitive ownership moves into the Taiji v1 contract.

## Public Beta

Seed ships as a self-contained Windows desktop build (dual-entry `Seed.exe` + `SeedBackend.exe`,
about 1.1GB): double-click launches the backend, activates the Seed native runtime, and serves the
web UI on `http://127.0.0.1:8000` within a few seconds. Development mode equivalent:

```bash
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000   # backend + web UI (serves frontend/dist)
python desktop/main.py                                         # desktop shell (window + backend + WebSocket 8765)
cd frontend && npm run dev                                     # frontend dev server
```

Environment knobs: `SEED_PORT` (default 8000), `SEED_HOST` (default 127.0.0.1),
`SEED_RUNTIME=1` (activate the Seed native runtime on startup).

Historical beta verification evidence lives in `reports/` (API stress 1000/1000, checkpoint
crash-recovery 10/10, latency/throughput baselines and frontend review) and
`reports/seed_public_beta_release_20260823.md`. It documents the TSK-v8 product shell rather than
Taiji v1 capability. Current checkpoints are early-stage: garbled replies are expected kernel
behavior, not a completed language architecture.

## Reproducible TSK-v8 kernel results

The committed verification uses two regions `[64, 48]`, seed `7`, and raw bytes:

| Metric | Result |
|---|---:|
| active learned parameters | 83,841 |
| fixed motor receptor edges | 224 (one per cortical coordinate) |
| actual learned scalar storage | 83,841 |
| dense-equivalent learned scalars | 138,161 |
| learned compressed topology | 81,792 int32 pre-indices |
| byte-cycle accuracy | 0% → 94.12% |
| mean surprise | 5.4041 → 0.1069 |
| surprise reduction | 98.02% |
| free generation | `a → bcdabcda` (all eight steps correct) |
| checkpoint exact-next-step | pass |

On the N7 ambiguous stream, full Taiji predicts all eight history-dependent `x → b/d` successors correctly. A first-order model and a full dynamic-state lesion both score 50%. N7's trace-only lesion remains at 100%, showing that its immediate history can live in fast state.

N8 inserts the shared distractors `1234` between cue and probe. Full and trace-only states score 100%; removing trace or all dynamic state scores 50%. This establishes that slow trace is necessary and sufficient for this fixed-delay behavior, but it is not yet evidence of episodic or autobiographical memory.

N9 trains the same 16-byte cycle under an explicit non-terminal stream contract, then feeds back 128 motor actions with no teacher forcing. All 128 positions are exact, all four actions remain present, and membrane/trace/threshold bounds hold at every tick. A terminal boundary is deliberately excluded from this benchmark because teaching “stop after the fourth cycle” would contradict an infinite-cycle target.

N10 replaces masked-dense synapses with compressed fixed-fan-in rows. Against a dense reference, forward differs by at most `2.98e-8`; backprojection and local update are exact. N5–N9 behavior regressions all still pass. Including the new field, the small v5 benchmark uses 111.22% of dense learned-weight bytes after int32 indices, while the default projects to 98.59%. This validates real edge execution, not a universal speedup claim; sparse indexing only wins storage at lower edge density.

N11 separates external sensation from action credit. On a two-cue environment where action changes both reward and the next `+/-` sensation, the last 40 online interactions reach 100% success versus 50% random and 57.5% with action learning disabled. Taiji receives only scalar reward and outcome sensation—never the correct action label. A pending action and its eligibility are atomically checkpointed until outcome settlement.

M5 stores eight one-shot active episodes in one shared 128-unit field. Writing uses a singleton demonstrated affordance with fabric/motor learning disabled; querying opens two actions, so this isolates associative recall rather than action discovery. Cross-episode action recall is 87.5%, versus 25% for equal-width trace-only execution and 25% after recurrent-association lesion. Outcome and provenance recall are 100%, episode identity is 75%, mean time-code cosine is 0.519, and recalled cortical state measurably changes the next fabric tick. The field allocates zero per-event slots.

Current reports: [Native v7](reports/taiji_native_v7_20260822.json), [M6 v7 seed panel](reports/taiji_m6_seed_panel_v7_20260822.json), [N10 v7 regression](reports/taiji_n10_v7_20260822.json), [N11 v7](reports/taiji_n11_v7_20260822.json), and [M5 v7](reports/taiji_m5_v7_20260822.json). Earlier Native v2–v6 reports remain migration evidence.

A single seed cannot separate a mechanism change from seed-specific idiosyncrasy, so mechanism-level decisions read the M6 seed panel (`verify_taiji_m6_endogenous_replay.py --panel`, 12 seeds) rather than one run, and a baseline is always re-executed from a clean worktree instead of read out of a committed report.

## Source layout

```text
seed/
├── config.py    Seed runtime compatibility configuration
└── model.py     product-facing compatibility wrapper around TSK-v8

taiji/
├── config.py    current kernel resource and dynamics contract
├── sparse.py    fixed fan-in synapses and local updates
├── state.py     persistent region and whole-system state
├── memory.py    distributed episodic encoding, completion, and readback
├── organs.py    raw-byte sensor, sparse receptor bank, and reward-aware motor
├── environment.py active environment protocol and outcome
├── fabric.py    predictive recurrent tick
└── model.py     observe, learn, score, generate, checkpoint

tests/taiji_native/                 kernel regression and ownership contracts
scripts/training/verify_taiji_native_v7.py
scripts/training/verify_taiji_m5_episodic_field.py
scripts/training/verify_taiji_n10_sparse_migration.py
scripts/training/verify_taiji_n11_active_environment.py
scripts/training/verify_taiji_n7_context.py
scripts/training/verify_taiji_n8_delayed_trace.py
scripts/training/verify_taiji_n9_long_free_run.py
plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md
plans/archive/implementation/TAIJI_SUBSTRATE_KERNEL_V8_SPEC.md
```

## Legacy NeuroPlex

`neuroplex/` contains the previous nine-member Transformer population, including all five dialogue members. It is retained only for reproducibility and future same-budget comparisons. It is not imported by Taiji.

Historical checkpoints that serialized `taiji.*` names are loaded through the scoped `neuroplex.legacy_checkpoint` compatibility utility; importing NeuroPlex no longer shadows the native `taiji` package.

Install legacy application dependencies only when reproducing that baseline:

```bash
python -m pip install -e ".[legacy]"
```

## Current implementation target

M6/M7 and N0–N11 remain reproducible TSK-v8 kernel evidence. They no longer define the research frontier. The current target is roadmap P1: versioned Taiji v1 observation/state/goal/plan/action contracts, cognitive-ownership tests, and a TSK-v8 compatibility adapter that preserves current checkpoints and product APIs. Learned temporal abstraction begins only after that boundary is executable.

## License

Apache License 2.0. See [LICENSE](LICENSE).
