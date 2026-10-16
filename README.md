# Seed — Runtime for the Taiji Native Cognitive Architecture

[简体中文](README.zh-CN.md) · [Project map](docs/README.md) · [Current execution](plans/active/roadmap/03_CURRENT_EXECUTION.md)

Seed hosts and exposes **Taiji**, a research architecture exploring persistent state, memory, local learning, and action. The repository includes the Python research runtime and the Taiji Harness product shell.

## Current status

The active research program is the **N mainline**, with N1 investigating learned reading, entity binding and state transitions. Results are bounded experiments, with training, unseen evaluation, retention and consumer behavior recorded separately. They do not establish general dialogue, AGI or autonomous self-improvement.

As of 2026-10-17, run88c finished three fresh position seeds under the final-cycle learning-rate reduction (0.001 for the first 18,432 updates, 0.00025 for the last 6,144), and an independent zero-CPU verifier recomputed from the sealed shards that all three seeds pass the six frozen training gates, that losses, update measurements and the score rows of all four milestones up to and including step 18,432 reproduce the run86c baseline item by item, and that nine independent restores and 110 source guards stayed unchanged inside 3,385.578125 of the frozen 4,800 CPU seconds. Training adequacy is not capability: the candidate is still not promotable, `TRAIN_SHA` stays None, confirmation v5 remains unconsumed, and the batch's own adjudication records the reverse readings too — the same reduction costs seed13 one item, and the failure it repaired had appeared on only one of the three seeds. Two control-arm batches (terminal, then cut) now train under the identical schedule so that the nine endpoints will differ only by readout arm, after which the frozen v5 binding queries that face once. See the [run88c batch adjudication](plans/reference/EXP-N1-88_run88c_batch_adjudication_20261017.md), the [control-arm and confirmation plan](plans/reference/PLAN-N1-89_lr_decay_control_arms_and_v5_confirmation_20261017.md), and [current execution](plans/active/roadmap/03_CURRENT_EXECUTION.md). Product delivery history is separate from research capability acceptance.

## Architecture and learning

- `taiji/` owns the native cognitive mechanisms and research prototypes.
- `seed/` exposes the compatibility API; `seed_platform/` supplies storage, lineage and runtime services; `api/` exposes the backend.
- `taiji-harness/` supplies the client and product shell in its own TypeScript workspace.
- `neuroplex/` retains legacy comparison and compatibility code.

The substrate uses local prediction-error learning. Some developmental sequence prototypes use optimizers and autograd; learning claims must identify the implementation and experiment. Long-term architecture goals are specified in the [core requirements](plans/active/TAIJI_CORE_REQUIREMENTS.md) and [architecture contract](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md). See the [code wiki](TAIJI_CODE_WIKI.md) for implementation details.

## Quick start

Requires Python 3.12+ and PyTorch. This example exercises a small byte-learning API; it is not a general chat model.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/seed/test_project_identity.py -q
```

```python
from seed import Seed

model = Seed()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)
checkpoint = model.checkpoint()
restored = Seed.from_checkpoint(checkpoint)
print(restored.generate(b"a", length=8))
```

For broader checks, run `python -m pytest tests -q`; some tests depend on optional packages or local artifacts. A targeted pass does not establish full CI health.

## Product shell

```bash
python -m pip install -e ".[dev,legacy]"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

See [Taiji Harness contribution instructions](taiji-harness/CONTRIBUTING.md) for the client workspace. The local provider route is `/api/chat/stream`.

## Repository navigation

| Location | Responsibility |
| --- | --- |
| `taiji/`, `seed/`, `seed_platform/`, `api/` | Model mechanisms, runtime and backend |
| `taiji-harness/` | Product client and packaging |
| `scripts/`, `instruments/`, `tests/` | Reusable tasks, measurement and verification |
| `plans/` | Requirements, active scheduling, frozen protocols and decisions |
| `docs/` | Contributor guidance and project navigation |
| `reports/` | Machine-readable research evidence |
| `design/` | Brand assets and their generators |
| `data/`, `checkpoints/`, `taiji_data/` | Local datasets and model state; preserve before migration |
| `output/`, `outputs/`, `artifacts/` | Legacy runtime/archive locations; tracked JSON evidence moved to `reports/legacy/` |

Follow the [folder rules](docs/FOLDER_STRUCTURE_RULES.md) and [repository hygiene rules](docs/REPO_HYGIENE_RULES.md). Training requires the applicable frozen protocol, budget and successful checkpoint save/restore checks. Historical failed experiments remain evidence.

## License

Apache License 2.0. See [LICENSE](LICENSE).
