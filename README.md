# Seed — Runtime for the Taiji Native Cognitive Architecture

[简体中文](README.zh-CN.md) · [Project map](docs/README.md) · [Current execution](plans/active/roadmap/03_CURRENT_EXECUTION.md)

Seed hosts and exposes **Taiji**, a research architecture exploring persistent state, memory, local learning, and action. The repository includes the Python research runtime and the Taiji Harness product shell.

## Current status

The active research program is the **N mainline**, with N1 investigating learned reading, entity binding and state transitions. Results are bounded experiments, with training, unseen evaluation, retention and consumer behavior recorded separately. They do not establish general dialogue, AGI or autonomous self-improvement.

As of 2026-10-17, run88c finished three fresh position seeds under the final-cycle learning-rate reduction (0.001 for the first 18,432 updates, 0.00025 for the last 6,144), and an independent zero-CPU verifier recomputed from the sealed shards that all three seeds pass the six frozen training gates, that losses, update measurements and the score rows of all four milestones up to and including step 18,432 reproduce the run86c baseline item by item, and that nine independent restores and 110 source guards stayed unchanged inside 3,385.578125 of the frozen 4,800 CPU seconds. Training adequacy is not capability: the candidate is still not promotable, `TRAIN_SHA` stays None, confirmation v5 remains unconsumed, and the batch's own adjudication records the reverse readings too — the same reduction costs seed13 one item, and the failure it repaired had appeared on only one of the three seeds. Both control-arm batches have since closed and been verified, and the frozen v5 binding then queried that face once on nine same-schedule endpoints: the four-gate conjunction **failed**. `advantage_cut` is false (position 79/96 against cut 80/96) and `each_position_kind` is false (seed13 position scored 0/8 on lookup and 0/8 on revision), while `each_position_seed` (31/32/16) and `advantage_terminal` (against 33/96) held. Reading only the sealed rows, position and cut emitted byte-identical output on 79 of 96 unseen item-model pairs and position was right on no item that cut missed, so the positional readout arm bought nothing observable over the cut control on unseen evidence; the failures fall along seed and item class, not along arm. The face is consumed and cannot be re-run, and one reporting field of that run (`sources_unchanged`) is a known false negative of my own binding instrument - it compared raw digests against CRLF-normalised ones stored by the batches, with zero real source changes (DEBT-G147) - so the run is not claimed as engineering complete either. See the [v5 confirmation adjudication](plans/reference/EXP-N1-90_v5_confirmation_adjudication_20261017.md), the [run88c batch adjudication](plans/reference/EXP-N1-88_run88c_batch_adjudication_20261017.md) and [current execution](plans/active/roadmap/03_CURRENT_EXECUTION.md). Product delivery history is separate from research capability acceptance.

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
