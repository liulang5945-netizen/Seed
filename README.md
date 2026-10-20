# Seed — Runtime for the Taiji Native Cognitive Model

[简体中文](README.zh-CN.md) · [Project map](docs/README.md) · [Contributing](CONTRIBUTING.md)

**Taiji (态极) is the native cognitive model being developed in this project.** Its architecture explores how perception, memory, world models, reasoning and action can work together in a model that develops through experience. **Seed** provides the runtime, experiments, APIs and product interface around the model.

The long-term goal is a model that carries knowledge and internal state across interactions, learns from the consequences of its actions, and acquires new skills while retaining existing abilities. The repository brings the cognitive mechanisms and the infrastructure needed to study them into one place.

## What Taiji explores

Taiji's design centers on five connected ideas:

- **Continuity:** persistent internal state and a world model that can be revised as new observations arrive.
- **Memory with different roles:** working memory for the current task, episodic memory for experiences, semantic memory for shared knowledge, and procedural memory for skills.
- **Specialization and cooperation:** heterogeneous groups that contribute different representations and learning rules, with computation selected according to the task, uncertainty and resource budget.
- **Learning through action:** language, tools and environment interaction connected to observable outcomes that can inform later decisions.
- **Lifelong development:** learning, replay, consolidation and structural adaptation that preserve useful prior capabilities.

These are architecture goals under investigation. The current implementation contains research kernels and bounded prototypes; the complete architecture and general conversational ability remain open research goals. The [core requirements](plans/active/TAIJI_CORE_REQUIREMENTS.md) explain the motivation and the [architecture design](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md) defines the intended mechanisms and interfaces.

## How the project fits together

**Taiji owns cognition:** learned representations, state, memory, goals, reasoning and action selection. **Seed hosts it:** processes, devices, datasets, experiments, checkpoint management, APIs and environment adapters. **Taiji Harness presents it:** the web and desktop client, interaction surfaces and packaging.

The intended interaction loop is:

```text
Observations → Perception → State, memory and reasoning → Action
     ↑                                                    │
     └──────────── Environment feedback and learning ──────┘
```

The Python implementation includes the TSK-v8 substrate and prototypes for learned reading, memory, planning and developmental learning. Learning methods vary by component: the substrate uses local prediction-error learning, while some sequence prototypes use optimizers and autograd. The [code wiki](TAIJI_CODE_WIKI.md) maps these mechanisms to their implementations.

## What is in this repository

| Location | Purpose |
| --- | --- |
| [`taiji/`](taiji/) | Native cognitive mechanisms and research prototypes |
| [`seed/`](seed/), [`seed_platform/`](seed_platform/) | Python API, runtime services, persistence and lineage |
| [`api/`](api/) | Backend interfaces for product integration |
| [`taiji-harness/`](taiji-harness/) | Web and desktop client in a separate TypeScript workspace |
| [`scripts/`](scripts/), [`instruments/`](instruments/), [`tests/`](tests/) | Experiment runners, measurement tools and verification |
| [`plans/`](plans/), [`reports/`](reports/) | Research designs, decisions and experimental evidence |
| [`docs/`](docs/), [`design/`](design/) | Contributor documentation and visual assets |
| [`neuroplex/`](neuroplex/) | Frozen legacy comparison and compatibility code |

## Quick start

Run these commands from the repository root with **Python 3.12+**. The base package depends on PyTorch; the client has its own setup.

```bash
python -m pip install -e ".[dev]"
python -m pytest tests/seed/test_project_identity.py -q
```

The small Python API below demonstrates byte learning and an in-memory checkpoint round trip:

```python
from seed import Seed

model = Seed()
model.learn_bytes(b"abcdabcdabcdabcd", epochs=200)
checkpoint = model.checkpoint()
restored = Seed.from_checkpoint(checkpoint)
print(restored.generate(b"a", length=8))
```

This is a small research example. It exercises the byte-learning interface rather than demonstrating general language understanding.

For backend development:

```bash
python -m pip install -e ".[dev,legacy]"
python -m uvicorn api.app:app --host 127.0.0.1 --port 8000
```

Client setup and build instructions are in [Taiji Harness contributing](taiji-harness/CONTRIBUTING.md). See [Seed contributing](CONTRIBUTING.md) for development checks and contribution guidance.

## Recent research

As of 2026-10-20, the N mainline is studying learned reading, entity binding, state transitions and question-surface robustness. N1-162B completed 55,296 matched continuation updates and 7,488 train-only scores with all six final bundles independently restored. Only two of six endpoints passed the original training gate, so no new face was queried and no N-series model was promoted. Offline attribution found that transition errors in two seeds often became the inverse-turn answer under question-surface training. See the [N1-162B closeout](plans/reference/PLAN-N1-162B_execution_closeout_20261020.md) and [current execution](plans/active/roadmap/03_CURRENT_EXECUTION.md) for the evidence and next step.

## Read more

- [Project map](docs/README.md): find documentation by purpose.
- [Core requirements](plans/active/TAIJI_CORE_REQUIREMENTS.md): the long-term goals and design principles.
- [Architecture design](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md): cognitive components and their contracts.
- [Active plans](plans/active/PLAN_INDEX.md) and [research reports](reports/README.md): ongoing work and supporting evidence.
- [Folder structure](docs/FOLDER_STRUCTURE_RULES.md) and [repository hygiene](docs/REPO_HYGIENE_RULES.md): where contributions and artifacts belong.

## License

[Apache License 2.0](LICENSE).
