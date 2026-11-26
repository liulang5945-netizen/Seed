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
- **Long-horizon development:** learning, replay, consolidation and structural adaptation that preserve useful prior capabilities. This is a research goal, not a demonstrated capability.

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

For the bounded `N36 / N37` delivery runtime and operational state machine CLI:

```bash
python scripts/taiji_n36_takeover.py build --bundle-dir ./taiji_bundle --seed 6801
python scripts/taiji_n36_takeover.py chat --bundle-dir ./taiji_bundle --utterance "请把红色杯子放到桌上。"
python scripts/taiji_n37_ops.py init-ops --bundle-dir ./taiji_bundle --initial-state Qualified
```

Client setup and build instructions are in [Taiji Harness contributing](taiji-harness/CONTRIBUTING.md). See [Seed contributing](CONTRIBUTING.md) for development checks and contribution guidance.

## Recent research

As of **2026-11-26**, the bounded mainline (`≤N30 → N31 → N32 → N33 → N34 → N35 → N36 → N37`) has completed full closeout within `bounded_closed_vocab_and_multimodule_v1` (**0 `check` / 0 `final` reads**):
- **Confirmed bounded capabilities (`N33–N35`)**: 100-session multi-module durability (`N34.5.90`), closed-vocab multi-turn Chinese coreference, synthetic two-family physics causal transfer, online Beta credibility learning, calibrated uncertainty, and in-place plastic growth (`N34.9.90`, `N35.1.91` 8/9 gates 3/3 pass).
- **Bounded delivery & §15 operational state machine (`N36 / N37`)**: Self-contained takeover runtime ([CLOSEOUT-N36](plans/reference/CLOSEOUT-N36_20261126.md), `D1..D5` 5/5 pass) and 8-state operational FSM with 6-step shadow upgrade & 3-tier change routing ([CLOSEOUT-N37](plans/reference/CLOSEOUT-N37_20261126.md), `O1..O6` 6/6 pass).
- **Active `N38` open-Chinese pretraining**: `mid_18m` (19.47M params, 8K byte-level BPE, persistent memory slots) is running 24h CPU pretraining (`~4,343 tokens/sec` in `bfloat16`, [MEMO-N38-STAGE0-PREFLIGHT](plans/reference/MEMO-N38-STAGE0-PREFLIGHT_20261126.md)).

Open-vocab natural Chinese, real physical sensors/actuators, 24h live field trials, and `check / final` splits remain explicitly **unverified / not signed**. See [CONSOLIDATION_20261126](plans/reference/CONSOLIDATION_20261126.md) and [MAINLINE](plans/MAINLINE.md).

## Read more

- [Project map](docs/README.md): find documentation by purpose.
- [Core requirements](plans/active/TAIJI_CORE_REQUIREMENTS.md): the long-term goals and design principles.
- [Architecture design](plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md): cognitive components and their contracts.
- [Active plans](plans/active/PLAN_INDEX.md) and [research reports](reports/README.md): ongoing work and supporting evidence.
- [Folder structure](docs/FOLDER_STRUCTURE_RULES.md) and [repository hygiene](docs/REPO_HYGIENE_RULES.md): where contributions and artifacts belong.

## License

[Apache License 2.0](LICENSE).
