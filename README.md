# Seed — Runtime for the Taiji Native Cognitive Architecture

[简体中文](README.zh-CN.md) · [Project map](docs/README.md) · [Current execution](plans/active/roadmap/03_CURRENT_EXECUTION.md)

Seed hosts and exposes **Taiji**, a research architecture exploring persistent state, memory, local learning, and action. The repository includes the Python research runtime and the Taiji Harness product shell.

## Current status

The active research program is the **N mainline**, with N1 investigating learned reading, entity binding and state transitions. Results are bounded experiments, with training, unseen evaluation, retention and consumer behavior recorded separately. They do not establish general dialogue, AGI or autonomous self-improvement.

As of 2026-10-17, run88c finished three fresh position seeds under the final-cycle learning-rate reduction (0.001 for the first 18,432 updates, 0.00025 for the last 6,144), and an independent zero-CPU verifier recomputed from the sealed shards that all three seeds pass the six frozen training gates, that losses, update measurements and the score rows of all four milestones up to and including step 18,432 reproduce the run86c baseline item by item, and that nine independent restores and 110 source guards stayed unchanged inside 3,385.578125 of the frozen 4,800 CPU seconds. Training adequacy is not capability: the candidate is still not promotable, `TRAIN_SHA` stays None, confirmation v5 remains unconsumed, and the batch's own adjudication records the reverse readings too — the same reduction costs seed13 one item, and the failure it repaired had appeared on only one of the three seeds. Both control-arm batches have since closed and been verified, and the frozen v5 binding then queried that face once on nine same-schedule endpoints: the four-gate conjunction **failed**. `advantage_cut` is false (position 79/96 against cut 80/96) and `each_position_kind` is false (seed13 position scored 0/8 on lookup and 0/8 on revision), while `each_position_seed` (31/32/16) and `advantage_terminal` (against 33/96) held. Reading only the sealed rows, position and cut emitted byte-identical output on 79 of 96 unseen item-model pairs and position was right on no item that cut missed, so the positional readout arm bought nothing observable over the cut control on unseen evidence; the failures fall along seed and item class, not along arm. The face is consumed and cannot be re-run, and one reporting field of that run (`sources_unchanged`) is a known false negative of my own binding instrument - it compared raw digests against CRLF-normalised ones stored by the batches, with zero real source changes (DEBT-G147) - so the run is not claimed as engineering complete either. The owner then chose exit (i) of the three pre-registered exits and it ran with zero queries: a read-only instrument attributed all 288 sealed rows and found `selection_side=96/96` with `compounding_only=0` (every row that fails in free generation also fails under teacher forcing), error shapes confined to `first_symbol_wrong=69` and `prefix_then_diverge=27`, the `transition` class error-free across all 72 of its rows, and all 16 groups where the three arms fail together sitting in seed13's 8 lookup plus 8 revision questions - inside those 16, no two arms emitted the same wrong string. Those are correlated readings about what to probe next: not a mechanism, not an arm advantage, not a filled capability rung. See the [error attribution report](reports/taiji_n1_confirmation_error_attribution_run89.json), the [run91 adjudication](plans/reference/EXP-N1-91_run91_error_attribution_adjudication_20261017.md), the [v5 confirmation adjudication](plans/reference/EXP-N1-90_v5_confirmation_adjudication_20261017.md), the [run88c batch adjudication](plans/reference/EXP-N1-88_run88c_batch_adjudication_20261017.md) and [current execution](plans/active/roadmap/03_CURRENT_EXECUTION.md). The follow-up oracle-address probe (PLAN-N1-92, owner-chosen) then ran to completion on the nine endpoints: both admission anchors were true (address extractor 96/96 against the pinned instrument, adapter native branch bit-identical to `teacher_logits` on all 144 pairs), 432 teacher sequences in 132.7969 of the 420-second window, and the pre-registered exit it names is **(4) the ruler lacks dynamic range** - five usable endpoints out of nine, because position and cut already answer 16/16 of the addressable items on seeds 11/12 while the three terminal endpoints answer 0/16 and stay 0/16 even when the tape address is pinned, so `oracle_lift_supported=false`. One disclosure-only asymmetry: inside seed13, pinning the address lifts cut to 11/16 but position only to 2/16, which is evidence for choosing the next question, not a mechanism claim and not an arm advantage. See the [oracle probe adjudication](plans/reference/EXP-N1-92_oracle_address_probe_adjudication_20261017.md). After that exit the owner ruled to build a fresh unseen face instead (PLAN-N1-93; its §1 shape parameters are still awaiting a signature, so no face has been generated and no endpoint has been queried for it). Before asking for that signature I ran an out-of-repo pre-flight of the three v6 instruments against the nine frozen endpoints - zero consumption, and the four real repo write-targets verified absent both before and after. It paid off twice: the success branches now provably execute end to end (20 items per endpoint across nine endpoints = 180 native sequences, 95-137 CPU seconds against the 180-second limit; the range instrument had to be upgraded to publish per-class error columns because its adjudication could not otherwise be recomputed from its own artifact - DEBT-G151), and the drafted difficulty knob does not open the ruler: across five values of observations-per-item and three sampling draws the usable-endpoint count stayed at 5/9 against the frozen 6/9 line, the single boundary pass appearing only for one draw and disappearing when the draw changed, with the same four endpoints (seeds 11/12 on position and cut) always below the line and three of the five above it contributed free by the damaged terminal arm. A proposed harder variant (multi-step update chains) was ruled out by code before writing any: the curriculum generators emit only single-step updates, so a longer chain would measure an unseen sentence form rather than capability. See the [v6 pre-flight adjudication](plans/reference/EXP-N1-95_v6_preflight_range_sweep_20261017.md) and its [nine-reading artifact](reports/taiji_n1_v6_preflight_range_sweep.json). A second pre-flight round then added one in-distribution difficulty axis - distractor entity names held at Hamming distance d from the subject, so the sentence forms stay exactly as trained and only the binding is stressed - and that axis does open the ruler: at d=1 the usable-endpoint count is 7/7/7/8 across four independent sampling draws against a frozen line of 6, and 9/9 at eight observations per item, while d=2 falls back to 5/9; the abstention class gains range for the first time (mean errors 0.56 to 3.89 of 5). It also exposes a structural exclusion: the terminal arm is already at its floor, so degrading position and cut together compresses both advantage gates toward zero and even inverts them (seed13 forecasts -2 and -3), and across the eight readings "passes the range gate" and "few forecast red cells" never co-occur - on these nine endpoints, a face with dynamic range and a face that can pass the frozen four gates are mutually exclusive. The queue head is therefore four rulings for the owner (sign d=1, whose first confirmation query is expected to land in the pre-registered exit (2); or sign 8/1, whose forecast is all nine cells red; or restate the four gates before consuming, which needs a fresh signature; or settle the §3 denominators and which line gates - DEBT-G152), plus the §1 signature. No confirmation query has been fired. Product delivery history is separate from research capability acceptance.

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
