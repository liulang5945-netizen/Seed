# Contributing to Seed / Taiji

Taiji is the native cognitive architecture; Seed is its product/runtime. Contributions should implement or falsify an explicit Taiji v1 capability, reuse mature methods where appropriate, and avoid both Transformer rewrapping and primitive reinvention.

## Setup

```bash
git clone https://github.com/<your-username>/Seed.git
cd Seed
python -m pip install -e ".[dev]"
python scripts/training/verify_taiji_native_v1.py
python -m pytest tests/taiji_native -q
```

Or use the **Dev Container** (recommended): open the project in VS Code with the Dev Containers extension — it auto-configures Python 3.12 + Node 22 + all extensions.

### Code quality

```bash
# Lint & format (pre-commit hooks run these automatically)
ruff check .
black --check .

# Frontend lint & test
cd frontend && npm run lint && npm test
```

Run the complete regression suite before submitting:

```bash
python -m pytest tests -q
```

### Building a release

```bash
python scripts/release.py            # Full build (frontend + PyInstaller + NSIS)
python scripts/release.py --skip-nsis # Without NSIS installer
python scripts/sync_version.py        # Sync version to all files
```

## Native-core rules

- `seed/` is a product/runtime host. It may depend on the public `taiji` API, but must not hide concept memory, planning or model decisions in the runtime layer.
- `taiji/` must not import `seed`, `neuroplex`, `transformers` or Legacy checkpoints.
- Mature embeddings, tokenizers/codecs, attention-like routing, state-space/graph operators, optimizers, reinforcement learning and CUDA implementations are allowed when Taiji owns their state and decision path.
- Developmental training must label `native-local` versus `native-assisted`; an external teacher may support an experiment but must not become a runtime cognitive dependency.
- Lifetime learning updates must state their credit signal, time span, persistent state, checkpoint and lesion behavior.
- Every persistent state must define its update, decay, reset, checkpoint, and lesion behavior.
- Parameter-count claims must distinguish active masked edges from dense tensor storage and measured FLOPs.
- Capability claims need a frozen holdout, relevant baselines and causal lesion/control.
- Increasing model size or epochs is not an accepted response to a failed mechanism gate.

## Pull requests

1. Create a branch, normally `feat/<name>` or `fix/<name>`.
2. Add the smallest failing test or falsification benchmark first.
3. Implement one architecture change with matching equations and code documentation.
4. Run native tests, the benchmark, and the full regression suite.
5. Update `plans/active/TAIJI_NATIVE_ARCHITECTURE_V1.md` when a capability contract changes, and keep the single execution source `plans/active/roadmap/03_CURRENT_EXECUTION.md` aligned with the implementation state.

## Legacy boundary

The previous NeuroPlex Transformer runtime remains under `neuroplex/` for reproducibility. Changes there must be labeled Legacy and must not become a dependency of native Taiji. Historical `taiji.*` pickle paths use the scoped compatibility loader rather than a global module alias.

## License

Contributions are licensed under Apache License 2.0.
