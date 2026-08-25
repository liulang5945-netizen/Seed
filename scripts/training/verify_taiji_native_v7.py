"""Verify Native v7 Taiji with dual-timescale cortical prediction."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import torch

# 直接运行时脚本目录在 sys.path；被 tests/ 以包形式导入时需手动补齐（F11）
try:
    import _verify_emit
except ModuleNotFoundError:  # pragma: no cover - import shim
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _verify_emit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import taiji
from taiji import Taiji, TaijiConfig

# 评估泄漏修复（2026-08-23）：原口径 learn_bytes(data) 后直接 score_bytes(data)
# 并作为通过条件（train==test）。现拆分为：
# - train_fit_accuracy：训练集拟合（原口径，仅信息性展示，不卡线）
# - heldout_accuracy：未参与 learn_bytes 的确定性字节流打分，作为通过条件
# heldout 选同字母表不同周期的 "abc" 循环（周期 3 ≠ 训练周期 4，且从未训练），
# 检验学到的是转移结构而非逐字节背诵。
HELDOUT_STREAM = b"abcabcabcabc"
HELDOUT_ACCURACY_FLOOR = 0.60  # 标定值见 run_benchmark 内注释


def _native_import_contract() -> bool:
    package = Path(taiji.__file__).resolve().parent
    imported = set()
    attributes = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
            elif isinstance(node, ast.Attribute):
                attributes.add(node.attr)
    return (
        not any(module.startswith(("neuroplex", "transformers")) for module in imported)
        and not {
            "backward",
            "topk",
            "MultiheadAttention",
            "TransformerEncoder",
        }
        & attributes
    )


def _learned_synapses(model: Taiji) -> tuple[object, ...]:
    return (
        *model.fabric.decoders,
        *model.fabric.consolidation_decoders,
        *model.fabric.transitions,
        model.motor.synapses,
        model.memory.association,
        model.memory.action_readout,
        model.memory.outcome_readout,
        model.memory.reward_readout,
        model.memory.familiarity_readout,
        model.memory.cortical_readout,
        model.memory.time_readout,
        model.memory.episode_readout,
        model.memory.provenance_readout,
    )


def _stored_synapses(model: Taiji) -> tuple[object, ...]:
    return (
        *_learned_synapses(model),
        model.memory.cue_encoder,
        model.memory.action_encoder,
        model.memory.outcome_encoder,
        model.memory.time_encoder,
        model.memory.episode_encoder,
        model.memory.provenance_encoder,
    )


def run_benchmark(*, epochs: int = 200, seed: int = 7) -> dict[str, object]:
    config = TaijiConfig(
        region_sizes=(64, 48),
        synapse_fan_in=16,
        motor_fan_in=48,
        seed=seed,
    )
    data = b"abcdabcdabcdabcd"
    model = Taiji(config, episode_id="native-v7")
    initial_parameters = [tensor.clone() for tensor in model.parameter_tensors()]
    before = model.score_bytes(data)

    started = time.perf_counter()
    training = model.learn_bytes(data, epochs=epochs)
    elapsed = time.perf_counter() - started
    after = model.score_bytes(data)
    heldout = model.score_bytes(HELDOUT_STREAM)
    generated = model.generate(b"a", 8)
    changed_tensors = sum(
        not torch.equal(before_tensor, after_tensor)
        for before_tensor, after_tensor in zip(
            initial_parameters, model.parameter_tensors(), strict=False
        )
    )

    checkpoint = model.checkpoint()
    restored = Taiji.from_checkpoint(checkpoint)
    next_left = model.observe(ord("!"), learn=True)
    next_right = restored.observe(ord("!"), learn=True)
    exact_next_step = (
        next_left.predicted_symbol == next_right.predicted_symbol
        and torch.equal(next_left.probabilities, next_right.probabilities)
        and all(
            torch.equal(left, right)
            for left, right in zip(
                model.parameter_tensors(), restored.parameter_tensors(), strict=False
            )
        )
    )

    synapses = _learned_synapses(model)
    stored_synapses = _stored_synapses(model)
    receptor_counts = torch.bincount(
        model.motor.receptors.channel.cpu(),
        minlength=config.motor_context_dim,
    )
    edge_count = sum(synapse.edge_count for synapse in synapses)
    topology_index_scalars = sum(synapse.pre_index.numel() for synapse in synapses)
    topology_index_bytes = sum(
        synapse.pre_index.numel() * synapse.pre_index.element_size() for synapse in synapses
    )
    edge_weight_bytes = sum(
        synapse.edge_weight.numel() * synapse.edge_weight.element_size() for synapse in synapses
    )
    active_parameters = model.parameter_count(active_only=True)
    actual_learned_storage = model.parameter_count(active_only=False)
    dense_equivalent = model.dense_equivalent_parameter_count()
    default_model = Taiji(TaijiConfig(), episode_id="v5-storage-projection")
    default_synapses = _learned_synapses(default_model)
    default_edges = sum(synapse.edge_count for synapse in default_synapses)
    default_dense_edges = sum(synapse.dense_equivalent_count for synapse in default_synapses)
    no_dense_synapse_tensor = all(
        synapse.edge_count == synapse.dense_equivalent_count
        or not any(
            isinstance(value, torch.Tensor)
            and value.shape == (synapse.out_features, synapse.in_features)
            for value in vars(synapse).values()
        )
        for synapse in stored_synapses
    )
    checks = {
        "native_namespace": Path(taiji.__file__).resolve().parent.name == "taiji",
        "no_legacy_or_transformer_dependency": _native_import_contract(),
        "no_autograd_parameters": all(
            tensor.requires_grad is False for tensor in model.parameter_tensors()
        ),
        "compressed_fixed_fan_in_storage": all(
            synapse.edge_weight.ndim == 2
            and synapse.edge_count == synapse.pre_index.numel()
            and synapse.edge_weight.shape == synapse.pre_index.shape
            for synapse in synapses
        ),
        "no_dense_synapse_tensor": no_dense_synapse_tensor,
        "all_cortical_coordinates_reach_motor": (
            int(receptor_counts.sum()) == config.cortical_context_dim
            and int(receptor_counts.max() - receptor_counts.min()) <= 1
        ),
        "shared_action_evidence_space": bool(
            model.motor.synapses.row_fan_in == config.motor_context_dim
        ),
        "online_learning_reduces_surprise": (
            after["mean_surprise"] <= before["mean_surprise"] * 0.30
        ),
        # 通过条件改为 heldout 口径；train 拟合准确率仅作信息性指标（metrics 内）。
        # 阈值 0.60 标定依据：训练 "abcd"(周期4) 后模型学到确定性转移 a→b→c→d→a，
        # 在 HELDOUT_STREAM(周期3 "abc") 上 a→b、b→c 命中、c→d 未命中，
        # 理论期望 2/3≈0.667；实测（seed=7, epochs=200, 2026-08-23）
        # heldout_accuracy=0.692，取 0.60 留出实现细节余量。
        # 纯位置记忆模型在此流上远低于该值。
        "heldout_accuracy": heldout["accuracy"] >= HELDOUT_ACCURACY_FLOOR,
        "free_generation_cycle": generated == b"bcdabcda",
        "local_parameters_changed": changed_tensors >= 3,
        "checkpoint_exact_next_step": exact_next_step,
        "active_action_api": all(hasattr(model, name) for name in ("act", "settle_action")),
        "native_episodic_field_api": all(
            hasattr(model.memory, name)
            for name in ("recall", "write", "association", "cortical_readout")
        ),
        "no_event_key_value_slots": not {
            "events",
            "keys",
            "values",
            "slots",
        }
        & set(vars(model.memory)),
    }
    return {
        "benchmark": "taiji_native_v7_dual_timescale",
        "seed": seed,
        "epochs": epochs,
        "architecture": {
            "checkpoint_format": model.CHECKPOINT_FORMAT,
            "state_version": model.STATE_VERSION,
            "reward_baseline_rate": config.reward_baseline_rate,
            "synapse_storage": "fixed-fan-in-v1",
            "sensor": "raw-byte-one-hot",
            "regions": list(config.region_sizes),
            "cortical_context_dim": config.cortical_context_dim,
            "motor_receptor_channels": config.motor_context_dim,
            "memory_units": config.memory_units,
            "memory_readout_fan_in": config.memory_readout_fan_in,
            "memory_iterations": config.memory_iterations,
            "episodic_write_count_in_passive_benchmark": model.memory.write_count,
            "fixed_receptor_edges": config.cortical_context_dim,
            "learned_synapse_edges": edge_count,
            "active_learned_parameters": active_parameters,
            "actual_learned_scalar_storage": actual_learned_storage,
            "dense_equivalent_learned_scalars": dense_equivalent,
            "avoided_dense_weight_scalars": dense_equivalent - actual_learned_storage,
            "topology_index_scalars": topology_index_scalars,
            "topology_index_bytes": topology_index_bytes,
            "edge_weight_bytes": edge_weight_bytes,
            "sparse_synapse_storage_bytes": (topology_index_bytes + edge_weight_bytes),
            "dense_synapse_weight_bytes": (
                (dense_equivalent - config.alphabet_size) * model.motor.bias.element_size()
            ),
            "sparse_to_dense_synapse_byte_ratio": (
                (topology_index_bytes + edge_weight_bytes)
                / ((dense_equivalent - config.alphabet_size) * model.motor.bias.element_size())
            ),
            "default_config_storage_projection": {
                "learned_synapse_edges": default_edges,
                "dense_equivalent_edges": default_dense_edges,
                "edge_density": default_edges / default_dense_edges,
                "sparse_to_dense_synapse_byte_ratio": (
                    default_edges * (4 + 4) / (default_dense_edges * 4)
                ),
            },
            "learning": "edge-local-predictive-and-motor-delta",
            "sequence_window": None,
        },
        "metrics": {
            "before": before,
            "training": training,
            "after": after,
            "train_fit_accuracy": after["accuracy"],
            "heldout_stream": HELDOUT_STREAM.decode("ascii"),
            "heldout": heldout,
            "heldout_accuracy": heldout["accuracy"],
            "surprise_reduction": (1.0 - after["mean_surprise"] / before["mean_surprise"]),
            "generated_hex": generated.hex(),
            "generated_text": generated.decode("latin-1"),
            "changed_parameter_tensors": changed_tensors,
            "training_seconds": elapsed,
        },
        "checks": checks,
        "status": "pass" if all(checks.values()) else "fail",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_benchmark(epochs=args.epochs, seed=args.seed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return _verify_emit.emit_and_exit("taiji_native_v7", report)


if __name__ == "__main__":
    raise SystemExit(main())
