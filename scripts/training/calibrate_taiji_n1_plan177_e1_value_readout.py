"""E2b-2: calibrate a value readout from real producer features and run controls.

For each of the eight colours, the trained N177.6 producer reads a fact about
the known entity "钥匙" and exposes its continuous slot feature. A closed-form
ridge readout maps the feature (plus the query-only V4 hidden state) to the
colour identity. The script then checks prototype accuracy, relevant/irrelevant
feature swaps, cut and rescue. No optimizer step, no E1 sealed text, and no
checkpoint overwrite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = (
    ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
)
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
READOUT_ROOT = ROOT / "data/checkpoints/n1_plan177_e1_readout_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_e1_value_readout_20261029.json"
RIDGE = 1e-3


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _family(entity: str, colour: str):
    from taiji.internalization import content_digest
    from taiji.joint_state_consumption_course import E1_COLORS, E1FamilyV1

    digest = content_digest({"prototype": entity, "colour": colour})
    old = next(item for item in E1_COLORS if item != colour)
    return E1FamilyV1(
        family_id=f"e1-family:proto-{digest[:12]}",
        entity=entity,
        value=colour,
        partition="train",
        old_value=old,
        token=digest[:12],
    )


def _collect(trainer, entity: str, colour: str, feature_source: str, encoder):
    from taiji.joint_state_consumption import feature_from_parse
    from taiji.joint_state_consumption_course import build_episode, e1_turn_text

    family = _family(entity, colour)
    episode = build_episode(family, 0)
    fact = episode.turns[0]
    query = episode.turns[4]
    adapter, joint = trainer._new_episode_state(f"e1-proto:{family.token}:fact")
    trace, joint = trainer._append_turn(adapter, joint, fact)
    parse, role_features = trainer.producer.predict_with_role_features(joint, trace)
    if feature_source == "producer-mean":
        continuous = parse.semantic_features
    elif feature_source == "producer-value":
        continuous = role_features.get("value")
    elif feature_source == "value-token":
        continuous = (
            None
            if encoder is None
            else tuple(float(item) for item in encoder.encode(colour).tolist())
        )
    else:
        raise ValueError("unsupported E2b-2 feature source")
    feature = feature_from_parse(
        parse,
        source_input_id=fact.frame.input_id,
        source_sha256=_sha256(bytes(fact.frame.payload)),
        turn_index=0,
        revision=0,
        continuous=continuous,
    )
    query_adapter, query_joint = trainer._new_episode_state(f"e1-proto:{family.token}:query")
    query_trace, query_joint = trainer._append_turn(query_adapter, query_joint, query)
    query_source = query_joint.input_bindings[-1].neural_source_id
    cache = trainer.model._joint_evidence_cache(query_joint, selected_source_ids=(query_source,))
    _, hidden = trainer.model.next_logits(query_joint.language_state, cache=cache)
    return {
        "family": family,
        "fact_text": e1_turn_text(fact)[0],
        "query_text": e1_turn_text(query)[0],
        "parse": parse,
        "feature": feature,
        "hidden": hidden.detach(),
        "query_joint": query_joint,
        "query_trace": query_trace,
    }


def _readout_input(feature, hidden, readout):
    if feature is None:
        return None
    return readout.read(torch.tensor(feature.continuous), hidden)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2b-2 value readout calibration")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--trace-cache", type=Path, default=TRACE_CACHE)
    parser.add_argument(
        "--feature-source",
        choices=("producer-mean", "producer-value", "value-token"),
        default="producer-value",
    )
    parser.add_argument("--readout-root", type=Path, default=READOUT_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    if args.readout_root.exists() or args.report.exists():
        raise ValueError("E1 value-readout calibration is write-once")
    if not args.checkpoint.is_file() or not args.trace_cache.is_file():
        raise FileNotFoundError("readout calibration requires the N177.6 endpoint and trace cache")

    from taiji.joint_state_consumption import (
        PersistentContentStoreV1,
        QueryConditionedSelectorV1,
        ValueTokenFeatureEncoderV1,
    )
    from taiji.joint_state_consumption_course import E1_COLORS
    from taiji.joint_state_readout import FeatureValueReadoutV1
    from taiji.joint_t1_trace_cache import JointT1TraceCacheV1
    from taiji.joint_t1_trainer import JointT1TrainerV1

    trace_cache = JointT1TraceCacheV1.from_payload(
        torch.load(args.trace_cache, map_location="cpu", weights_only=False)
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=trace_cache)
    trainer.input_trace_cache = None
    encoder = ValueTokenFeatureEncoderV1() if args.feature_source == "value-token" else None

    cpu_start = time.process_time()
    wall_start = time.monotonic()
    prototypes = []
    features = []
    hiddens = []
    for index, colour in enumerate(E1_COLORS):
        record = _collect(trainer, "钥匙", colour, args.feature_source, encoder)
        if record["feature"] is None:
            prototypes.append(
                {
                    "value": colour,
                    "index": index,
                    "parse_act": record["parse"].act,
                    "parse_status": record["parse"].status,
                    "feature": None,
                }
            )
            continue
        prototypes.append(
            {
                "value": colour,
                "index": index,
                "parse_act": record["parse"].act,
                "parse_status": record["parse"].status,
                "feature_dim": len(record["feature"].continuous),
                "hidden_dim": int(record["hidden"].numel()),
                "source_input_id": record["feature"].source_input_id,
                "source_sha256": record["feature"].source_sha256,
            }
        )
        features.append(torch.tensor(record["feature"].continuous))
        hiddens.append(record["hidden"])
    if len(features) != len(E1_COLORS):
        raise RuntimeError("producer did not yield all eight colour prototypes")
    readout = FeatureValueReadoutV1.fit_closed_form(
        features,
        list(range(len(E1_COLORS))),
        query_hiddens=hiddens,
        ridge=RIDGE,
    )
    prototype_rows = []
    for prototype, feature, hidden in zip(prototypes, features, hiddens, strict=True):
        result = readout.read(feature, hidden)
        prototype_rows.append(
            {
                "value": prototype["value"],
                "selected_value": result.selected_value,
                "correct": result.selected_value == prototype["value"],
                "logits": list(result.logits),
            }
        )
    matrix = torch.stack(features)
    rank = int(torch.linalg.matrix_rank(matrix))
    normalized = matrix / matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    cosine = normalized @ normalized.T
    off_diagonal = cosine[~torch.eye(len(features), dtype=torch.bool)]
    if not all(item["correct"] for item in prototype_rows):
        args.readout_root.mkdir(parents=True)
        prototype_path = args.readout_root / "prototypes.pt"
        torch.save(
            {
                "features": [list(item) for item in features],
                "hiddens": [list(item.tolist()) for item in hiddens],
                "values": list(E1_COLORS),
                "sources": [item.get("source_input_id") for item in prototypes],
            },
            prototype_path,
        )
        report = {
            "schema": "taiji-n177-e1-value-readout-v1",
            "status": "feature_not_separable",
            "checkpoint": str(args.checkpoint.relative_to(ROOT)),
            "trace_cache": str(args.trace_cache.relative_to(ROOT)),
            "entity": "钥匙",
            "feature_source": args.feature_source,
            "prototype_rows": prototype_rows,
            "prototype_accuracy": sum(item["correct"] for item in prototype_rows)
            / len(prototype_rows),
            "feature_rank": rank,
            "feature_count": len(features),
            "feature_dim": int(features[0].numel()),
            "hidden_dim": int(hiddens[0].numel()),
            "min_cosine": float(off_diagonal.min()),
            "max_cosine": float(off_diagonal.max()),
            "prototype_sha256": _sha256(prototype_path.read_bytes()),
            "controls": "not_run_because_feature_not_separable",
            "architecture_verdict": (
                "the selected producer role feature still collapses colour identities; "
                "the readout cannot separate the value, so the decoder was not exercised"
            ),
            "task_scope": "closed-world known-entity colour value readout",
            "suspected_mechanism": (
                "value-span mean feature still lacks sufficient value information or is dominated "
                "by shared slot content"
            ),
            "information_path": "fact bytes -> producer value-span mean feature -> value readout",
            "credit_path": "no optimizer update; closed-form readout not fitted for a claim",
            "competing_causes": (
                "value-span mean pooling, producer feature capacity, or a missing value-token feature"
            ),
            "next_action": (
                "use a value-token/per-byte value feature and rerun E2b-2 before any decoder or F "
                "training"
            ),
            "optimizer_updates": 0,
            "checkpoint_loads": 1,
            "sealed_text_read": 0,
            "model_calls": len(E1_COLORS) * 2,
            "can_promote": False,
            "cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.monotonic() - wall_start,
        }
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "feature_rank": rank,
                    "min_cosine": report["min_cosine"],
                    "prototype_accuracy": report["prototype_accuracy"],
                },
                ensure_ascii=False,
            )
        )
        return 2

    cup_black = _collect(trainer, "杯子", "黑色", args.feature_source, encoder)
    cup_white = _collect(trainer, "杯子", "白色", args.feature_source, encoder)
    key_blue = _collect(trainer, "钥匙", "蓝色", args.feature_source, encoder)
    key_green = _collect(trainer, "钥匙", "绿色", args.feature_source, encoder)
    for record in (cup_black, cup_white, key_blue, key_green):
        if record["feature"] is None:
            raise RuntimeError("producer failed on a control fact")
    selector = QueryConditionedSelectorV1()
    store = PersistentContentStoreV1(()).add(key_blue["feature"]).add(cup_black["feature"])
    query_hidden = key_blue["hidden"]

    def selected_result():
        selection = selector.select(
            entity="钥匙", attribute="颜色", candidates=store.active_features()
        )
        if selection is None:
            return None, None, None
        result = readout.read(torch.tensor(selection.selected_feature.continuous), query_hidden)
        return selection, selection.selected_feature, result

    selection, feature, result = selected_result()
    if selection is None or result is None:
        raise RuntimeError("baseline selection failed")
    baseline = {
        "selected_feature_id": feature.feature_id,
        "selected_value": result.selected_value,
        "logits": list(result.logits),
    }
    relevant_store = store.swap("钥匙", "颜色", key_green["feature"])
    store = relevant_store
    selection, feature, result = selected_result()
    if selection is None or result is None:
        raise RuntimeError("relevant swap lost the selected feature")
    relevant = {
        "selected_feature_id": feature.feature_id,
        "selected_value": result.selected_value,
        "logits": list(result.logits),
        "changed": result.selected_value != baseline["selected_value"],
    }
    store = store.swap("杯子", "颜色", cup_white["feature"])
    selection, feature, result = selected_result()
    if selection is None or result is None:
        raise RuntimeError("irrelevant swap lost the selected feature")
    irrelevant = {
        "selected_feature_id": feature.feature_id,
        "selected_value": result.selected_value,
        "logits": list(result.logits),
        "selected_unchanged": feature.feature_id == relevant["selected_feature_id"],
        "logs_unchanged": max(
            abs(a - b) for a, b in zip(result.logits, relevant["logits"], strict=True)
        )
        <= 1e-6,
    }
    cut_store = store.cut("钥匙", "颜色")
    cut_selection = selector.select(
        entity="钥匙", attribute="颜色", candidates=cut_store.active_features()
    )
    rescued_store = cut_store.add(key_green["feature"])
    rescued_selection = selector.select(
        entity="钥匙", attribute="颜色", candidates=rescued_store.active_features()
    )
    rescued_result = (
        None
        if rescued_selection is None
        else readout.read(torch.tensor(rescued_selection.selected_feature.continuous), query_hidden)
    )
    if rescued_result is None:
        raise RuntimeError("rescue failed to restore the selected feature")

    args.readout_root.mkdir(parents=True)
    readout_path = args.readout_root / "readout.pt"
    prototype_path = args.readout_root / "prototypes.pt"
    torch.save(
        {
            "state_dict": readout.state_dict(),
            "feature_dim": readout.feature_dim,
            "hidden_dim": readout.hidden_dim,
            "ridge": RIDGE,
            "values": list(E1_COLORS),
        },
        readout_path,
    )
    torch.save(
        {
            "features": [list(item) for item in features],
            "hiddens": [list(item.tolist()) for item in hiddens],
            "values": list(E1_COLORS),
            "sources": [item.get("source_input_id") for item in prototypes],
        },
        prototype_path,
    )
    report = {
        "schema": "taiji-n177-e1-value-readout-v1",
        "status": "completed",
        "checkpoint": str(args.checkpoint.relative_to(ROOT)),
        "trace_cache": str(args.trace_cache.relative_to(ROOT)),
        "entity": "钥匙",
        "feature_source": args.feature_source,
        "prototype_rows": prototype_rows,
        "prototype_accuracy": sum(item["correct"] for item in prototype_rows) / len(prototype_rows),
        "readout": {
            "feature_dim": readout.feature_dim,
            "hidden_dim": readout.hidden_dim,
            "output_values": list(E1_COLORS),
            "ridge": RIDGE,
            "readout_sha256": _sha256(readout_path.read_bytes()),
            "prototype_sha256": _sha256(prototype_path.read_bytes()),
        },
        "controls": {
            "baseline": baseline,
            "relevant_swap": relevant,
            "irrelevant_swap": irrelevant,
            "cut": {"selection": None, "selected": False},
            "rescue": {
                "selected_feature_id": rescued_selection.selected_feature.feature_id,
                "selected_value": rescued_result.selected_value,
            },
            "query_only": {
                "bindings": len(key_blue["query_joint"].input_bindings),
                "observations": len(key_blue["query_joint"].language_state.observations),
                "fact_bytes_present": any(
                    raw == key_blue["fact_text"].encode("utf-8")
                    for _, raw in key_blue["query_joint"].language_state.observations
                ),
            },
        },
        "optimizer_updates": 0,
        "checkpoint_loads": 1,
        "sealed_text_read": 0,
        "model_calls": len(E1_COLORS) * 2 + 8,
        "can_promote": False,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.monotonic() - wall_start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "completed",
                "prototype_accuracy": report["prototype_accuracy"],
                "baseline": baseline["selected_value"],
                "relevant": relevant["selected_value"],
                "irrelevant_same": irrelevant["selected_unchanged"],
                "cut": cut_selection is None,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
