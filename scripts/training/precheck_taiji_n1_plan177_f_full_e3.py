"""Full E3 precheck: producer + V4 + value encoder + selector/store + byte bias.

The parent and a fresh child each load the frozen N177.6 endpoint and its trace
cache, run the same E1 fact/query episode, build the persistent feature from the
producer value span through the F0 byte encoder, select it, and add a feature-
conditioned byte bias to the query-only V4 next-byte logits. Parse, selected
feature and conditioned logits must match exactly. No optimizer update and no
E1 query/holdout text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_CHECKPOINT = (
    ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
)
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
F0_FINAL = ROOT / "data/checkpoints/n1_plan177_f_value_encoder_20261029/final.pt"
E2B3_BUNDLE = ROOT / "data/checkpoints/n1_plan177_e1_value_feature_bundle_20261029/bundle.pt"
BUNDLE_ROOT = ROOT / "data/checkpoints/n1_plan177_f_full_e3_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f_full_e3_20261029.json"
BUNDLE_FORMAT = "taiji-n177-f-full-e3-bundle-v1"
TOLERANCE = 1e-9


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bundle() -> dict:
    from taiji.joint_state_consumption import (
        PersistentContentFeatureV1,
        PersistentContentStoreV1,
        QueryConditionedSelectorV1,
    )
    from taiji.joint_state_value_encoder import FeatureByteBiasV1, ValueByteEncoderV1

    f0 = torch.load(F0_FINAL, map_location="cpu", weights_only=False)
    e2b3 = torch.load(E2B3_BUNDLE, map_location="cpu", weights_only=False)
    if e2b3.get("format") != "taiji-n177-e1-value-feature-bundle-v1":
        raise ValueError("E2b-3 bundle format differs")
    byte_dim = int(f0["byte_dim"])
    feature_dim = int(f0["hidden_dim"])
    encoder = ValueByteEncoderV1(byte_dim=byte_dim, hidden_dim=feature_dim)
    encoder.load_state_dict(f0["encoder"])
    selector = QueryConditionedSelectorV1()
    selector.load_state_dict(e2b3["selector_state"])
    torch.manual_seed(1771029)
    bias = FeatureByteBiasV1(feature_dim=feature_dim, byte_vocab=257)
    e2b3_store = PersistentContentStoreV1.from_payload(e2b3["store"])
    key_blue = next(
        item for item in e2b3_store.features if item.entity == "钥匙" and item.value == "蓝色"
    )
    cup_black = next(
        item for item in e2b3_store.features if item.entity == "杯子" and item.value == "黑色"
    )
    replacements = {
        key: PersistentContentFeatureV1.from_payload(value)
        for key, value in e2b3["replacements"].items()
    }

    def rebind(feature: PersistentContentFeatureV1) -> PersistentContentFeatureV1:
        return PersistentContentFeatureV1.create(
            entity=feature.entity,
            attribute=feature.attribute,
            value=feature.value,
            source_input_id=feature.source_input_id,
            source_sha256=feature.source_sha256,
            turn_index=feature.turn_index,
            revision=feature.revision,
            valid_from=feature.valid_from,
            confidence=feature.confidence,
            continuous=tuple(float(item) for item in encoder.encode_text(feature.value).tolist()),
        )

    store = PersistentContentStoreV1(()).add(rebind(key_blue)).add(rebind(cup_black))
    return {
        "format": BUNDLE_FORMAT,
        "version": 1,
        "base_checkpoint": str(BASE_CHECKPOINT.relative_to(ROOT)),
        "base_checkpoint_sha256": _sha256_file(BASE_CHECKPOINT),
        "trace_cache": str(TRACE_CACHE.relative_to(ROOT)),
        "trace_cache_sha256": _sha256_file(TRACE_CACHE),
        "byte_dim": byte_dim,
        "feature_dim": feature_dim,
        "byte_vocab": 257,
        "encoder_state": encoder.state_dict(),
        "selector_state": selector.state_dict(),
        "bias_state": bias.state_dict(),
        "readout_state": f0["readout"],
        "store": store.to_payload(),
        "replacements": {key: rebind(value).to_payload() for key, value in replacements.items()},
        "source_sha256": {
            "f0_final": _sha256_file(F0_FINAL),
            "e2b3_bundle": _sha256_file(E2B3_BUNDLE),
            "base_checkpoint": _sha256_file(BASE_CHECKPOINT),
            "trace_cache": _sha256_file(TRACE_CACHE),
        },
    }


def _episode():
    from taiji.internalization import content_digest
    from taiji.joint_state_consumption_course import E1FamilyV1, build_episode

    digest = content_digest({"full_e3": "钥匙|蓝色"})
    family = E1FamilyV1(
        family_id=f"e1-family:full-e3-{digest[:12]}",
        entity="钥匙",
        value="蓝色",
        partition="train",
        old_value="黄色",
        token=digest[:12],
    )
    return build_episode(family, 0)


def _run(bundle_path: Path) -> dict:
    from taiji.internalization import content_digest
    from taiji.joint_state_consumption import (
        PersistentContentFeatureV1,
        PersistentContentStoreV1,
        QueryConditionedSelectorV1,
        feature_from_parse,
    )
    from taiji.joint_state_consumption_course import E1_COLORS
    from taiji.joint_state_value_encoder import FeatureByteBiasV1, ValueByteEncoderV1
    from taiji.joint_t1_trace_cache import JointT1TraceCacheV1
    from taiji.joint_t1_trainer import JointT1TrainerV1

    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    if bundle.get("format") != BUNDLE_FORMAT or bundle.get("version") != 1:
        raise ValueError("full E3 bundle format differs")
    if _sha256_file(ROOT / bundle["base_checkpoint"]) != bundle["base_checkpoint_sha256"]:
        raise ValueError("base checkpoint changed after bundle creation")
    if _sha256_file(ROOT / bundle["trace_cache"]) != bundle["trace_cache_sha256"]:
        raise ValueError("trace cache changed after bundle creation")
    trace_cache = JointT1TraceCacheV1.from_payload(
        torch.load(ROOT / bundle["trace_cache"], map_location="cpu", weights_only=False)
    )
    payload = torch.load(ROOT / bundle["base_checkpoint"], map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=trace_cache)
    trainer.input_trace_cache = None
    encoder = ValueByteEncoderV1(
        byte_dim=int(bundle["byte_dim"]), hidden_dim=int(bundle["feature_dim"])
    )
    bias = FeatureByteBiasV1(
        feature_dim=int(bundle["feature_dim"]), byte_vocab=int(bundle["byte_vocab"])
    )
    selector = QueryConditionedSelectorV1()
    encoder.load_state_dict(bundle["encoder_state"])
    bias.load_state_dict(bundle["bias_state"])
    selector.load_state_dict(bundle["selector_state"])
    episode = _episode()
    fact = episode.turns[3]
    query = episode.turns[4]
    with torch.no_grad():
        adapter, joint = trainer._new_episode_state(f"e1-full-e3:{episode.episode_id}:fact")
        trace, joint = trainer._append_turn(adapter, joint, fact)
        parse = trainer.producer.predict(joint, trace)
        spans = {span.role: span.text for span in parse.spans}
        if "entity" not in spans or "value" not in spans:
            raise RuntimeError("trained producer did not parse entity/value on the E1 fact")
        value = spans["value"]
        feature = feature_from_parse(
            parse,
            source_input_id=fact.frame.input_id,
            source_sha256=hashlib.sha256(bytes(fact.frame.payload)).hexdigest(),
            turn_index=0,
            revision=0,
            continuous=encoder.encode_text(value).tolist(),
        )
        if feature is None:
            raise RuntimeError("producer parse did not yield a complete content feature")
        store = PersistentContentStoreV1(()).add(feature)
        bundle_store = PersistentContentStoreV1.from_payload(bundle["store"])
        bundle_selection = selector.select(
            entity="钥匙", attribute="颜色", candidates=bundle_store.active_features()
        )
        if bundle_selection is None:
            raise RuntimeError("restored bundle store has no selectable key feature")
        query_adapter, query_joint = trainer._new_episode_state(
            f"e1-full-e3:{episode.episode_id}:query"
        )
        query_trace, query_joint = trainer._append_turn(query_adapter, query_joint, query)
        query_source = query_joint.input_bindings[-1].neural_source_id
        cache = trainer.model._joint_evidence_cache(
            query_joint, selected_source_ids=(query_source,)
        )
        base_logits, _hidden = trainer.model.next_logits(query_joint.language_state, cache=cache)
        selection = selector.select(
            entity="钥匙", attribute="颜色", candidates=store.active_features()
        )
        if selection is None:
            raise RuntimeError("selector did not select the parsed value feature")
        selected = selection.selected_feature
        conditioned = base_logits + bias(torch.tensor(selected.continuous))
        copy_feature = torch.zeros(int(bundle["feature_dim"]))
        copy_feature[E1_COLORS.index(selected.value)] = 1.0
        copy_logits = base_logits + bias(copy_feature)
        no_state_logits = base_logits + bias(torch.zeros(int(bundle["feature_dim"])))
        green = PersistentContentFeatureV1.create(
            entity="钥匙",
            attribute="颜色",
            value="绿色",
            source_input_id="full-e3-fixture:green",
            source_sha256=_sha256_text("钥匙的颜色是绿色。"),
            turn_index=0,
            revision=1,
            valid_from=0,
            confidence=0.9,
            continuous=tuple(float(item) for item in encoder.encode_text("绿色").tolist()),
        )
        swapped = store.swap("钥匙", "颜色", green)
        swapped_selection = selector.select(
            entity="钥匙", attribute="颜色", candidates=swapped.active_features()
        )
        if swapped_selection is None:
            raise RuntimeError("selector did not select the relevant swap feature")
        green_conditioned = base_logits + bias(
            torch.tensor(swapped_selection.selected_feature.continuous)
        )
        relevant_delta = float((green_conditioned - conditioned).abs().max())
    return {
        "parse_digest": content_digest(parse.to_payload()),
        "parse_features": [float(item) for item in parse.semantic_features],
        "parse_confidence": float(parse.confidence),
        "parse_ambiguity": float(parse.ambiguity),
        "parse_meta": {
            "joint_bundle_id": parse.joint_bundle_id,
            "owner_id": parse.owner_id,
            "scope_id": parse.scope_id,
            "session_id": parse.session_id,
            "task_id": parse.task_id,
            "source_joint_revision": parse.source_joint_revision,
            "producer_bundle_id": parse.producer_bundle_id,
            "producer_parameter_revision": parse.producer_parameter_revision,
            "input_id": parse.input_id,
            "source_sha256": parse.source_sha256,
            "parse_sha256": parse.parse_sha256,
            "span_lineage_digest": content_digest(
                [
                    {
                        "role": span.role,
                        "start_byte": span.start_byte,
                        "end_byte": span.end_byte,
                        "percept_event_ids": list(span.percept_event_ids),
                        "assembly_ids": list(span.assembly_ids),
                    }
                    for span in parse.spans
                ]
            ),
        },
        "parse_act": parse.act,
        "parse_status": parse.status,
        "parsed_entity": spans["entity"],
        "parsed_value": value,
        "selected_feature_id": selected.feature_id,
        "selected_value": selected.value,
        "store_digest": content_digest(store.to_payload()),
        "bundle_store_digest": content_digest(bundle_store.to_payload()),
        "bundle_selected_feature_id": bundle_selection.selected_feature.feature_id,
        "base_logits": [float(item) for item in base_logits.tolist()],
        "conditioned_logits": [float(item) for item in conditioned.tolist()],
        "copy_only_logits": [float(item) for item in copy_logits.tolist()],
        "no_state_logits": [float(item) for item in no_state_logits.tolist()],
        "conditioned_argmax_byte": int(conditioned.argmax()),
        "copy_only_argmax_byte": int(copy_logits.argmax()),
        "no_state_argmax_byte": int(no_state_logits.argmax()),
        "relevant_swap_delta": relevant_delta,
        "query_only": {
            "bindings": len(query_joint.input_bindings),
            "observations": len(query_joint.language_state.observations),
            "fact_bytes_present": any(
                raw == bytes(fact.frame.payload)
                for _, raw in query_joint.language_state.observations
            ),
        },
    }


def _compare(parent: dict, child: dict) -> dict:
    same_structure = True
    max_base = 0.0
    max_conditioned = 0.0
    for key in (
        "parse_act",
        "parse_status",
        "parsed_entity",
        "parsed_value",
        "selected_feature_id",
        "selected_value",
        "store_digest",
        "bundle_store_digest",
        "bundle_selected_feature_id",
    ):
        same_structure &= parent[key] == child[key]
    for a, b in zip(parent["base_logits"], child["base_logits"], strict=True):
        max_base = max(max_base, abs(a - b))
    for a, b in zip(parent["conditioned_logits"], child["conditioned_logits"], strict=True):
        max_conditioned = max(max_conditioned, abs(a - b))
    max_parse_features = max(
        (
            abs(a - b)
            for a, b in zip(parent["parse_features"], child["parse_features"], strict=True)
        ),
        default=0.0,
    )
    max_parse_confidence = abs(parent["parse_confidence"] - child["parse_confidence"])
    max_parse_ambiguity = abs(parent["parse_ambiguity"] - child["parse_ambiguity"])
    meta_differences = sorted(
        key for key in parent["parse_meta"] if parent["parse_meta"][key] != child["parse_meta"][key]
    )
    same_structure &= (
        parent["conditioned_argmax_byte"] == child["conditioned_argmax_byte"]
        and parent["copy_only_argmax_byte"] == child["copy_only_argmax_byte"]
        and parent["no_state_argmax_byte"] == child["no_state_argmax_byte"]
    )
    return {
        "same_parse_and_selection": bool(same_structure),
        "parse_digest_equal": parent["parse_digest"] == child["parse_digest"],
        "max_parse_feature_difference": max_parse_features,
        "max_parse_confidence_difference": max_parse_confidence,
        "max_parse_ambiguity_difference": max_parse_ambiguity,
        "parse_meta_differences": meta_differences,
        "max_base_logit_difference": max_base,
        "max_conditioned_logit_difference": max_conditioned,
        "passed": bool(
            same_structure
            and max_parse_features <= 1e-5
            and max_parse_confidence <= 1e-5
            and max_parse_ambiguity <= 1e-5
            and max_base <= TOLERANCE
            and max_conditioned <= TOLERANCE
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Full E3 zero-step candidate restore")
    parser.add_argument("--bundle-root", type=Path, default=BUNDLE_ROOT)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    args.bundle_root = args.bundle_root.resolve()
    args.report = args.report.resolve()
    if args.bundle_root.exists() or args.report.exists():
        raise ValueError("full E3 bundle/report are write-once")
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    args.bundle_root.mkdir(parents=True)
    bundle_path = args.bundle_root / "bundle.pt"
    bundle = _bundle()
    torch.save(bundle, bundle_path)
    parent = _run(bundle_path)
    child_code = (
        "import json, sys\n"
        "from scripts.training.precheck_taiji_n1_plan177_f_full_e3 import _run\n"
        "print(json.dumps(_run(__import__('pathlib').Path(sys.argv[1]))))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", child_code, str(bundle_path)],
        cwd=str(ROOT),
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        child = None
        comparison = {"passed": False, "child_error": result.stderr}
    else:
        child = json.loads(result.stdout.strip().splitlines()[-1])
        comparison = _compare(parent, child)
    strict_lineage = bool(comparison.get("parse_digest_equal"))
    status = (
        "full_e3_zero_step_passed"
        if comparison["passed"] and strict_lineage
        else (
            "full_e3_content_passed_lineage_gap"
            if comparison["passed"]
            else "full_e3_zero_step_failed"
        )
    )
    report = {
        "schema": "taiji-n177-f-full-e3-v1",
        "status": status,
        "strict_lineage_passed": strict_lineage,
        "bundle": str(bundle_path.relative_to(ROOT)),
        "bundle_sha256": _sha256_file(bundle_path),
        "bundle_source_sha256": bundle["source_sha256"],
        "parent": parent,
        "child": child,
        "comparison": comparison,
        "feature_source": "value_span_bytes_gru",
        "copy_only": {
            "main_selected_value": parent["selected_value"],
            "copy_only_argmax_byte": parent["copy_only_argmax_byte"],
            "main_argmax_byte": parent["conditioned_argmax_byte"],
            "interpretation": (
                "copy-only uses the discrete value address; it remains a lower-bound control"
            ),
        },
        "optimizer_updates": 0,
        "model_calls": 2,
        "sealed_text_read": 0,
        "checkpoint_loads": 2,
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
                "same_parse": comparison.get("same_parse_and_selection"),
                "max_conditioned_diff": comparison.get("max_conditioned_logit_difference"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if comparison["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
