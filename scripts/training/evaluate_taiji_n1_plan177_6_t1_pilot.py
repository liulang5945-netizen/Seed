"""Post-training, read-only F0/F1 scorer for the N177.6 joint T1 pilot."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from taiji.adapter import TSKV8Adapter
from taiji.config import TaijiConfig
from taiji.internalization import content_digest
from taiji.joint_input_producer import ACT_NAMES, TAG_NAMES, JointInputProducerV1
from taiji.joint_t1_course import (
    JOINT_T1_STOP_BYTES,
    JointT1EpisodeV1,
    build_joint_t1_split,
    joint_t1_course_contract,
    joint_t1_split_digest,
)
from taiji.joint_t1_trace_cache import JointT1TraceCacheV1, build_joint_t1_trace_cache
from taiji.joint_t1_trainer import (
    JOINT_T1_MODES,
    JointT1TrainerV1,
    _carry_t1_world_workspace,
)
from taiji.joint_task_state import JointTaskStateV1

DEFAULT_RUN_ROOT = REPOSITORY_ROOT / "data" / "checkpoints" / "n1_plan177_6_t1_pilot_20261027"
DEFAULT_PARENT = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan176_foundation_pilot_20261026"
    / "curriculum_v4"
    / "batch_02560.pt"
)
DEFAULT_ZERO_CORRECT_SEED_0 = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan177_5a_preflight_20261027"
    / "correct_seed_0_zero_cached.pt"
)
DEFAULT_TRAIN_CACHE = (
    REPOSITORY_ROOT
    / "data"
    / "checkpoints"
    / "n1_plan177_5a_preflight_20261027"
    / "train_trace_cache_v1.pt"
)
DEFAULT_REPORT = REPOSITORY_ROOT / "reports" / "taiji_n1_plan177_6_t1_pilot_20261027.json"
REPORT_V2 = REPOSITORY_ROOT / "reports" / "taiji_n1_plan177_5a_preflight_20261027_v2.json"
CPU_CAP_SECONDS = 2_400_000
WALL_CAP_SECONDS = 220_000
ARTIFACT_CAP_BYTES = 8 * 1024**3
MAX_GENERATION_BYTES = 64
# Leave enough budget for one longest bounded generation turn. Split-cache
# construction is separately reserved from the measured train-cache upper bound.
TURN_CPU_RESERVE_SECONDS = 600.0
TURN_WALL_RESERVE_SECONDS = 600.0
CACHE_BUILD_CPU_UPPER_SECONDS = 16_672.5
CACHE_BUILD_WALL_UPPER_SECONDS = 1_414.572
CACHE_BUILD_ARTIFACT_UPPER_BYTES = 590_512_321
REPORT_ARTIFACT_RESERVE_BYTES = 64 * 1024**2

_TAG_BEGIN = {
    TAG_NAMES.index("B_ENTITY"): "entity",
    TAG_NAMES.index("B_ATTRIBUTE"): "attribute",
    TAG_NAMES.index("B_VALUE"): "value",
    TAG_NAMES.index("B_NEGATION"): "negation",
}
_TAG_INSIDE = {
    TAG_NAMES.index("I_ENTITY"): "entity",
    TAG_NAMES.index("I_ATTRIBUTE"): "attribute",
    TAG_NAMES.index("I_VALUE"): "value",
    TAG_NAMES.index("I_NEGATION"): "negation",
}


class _EvaluationBudgetStop(RuntimeError):
    """Stop scoring at a turn boundary while preserving complete episode rows."""


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _arm_id(mode: str, seed: int) -> str:
    return f"{mode}__seed_{seed}"


def _tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _decode_tag_spans(turn) -> tuple[tuple[str, int, int], ...]:
    tags = turn.semantic_target.tag_ids
    spans = []
    active_role: str | None = None
    active_start = 0
    for index, tag in enumerate((*tags, 0)):
        begin_role = _TAG_BEGIN.get(int(tag))
        inside_role = _TAG_INSIDE.get(int(tag))
        if begin_role is not None:
            if active_role is not None:
                spans.append((active_role, active_start, index))
            active_role = begin_role
            active_start = index
        elif inside_role == active_role:
            continue
        else:
            if active_role is not None:
                spans.append((active_role, active_start, index))
            active_role = None
    return tuple(spans)


def _predicted_tag_ids(parse, byte_count: int) -> tuple[int, ...]:
    tags = [0] * byte_count
    for span in parse.spans:
        role = span.role.upper()
        if not 0 <= span.start_byte < span.end_byte <= byte_count:
            continue
        tags[span.start_byte] = TAG_NAMES.index(f"B_{role}")
        for position in range(span.start_byte + 1, span.end_byte):
            tags[position] = TAG_NAMES.index(f"I_{role}")
    return tuple(tags)


def _text_from_span(raw: bytes, span: tuple[str, int, int]) -> str:
    return raw[span[1] : span[2]].decode("utf-8")


def _normalized(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return re.sub(r"[\s，。！？、；：,.!?;:（）()\"'“”‘’]", "", text)


def _target_answer(turn) -> bytes:
    target = turn.answer_target.target_bytes
    if not target.endswith(JOINT_T1_STOP_BYTES):
        raise ValueError("evaluation target lacks the fixed stop string")
    return target[: -len(JOINT_T1_STOP_BYTES)]


def _read_cache_for_split(
    *,
    split: str,
    episodes: tuple[JointT1EpisodeV1, ...],
    adapter_checkpoint: dict[str, Any],
    run_root: Path,
) -> JointT1TraceCacheV1:
    cache_path = run_root / "evaluation" / f"{split}_trace_cache_v1.pt"
    if cache_path.is_file():
        cache = JointT1TraceCacheV1.from_payload(
            torch.load(cache_path, map_location="cpu", weights_only=False)
        )
    else:
        cache = build_joint_t1_trace_cache(
            base_adapter_checkpoint=adapter_checkpoint,
            train_episodes=episodes,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache.save(cache_path)
    cache.validate_for(
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=episodes,
    )
    return cache


def _read_cache(
    cache: JointT1TraceCacheV1,
    episode_id: str,
    turn_index: int,
):
    return cache.get(episode_id, turn_index)


def _selected_v4_cache(model, joint):
    binding = joint.input_bindings[-1]
    view = joint.content_view()
    if view.selected_candidates and all(
        item.source_status == "source_linked" for item in view.selected_candidates
    ):
        return model._joint_evidence_cache(joint)
    return model._joint_evidence_cache(
        joint,
        selected_source_ids=(binding.neural_source_id,),
    )


@torch.no_grad()
def _generate_answer(model, joint) -> tuple[bytes, JointTaskStateV1]:
    cache = _selected_v4_cache(model, joint)
    return model._generate_from_joint_cache(
        joint,
        cache,
        max_new_bytes=min(MAX_GENERATION_BYTES, model.config.max_generation_bytes),
    )


def _world_facts(joint: JointTaskStateV1) -> set[tuple[str, str, str]]:
    return {
        (obj.object_id, str(attribute), str(value))
        for obj in joint.cognitive_state.world.objects
        for attribute, value in obj.attributes
    }


def _expected_world_facts(
    joint: JointTaskStateV1,
    facts: dict[tuple[str, str], str],
    producer: JointInputProducerV1,
) -> set[tuple[str, str, str]]:
    return {
        (producer.entity_object_id(joint, entity), attribute, value)
        for (entity, attribute), value in facts.items()
    }


def _score_episode(
    *,
    trainer: JointT1TrainerV1,
    episode: JointT1EpisodeV1,
    cache: JointT1TraceCacheV1,
    state_writes: bool,
    budget_check: Callable[..., bool] | None = None,
) -> dict[str, Any]:
    model = trainer.model
    producer = trainer.producer
    scope = f"n177-t1-eval:{trainer.mode}:seed-{trainer.seed}:{episode.episode_id}"
    session = f"n177-t1-eval:seed-{trainer.seed}:{episode.episode_id}"
    joint = JointTaskStateV1.begin(
        model,
        cache.get_initial_state(episode.episode_id),
        joint_bundle_id=trainer.joint_bundle_id,
        owner_id=trainer.owner_id,
        scope_id=scope,
        session_id=session,
    )
    expected_facts: dict[tuple[str, str], str] = {}
    turn_rows = []
    query_answer_hits = []
    negative_false_writes = 0
    ambiguous_false_writes = 0
    total_answer_hits = 0
    total_turns = len(episode.turns)
    span_exact_count = 0
    span_tp = span_fp = span_fn = 0
    act_exact_count = 0
    role_exact_hits: dict[str, list[float]] = defaultdict(list)

    for turn_index, turn in enumerate(episode.turns):
        if budget_check is not None and not budget_check():
            raise _EvaluationBudgetStop("scoring stopped before the next turn")
        cached = _read_cache(cache, episode.episode_id, turn_index)
        cognitive_state = cached.cognitive_state
        if joint.input_bindings:
            cognitive_state = _carry_t1_world_workspace(joint.cognitive_state, cognitive_state)
        joint = joint.append_input(
            model,
            cached.trace,
            cognitive_state,
            resource_snapshot={
                "remaining_steps": 1,
                "max_observation_bytes": model.config.max_observation_bytes,
            },
        )
        parse = producer.predict(joint, cached.trace)
        raw = bytes(turn.frame.payload)
        gold_spans = _decode_tag_spans(turn)
        predicted_tags = _predicted_tag_ids(parse, len(raw))
        gold_tags = turn.semantic_target.tag_ids
        span_exact = predicted_tags == gold_tags
        span_exact_count += int(span_exact)
        predicted_spans = {(span.role, span.start_byte, span.end_byte) for span in parse.spans}
        for role in ("entity", "attribute", "value"):
            gold_role_spans = {
                (item_role, start, end) for item_role, start, end in gold_spans if item_role == role
            }
            if gold_role_spans:
                predicted_role_spans = {item for item in predicted_spans if item[0] == role}
                role_exact_hits[role].append(float(predicted_role_spans == gold_role_spans))
        for predicted, gold in zip(predicted_tags, gold_tags, strict=True):
            if predicted == gold and gold != 0:
                span_tp += 1
            elif predicted != gold:
                if predicted != 0:
                    span_fp += 1
                if gold != 0:
                    span_fn += 1
        gold_act = ACT_NAMES[turn.semantic_target.act_id]
        act_exact_count += int(parse.act == gold_act)

        gold_by_role = {
            role: _text_from_span(raw, (role, start, end)) for role, start, end in gold_spans
        }
        before_world = content_digest(joint.cognitive_state.world.to_payload())
        if state_writes:
            joint = producer.commit_to_joint_state(joint, cached.trace, parse, model)
        after_world = content_digest(joint.cognitive_state.world.to_payload())

        if episode.level == "F1" and gold_act in {"assertion", "correction"}:
            entity = gold_by_role.get("entity")
            attribute = gold_by_role.get("attribute")
            value = gold_by_role.get("value")
            if entity and attribute and value:
                expected_facts[(entity, attribute)] = value
        elif episode.level == "F1" and gold_act == "ambiguous":
            if before_world != after_world:
                negated = any(role == "negation" for role, _, _ in gold_spans)
                if negated:
                    negative_false_writes += 1
                else:
                    ambiguous_false_writes += 1

        generated, joint = _generate_answer(model, joint)
        try:
            generated_text = generated.decode("utf-8")
            valid_utf8 = True
        except UnicodeDecodeError:
            generated_text = generated.decode("utf-8", errors="replace")
            valid_utf8 = False
        ended = generated.endswith(JOINT_T1_STOP_BYTES)
        generated_body = generated[: -len(JOINT_T1_STOP_BYTES)] if ended else generated
        try:
            generated_body_text = generated_body.decode("utf-8")
        except UnicodeDecodeError:
            generated_body_text = generated_body.decode("utf-8", errors="replace")
            valid_utf8 = False
        target_text = _target_answer(turn).decode("utf-8")
        exact = _normalized(generated_body_text) == _normalized(target_text)
        total_answer_hits += int(exact)
        if episode.level == "F1" and gold_act == "query":
            query_answer_hits.append(int(exact))

        post_correction_state = None
        post_correction_answer = None
        unrelated_keep_state = None
        if episode.level == "F1" and turn_index == 2:
            entity = gold_by_role.get("entity")
            attribute = gold_by_role.get("attribute")
            actual_facts = _world_facts(joint)
            expected = _expected_world_facts(joint, expected_facts, producer)
            post_correction_state = float(bool(expected_facts) and actual_facts == expected)
        if episode.level == "F1" and turn_index == 3:
            post_correction_answer = float(exact)
        if episode.level == "F1" and turn_index == 4:
            distractor = gold_by_role.get("entity")
            primary = next(
                (
                    (entity, attribute, value)
                    for (entity, attribute), value in expected_facts.items()
                    if entity != distractor
                ),
                None,
            )
            if primary is not None:
                primary_entity, primary_attribute, primary_value = primary
                object_id = producer.entity_object_id(joint, primary_entity)
                actual_object = next(
                    (
                        item
                        for item in joint.cognitive_state.world.objects
                        if item.object_id == object_id
                    ),
                    None,
                )
                unrelated_keep_state = float(
                    actual_object is not None
                    and actual_object.attribute(primary_attribute) == primary_value
                )

        turn_rows.append(
            {
                "turn_index": turn_index,
                "input_id": turn.frame.input_id,
                "gold_act": gold_act,
                "predicted_act": parse.act,
                "parse_status": parse.status,
                "span_exact": span_exact,
                "entity_span_exact": (
                    None
                    if "entity" not in role_exact_hits or not role_exact_hits["entity"]
                    else role_exact_hits["entity"][-1]
                ),
                "attribute_span_exact": (
                    None
                    if "attribute" not in role_exact_hits or not role_exact_hits["attribute"]
                    else role_exact_hits["attribute"][-1]
                ),
                "value_span_exact": (
                    None
                    if "value" not in role_exact_hits or not role_exact_hits["value"]
                    else role_exact_hits["value"][-1]
                ),
                "answer_exact": exact,
                "valid_utf8": valid_utf8,
                "ended": ended,
                "post_correction_state_exact": post_correction_state,
                "post_correction_answer_exact": post_correction_answer,
                "unrelated_keep_exact": unrelated_keep_state,
                "negative_or_ambiguous_false_write": (
                    episode.level == "F1"
                    and gold_act == "ambiguous"
                    and before_world != after_world
                ),
                "prompt_utf8": raw.decode("utf-8"),
                "answer_utf8": generated_text,
                "target_utf8": target_text,
            }
        )

    span_denominator = 2 * span_tp + span_fp + span_fn
    query_rate = (
        None if episode.level != "F1" else sum(query_answer_hits) / max(1, len(query_answer_hits))
    )
    post_correction_rate = (
        None if episode.level != "F1" else turn_rows[2]["post_correction_state_exact"]
    )
    return {
        "episode_id": episode.episode_id,
        "family_id": episode.family_id,
        "level": episode.level,
        "partition": episode.partition,
        "turn_count": total_turns,
        "answer_exact_rate": total_answer_hits / max(1, total_turns),
        "valid_utf8_rate": sum(int(item["valid_utf8"]) for item in turn_rows) / max(1, total_turns),
        "stop_marker_rate": sum(int(item["ended"]) for item in turn_rows) / max(1, total_turns),
        "span_exact_rate": span_exact_count / max(1, total_turns),
        "act_exact_rate": act_exact_count / max(1, total_turns),
        "entity_span_exact_rate": (
            None
            if not role_exact_hits["entity"]
            else sum(role_exact_hits["entity"]) / len(role_exact_hits["entity"])
        ),
        "attribute_span_exact_rate": (
            None
            if not role_exact_hits["attribute"]
            else sum(role_exact_hits["attribute"]) / len(role_exact_hits["attribute"])
        ),
        "value_span_exact_rate": (
            None
            if not role_exact_hits["value"]
            else sum(role_exact_hits["value"]) / len(role_exact_hits["value"])
        ),
        "span_f1": 0.0 if span_denominator == 0 else 2 * span_tp / span_denominator,
        "f1_query_answer_exact_rate": query_rate,
        "f1_post_correction_state_exact": post_correction_rate,
        "f1_post_correction_answer_exact": (
            None
            if episode.level != "F1"
            else float(turn_rows[3]["post_correction_answer_exact"] or 0.0)
        ),
        "f1_state_query_score": (
            None
            if query_rate is None or post_correction_rate is None
            else (query_rate + post_correction_rate) / 2
        ),
        "f1_unrelated_keep_exact": (
            None if episode.level != "F1" else turn_rows[4]["unrelated_keep_exact"]
        ),
        "f1_false_write_rate": (
            None if episode.level != "F1" else (negative_false_writes + ambiguous_false_writes) / 2
        ),
        "turns": turn_rows,
    }


def _macro_family_metric(episodes: list[dict[str, Any]], key: str) -> float | None:
    by_family: dict[str, list[float]] = defaultdict(list)
    for episode in episodes:
        value = episode.get(key)
        if value is not None:
            by_family[episode["family_id"]].append(float(value))
    if not by_family:
        return None
    return mean(mean(values) for values in by_family.values())


def _score_split(
    *,
    trainer: JointT1TrainerV1,
    episodes: tuple[JointT1EpisodeV1, ...],
    cache: JointT1TraceCacheV1,
    state_writes: bool,
    budget_check: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    episode_scores = []
    for episode in episodes:
        if budget_check is not None and not budget_check():
            break
        try:
            episode_score = _score_episode(
                trainer=trainer,
                episode=episode,
                cache=cache,
                state_writes=state_writes,
                budget_check=budget_check,
            )
        except _EvaluationBudgetStop:
            break
        episode_scores.append(episode_score)
    metric_keys = (
        "answer_exact_rate",
        "valid_utf8_rate",
        "stop_marker_rate",
        "span_exact_rate",
        "act_exact_rate",
        "entity_span_exact_rate",
        "attribute_span_exact_rate",
        "value_span_exact_rate",
        "span_f1",
        "f1_query_answer_exact_rate",
        "f1_post_correction_state_exact",
        "f1_post_correction_answer_exact",
        "f1_state_query_score",
        "f1_unrelated_keep_exact",
        "f1_false_write_rate",
    )
    by_level = {}
    for level in ("F0", "F1"):
        level_episodes = [item for item in episode_scores if item["level"] == level]
        by_level[level] = {
            "episodes": len(level_episodes),
            "families": len({item["family_id"] for item in level_episodes}),
            "family_macro": {key: _macro_family_metric(level_episodes, key) for key in metric_keys},
        }
    return {
        "split": episodes[0].partition,
        "level": "mixed",
        "episodes": len(episodes),
        "episodes_scored": len(episode_scores),
        "partial": len(episode_scores) != len(episodes),
        "families": len({item.family_id for item in episodes}),
        "state_writes_enabled": state_writes,
        "family_macro": {key: _macro_family_metric(episode_scores, key) for key in metric_keys},
        "by_level": by_level,
        "episodes_detail": episode_scores,
    }


def _run_evaluation() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--train-cache", type=Path, default=DEFAULT_TRAIN_CACHE)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    state_path = args.run_root / "run_state.json"
    if not state_path.is_file():
        raise FileNotFoundError("N177.6 scorer requires the run_state checkpoint ledger")
    run_state = json.loads(state_path.read_text(encoding="utf-8"))
    if run_state.get("status") not in {
        "training_complete_evaluation_pending",
        "budget_stop",
        "evaluation_in_progress",
    }:
        raise ValueError("N177.6 scorer runs only after training stops at a fixed boundary")
    if not args.parent.is_file() or not args.train_cache.is_file():
        raise FileNotFoundError("N177.6 scorer requires the fixed parent and train cache")

    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    total_cpu_before = float(run_state["process_cpu_seconds"])
    total_wall_before = float(run_state["wall_seconds"])

    evaluation_context = {"split": "startup", "arm_id": None, "checks": 0}

    def current_artifact_bytes() -> int:
        return (
            _tree_bytes(args.run_root)
            + _tree_bytes(args.train_cache.parent)
            + (args.report.stat().st_size if args.report.exists() else 0)
            + (REPORT_V2.stat().st_size if REPORT_V2.exists() else 0)
        )

    def budget_check(*, cache_build: bool = False) -> bool:
        evaluation_context["checks"] += 1
        current_cpu = total_cpu_before + time.process_time() - start_cpu
        current_wall = total_wall_before + time.perf_counter() - start_wall
        current_artifacts = current_artifact_bytes()
        reserve_cpu = CACHE_BUILD_CPU_UPPER_SECONDS if cache_build else TURN_CPU_RESERVE_SECONDS
        reserve_wall = CACHE_BUILD_WALL_UPPER_SECONDS if cache_build else TURN_WALL_RESERVE_SECONDS
        reserve_artifacts = REPORT_ARTIFACT_RESERVE_BYTES
        if cache_build:
            reserve_artifacts += CACHE_BUILD_ARTIFACT_UPPER_BYTES
        if evaluation_context["checks"] % 32 == 0:
            run_state["status"] = "evaluation_in_progress"
            run_state["process_cpu_seconds"] = current_cpu
            run_state["wall_seconds"] = current_wall
            run_state["artifact_bytes"] = current_artifacts
            run_state["evaluation_progress"] = dict(evaluation_context)
            _atomic_json(state_path, run_state)
        if current_cpu + reserve_cpu >= CPU_CAP_SECONDS:
            evaluation_context["budget_stop_reason"] = "CPU cap safety reserve reached"
            return False
        if current_wall + reserve_wall >= WALL_CAP_SECONDS:
            evaluation_context["budget_stop_reason"] = "wall cap safety reserve reached"
            return False
        if current_artifacts + reserve_artifacts >= ARTIFACT_CAP_BYTES:
            evaluation_context["budget_stop_reason"] = "artifact cap safety reserve reached"
            return False
        return True

    parent_sha = _sha256_file(args.parent)
    adapter = TSKV8Adapter(TaijiConfig(seed=17751027), episode_id="n177-t1:adapter-baseline")
    adapter_checkpoint = adapter.native_checkpoint()
    train_episodes = build_joint_t1_split("train")
    train_cache = JointT1TraceCacheV1.from_payload(
        torch.load(args.train_cache, map_location="cpu", weights_only=False)
    )
    train_cache.validate_for(
        base_adapter_checkpoint=adapter_checkpoint,
        train_episodes=train_episodes,
    )
    evaluation_results = []
    completed = set(run_state.get("completed_arms", []))
    arm_order = tuple((mode, seed) for seed in range(3) for mode in JOINT_T1_MODES)

    evaluation_budget_stop = False
    for split in ("query", "holdout"):
        evaluation_context["split"] = split
        evaluation_context["arm_id"] = None
        episodes = build_joint_t1_split(split)
        split_cache_path = args.run_root / "evaluation" / f"{split}_trace_cache_v1.pt"
        if not budget_check(cache_build=not split_cache_path.is_file()):
            evaluation_budget_stop = True
            break
        cache = _read_cache_for_split(
            split=split,
            episodes=episodes,
            adapter_checkpoint=adapter_checkpoint,
            run_root=args.run_root,
        )
        cache.validate_for(
            base_adapter_checkpoint=adapter_checkpoint,
            train_episodes=episodes,
        )
        if not budget_check():
            evaluation_budget_stop = True
            break
        split_results = []
        for mode, seed in arm_order:
            if _arm_id(mode, seed) not in completed:
                continue
            evaluation_context["arm_id"] = _arm_id(mode, seed)
            checkpoint_info = run_state.get("latest_checkpoint_by_arm", {}).get(_arm_id(mode, seed))
            if not checkpoint_info:
                continue
            checkpoint_path = Path(checkpoint_info["path"])
            trainer = JointT1TrainerV1.from_native_checkpoint(
                torch.load(checkpoint_path, map_location="cpu", weights_only=False),
                input_trace_cache=train_cache,
            )
            result = _score_split(
                trainer=trainer,
                episodes=episodes,
                cache=cache,
                state_writes=bool(trainer.policy["cognitive_state_writes"]),
                budget_check=budget_check,
            )
            split_results.append(
                {
                    "mode": mode,
                    "seed": seed,
                    "checkpoint": str(checkpoint_path),
                    "update_cursor": trainer.update_cursor,
                    **result,
                }
            )
            del trainer
            run_state["status"] = "evaluation_in_progress"
            run_state["process_cpu_seconds"] = total_cpu_before + time.process_time() - start_cpu
            run_state["wall_seconds"] = total_wall_before + time.perf_counter() - start_wall
            run_state["artifact_bytes"] = current_artifact_bytes()
            run_state["evaluation_progress"] = {
                "split": split,
                "arm_id": _arm_id(mode, seed),
                "arms_scored": len(split_results),
            }
            _atomic_json(state_path, run_state)
            if result["partial"]:
                evaluation_budget_stop = True
                break
        evaluation_results.append({"split": split, "arms": split_results})
        if evaluation_budget_stop:
            break

    # State-write ablation: same Correct endpoint and holdout episodes, with writes disabled.
    holdout_episodes = build_joint_t1_split("holdout")
    holdout_cache = None
    if len(evaluation_results) > 1 and not evaluation_budget_stop:
        holdout_cache = _read_cache_for_split(
            split="holdout",
            episodes=holdout_episodes,
            adapter_checkpoint=adapter_checkpoint,
            run_root=args.run_root,
        )
    state_write_diagnostics = []
    for seed in (range(3) if holdout_cache is not None and not evaluation_budget_stop else ()):
        arm_id = _arm_id("correct", seed)
        evaluation_context["split"] = "holdout_state_write_diagnostic"
        evaluation_context["arm_id"] = arm_id
        if arm_id not in completed:
            continue
        checkpoint_info = run_state.get("latest_checkpoint_by_arm", {}).get(arm_id)
        if not checkpoint_info:
            continue
        checkpoint_path = Path(checkpoint_info["path"])
        trainer = JointT1TrainerV1.from_native_checkpoint(
            torch.load(checkpoint_path, map_location="cpu", weights_only=False),
            input_trace_cache=train_cache,
        )
        read_only = _score_split(
            trainer=trainer,
            episodes=holdout_episodes,
            cache=holdout_cache,
            state_writes=False,
            budget_check=budget_check,
        )
        if read_only["partial"]:
            evaluation_budget_stop = True
            break
        state_write_diagnostics.append(
            {
                "seed": seed,
                "checkpoint": str(checkpoint_path),
                "write_enabled_holdout": next(
                    item
                    for item in evaluation_results[1]["arms"]
                    if item["mode"] == "correct" and item["seed"] == seed
                )["family_macro"],
                "write_disabled_holdout": read_only["family_macro"],
            }
        )
        del trainer

    f0_retention = []
    for seed in (range(3) if holdout_cache is not None and not evaluation_budget_stop else ()):
        arm_id = _arm_id("correct", seed)
        endpoint = next(
            (
                item
                for item in evaluation_results[1]["arms"]
                if item["mode"] == "correct" and item["seed"] == seed
            ),
            None,
        )
        if arm_id not in completed or endpoint is None:
            continue
        zero_path = (
            DEFAULT_ZERO_CORRECT_SEED_0
            if seed == 0
            else args.run_root / arm_id / "update_000000_zero.pt"
        )
        zero_trainer = JointT1TrainerV1.from_native_checkpoint(
            torch.load(zero_path, map_location="cpu", weights_only=False),
            input_trace_cache=train_cache,
        )
        f0_episodes = tuple(item for item in holdout_episodes if item.level == "F0")
        zero_scores = _score_split(
            trainer=zero_trainer,
            episodes=f0_episodes,
            cache=holdout_cache,
            state_writes=True,
            budget_check=budget_check,
        )
        if zero_scores["partial"]:
            evaluation_budget_stop = True
            break
        zero_exact = zero_scores["family_macro"]["answer_exact_rate"]
        endpoint_exact = endpoint["by_level"]["F0"]["family_macro"]["answer_exact_rate"]
        f0_retention.append(
            {
                "seed": seed,
                "zero_f0_holdout_answer_exact": zero_exact,
                "final_f0_holdout_answer_exact": endpoint_exact,
                "delta": endpoint_exact - zero_exact,
                "passed_drop_limit": endpoint_exact >= zero_exact - 0.03,
            }
        )
        del zero_trainer

    holdout_arms = evaluation_results[1]["arms"] if len(evaluation_results) > 1 else []
    fully_scored_holdout_arms = [
        item for item in holdout_arms if item["episodes_scored"] == item["episodes"]
    ]
    f1_scores = {
        (item["mode"], item["seed"]): item["by_level"]["F1"]["family_macro"]
        for item in fully_scored_holdout_arms
    }
    learning_contrasts = {}
    for control in ("no_update", "parser_permuted"):
        deltas = []
        for seed in range(3):
            correct = f1_scores.get(("correct", seed))
            baseline = f1_scores.get((control, seed))
            if correct is None or baseline is None:
                continue
            correct_score = correct["f1_state_query_score"]
            baseline_score = baseline["f1_state_query_score"]
            if correct_score is not None and baseline_score is not None:
                deltas.append(
                    {
                        "seed": seed,
                        "delta": correct_score - baseline_score,
                        "correct": correct_score,
                        "control": baseline_score,
                    }
                )
        values = [item["delta"] for item in deltas]
        learning_contrasts[control] = {
            "paired_seed_deltas": deltas,
            "median_delta": median(values) if values else None,
            "positive_seed_count": sum(value > 0.0 for value in values),
            "minimum_delta": min(values) if values else None,
            "passed": (
                len(values) == 3
                and median(values) >= 0.10
                and sum(value > 0.0 for value in values) >= 2
                and min(values) >= -0.05
            ),
        }
    learning_gate_passed = len(learning_contrasts) == 2 and all(
        item["passed"] for item in learning_contrasts.values()
    )
    correct_f0 = [
        item["by_level"]["F0"]["family_macro"]
        for item in fully_scored_holdout_arms
        if item["mode"] == "correct"
    ]
    correct_f1 = [
        item["by_level"]["F1"]["family_macro"]
        for item in fully_scored_holdout_arms
        if item["mode"] == "correct"
    ]
    absolute_means = {
        "f0_answer_exact": (
            mean(item["answer_exact_rate"] for item in correct_f0) if correct_f0 else None
        ),
        "f0_valid_utf8": (
            mean(item["valid_utf8_rate"] for item in correct_f0) if correct_f0 else None
        ),
        "f0_stop_marker": (
            mean(item["stop_marker_rate"] for item in correct_f0) if correct_f0 else None
        ),
        "f1_entity_span_exact": (
            mean(item["entity_span_exact_rate"] for item in correct_f1) if correct_f1 else None
        ),
        "f1_attribute_span_exact": (
            mean(item["attribute_span_exact_rate"] for item in correct_f1) if correct_f1 else None
        ),
        "f1_value_span_exact": (
            mean(item["value_span_exact_rate"] for item in correct_f1) if correct_f1 else None
        ),
        "f1_post_correction_state_exact": (
            mean(item["f1_post_correction_state_exact"] for item in correct_f1)
            if correct_f1
            else None
        ),
        "f1_post_correction_answer_exact": (
            mean(item["f1_post_correction_answer_exact"] for item in correct_f1)
            if correct_f1
            else None
        ),
        "f1_unrelated_keep_exact": (
            mean(item["f1_unrelated_keep_exact"] for item in correct_f1) if correct_f1 else None
        ),
        "f1_false_write_rate": (
            mean(item["f1_false_write_rate"] for item in correct_f1) if correct_f1 else None
        ),
    }
    absolute_gates = {
        "f0_answer_exact": len(correct_f0) == 3
        and absolute_means["f0_answer_exact"] is not None
        and absolute_means["f0_answer_exact"] >= 0.85,
        "f0_valid_utf8": len(correct_f0) == 3
        and absolute_means["f0_valid_utf8"] is not None
        and absolute_means["f0_valid_utf8"] >= 0.99,
        "f0_stop_marker": len(correct_f0) == 3
        and absolute_means["f0_stop_marker"] is not None
        and absolute_means["f0_stop_marker"] >= 0.95,
        "f1_entity_span_exact": len(correct_f1) == 3
        and absolute_means["f1_entity_span_exact"] is not None
        and absolute_means["f1_entity_span_exact"] >= 0.90,
        "f1_attribute_span_exact": len(correct_f1) == 3
        and absolute_means["f1_attribute_span_exact"] is not None
        and absolute_means["f1_attribute_span_exact"] >= 0.90,
        "f1_value_span_exact": len(correct_f1) == 3
        and absolute_means["f1_value_span_exact"] is not None
        and absolute_means["f1_value_span_exact"] >= 0.90,
        "f1_post_correction_state_exact": len(correct_f1) == 3
        and absolute_means["f1_post_correction_state_exact"] is not None
        and absolute_means["f1_post_correction_state_exact"] >= 0.90,
        "f1_post_correction_answer_exact": len(correct_f1) == 3
        and absolute_means["f1_post_correction_answer_exact"] is not None
        and absolute_means["f1_post_correction_answer_exact"] >= 0.90,
        "f1_unrelated_keep_exact": len(correct_f1) == 3
        and absolute_means["f1_unrelated_keep_exact"] is not None
        and absolute_means["f1_unrelated_keep_exact"] >= 0.95,
        "f1_false_write_rate": len(correct_f1) == 3
        and absolute_means["f1_false_write_rate"] is not None
        and absolute_means["f1_false_write_rate"] <= 0.05,
    }
    f0_retention_passed = len(f0_retention) == 3 and all(
        item["passed_drop_limit"] for item in f0_retention
    )
    architecture_attribution = {
        "information_path": "frozen Adapter source traces are hash-bound; each update rebuilds V4 TaskState and producer parse/state from the current arm",
        "credit_path": "preflight proved the declared gradient receivers; formal update outcomes are recorded per arm",
        "state_lifecycle": "cached Adapter snapshots carry producer-owned world/workspace forward between turns",
        "learning_gain_observed": learning_gate_passed,
        "f0_retention_observed": f0_retention_passed,
        "conclusion": (
            "Correct supervision has fixed-slice behavioral gain over both controls; this supports the T1 path only."
            if learning_gate_passed
            else "Gradient paths are reachable, but fixed-slice behavioral gain is not established; objective, representation and optimization remain competing causes."
        ),
    }

    current_cpu = total_cpu_before + time.process_time() - start_cpu
    current_wall = total_wall_before + time.perf_counter() - start_wall
    artifact_bytes = (
        _tree_bytes(args.run_root)
        + _tree_bytes(args.train_cache.parent)
        + (REPORT_V2.stat().st_size if REPORT_V2.exists() else 0)
        + (args.report.stat().st_size if args.report.exists() else 0)
    )
    report = {
        "format": "taiji-n177-6-t1-evaluation-v1",
        "status": (
            "evaluation_budget_stop"
            if evaluation_budget_stop
            else "complete" if len(completed) == 15 else "partial_training_evaluation"
        ),
        "parent_v4_sha256": parent_sha,
        "course_sha256": str(joint_t1_course_contract()["course_sha256"]),
        "train_split_sha256": joint_t1_split_digest(train_episodes),
        "train_trace_cache_sha256": train_cache.cache_sha256,
        "arms_completed": len(completed),
        "optimizer_updates": run_state.get("optimizer_updates", 0),
        "resources": {
            "process_cpu_seconds": current_cpu,
            "wall_seconds": current_wall,
            "artifact_bytes": artifact_bytes,
            "hard_caps": {
                "process_cpu_seconds": CPU_CAP_SECONDS,
                "wall_seconds": WALL_CAP_SECONDS,
                "artifact_bytes": ARTIFACT_CAP_BYTES,
            },
        },
        "evaluation_stop_reason": evaluation_context.get("budget_stop_reason"),
        "fully_scored_holdout_arms": len(fully_scored_holdout_arms),
        "query_and_holdout": evaluation_results,
        "state_write_diagnostics": state_write_diagnostics,
        "learning_contrasts": learning_contrasts,
        "absolute_means": absolute_means,
        "absolute_gates": absolute_gates,
        "f0_retention": f0_retention,
        "f0_retention_passed": f0_retention_passed,
        "architecture_attribution": architecture_attribution,
        "sealed_reads": 0,
        "parameters_updated_during_scoring": False,
    }
    _atomic_json(args.report, report)
    run_state["status"] = (
        "evaluation_budget_stop"
        if evaluation_budget_stop
        else "evaluation_complete" if len(completed) == 15 else "partial_evaluation_complete"
    )
    run_state["process_cpu_seconds"] = current_cpu
    run_state["wall_seconds"] = current_wall
    run_state["artifact_bytes"] = artifact_bytes + len(
        json.dumps(report, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    run_state["evaluation_report"] = str(args.report)
    run_state["evaluation_stop_reason"] = evaluation_context.get("budget_stop_reason")
    _atomic_json(state_path, run_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_evaluation())
