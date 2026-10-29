"""N177 F1 一次性 holdout 评分（只读一次；读后不许重跑）。

候选：冻结 producer/V4 + 训练后的 value encoder 与 byte bias。答案按 F1 合同 §2.5 生成
（冻结 V4 贪心逐步生成，每步加同一 `bias(feature)`）。

**特征不喂课程真值**：事实由 producer 自己解析，写进 `PersistentContentStoreV1`，查询时
由 selector 选出被服务的那条事实，再取它的值过 encoder。否则答案是被老师喂出来的，
`state_exact` 也就成了"课程表抄得对不对"。

指标按 F 合同 §5：answer_exact / value_exact / state_exact / relevant_swap_delta /
irrelevant_swap_keep / cut_drop / rescue_gain，以及 copy_only_match / no_state_gap /
credit_gap；逐 episode 内轮次评分 → family macro → 三 seed 中位。数值门见 F1 冻结提案 §4。
`old_retention`（N176 F0 16 题）本件未实现，报告写 `not_run`。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training.run_taiji_n1_plan177_f_f0b import (  # noqa: E402
    ARMS,
    _Arm,
    _rel,
    _sha256_file,
    load_trainer,
)
from taiji.internalization import content_digest  # noqa: E402
from taiji.joint_state_consumption import (  # noqa: E402
    PersistentContentFeatureV1,
    PersistentContentStoreV1,
    QueryConditionedSelectorV1,
)
from taiji.joint_state_consumption_course import (  # noqa: E402
    E1_ATTRIBUTE,
    E1_COLORS,
    E1_EPISODES_PER_FAMILY,
    build_episode,
    e1_families,
)

RUN_ROOT = ROOT / "data/checkpoints/n1_plan177_f_f1_20261029"
REPORT = ROOT / "reports/taiji_n1_plan177_f_f1_holdout_20261029.json"
MAX_NEW_BYTES = 64
GATES = {
    "answer_exact": 0.85,
    "value_exact": 0.90,
    "state_exact": 0.90,
    "relevant_swap_delta": 0.30,
    "irrelevant_swap_keep": 0.95,
    "cut_drop": 0.30,
    "rescue_gain": 0.20,
    "main_minus_copy_only": 0.10,
    "main_minus_no_state": 0.10,
    "credit_gap": 0.10,
}
METRICS = (
    "answer_exact",
    "value_exact",
    "state_exact",
    "relevant_swap_delta",
    "irrelevant_swap_keep",
    "cut_drop",
    "rescue_gain",
)


def normalize(raw: bytes) -> str:
    text = raw.decode("utf-8", errors="replace")
    if "<|end|>" in text:
        text = text.split("<|end|>", 1)[0]
    return text.strip().strip("。.!！?？ ")


def answer_colour(text: str) -> str | None:
    """答案是一整句话；指标要比的是它说的是哪个颜色，不是整句相等。"""

    for colour in E1_COLORS:
        if colour in text:
            return colour
    return None


def exchanged(value: str) -> str:
    """同一实体换成另一个合法值：按冻结色表顺序取下一个同名外的值。"""

    index = E1_COLORS.index(value)
    return E1_COLORS[(index + 1) % len(E1_COLORS)]


def family_macro(rows: list[dict[str, Any]]) -> dict[str, float]:
    """每个指标各自的分母：问句类指标只在有问句的 episode 上平均。"""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["family_id"], []).append(row)
    result: dict[str, float] = {}
    for key in METRICS:
        family_values = []
        for family_rows in grouped.values():
            values = [row[key] for row in family_rows if row[key] is not None]
            if values:
                family_values.append(statistics.mean(values))
        result[key] = statistics.mean(family_values) if family_values else 0.0
    return result


def contrasts(
    macro_by_arm: dict[str, dict[str, float]], seeds: tuple[int, ...]
) -> dict[str, float]:
    def median(key: str, arm: str) -> float:
        return statistics.median(macro_by_arm[f"{arm}__seed_{seed}"][key] for seed in seeds)

    main = median("answer_exact", "main_credit_on")
    return {
        "answer_exact": main,
        "main_minus_copy_only": main - median("answer_exact", "copy_only"),
        "main_minus_no_state": main - median("answer_exact", "no_state"),
        "credit_gap": main - median("answer_exact", "main_credit_stop"),
    }


def gate_report(scores: dict[str, float]) -> dict[str, Any]:
    return {
        key: {
            "threshold": threshold,
            "value": scores.get(key),
            "passed": scores.get(key) is not None and scores[key] >= threshold,
        }
        for key, threshold in GATES.items()
    }


def _store_from_parse(
    store: PersistentContentStoreV1, spans: dict[str, str], *, turn_index: int, revision: int
) -> PersistentContentStoreV1:
    entity = spans.get("entity")
    attribute = spans.get("attribute")
    value = spans.get("value")
    if not entity or attribute != E1_ATTRIBUTE or value not in E1_COLORS:
        return store
    feature = PersistentContentFeatureV1.create(
        entity=entity,
        attribute=attribute,
        value=value,
        source_input_id=f"f1-eval:{entity}:{turn_index}",
        source_sha256=content_digest({"entity": entity, "value": value, "turn": turn_index}),
        turn_index=turn_index,
        revision=revision,
        valid_from=turn_index,
        confidence=1.0,
        continuous=[0.0] * 64,
    )
    return store.add(feature)


def _feature(arm: _Arm, value: str | None) -> torch.Tensor:
    if value is None:
        return torch.zeros(arm.feature_dim)
    return arm.feature({"value": value})


def _answer(trainer: Any, arm: _Arm, joint: Any, feature: torch.Tensor) -> str:
    with torch.no_grad():
        source = joint.input_bindings[-1].neural_source_id
        cache = trainer.model._joint_evidence_cache(joint, selected_source_ids=(source,))
        produced, _updated = trainer.model._generate_from_joint_cache(
            joint,
            cache,
            max_new_bytes=MAX_NEW_BYTES,
            logit_bias=arm.bias(feature),
        )
    return normalize(produced)


def _replay(trainer: Any, episode: Any) -> list[dict[str, Any]]:
    rows = []
    with torch.no_grad():
        adapter, joint = trainer._new_episode_state(f"f1-eval:{episode.episode_id}")
        for turn in episode.turns:
            trace, joint = trainer._append_turn(adapter, joint, turn)
            parse = trainer.producer.predict(joint, trace)
            spans = {span.role: span.text for span in getattr(parse, "spans", ())}
            rows.append(
                {
                    "turn": turn,
                    "parse": parse,
                    "spans": spans,
                    "joint": joint,
                    "act": getattr(parse, "act", None),
                }
            )
    return rows


def score_episode(
    trainer: Any, arm: _Arm, selector: QueryConditionedSelectorV1, episode: Any
) -> dict[str, Any]:
    rows = _replay(trainer, episode)
    store = PersistentContentStoreV1(())
    answer_hits = 0
    query_answer_hits = state_hits = 0
    relevant_hits = irrelevant_hits = cut_hits = rescue_hits = 0
    query_turns = irrelevant_turns = query_unparsed = 0
    for index, row in enumerate(rows):
        store = _store_from_parse(store, row["spans"], turn_index=index, revision=index)
        truth = normalize(bytes(row["turn"].answer_target.target_bytes))
        entity = row["spans"].get("entity")
        current_feature = store.current(entity, E1_ATTRIBUTE) if entity is not None else None
        current = current_feature.value if current_feature is not None else None
        produced = _answer(trainer, arm, row["joint"], _feature(arm, current))
        answer_hits += produced == truth
        if row["act"] != "query":
            continue
        if entity is None:
            query_unparsed += 1
            continue
        expected_value = answer_colour(truth)
        query_turns += 1
        produced_colour = answer_colour(produced)
        query_answer_hits += produced_colour is not None and produced_colour == expected_value
        state_hits += current is not None and current == expected_value
        baseline = produced_colour
        if current is not None:
            swapped_value = exchanged(current)
            swapped = answer_colour(
                _answer(trainer, arm, row["joint"], _feature(arm, swapped_value))
            )
            relevant_hits += swapped == swapped_value and swapped != baseline
        other = next(
            (feature for feature in store.active_features() if feature.entity != entity), None
        )
        if other is not None:
            irrelevant_turns += 1
            swapped_store = store.add(
                PersistentContentFeatureV1.create(
                    entity=other.entity,
                    attribute=other.attribute,
                    value=exchanged(other.value),
                    source_input_id=f"f1-eval:irrelevant:{index}",
                    source_sha256=content_digest({"irrelevant": index}),
                    turn_index=index,
                    revision=index + 1000,
                    valid_from=index,
                    confidence=1.0,
                    continuous=[0.0] * 64,
                )
            )
            other_selection = selector.select(
                entity=entity, attribute=E1_ATTRIBUTE, candidates=swapped_store.active_features()
            )
            kept_value = (
                other_selection.selected_feature.value if other_selection is not None else current
            )
            kept = answer_colour(_answer(trainer, arm, row["joint"], _feature(arm, kept_value)))
            irrelevant_hits += kept == baseline
        cut = answer_colour(_answer(trainer, arm, row["joint"], _feature(arm, None)))
        cut_hits += cut != baseline
        rescue = answer_colour(_answer(trainer, arm, row["joint"], _feature(arm, current)))
        rescue_hits += rescue == baseline

    def ratio(count: int, denominator: int) -> float | None:
        return count / denominator if denominator else None

    return {
        "family_id": episode.family_id,
        "turns": len(rows),
        "query_turns": query_turns,
        "irrelevant_turns": irrelevant_turns,
        "query_unparsed": query_unparsed,
        "answer_exact": ratio(answer_hits, len(rows)),
        "value_exact": ratio(query_answer_hits, query_turns),
        "state_exact": ratio(state_hits, query_turns),
        "relevant_swap_delta": ratio(relevant_hits, query_turns),
        "irrelevant_swap_keep": ratio(irrelevant_hits, irrelevant_turns),
        "cut_drop": ratio(cut_hits, query_turns),
        "rescue_gain": ratio(rescue_hits, query_turns),
    }


def holdout_episodes() -> list[Any]:
    return [
        build_episode(family, variant)
        for family in e1_families()["holdout"]
        for variant in range(E1_EPISODES_PER_FAMILY)
    ]


def assert_runs_complete(run_root: Path, seeds: tuple[int, ...] = (0, 1, 2)) -> None:
    """任何一臂没跑完就拒跑——而且要在物化 holdout、写 attempt 之前拒。"""

    missing = []
    for seed in seeds:
        for arm_name in ARMS:
            state_path = run_root / f"{arm_name}__seed_{seed}" / "run_state.json"
            if not state_path.is_file():
                missing.append(f"{arm_name}__seed_{seed}:missing")
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("status") != "completed":
                missing.append(f"{arm_name}__seed_{seed}:{state.get('status')}")
    if missing:
        raise ValueError(f"F1 runs are not all completed: {missing}")


def arm_scores(trainer: Any, arm_name: str, seed: int, episodes: list[Any]) -> dict[str, Any]:
    state_path = RUN_ROOT / f"{arm_name}__seed_{seed}" / "run_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "completed":
        raise ValueError(f"F1 run {arm_name}__seed_{seed} is not completed: {state.get('status')}")
    endpoint = ROOT / state["endpoint"]
    arm = _Arm.from_checkpoint(torch.load(endpoint, map_location="cpu", weights_only=False))
    selector = QueryConditionedSelectorV1()
    per_episode = [score_episode(trainer, arm, selector, episode) for episode in episodes]
    return {
        "arm": arm_name,
        "seed": seed,
        "endpoint": _rel(endpoint),
        "endpoint_sha256": _sha256_file(endpoint),
        "macro": family_macro(per_episode),
        "per_episode": per_episode,
    }


VERDICT_PRECEDENCE = (
    "FAIL_COPY_ONLY",
    "FAIL_NO_STATE_EFFECT",
    "FAIL_NO_LEARNING",
    "FAIL_NO_GENERALIZATION",
)


def verdict(scores: dict[str, float], retention_pp: float | None) -> dict[str, Any]:
    """F 合同 §5.1 的出口判定；规则先于任何 holdout 读数写死。

    先判"没超出复制下界"，再判"特征没用"，再判"信用没到"，最后才判"只学会了 train"。
    `old_retention` 没测就不能算 PASS——合同把它写进 PASS 的合取项，缺项判 INCONCLUSIVE。
    """

    main = scores.get("answer_exact")
    copy_only = scores.get("copy_only_median")
    fired: set[str] = set()
    if main is not None and copy_only is not None and copy_only >= main:
        fired.add("FAIL_COPY_ONLY")
    if (
        scores.get("main_minus_no_state", 0.0) < GATES["main_minus_no_state"]
        or scores.get("relevant_swap_delta", 0.0) < GATES["relevant_swap_delta"]
    ):
        fired.add("FAIL_NO_STATE_EFFECT")
    if scores.get("credit_gap", 0.0) < GATES["credit_gap"]:
        fired.add("FAIL_NO_LEARNING")
    if main is not None and main < GATES["answer_exact"]:
        fired.add("FAIL_NO_GENERALIZATION")
    all_gates = all(
        scores.get(key) is not None and scores[key] >= threshold for key, threshold in GATES.items()
    )
    retention_ok = retention_pp is not None and retention_pp <= 3.0
    reasons = [code for code in VERDICT_PRECEDENCE if code in fired]
    if all_gates and retention_ok and not fired:
        label = "PASS_LEARNED_VALUE"
    elif reasons:
        label = reasons[0]
    else:
        label = "INCONCLUSIVE"
    return {
        "verdict": label,
        "reasons": reasons,
        "all_numeric_gates_passed": all_gates,
        "old_retention_pp": retention_pp,
        "old_retention_measured": retention_pp is not None,
        "pass_blocked_by": (
            []
            if label == "PASS_LEARNED_VALUE"
            else (
                ["old_retention_not_run"]
                if all_gates and not fired and retention_pp is None
                else reasons or ["effect_below_gate"]
            )
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 F1 holdout 一次性评分")
    parser.add_argument("--report", type=Path, default=REPORT)
    parser.add_argument(
        "--smoke-train",
        action="store_true",
        help="只在 train 的一条 episode 上跑通链路（不碰 holdout、不写 holdout 报告）",
    )
    parser.add_argument(
        "--smoke-checkpoint",
        type=Path,
        default=ROOT / "data/checkpoints/n1_plan177_f_f0b_20261029/copy_only__seed_0/zero.pt",
    )
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.smoke_train:
        from taiji.joint_state_consumption_course import build_train_episodes

        trainer = load_trainer()
        arm = _Arm.from_checkpoint(
            torch.load(args.smoke_checkpoint, map_location="cpu", weights_only=False)
        )
        episode = build_train_episodes()[0]
        row = score_episode(trainer, arm, QueryConditionedSelectorV1(), episode)
        print(json.dumps(row, ensure_ascii=False, sort_keys=True))
        return 0
    if args.report.exists():
        raise ValueError("F1 holdout report is write-once")
    assert_runs_complete(RUN_ROOT)
    attempt = args.report.with_suffix(".attempt.json")
    if attempt.exists():
        raise ValueError("F1 holdout already has one attempt")
    attempt.write_text(
        json.dumps({"started": time.time(), "pid": os.getpid()}) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    trainer = load_trainer()
    episodes = holdout_episodes()
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    seeds = (0, 1, 2)
    runs: dict[str, Any] = {}
    macro_by_arm: dict[str, dict[str, float]] = {}
    for seed in seeds:
        for arm_name in ARMS:
            result = arm_scores(trainer, arm_name, seed, episodes)
            runs[f"{arm_name}__seed_{seed}"] = result
            macro_by_arm[f"{arm_name}__seed_{seed}"] = result["macro"]
    scores = contrasts(macro_by_arm, seeds)
    scores["copy_only_median"] = statistics.median(
        macro_by_arm[f"copy_only__seed_{seed}"]["answer_exact"] for seed in seeds
    )
    scores["no_state_median"] = statistics.median(
        macro_by_arm[f"no_state__seed_{seed}"]["answer_exact"] for seed in seeds
    )
    gates = gate_report(scores)
    decision = verdict(scores, None)
    report = {
        "format": "taiji-n177-f-f1-holdout-v1",
        "date": "2026-10-29",
        "scope": "12 endpoints; E1 holdout materialized once here; query never read",
        "holdout_families": len({episode.family_id for episode in episodes}),
        "holdout_episodes": len(episodes),
        "scores": scores,
        "gates": gates,
        "passed_all": all(item["passed"] for item in gates.values()),
        "decision": decision,
        "old_retention": "not_run (N176 F0 16-probe file is not part of this instrument)",
        "cost": {
            "process_cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.perf_counter() - wall_start,
        },
        "runs": runs,
    }
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    for key, item in gates.items():
        print(f"{key}: {item['value']} >= {item['threshold']} -> {item['passed']}")
    print(f"PASSED_ALL={report['passed_all']} VERDICT={decision['verdict']}")
    print(f"out={args.report.name}")
    return 0 if decision["verdict"] == "PASS_LEARNED_VALUE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
