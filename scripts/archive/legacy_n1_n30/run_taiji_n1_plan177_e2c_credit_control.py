"""E2c 信用 on/stop 的同图控制（合成 + 真实 train episode 各一次，只读检查）。

两臂共享同一 seed 的父参数与同一用例，只切换"回答 CE 是否留在连续特征上"：
`credit_on` 应让编码器拿到正梯度，`credit_stop` 应在读数入口截断、编码器梯度为零，
而读数头两臂都更新。选择器按实体/属性身份打分，回答 CE 对它在两臂都是零——这是
本候选的事实，单列披露，不计作"选择器已受任务纠正"。

不训练 producer/V4，不读 query/holdout/sealed；每个用例只做一次前向+反向+一步诊断。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.internalization import content_digest  # noqa: E402
from taiji.joint_state_consumption import PersistentContentFeatureV1  # noqa: E402
from taiji.joint_state_consumption_course import (  # noqa: E402
    E1_ATTRIBUTE,
    E1_COLORS,
    build_train_episodes,
    e1_families,
)
from taiji.joint_state_credit import (  # noqa: E402
    CREDIT_MODES,
    MODULE_GROUPS,
    CreditCandidateV1,
    CreditCaseV1,
    StateCreditCandidateV1,
)

CHECKPOINT = (
    ROOT / "data/checkpoints/n1_plan177_6_t1_pilot_20261027/correct__seed_0/update_002560.pt"
)
TRACE_CACHE = ROOT / "data/checkpoints/n1_plan177_5a_preflight_20261027/train_trace_cache_v1.pt"
REPORT = ROOT / "reports/taiji_n1_plan177_e2c_credit_control_20261029.json"
FEATURE_DIM = 64
SEED = 0


def _feature(*, entity: str, attribute: str, value: str, case_id: str, index: int):
    return PersistentContentFeatureV1.create(
        entity=entity,
        attribute=attribute,
        value=value,
        source_input_id=f"e2c:{case_id}:{index}",
        source_sha256=content_digest({"case": case_id, "index": index, "value": value}),
        turn_index=0,
        revision=1,
        valid_from=0,
        confidence=1.0,
        # 训练侧连续特征由字节编码器现场算；持久化向量属F，这里只占位、不参与前向。
        continuous=[0.0] * FEATURE_DIM,
    )


def synthetic_case() -> CreditCaseV1:
    """Three entities, one colour fact each: identity scoring must pick the queried one."""

    facts = (("钥匙", "黄色"), ("杯子", "红色"), ("球", "蓝色"))
    candidates = tuple(
        CreditCandidateV1.from_value(
            feature=_feature(
                entity=entity,
                attribute=E1_ATTRIBUTE,
                value=value,
                case_id="synthetic",
                index=index,
            ),
            value=value,
        )
        for index, (entity, value) in enumerate(facts)
    )
    entity, value = facts[0]
    return CreditCaseV1(
        case_id="synthetic:three-entities",
        entity=entity,
        attribute=E1_ATTRIBUTE,
        candidates=candidates,
        label=E1_COLORS.index(value),
    )


def real_case() -> tuple[CreditCaseV1, dict[str, Any]]:
    """真实链：N177.6 producer 解析 E1 train episode，取实体/属性/值 span。

    用例挑选规则事先固定、按语料顺序试：取第一条"producer 把该实体当前值解析成
    课程真值"的 episode（且答案色与合成用例不同）。试过而失败的家族逐个登记，
    不静默跳过，避免"挑一个让它过"。
    """

    from taiji.joint_t1_trace_cache import JointT1TraceCacheV1
    from taiji.joint_t1_trainer import JointT1TrainerV1

    trace_cache = JointT1TraceCacheV1.from_payload(
        torch.load(TRACE_CACHE, map_location="cpu", weights_only=False)
    )
    payload = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    trainer = JointT1TrainerV1.from_native_checkpoint(payload, input_trace_cache=trace_cache)
    trainer.input_trace_cache = None
    episodes = build_train_episodes()
    attempts: list[dict[str, Any]] = []
    chosen: tuple[Any, Any, dict[str, tuple[str, str, int]], list[dict[str, Any]]] | None = None
    with torch.no_grad():
        for family in e1_families()["train"]:
            if family.value not in E1_COLORS or family.value == "黄色":
                continue
            episode = next(
                (
                    item
                    for item in episodes
                    if item.family_id == family.family_id and item.episode_id.endswith(":0")
                ),
                None,
            )
            if episode is None:
                continue
            adapter, joint = trainer._new_episode_state(episode.episode_id)
            facts: list[tuple[str, str, str, int]] = []
            parses: list[dict[str, Any]] = []
            for turn in episode.turns:
                trace, joint = trainer._append_turn(adapter, joint, turn)
                parse = trainer.producer.predict(joint, trace)
                parses.append(
                    {
                        "turn_index": turn.turn_index,
                        "act": parse.act,
                        "spans": [
                            {"role": span.role, "text": span.text}
                            for span in getattr(parse, "spans", ())
                        ],
                    }
                )
                by_role: dict[str, list[str]] = {"entity": [], "attribute": [], "value": []}
                for span in getattr(parse, "spans", ()):
                    role = getattr(span, "role", None)
                    if role in by_role:
                        by_role[role].append(str(getattr(span, "text", "")))
                if all(len(by_role[role]) == 1 for role in by_role):
                    facts.append(
                        (
                            by_role["entity"][0],
                            by_role["attribute"][0],
                            by_role["value"][0],
                            turn.turn_index,
                        )
                    )
            current: dict[str, tuple[str, str, int]] = {}
            for entity, attribute, value, turn_index in facts:
                if attribute == E1_ATTRIBUTE:
                    current[entity] = (attribute, value, turn_index)
            parsed = current.get(family.entity)
            attempts.append(
                {
                    "family_id": family.family_id,
                    "entity": family.entity,
                    "true_value": family.value,
                    "parsed_value": None if parsed is None else parsed[1],
                    "matches": parsed is not None and parsed[1] == family.value,
                }
            )
            if parsed is not None and parsed[1] == family.value:
                chosen = (family, episode, current, parses)
                break
    if chosen is None:
        raise RuntimeError(f"real chain parsed no family truth; attempts={attempts}")
    family, episode, current, parses = chosen
    candidates = tuple(
        CreditCandidateV1.from_value(
            feature=_feature(
                entity=entity,
                attribute=attribute,
                value=value,
                case_id="real",
                index=turn_index,
            ),
            value=value,
        )
        for entity, (attribute, value, turn_index) in current.items()
    )
    case = CreditCaseV1(
        case_id=f"real:{episode.episode_id}",
        entity=family.entity,
        attribute=E1_ATTRIBUTE,
        candidates=candidates,
        label=E1_COLORS.index(family.value),
    )
    provenance = {
        "episode_id": episode.episode_id,
        "family_id": family.family_id,
        "true_value": family.value,
        "current_facts": {
            entity: {"value": value, "turn_index": turn_index}
            for entity, (_, value, turn_index) in current.items()
        },
        "producer_matched_family_value": True,
        "candidate_values": [item.feature.value for item in candidates],
        "producer_checkpoint": str(CHECKPOINT.relative_to(ROOT)).replace("\\", "/"),
        "trace_cache": str(TRACE_CACHE.relative_to(ROOT)).replace("\\", "/"),
        "case_selection_rule": "first train family (value != 黄色) whose current fact parses to truth",
        "attempts": attempts,
        "parses": parses,
    }
    return case, provenance


def _group_delta(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, float]:
    deltas: dict[str, float] = {}
    for group in MODULE_GROUPS:
        squares = 0.0
        for name, value in before[group].items():
            squares += float((after[group][name] - value).square().sum())
        deltas[group] = squares**0.5
    return deltas


def run_case(case: CreditCaseV1) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for mode in CREDIT_MODES:
        candidate = StateCreditCandidateV1(mode=mode, seed=SEED, feature_dim=FEATURE_DIM)
        out = candidate.forward(case)
        out["feature"].retain_grad()
        logits_before = [float(item) for item in out["logits"].detach().tolist()]
        loss_value = float(out["loss"].detach())
        selection = int(out["selection_index"])
        feature_vector = out["feature"].detach()
        feature_digest = content_digest({"v": [float(x) for x in feature_vector.tolist()]})
        before_parameters = {
            group: {
                name: value.detach().clone()
                for name, value in getattr(candidate, group).state_dict().items()
            }
            for group in MODULE_GROUPS
        }
        gradients = candidate.backward(out["loss"])
        feature_gradient = (
            None if out["feature"].grad is None else float(out["feature"].grad.detach().norm())
        )
        candidate.step()
        after_parameters = {
            group: {
                name: value.detach().clone()
                for name, value in getattr(candidate, group).state_dict().items()
            }
            for group in MODULE_GROUPS
        }
        arms[mode] = {
            "logits": logits_before,
            "logits_digest": content_digest({"logits": logits_before}),
            "loss": loss_value,
            "selection_index": selection,
            "selected_value": case.candidates[selection].feature.value,
            "feature_digest": feature_digest,
            "feature_gradient_norm": feature_gradient,
            "gradient_norms": gradients,
            "gradients_finite": all(
                value == value and abs(value) != float("inf") for value in gradients.values()
            ),
            "parameter_delta": _group_delta(before_parameters, after_parameters),
            "cursor_after_step": candidate.cursor,
            "parent_digest": candidate.parent_digest,
        }
    on, stop = arms["credit_on"], arms["credit_stop"]
    answer_colour = E1_COLORS[case.label]
    verdicts = {
        "arms_share_initial_forward": on["logits_digest"] == stop["logits_digest"],
        "arms_share_parent_parameters": on["parent_digest"] == stop["parent_digest"],
        "same_selection": on["selection_index"] == stop["selection_index"],
        "selection_matches_label": all(
            case.candidates[arms[mode]["selection_index"]].feature.value == answer_colour
            for mode in CREDIT_MODES
        ),
        "credit_on_reaches_encoder": on["gradient_norms"]["encoder"] > 0.0
        and (on["feature_gradient_norm"] or 0.0) > 0.0,
        "credit_stop_cuts_encoder": stop["gradient_norms"]["encoder"] == 0.0
        and stop["feature_gradient_norm"] in (None, 0.0),
        "readout_trains_in_both": on["gradient_norms"]["readout"] > 0.0
        and stop["gradient_norms"]["readout"] > 0.0,
        "gradients_finite_in_both": on["gradients_finite"] and stop["gradients_finite"],
        "one_diagnostic_step_in_both": on["cursor_after_step"] == 1
        and stop["cursor_after_step"] == 1,
        "encoder_moves_only_under_credit_on": on["parameter_delta"]["encoder"] > 0.0
        and stop["parameter_delta"]["encoder"] == 0.0,
        "readout_moves_in_both": on["parameter_delta"]["readout"] > 0.0
        and stop["parameter_delta"]["readout"] > 0.0,
        "selector_gets_no_answer_credit_in_both": on["gradient_norms"]["selector"] == 0.0
        and stop["gradient_norms"]["selector"] == 0.0,
    }
    return {"case": case.to_payload(), "arms": arms, "verdicts": verdicts}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N177 E2c 信用 on/stop 同图控制（只读检查）")
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    torch.set_num_threads(1)
    if args.report.exists():
        raise ValueError("E2c credit control report is write-once")

    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    synthetic = run_case(synthetic_case())
    real, provenance = real_case()
    real_result = run_case(real)
    cases = {
        "synthetic": synthetic,
        "real_train_episode": {**real_result, "provenance": provenance},
    }
    verdicts = {
        "synthetic_all": all(synthetic["verdicts"].values()),
        "real_train_episode_all": all(real_result["verdicts"].values()),
        "both_cases_present": len(cases) == 2,
        "no_optimizer_training_beyond_diagnostics": True,
        "auxiliary_slot_act_term": "absent (this bundle has no slot/act head)",
    }
    report = {
        "format": "taiji-n177-e2c-credit-control-v1",
        "date": "2026-10-29",
        "scope": (
            "one synthetic case + one real E1 train episode; per arm one forward/backward/"
            "single diagnostic step; no producer/V4 training, no query/holdout/sealed read"
        ),
        "credit_modes": list(CREDIT_MODES),
        "seed": SEED,
        "feature_dim": FEATURE_DIM,
        "cases": cases,
        "verdicts": verdicts,
        "all_verdicts_ok": (verdicts["synthetic_all"] and verdicts["real_train_episode_all"]),
        "cost": {
            "process_cpu_seconds": time.process_time() - cpu_start,
            "wall_seconds": time.perf_counter() - wall_start,
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.report.with_name(args.report.name + ".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(args.report)
    for name, values in cases.items():
        for key, value in values["verdicts"].items():
            print(f"{name}:{key}={value}")
    print(f"ALL_VERDICTS_OK={report['all_verdicts_ok']} out={args.report.name}")
    return 0 if report["all_verdicts_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
