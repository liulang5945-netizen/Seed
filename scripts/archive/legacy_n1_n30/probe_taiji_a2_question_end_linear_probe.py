"""A2：提问读完那一刻的状态，能否**线性读出**"该查哪条"（零训练线性探针）。

预注册：`PLAN-A-24` §2"未测"第 2 项／§5 队列 A2 行。它区分两件事：
* **表征里没有**：线性探针在提问末端状态上分不出"答案是哪条告知" ⇒ 提问条件化的检索意图
  无处安放，修接线没用（转 G4=B1／表征侧）；
* **有但没被拿去检索**：探针读得出，而生产选择机构（电路的余弦 top-1）在同一状态上挑错
  ⇒ 信息在场、检索打分没接上（修接线/承诺，G3/G1 方向）。

**任务**：v3 全部 104 题（恰好 3 轮：两条告知＋一问，`answer_tell_position` 52/52 均衡）。
标签＝哪条告知含答案（二分类，chance=0.5）。特征＝提问末端（整份序列化提问喂完那一刻）
的三种口径：`cue_full`（`fabric.cortical_context`，1152 维——生产选择用的就是它）、
`f1_context`（`motor_context`，96 维）、`region0`（区 0 activity＋trace，512 维）。

**对照三件**（缺一不出结论）：
1. 打乱标签的同协议探针（应为 chance）——仪器有效性；
2. 生产选择机构在同一状态、同一库上的挑对率——"有没被拿去用"的量化；
3. 两电路（seed-A／seed-B）各自跑，同向才判（`not_resolved` 兜底）。

**判读线（跑前冻结）**：CV 准确率 ≥0.8 ⇒ 表征有"该查哪条"；≤0.6（近 chance）⇒ 表征缺；
中间 ⇒ 分层报数不裁定。历史同形读数：a25 探针在 v1 扩展集上静态神经 cue 挑对 37/48、
v2 上 16/104——本件把"静态余弦这一把现成的线性打分"换成"训练出的最优线性打分"，
是同一问题的**上限**版。

纪律：零训练（探针权重不进模型）、`learn=False`、基座 sha 复核、记分链历史用 A2.6 的
`_scored_history`（产品原语，不重抄）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for entry in (PROJECT_ROOT, PROJECT_ROOT / "scripts" / "training"):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

MANIFEST = PROJECT_ROOT / "plans/manifests/r2_copy_surface_extension_v3_position_random.json"
SURFACE_REPORT = PROJECT_ROOT / "reports/taiji_r2_a25_lock_only_surface_v3_20260926.json"
FOLDS = 5
EPOCHS = 400
LR = 0.05
WEIGHT_DECAY = 1e-3
SEED = 20260926
HIGH_LINE = 0.8
CHANCE_LINE = 0.6


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _feed_question(runtime: Any, prompt: str, history: list[tuple[str, str]]) -> None:
    """产品链喂法（与 A2.6 trace_item 同形）：reset → 边界符 → 逐字节 observe(learn=False)。"""
    from api.seed_runtime import SeedRuntime

    text = SeedRuntime._serialize(prompt, history)
    substrate = runtime.model.substrate
    substrate.reset_dynamics(episode_id="a2-probe")
    substrate.observe(
        int(substrate.config.boundary_symbol), learn=False, readout="predictive", use_memory=False
    )
    for symbol in text.encode("utf-8"):
        substrate.observe(int(symbol), learn=False, readout="predictive", use_memory=False)


def collect_features(
    items: list[dict[str, Any]], checkpoint: Path, circuit_path: str
) -> dict[str, Any]:
    """每题：记分链历史 → 提问末端三种特征 + 生产选择是否挑中目标事件。"""
    from probe_taiji_r2_a26_emission_trace import _scored_history

    from api.seed_runtime import SeedRuntime

    runtime = SeedRuntime.load(checkpoint)
    runtime.enable_copy_circuit(
        PROJECT_ROOT / circuit_path if not Path(circuit_path).is_absolute() else Path(circuit_path)
    )
    substrate = runtime.model.substrate
    circuit = substrate.copy_circuit

    rows: list[dict[str, Any]] = []
    for item in items:
        turns = [str(turn) for turn in item["turns"]]
        told = turns[int(item["answer_tell_position"])]
        told_bytes = told.encode("utf-8")
        history = _scored_history(runtime, turns)
        _feed_question(runtime, turns[-1], history)
        state = substrate._state
        cue = substrate.fabric.cortical_context(state.regions).detach().cpu()
        f1 = state.motor_context.detach().cpu()
        region0 = torch.cat(
            [state.regions[0].activity.detach().cpu(), state.regions[0].trace.detach().cpu()]
        )
        snap = circuit.addressing(
            cue=substrate.fabric.cortical_context(state.regions),
            f1_context=state.motor_context,
            prev_byte=int(substrate.config.boundary_symbol),
        )
        production_hit = bool(snap is not None and bytes(snap["event"].content) == told_bytes)
        store_contents = [
            event.content.decode("utf-8", errors="replace") for event in circuit.store.events()
        ]
        rows.append(
            {
                "id": item["id"],
                "label": int(item["answer_tell_position"]),
                "cue_full": cue.tolist(),
                "f1_context": f1.tolist(),
                "region0": region0.tolist(),
                "production_selection_hits_target": production_hit,
                "store_contents": store_contents,
            }
        )
    return {"circuit": circuit_path, "rows": rows}


def _cv_accuracy(features: torch.Tensor, labels: torch.Tensor, seed: int) -> dict[str, Any]:
    """5 折 CV 的逻辑回归（torch 全批 Adam；样本 104、维度 ≤1152，全批足够）。"""
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(features.shape[0], generator=generator).tolist()
    fold_accs: list[float] = []
    for fold in range(FOLDS):
        test_idx = order[fold::FOLDS]
        train_idx = [index for index in order if index not in set(test_idx)]
        x_train, y_train = features[train_idx], labels[train_idx]
        x_test, y_test = features[test_idx], labels[test_idx]
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        weight = torch.zeros(x_train.shape[1], requires_grad=True)
        bias = torch.zeros(1, requires_grad=True)
        optimizer = torch.optim.Adam([weight, bias], lr=LR, weight_decay=WEIGHT_DECAY)
        for _ in range(EPOCHS):
            optimizer.zero_grad()
            logits = x_train @ weight + bias
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y_train.float())
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            predictions = ((x_test @ weight + bias) > 0).long()
            fold_accs.append(float((predictions == y_test.long()).float().mean()))
    return {
        "fold_accuracies": [round(value, 4) for value in fold_accs],
        "mean_accuracy": round(sum(fold_accs) / len(fold_accs), 4),
    }


def evaluate_arm(arm: dict[str, Any]) -> dict[str, Any]:
    rows = arm["rows"]
    labels = torch.tensor([row["label"] for row in rows])
    feature_sets = {
        "cue_full": torch.tensor([row["cue_full"] for row in rows]),
        "f1_context": torch.tensor([row["f1_context"] for row in rows]),
        "region0": torch.tensor([row["region0"] for row in rows]),
    }
    results: dict[str, Any] = {}
    for name, features in feature_sets.items():
        real = _cv_accuracy(features, labels, seed=SEED)
        generator = torch.Generator().manual_seed(SEED + 1)
        shuffled = labels[torch.randperm(labels.shape[0], generator=generator)]
        control = _cv_accuracy(features, shuffled, seed=SEED)
        results[name] = {
            "real": real,
            "shuffled_label_control": control,
            "above_chance_margin": round(real["mean_accuracy"] - control["mean_accuracy"], 4),
        }
    production = sum(1 for row in rows if row["production_selection_hits_target"])
    return {
        "circuit": arm["circuit"],
        "features": results,
        "production_selection_hits_target": f"{production}/{len(rows)}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/seed_beta.pt")
    parser.add_argument("--limit", type=int, default=None, help="smoke 用；缺省 104 题")
    parser.add_argument("--out-report", required=True)
    args = parser.parse_args()

    checkpoint = PROJECT_ROOT / args.checkpoint
    sha_before = _sha256(checkpoint)
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    eligible = [
        dict(item)
        for item in payload["dimensions"]["X"]["items"]
        if len(item["turns"]) == 3 and int(item["answer_tell_position"]) in (0, 1)
    ]
    if args.limit is not None:
        #: smoke 分层抽样：两位置各取一半，别让标签均衡守卫在小样本上误报。
        half = max(args.limit // 2, 1)
        stratified: list[dict[str, Any]] = []
        for position in (0, 1):
            subset = [item for item in eligible if int(item["answer_tell_position"]) == position]
            stratified.extend(subset[:half])
        items = stratified
    else:
        items = eligible
    data = json.loads(SURFACE_REPORT.read_text(encoding="utf-8"))
    circuits = [str(arm["circuit"]) for arm in data["treated_arms"]]
    if not items or len(circuits) != 2:
        print(
            json.dumps(
                {"guard_ok": False, "error": f"items={len(items)} circuits={len(circuits)}"},
                ensure_ascii=False,
            )
        )
        return 2

    arms = [collect_features(items, checkpoint, circuit) for circuit in circuits]
    evaluations = [evaluate_arm(arm) for arm in arms]

    def best_real(evaluation: dict[str, Any]) -> float:
        return max(result["real"]["mean_accuracy"] for result in evaluation["features"].values())

    all_above = all(best_real(evaluation) >= HIGH_LINE for evaluation in evaluations)
    all_below = all(best_real(evaluation) <= CHANCE_LINE for evaluation in evaluations)
    verdict = (
        "representation_has_query_target"
        if all_above
        else ("representation_lacks_query_target" if all_below else "mixed_report_by_feature")
    )
    report = {
        "format": "taiji-a2-question-end-linear-probe-v1",
        "prereg": "plans/reference/M5_R2_A_BRANCH_PLAN_REV2_20260926.md §2 未测②／§5 A2",
        "manifest": str(MANIFEST.relative_to(PROJECT_ROOT)),
        "items": len(items),
        "label_balance": {
            "position_0": sum(1 for item in items if int(item["answer_tell_position"]) == 0),
            "position_1": sum(1 for item in items if int(item["answer_tell_position"]) == 1),
        },
        "cv_protocol": {
            "folds": FOLDS,
            "epochs": EPOCHS,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "seed": SEED,
        },
        "reading_lines": {
            "representation_has": HIGH_LINE,
            "chance_band": CHANCE_LINE,
            "control": "shuffled-label same protocol must sit at chance",
        },
        "arms": evaluations,
        "rows_minimal": [
            {
                "id": row["id"],
                "label": row["label"],
                "production_selection_hits_target": row["production_selection_hits_target"],
            }
            for arm in arms
            for row in arm["rows"]
        ],
        "verdict": verdict,
        "what_would_overturn": (
            "打乱标签对照若明显高于 chance ⇒ 特征泄漏，本件作废；两电路 verdict 方向不一致 ⇒ not_resolved；"
            "生产选择挑对率若与本件 CV 同高 ⇒ '有但没被拿去检索'不成立（选择机构本身已够）"
        ),
        "base_sha256_unchanged": _sha256(checkpoint) == sha_before,
    }
    report["instrument_guard"] = {
        "items_nonzero": bool(items),
        "label_balanced": report["label_balance"]["position_0"]
        == report["label_balance"]["position_1"],
        "shuffled_controls_at_chance": all(
            result["shuffled_label_control"]["mean_accuracy"] <= CHANCE_LINE + 0.05
            for evaluation in evaluations
            for result in evaluation["features"].values()
        ),
        "base_unchanged": report["base_sha256_unchanged"],
    }

    out = Path(args.out_report)
    if not out.is_absolute():
        out = PROJECT_ROOT / out
    if out.exists():
        out = out.with_name(f"{out.stem}-{datetime.now(UTC).strftime('%H%M%S')}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    report_payload = dict(report)
    report_payload.pop("rows_minimal", None)
    report_payload["rows"] = report.pop("rows_minimal")
    out.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    ok = all(report["instrument_guard"].values())
    print(
        json.dumps(
            {
                "guard_ok": ok,
                "items": len(items),
                "arms": [
                    {
                        "circuit": evaluation["circuit"],
                        "cue_full": evaluation["features"]["cue_full"]["real"]["mean_accuracy"],
                        "f1": evaluation["features"]["f1_context"]["real"]["mean_accuracy"],
                        "region0": evaluation["features"]["region0"]["real"]["mean_accuracy"],
                        "production": evaluation["production_selection_hits_target"],
                    }
                    for evaluation in evaluations
                ],
                "verdict": verdict,
                "out": (
                    out.relative_to(PROJECT_ROOT).as_posix()
                    if out.is_relative_to(PROJECT_ROOT)
                    else str(out)
                ),
            },
            ensure_ascii=False,
        )
    )
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
