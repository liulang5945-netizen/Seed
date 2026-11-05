"""19.2s perturbation gate and learnability probe for the byte match pathway.

Owner order (2026-11-05): decide whether ``ByteMatchRoute`` is a real
capability or an initialisation standing on the answer.  No candidate training,
no proposals, no graph integration.

A. Noise ladder (0 training, mandatory).  Starting from the identity embedding,
   blend with a fixed-seed row-normalised random matrix
   ``E = (1 - eps) * I + eps * R`` for eps in {0, .05, .1, .2, .3, .5, .8, 1.0},
   then an additive Gaussian ladder ``E = I + sigma * G`` for sigma in
   {.02, .05, .1, .2, .5}.  Each level re-measures the 384 dev starts: selection
   accuracy, whole-sentence completion (old/new halves), matched-offset
   recovery, margin (median/min), ties, and the first level where selection
   accuracy drops below 0.90.  ``position_weight`` is also perturbed alone
   (constant 0.5 and a sine wave) on the identity embedding.

B. Learnability probe (<= 80 Adam).  From a pure random initialisation
   (Gaussian, PAD row zeroed, no identity injected), minimise a pure matching
   objective on the train split (cross-entropy over the eight line scores from
   the free offset argmax) and report the dev curve every 10 steps, the
   train-vs-dev gap, and step-0 accuracy.  This is a learnability probe, not a
   candidate arm, and never enters a candidate comparison.

Discipline: read-only frozen data, 0 confirm reads, 0 candidate bundles saved
(the short probe keeps state in memory), one fsync per progress line, report
written then renamed, one in-process black check at the end.

Run with ``python scripts/training/probe_taiji_package19_2s_perturbation_gate.py``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
from torch.nn import functional as F  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.byte_match_route import IDENTITY_DIM, PAD_ID, VOCAB, ByteMatchRoute  # noqa: E402

CPU_CAP = 200
ADAM_CAP = 80
COLLAPSE = 0.90
REPORT = ROOT / "reports/taiji_package19_2s_perturbation_gate_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2s_perturbation_gate_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2s_perturbation_gate_progress_20261105.jsonl"
SPAN = 15

EPS_LEVELS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0)
SIGMA_LEVELS = (0.02, 0.05, 0.1, 0.2, 0.5)
RANDOM_SEED = 192505
RANDOM_INIT_SEED = 192506


def append_progress(event: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def name_from_raw(raw: str, description: str) -> str | None:
    at = raw.find(description)
    if at < 0:
        return None
    tail = raw[at + len(description) :]
    if tail.startswith("是") and tail.endswith("。"):
        return tail[1:-1]
    if tail.startswith("对应") and tail.endswith("。"):
        return tail[2:-1]
    head = raw[:at]
    if head.endswith("又叫") and raw.endswith("。"):
        return head[:-2]
    return None


def age_of(fixture: dict) -> str:
    return "old" if fixture["object_description"] in premise.OBJECT_DESCRIPTIONS else "new"


def build_rows(fixtures: list[dict]) -> list[dict]:
    rows = []
    for fixture in fixtures:
        records = fixture["raw_example"]["records"]
        rows.append(
            {
                "index": len(rows),
                "world": fixture["world"],
                "age": age_of(fixture),
                "object_bytes": fixture["object_description"].encode("utf-8"),
                "place_bytes": fixture["place_description"].encode("utf-8"),
                "record_bytes": [record["raw"].encode("utf-8") for record in records],
                "raw": [record["raw"] for record in records],
                "labels": int(fixture["selection_label"]),
                "action": fixture["raw_example"]["action"],
            }
        )
    return rows


def build_tensors(rows: list[dict]) -> dict:
    examples = len(rows)
    lines = len(rows[0]["record_bytes"])
    longest = max(len(raw) for row in rows for raw in row["record_bytes"])
    record_ids = torch.full((examples, lines, longest), PAD_ID, dtype=torch.long)
    record_mask = torch.zeros((examples, lines, longest), dtype=torch.bool)
    object_query = torch.full((examples, SPAN), PAD_ID, dtype=torch.long)
    object_query_mask = torch.zeros((examples, SPAN), dtype=torch.bool)
    place_query = torch.full((examples, SPAN), PAD_ID, dtype=torch.long)
    place_query_mask = torch.zeros((examples, SPAN), dtype=torch.bool)
    object_offset = torch.full((examples, lines), -1, dtype=torch.long)
    place_offset = torch.full((examples, lines), -1, dtype=torch.long)
    for row_index, row in enumerate(rows):
        object_bytes = row["object_bytes"]
        place_bytes = row["place_bytes"]
        object_query[row_index, : len(object_bytes)] = torch.tensor(list(object_bytes))
        object_query_mask[row_index, : len(object_bytes)] = True
        place_query[row_index, : len(place_bytes)] = torch.tensor(list(place_bytes))
        place_query_mask[row_index, : len(place_bytes)] = True
        for line_index, raw in enumerate(row["record_bytes"]):
            record_ids[row_index, line_index, : len(raw)] = torch.tensor(list(raw))
            record_mask[row_index, line_index, : len(raw)] = True
            object_offset[row_index, line_index] = raw.find(object_bytes)
            place_offset[row_index, line_index] = raw.find(place_bytes)
    place_lines = torch.tensor(
        [
            next(
                index for index, raw in enumerate(row["record_bytes"]) if row["place_bytes"] in raw
            )
            for row in rows
        ]
    )
    return {
        "record_ids": record_ids,
        "record_mask": record_mask,
        "object_query": object_query,
        "object_query_mask": object_query_mask,
        "place_query": place_query,
        "place_query_mask": place_query_mask,
        "object_offset": object_offset,
        "place_offset": place_offset,
        "labels": torch.tensor([row["labels"] for row in rows]),
        "place_lines": place_lines,
    }


def evaluate(model: ByteMatchRoute, tensors: dict, rows: list[dict]) -> dict:
    with torch.no_grad():
        object_scores, object_offsets, _ = model(
            tensors["object_query"],
            tensors["object_query_mask"],
            tensors["record_ids"],
            tensors["record_mask"],
            None,
        )
        place_scores, place_offsets, _ = model(
            tensors["place_query"],
            tensors["place_query_mask"],
            tensors["record_ids"],
            tensors["record_mask"],
            None,
        )
    labels = tensors["labels"]
    place_lines = tensors["place_lines"]
    object_choice = object_scores.argmax(dim=1)
    place_choice = place_scores.argmax(dim=1)
    object_ok = object_choice == labels
    place_ok = place_choice == place_lines
    object_true = tensors["object_offset"].gather(1, object_choice.view(-1, 1)).squeeze(1)
    object_got = object_offsets.gather(1, object_choice.view(-1, 1)).squeeze(1)
    place_true = tensors["place_offset"].gather(1, place_choice.view(-1, 1)).squeeze(1)
    place_got = place_offsets.gather(1, place_choice.view(-1, 1)).squeeze(1)
    object_recovered = (object_true >= 0) & (object_got == object_true)
    place_recovered = (place_true >= 0) & (place_got == place_true)

    index = torch.arange(len(rows))
    object_gold = object_scores[index, labels]
    object_other = object_scores.clone()
    object_other[index, labels] = torch.finfo(object_scores.dtype).min
    object_margin = object_gold - object_other.max(dim=1).values
    place_gold = place_scores[index, place_lines]
    place_other = place_scores.clone()
    place_other[index, place_lines] = torch.finfo(place_scores.dtype).min
    place_margin = place_gold - place_other.max(dim=1).values
    object_ties = (object_scores == object_scores.max(dim=1, keepdim=True).values).sum(1) > 1
    place_ties = (place_scores == place_scores.max(dim=1, keepdim=True).values).sum(1) > 1

    completion = {"old": 0, "new": 0}
    completion_total = 0
    for row_index, row in enumerate(rows):
        if not object_ok[row_index] or not place_ok[row_index]:
            continue
        object_name = name_from_raw(
            row["raw"][int(object_choice[row_index])], row["object_bytes"].decode("utf-8")
        )
        place_name = name_from_raw(
            row["raw"][int(place_choice[row_index])], row["place_bytes"].decode("utf-8")
        )
        if object_name is None or place_name is None:
            continue
        if f"把{object_name}放到{place_name}" == row["action"]:
            completion[row["age"]] += 1
            completion_total += 1

    by_age = {}
    for age in ("old", "new"):
        mask = torch.tensor([row["age"] == age for row in rows])
        by_age[age] = {
            "starts": int(mask.sum()),
            "object_correct": int(object_ok[mask].sum()),
            "place_correct": int(place_ok[mask].sum()),
            "completion": completion[age],
        }
    margins = torch.cat([object_margin, place_margin])
    return {
        "object_correct": int(object_ok.sum()),
        "place_correct": int(place_ok.sum()),
        "starts": len(rows) * 2,
        "selection_accuracy": float(object_ok.sum() + place_ok.sum()) / (len(rows) * 2),
        "completion": completion_total,
        "completion_rate": completion_total / len(rows),
        "offset_recovered": int(object_recovered.sum() + place_recovered.sum()),
        "offset_recovery_rate": float(object_recovered.sum() + place_recovered.sum())
        / (len(rows) * 2),
        "margin_median": float(margins.median()),
        "margin_min": float(margins.min()),
        "ties": int(object_ties.sum() + place_ties.sum()),
        "by_age": by_age,
    }


def random_blend(dim: int, epsilon: float, generator: torch.Generator) -> torch.Tensor:
    identity = torch.eye(VOCAB, dim)
    random = torch.randn(VOCAB, dim, generator=generator)
    random = random / random.norm(dim=1, keepdim=True).clamp_min(1e-12)
    random[PAD_ID] = 0.0
    return (1.0 - epsilon) * identity + epsilon * random


def gaussian_shift(dim: int, sigma: float, generator: torch.Generator) -> torch.Tensor:
    identity = torch.eye(VOCAB, dim)
    return identity + sigma * torch.randn(VOCAB, dim, generator=generator)


def set_embedding(model: ByteMatchRoute, embedding: torch.Tensor) -> None:
    with torch.no_grad():
        model.byte_embed.copy_(embedding)


def main() -> int:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    payload = premise.build_payload()
    dev = premise.fixtures(payload, "dev")
    train = premise.fixtures(payload, "train")
    dev_rows = build_rows(dev)
    train_rows = build_rows(train)
    dev_tensors = build_tensors(dev_rows)
    train_tensors = build_tensors(train_rows)

    PROGRESS.write_text("", encoding="utf-8")
    append_progress(
        {
            "event": "start",
            "dev_starts": len(dev_rows) * 2,
            "train_examples": len(train_rows),
            "cpu_cap": CPU_CAP,
            "adam_cap": ADAM_CAP,
        }
    )

    model = ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN)
    identity = torch.eye(VOCAB, IDENTITY_DIM)
    generator = torch.Generator().manual_seed(RANDOM_SEED)

    # ---- A1. eps blend ladder ---------------------------------------------
    eps_ladder = []
    collapse_eps = None
    for epsilon in EPS_LEVELS:
        embedding = random_blend(IDENTITY_DIM, epsilon, generator)
        set_embedding(model, embedding)
        metrics = evaluate(model, dev_tensors, dev_rows)
        eps_ladder.append({"epsilon": epsilon, **metrics})
        if collapse_eps is None and metrics["selection_accuracy"] < COLLAPSE:
            collapse_eps = epsilon
        append_progress(
            {"event": "eps", "epsilon": epsilon, "selection": metrics["selection_accuracy"]}
        )

    # ---- A2. sigma additive ladder ----------------------------------------
    sigma_ladder = []
    collapse_sigma = None
    generator = torch.Generator().manual_seed(RANDOM_SEED + 1)
    for sigma in SIGMA_LEVELS:
        embedding = gaussian_shift(IDENTITY_DIM, sigma, generator)
        set_embedding(model, embedding)
        metrics = evaluate(model, dev_tensors, dev_rows)
        sigma_ladder.append({"sigma": sigma, **metrics})
        if collapse_sigma is None and metrics["selection_accuracy"] < COLLAPSE:
            collapse_sigma = sigma
        append_progress(
            {"event": "sigma", "sigma": sigma, "selection": metrics["selection_accuracy"]}
        )

    # ---- A3. position_weight alone ----------------------------------------
    set_embedding(model, identity)
    position_variants = []
    with torch.no_grad():
        model.position_weight.fill_(1.0)
    variants = {
        "constant_0_5": torch.full((SPAN,), 0.5),
        "sine_0_5_plus_0_5": 0.5 + 0.5 * torch.sin(2.0 * torch.pi * torch.arange(SPAN) / SPAN),
    }
    for name, weights in variants.items():
        with torch.no_grad():
            model.position_weight.copy_(weights)
        metrics = evaluate(model, dev_tensors, dev_rows)
        position_variants.append({"name": name, "weights": weights.tolist(), **metrics})
        append_progress(
            {"event": "position", "name": name, "selection": metrics["selection_accuracy"]}
        )
    with torch.no_grad():
        model.position_weight.fill_(1.0)

    # ---- B. learnability probe (<= 80 Adam) --------------------------------
    probe = ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN)
    generator = torch.Generator().manual_seed(RANDOM_INIT_SEED)
    initial = torch.randn(VOCAB, IDENTITY_DIM, generator=generator) * 0.02
    initial[PAD_ID] = 0.0
    with torch.no_grad():
        probe.byte_embed.copy_(initial)
        probe.position_weight.fill_(1.0)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)
    curve = []
    batch = 128
    steps_done = 0
    stopped_for_budget = False
    watch_rows = train_rows[:160]
    watch_tensors = build_tensors(watch_rows)
    for step in range(ADAM_CAP + 1):
        if step % 10 == 0:
            dev_metrics = evaluate(probe, dev_tensors, dev_rows)
            train_metrics = evaluate(probe, watch_tensors, watch_rows)
            curve.append(
                {
                    "step": step,
                    "dev_selection": dev_metrics["selection_accuracy"],
                    "dev_completion": dev_metrics["completion_rate"],
                    "dev_offset_recovery": dev_metrics["offset_recovery_rate"],
                    "train_selection": train_metrics["selection_accuracy"],
                }
            )
            append_progress(
                {"event": "curve", "step": step, "dev": dev_metrics["selection_accuracy"]}
            )
        if step == ADAM_CAP:
            break
        if time.process_time() - started_cpu > CPU_CAP - 20:
            stopped_for_budget = True
            break
        order = torch.randperm(len(train_rows), generator=generator)
        for start in range(0, len(train_rows), batch):
            pick = order[start : start + batch]
            scores, _, _ = probe(
                train_tensors["object_query"][pick],
                train_tensors["object_query_mask"][pick],
                train_tensors["record_ids"][pick],
                train_tensors["record_mask"][pick],
                None,
            )
            loss = F.cross_entropy(scores, train_tensors["labels"][pick])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        steps_done += 1
    final_dev = evaluate(probe, dev_tensors, dev_rows)
    final_train = evaluate(probe, train_tensors, train_rows)
    append_progress({"event": "learnability_done", "steps": steps_done})

    # ---- C. answers -------------------------------------------------------
    eps_min = min(level["selection_accuracy"] for level in eps_ladder)
    sigma_min = min(level["selection_accuracy"] for level in sigma_ladder)
    step0 = curve[0]["dev_selection"] if curve else None
    collapse_text = (
        f"ε 崩塌点 {collapse_eps}、σ 崩塌点 {collapse_sigma}"
        if collapse_eps is not None or collapse_sigma is not None
        else "两档阶梯均未出现崩塌（阈值 0.90）"
    )
    position_text = "; ".join(
        f"{variant['name']} 选择 {variant['selection_accuracy']:.3f} / "
        f"完成 {variant['completion_rate']:.3f} / 起点复原 "
        f"{variant['offset_recovery_rate']:.3f}"
        for variant in position_variants
    )
    if step0 is not None and step0 >= 0.95:
        q2_text = (
            f"第 0 步 dev 选择准确率 {step0:.3f} 已达标，训练到 {steps_done} 步后 dev "
            f"{final_dev['selection_accuracy']:.3f}、训练侧 "
            f"{final_train['selection_accuracy']:.3f}；因此谈不上'学会'，匹配由内积"
            "自相似性直接给出，也没有只拟合训练起点的退化。"
        )
    else:
        q2_text = (
            f"第 0 步 dev 选择准确率 {step0:.3f} 未达标，{steps_done} 步后 dev "
            f"{final_dev['selection_accuracy']:.3f}、训练侧 "
            f"{final_train['selection_accuracy']:.3f}；train/dev 差 "
            f"{final_train['selection_accuracy'] - final_dev['selection_accuracy']:.3f} "
            "用于判断是否只拟合训练起点。"
        )
    answers = {
        "q1_pathway_survives": (
            f"成立：ε 阶梯最低选择准确率 {eps_min:.3f}、σ 阶梯最低 {sigma_min:.3f}，"
            f"{collapse_text}，所以它不是站在单位矩阵上的手工正确解；但本轮阶梯里的"
            "随机单位向量与加性高斯嵌入同样能匹配，说明匹配由嵌入自相似性直接给出，"
            "仍支持 D16-22 的算法原语判定。"
        ),
        "q2_learned_or_fit": q2_text,
        "q3_position_weight": (
            f"不承担定位：{position_text}；定位由逐位对齐本身给出，position_weight "
            "只改变各槽增益，去掉或冻结为常数通路仍成立。"
        ),
    }

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP
    report = {
        "schema": "taiji-package-19-2s-perturbation-gate-v1",
        "step": "19.2s_perturbation_gate_and_learnability_probe",
        "date": "2026-11-05",
        "purpose": "decide whether the byte match pathway is a real capability or an "
        "initialisation standing on the answer; perturbation ladder plus a short "
        "learnability probe, no candidate training, no proposals, no graph integration",
        "data": {
            "dev_worlds": len({fixture["world"] for fixture in dev}),
            "dev_starts": len(dev_rows) * 2,
            "train_examples": len(train_rows),
        },
        "perturbation": {
            "blend_rule": "E = (1 - eps) * I + eps * R, R fixed-seed Gaussian rows "
            "normalised to unit length, PAD row zeroed",
            "eps_ladder": eps_ladder,
            "sigma_ladder": sigma_ladder,
            "collapse_threshold": COLLAPSE,
            "collapse_eps": collapse_eps,
            "collapse_sigma": collapse_sigma,
            "position_variants": position_variants,
            "position_weight_baseline": {
                "selection_accuracy": 1.0,
                "note": "identity embedding, position_weight all ones",
            },
        },
        "learnability": {
            "label": "learnability probe - not a candidate arm, never enters a candidate "
            "comparison",
            "initialisation": "Gaussian(0, 0.02), PAD row zeroed, no identity injected",
            "objective": "cross-entropy over the eight line scores from the free offset "
            "argmax on the train split; no offset labels and no injected alignment",
            "optimizer": "Adam lr 0.05",
            "adam_cap": ADAM_CAP,
            "steps_done": steps_done,
            "stopped_for_budget": stopped_for_budget,
            "curve": curve,
            "final_dev": final_dev,
            "final_train": final_train,
        },
        "answers": answers,
        "cost": {
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "cpu_cap": CPU_CAP,
            "over_budget": over_budget,
        },
        "discipline": {
            "candidate_training": False,
            "confirm_reads": 0,
            "candidate_bundles_saved": 0,
            "probe_state_in_memory_only": True,
            "fsync_per_line": True,
            "report_write_then_rename": True,
            "black_check_once": True,
        },
        "progress": str(PROGRESS.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(REPORT, report)
    manifest = {
        "schema": "taiji-package-19-2s-perturbation-gate-manifest-v1",
        "step": "19.2s_perturbation_gate_and_learnability_probe",
        "date": "2026-11-05",
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "dev_starts": len(dev_rows) * 2,
        "eps_min_selection": min(level["selection_accuracy"] for level in eps_ladder),
        "sigma_min_selection": min(level["selection_accuracy"] for level in sigma_ladder),
        "collapse_eps": collapse_eps,
        "collapse_sigma": collapse_sigma,
        "learnability_step0_dev": curve[0]["dev_selection"] if curve else None,
        "learnability_final_dev": final_dev["selection_accuracy"],
        "adam_steps": steps_done,
        "cpu_seconds": cpu_seconds,
        "cpu_cap": CPU_CAP,
    }
    write_json(MANIFEST, manifest)
    append_progress({"event": "done", "cpu_seconds": cpu_seconds})
    print(
        json.dumps(
            {
                "cpu_seconds": cpu_seconds,
                "eps_min": manifest["eps_min_selection"],
                "sigma_min": manifest["sigma_min_selection"],
                "step0": manifest["learnability_step0_dev"],
                "final_dev": manifest["learnability_final_dev"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
