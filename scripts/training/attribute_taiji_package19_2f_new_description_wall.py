"""19.2f: zero-update attribution of the "new description" wall.

The 19.5 six cells are already scored.  Their failures are not spread over the
192 dev examples: the 96 examples whose object description is unseen (indices
8-15) carry almost all of them.  This script answers three questions on the
finished step300 bundles, without training, saving or touching confirm:

1. Stratification, on the same 192 examples and the formal ``completed``
   predicate (exact action bytes + EOS).  For both B cells (and the C / healthy
   A controls) it reports sentence, route, composite start, conditional byte and
   unconditional content byte separately for the 96 old-description and the 96
   new-description examples.
2. An order-swap probe.  Each witness pair keeps the goal byte multiset and only
   exchanges the two description tokens (``让甲到乙。`` <-> ``让乙到甲。``).  The
   readings are the frozen 17.3 T2 shape: ``representation_relative_l2`` and
   ``score_max_abs_diff`` against theta_repr = 1e-2 / theta_score = 1e-3, plus
   the argmax switch, the softmax total-variation distance and whether the route
   follows the first-mentioned entity.
3. A mutually exclusive six-class attribution restricted to the 96
   new-description examples: route_miss / start_miss / content_byte_wrong /
   frame_byte_wrong / eos_not_stopped / order_or_extra, with the passing cases
   reported separately so the table sums to 96.

The route peak is recorded as -1 when the null row wins (the earlier residual
script read ``argmax(a_hard[:-1])``, which reports record 0 on a null step).

gamma is the generation weight and 1 - gamma the copy weight; nothing here
infers training credit from gamma (M3).  D16-6 (``28031-A_route_lock`` all-NaN)
is skipped and never pooled.

Run: ``python scripts/training/attribute_taiji_package19_2f_new_description_wall.py``
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

torch.set_num_threads(1)

import probe_taiji_content_transport_18_1 as probe_mod  # noqa: E402

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.binding_copy import (  # noqa: E402
    MAX_PREDICTIONS,
    initial_binding_state,
    observe_goal,
    prepare,
)
from taiji.living_core import EOS  # noqa: E402
from taiji.record_relation_core import decode_endpoint  # noqa: E402
from taiji.selection_routing import (  # noqa: E402
    _advanced_state,
    _s_t,
    build_context,
    routing_step,
)
from taiji.selection_routing_checkpoint import (  # noqa: E402
    load_routing_bundle,
    parameter_tensors,
)

SEEDS = (28031, 28032)
ARMS = ("A_route_lock", "B_stop_lock", "C_unlocked")
BUNDLE_DIR = ROOT / "output/taiji_package19_compare"
EVAL_DIR = BUNDLE_DIR
REPORT = ROOT / "reports/taiji_package19_2f_new_description_wall_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2f_new_description_wall_20261104.json"

CLASSES = (
    "route_miss",
    "start_miss",
    "content_byte_wrong",
    "frame_byte_wrong",
    "eos_not_stopped",
    "order_or_extra",
)

THETA_REPR = 1e-2
THETA_SCORE = 1e-3
PAIRS_PER_STRATUM = 20

CONCLUSIONS = (
    "B 臂两 seed 旧描述 96 例整句 1.000/0.9375（池化 0.9688），新描述 96 例 0.2604/0.2292"
    "（池化 0.2448）：全部损失集中在新描述半，分层结论钉死。",
    "新描述半的选择/起点才是缺口：物件 route 0.469/0.375、地点 route 0.844/0.823；"
    "composite_start 物件 0.385/0.375、地点 0.531/0.417；而起点对时 conditional_byte "
    "物件 0.761/0.829、地点 1.000/0.946——起点错了以后内容字节才跟着崩。",
    "换序读数：表示相对 L2 中位 0.246（新）/0.255（易），得分最大差中位 1.19/1.21；"
    "40/40 对超 theta_score=1e-3、40/40 对超 theta_repr=1e-2 → 有序信息确实进了编码。",
    "但硬 argmax 改变 0/40（新）/3/40（易），top 概率 TV 中位 0.016/0.059 → 选择饱和："
    "排名被单一记录先验压住，换序造成的 1.2 分差翻不动排名；这不是'顺序未接线'，"
    "是'有序信号被边距吞掉'。",
    "前 3 字节 teacher-forced 的 decode 起点打分：易层 0/40 改变、40/40→0/40 跟对首实体，"
    "新描述 14/40 改变、27/40→5/40 跟对首实体——易层是记住的常数选择，"
    "新描述换序后仍多数落到物件记录（35/40），硬路由基本不看顺序。",
    "新描述 96 例六类（B 池化，失败占比）：route_miss 51.7%、content_byte_wrong 44.2%、"
    "start_miss 3.4%、eos_not_stopped 0.7%、frame 0%、order_or_extra 0%；"
    "route_miss 占全部 96 例的 39.1%。",
    "判定：顺序信息已编码、但选择塌缩；新描述主因 route_miss（过半失败）→ 墙在"
    "描述↔记录的组合匹配与边距，属训练目标，不是图形接线缺陷。",
)

MINIMAL_NEXT_STEP = {
    "id": "19.2g（提议，未启动，等 owner 裁定）",
    "name": "记录匹配对比边距（slot-aware contrastive margin）",
    "graph_change": "不改图；复用现有 37,120 参数选择头与 4,194 参数 span 头",
    "new_parameters": 0,
    "loss_change": (
        "每个内容起点在现有 route 交叉熵之外，加同 world 同类 hard-negative 的边距项 "
        "max(0, m - (score_gold - score_decoy))，m = 1.0；decoy = 同 world 另一条同类记录"
        "（物件对物件、地点对地点）；两项等权，同批反向传播"
    ),
    "why_this": (
        "读数显示有序信号（1.2 分差）已经存在，但 top 记录领先过大（TV 0.016），"
        "硬选择翻不动。边距项直接把'必须分开 gold 与同类 decoy'写进目标，"
        "不依赖新增容量，也不动 transport/pointer 的既有合成。"
    ),
    "criteria": {
        "new_description_route_miss": "<= 0.05（现 0.391）",
        "composite_start_hit_object": ">= 0.70",
        "composite_start_hit_place": ">= 0.70",
        "conditional_byte_object": ">= 0.90",
        "conditional_byte_place": ">= 0.90",
        "new_description_sentence": ">= 0.80",
        "old_description_sentence": ">= 0.9375（不许回退）",
        "T2_hard_argmax_change": ">= 18/20（现 0/40）",
        "confirm_reads": 0,
    },
    "budget": {
        "adam": 30,
        "cpu": "<= 220（按 19.2c 实测 5.34 CPU/Adam × 30 = 160，加 192 例评与 40 对换序 ≈ 60）",
        "bundles_saved": 1,
        "confirm_reads": 0,
    },
    "preflight": (
        "先做参数健康检查（NaN 即弃，D16-6 教训）；checkpoint 保存路径先 dry-run 再训练；"
        "沿用 19.4 式身份绑定"
    ),
    "extrapolation": (
        "若 30 步诊断过，正式六格重跑按 19.5 实测 ≈29,574 CPU（不在本方案内，需 owner 另批）；"
        "若 30 步 route_miss 不动则停，不追加剂量"
    ),
}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def payload_bytes(fixture: dict) -> tuple[int, ...]:
    target = tuple(fixture["target_bytes"])
    return target[:-1] if target and target[-1] == EOS else target


def place_record(fixture: dict) -> int:
    for index, record in enumerate(fixture["raw_example"]["records"]):
        if record["kind"] == "place" and record["description"] == fixture["place_description"]:
            return index
    raise RuntimeError("place record not found")


def decode_fixture(probe, fixture: dict) -> dict:
    """Free decode; the route, composite row and source are read per position."""
    prepared, context = probe.prepare(fixture)
    state, gate_index = observe_goal(probe.candidate, prepared, gate_index=0)
    previous = EOS
    raw: list[int] = []
    routes: list[int] = []
    composites: list[int] = []
    terminated = False
    with torch.no_grad():
        for _prediction in range(MAX_PREDICTIONS):
            step = routing_step(
                probe.candidate,
                prepared,
                state,
                previous,
                context,
                probe.candidate.head,
                arm=probe.arm,
                gate_index=gate_index,
                span_head=probe.candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
            endpoint = decode_endpoint(step.probabilities, prefix=bytes(raw))
            if endpoint["eos"]:
                terminated = True
                break
            peak = int(torch.argmax(step.a_hard))
            record_count = int(step.a_hard.numel()) - 1
            route = peak if peak < record_count else -1
            row = step.pointer_logits[route] if 0 <= route < len(step.pointer_logits) else None
            composite = int(torch.argmax(row)) if row is not None else -1
            raw.append(endpoint["byte"])
            routes.append(route)
            composites.append(composite)
            if len(raw) > 64:
                break
            state, _, gate_index = _advanced_state(
                probe.candidate, prepared, step, endpoint["byte"]
            )
            previous = endpoint["byte"]
    return {
        "raw": raw,
        "routes": routes,
        "composites": composites,
        "terminated": terminated,
    }


def example_metrics(fixture: dict, decoded: dict) -> dict:
    target = payload_bytes(fixture)
    raw = decoded["raw"]
    mapping = probe_mod.content_positions(fixture)
    out: dict = {
        "world": fixture["world"],
        "index": fixture["index"],
        "new_description": bool(fixture["flags"]["new_description"]),
        "completed": bool(decoded["terminated"] and tuple(raw) == target),
    }
    for name, positions in (
        ("object", probe_mod.OBJECT_POSITIONS),
        ("place", probe_mod.PLACE_POSITIONS),
    ):
        start = positions[0]
        gold_record, gold_offset = mapping[start]
        route = decoded["routes"][start] if start < len(decoded["routes"]) else -1
        composite = decoded["composites"][start] if start < len(decoded["composites"]) else -1
        route_ok = route == gold_record
        start_ok = route_ok and composite == gold_offset
        hits = sum(
            1
            for position in positions
            if position < len(raw) and position < len(target) and raw[position] == target[position]
        )
        out[name] = {
            "route_ok": bool(route_ok),
            "start_ok": bool(start_ok),
            "byte_hits": hits,
            "byte_total": len(positions),
        }
    out["route_both"] = bool(out["object"]["route_ok"] and out["place"]["route_ok"])
    out["start_both"] = bool(out["object"]["start_ok"] and out["place"]["start_ok"])
    return out


def classify(fixture: dict, decoded: dict) -> dict:
    """First divergence against the target, content positions before frame ones."""
    target = payload_bytes(fixture)
    raw = decoded["raw"]
    mapping = probe_mod.content_positions(fixture)
    start_positions = {probe_mod.OBJECT_POSITIONS[0], probe_mod.PLACE_POSITIONS[0]}
    first_mismatch = None
    for position in range(min(len(raw), len(target))):
        if raw[position] != target[position]:
            first_mismatch = position
            break
    if first_mismatch is None:
        if len(raw) == len(target) and decoded["terminated"]:
            return {"bucket": "pass", "position": None, "route": None}
        if len(raw) > len(target):
            return {"bucket": "order_or_extra", "position": len(target), "route": None}
        return {"bucket": "eos_not_stopped", "position": len(raw), "route": None}
    position = first_mismatch
    if position in mapping:
        gold_record, _offset = mapping[position]
        route = decoded["routes"][position] if position < len(decoded["routes"]) else -1
        if position in start_positions:
            if route != gold_record:
                return {"bucket": "route_miss", "position": position, "route": route}
            return {"bucket": "start_miss", "position": position, "route": route}
        return {"bucket": "content_byte_wrong", "position": position, "route": route}
    return {"bucket": "frame_byte_wrong", "position": position, "route": None}


def aggregate(rows: list[dict], new_description: bool | None) -> dict:
    selected = [
        row for row in rows if new_description is None or row["new_description"] is new_description
    ]
    total = len(selected)
    out: dict = {"examples": total}
    if not total:
        return out
    out["completed_count"] = sum(1 for row in selected if row["completed"])
    out["completed"] = out["completed_count"] / total
    for name in ("object", "place"):
        out[f"route_hit_{name}"] = sum(1 for row in selected if row[name]["route_ok"]) / total
        out[f"composite_start_hit_{name}"] = (
            sum(1 for row in selected if row[name]["start_ok"]) / total
        )
        unconditional_hits = sum(row[name]["byte_hits"] for row in selected)
        unconditional_total = sum(row[name]["byte_total"] for row in selected)
        out[f"content_byte_{name}"] = (
            unconditional_hits / unconditional_total if unconditional_total else 0.0
        )
        conditioned = [row for row in selected if row[name]["start_ok"]]
        conditional_hits = sum(row[name]["byte_hits"] for row in conditioned)
        conditional_total = sum(row[name]["byte_total"] for row in conditioned)
        out[f"conditional_byte_{name}"] = (
            conditional_hits / conditional_total if conditional_total else 0.0
        )
        out[f"conditional_examples_{name}"] = len(conditioned)
    out["route_hit_both"] = sum(1 for row in selected if row["route_both"]) / total
    out["composite_start_hit_both"] = sum(1 for row in selected if row["start_both"]) / total
    counts = {bucket: 0 for bucket in CLASSES}
    frame = {"model_generation": 0, "decode_eos_logic": 0}
    passing = 0
    for row in selected:
        bucket = row["bucket"]
        if bucket == "pass":
            passing += 1
        else:
            counts[bucket] += 1
            if bucket == "frame_byte_wrong":
                frame[row["frame_subtype"]] += 1
    out["pass"] = passing
    out["buckets_of_all"] = {bucket: counts[bucket] / total for bucket in CLASSES}
    out["bucket_counts"] = counts
    failures = total - passing
    out["failures"] = failures
    out["buckets_of_failures"] = {
        bucket: (counts[bucket] / failures if failures else 0.0) for bucket in CLASSES
    }
    out["frame_split"] = frame
    return out


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    denominator = max(float(left.norm()), float(right.norm()), 1e-30)
    return float(torch.dist(left, right) / denominator)


def head_scores(head, binding, goal: bytes, keys: tuple[str, ...]) -> torch.Tensor:
    """Frozen 17.3 T2 shape: zero state, records only (null excluded)."""
    context = build_context(binding, goal, keys)
    s_t = torch.zeros(128, dtype=torch.float64)
    previous = torch.zeros(64, dtype=torch.float64)
    with torch.no_grad():
        return head.score_rows(context.goal_vec, context.key_vecs, s_t, previous)[:-1]


def decode_route_scores(candidate, fixture: dict, goal: bytes, arm: str) -> torch.Tensor:
    """Score vector at the object-start position (first three bytes teacher-forced)."""
    binding = candidate.binding
    state = initial_binding_state(binding, f"swap-{fixture['world']}-{fixture['index']}")
    prepared = prepare(binding, state, goal, fixture["records"])
    context = build_context(binding, goal, tuple(fixture["record_keys"]))
    observed, gate_index = observe_goal(candidate, prepared, gate_index=0)
    target = payload_bytes(fixture)
    previous = EOS
    with torch.no_grad():
        for position in range(3):
            step = routing_step(
                candidate,
                prepared,
                observed,
                previous,
                context,
                candidate.head,
                arm=arm,
                gate_index=gate_index,
                span_head=candidate.span_head,
                pointer_temperature=1.0,
                pointer_hard=True,
            )
            observed, _coverage, gate_index = _advanced_state(
                candidate, prepared, step, target[position]
            )
            previous = target[position]
        s_t = _s_t(candidate, prepared, observed, previous)
        previous_embedding = candidate.subject.encoder.embedding(
            torch.tensor(previous, device=candidate.subject.W.device)
        )
        return candidate.head.score_rows(
            context.goal_vec, context.key_vecs, s_t, previous_embedding
        )[:-1]


def swap_pair(candidate, fixture: dict, arm: str) -> dict:
    binding = candidate.binding
    relation = candidate.base.relation
    subject = candidate.subject
    head = candidate.head
    goal_a = fixture["goal"].encode("utf-8")
    goal_b = (
        "让" + fixture["place_description"] + "到" + fixture["object_description"] + "。"
    ).encode("utf-8")
    assert len(goal_a) == len(goal_b) == 36
    assert sorted(goal_a) == sorted(goal_b)
    keys = tuple(fixture["record_keys"])
    with torch.no_grad():
        rep_a = head.pooled(relation, subject, goal_a)
        rep_b = head.pooled(relation, subject, goal_b)
        scores_a = head_scores(head, binding, goal_a, keys)
        scores_b = head_scores(head, binding, goal_b, keys)
        probs_a = torch.softmax(scores_a, dim=0)
        probs_b = torch.softmax(scores_b, dim=0)
        tv = 0.5 * float((probs_a - probs_b).abs().sum())
        arg_a = int(torch.argmax(scores_a))
        arg_b = int(torch.argmax(scores_b))
        bias_free_a = decode_route_scores(candidate, fixture, goal_a, arm)
        bias_free_b = decode_route_scores(candidate, fixture, goal_b, arm)
        actual_arg_a = int(torch.argmax(bias_free_a))
        actual_arg_b = int(torch.argmax(bias_free_b))
        actual_diff = float((bias_free_a - bias_free_b).abs().max())
    gold_a = int(fixture["selection_label"])
    gold_b = place_record(fixture)
    return {
        "world": fixture["world"],
        "index": fixture["index"],
        "new_description": bool(fixture["flags"]["new_description"]),
        "representation_relative_l2": relative_l2(rep_a, rep_b),
        "score_max_abs_diff": float((scores_a - scores_b).abs().max()),
        "top_tv_distance": tv,
        "argmax_a": arg_a,
        "argmax_b": arg_b,
        "argmax_changed": arg_a != arg_b,
        "zero_state_follow_a": arg_a == gold_a,
        "zero_state_follow_b": arg_b == gold_b,
        "decode_route_score_max_abs_diff": actual_diff,
        "decode_route_argmax_changed": actual_arg_a != actual_arg_b,
        "decode_route_follow_a": actual_arg_a == gold_a,
        "decode_route_follow_b": actual_arg_b == gold_b,
        "gold_a": gold_a,
        "gold_b": gold_b,
    }


def swap_summary(pairs: list[dict]) -> dict:
    if not pairs:
        return {"pairs": 0}
    repr_values = [pair["representation_relative_l2"] for pair in pairs]
    score_values = [pair["score_max_abs_diff"] for pair in pairs]
    actual_values = [pair["decode_route_score_max_abs_diff"] for pair in pairs]
    return {
        "pairs": len(pairs),
        "median_repr": statistics.median(repr_values),
        "min_repr": min(repr_values),
        "median_score": statistics.median(score_values),
        "min_score": min(score_values),
        "pairs_repr_ge_theta": sum(1 for value in repr_values if value >= THETA_REPR),
        "pairs_score_ge_theta": sum(1 for value in score_values if value >= THETA_SCORE),
        "pairs_argmax_changed": sum(1 for pair in pairs if pair["argmax_changed"]),
        "median_top_tv": statistics.median(pair["top_tv_distance"] for pair in pairs),
        "zero_state_follow_a": sum(1 for pair in pairs if pair["zero_state_follow_a"]),
        "zero_state_follow_b": sum(1 for pair in pairs if pair["zero_state_follow_b"]),
        "median_decode_route_score": statistics.median(actual_values),
        "pairs_decode_route_score_ge_theta": sum(
            1 for value in actual_values if value >= THETA_SCORE
        ),
        "pairs_decode_route_argmax_changed": sum(
            1 for pair in pairs if pair["decode_route_argmax_changed"]
        ),
        "decode_route_follow_a": sum(1 for pair in pairs if pair["decode_route_follow_a"]),
        "decode_route_follow_b": sum(1 for pair in pairs if pair["decode_route_follow_b"]),
        "order_sensitive": bool(
            statistics.median(score_values) >= THETA_SCORE
            and sum(1 for value in score_values if value >= THETA_SCORE) >= 18
        ),
    }


def pair_fixtures(dev: tuple[dict, ...]) -> tuple[list[dict], list[dict]]:
    """20 pairs per stratum, spread over the 12 worlds."""
    easy: list[dict] = []
    new: list[dict] = []
    for world in range(12):
        easy_index = world % 8
        new_index = 8 + (world % 8)
        easy.extend(
            fixture
            for fixture in dev
            if fixture["world"] == world and fixture["index"] == easy_index
        )
        new.extend(
            fixture
            for fixture in dev
            if fixture["world"] == world and fixture["index"] == new_index
        )
    for world in range(8):
        easy_index = (world + 1) % 8
        new_index = 8 + (world + 1) % 8
        easy.extend(
            fixture
            for fixture in dev
            if fixture["world"] == world and fixture["index"] == easy_index
        )
        new.extend(
            fixture
            for fixture in dev
            if fixture["world"] == world and fixture["index"] == new_index
        )
    unique_easy: list[dict] = []
    seen = set()
    for fixture in easy:
        key = (fixture["world"], fixture["index"])
        if key not in seen:
            seen.add(key)
            unique_easy.append(fixture)
    unique_new: list[dict] = []
    seen = set()
    for fixture in new:
        key = (fixture["world"], fixture["index"])
        if key not in seen:
            seen.add(key)
            unique_new.append(fixture)
    return unique_easy[:PAIRS_PER_STRATUM], unique_new[:PAIRS_PER_STRATUM]


def archived_eval(seed: int, arm: str) -> dict:
    path = EVAL_DIR / f"eval300-{seed}-{arm}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    overall = data["overall"]
    return {
        "path": str(path.relative_to(ROOT)),
        "examples": int(overall["examples"]),
        "completed": float(overall["completed"]),
        "new_description": float(overall["subsets"]["new_description"]),
        "new_description_examples": int(overall["subset_denominators"]["new_description"]),
        "easy_stratum": (
            (
                float(overall["completed"]) * int(overall["examples"])
                - float(overall["subsets"]["new_description"])
                * int(overall["subset_denominators"]["new_description"])
            )
            / (int(overall["examples"]) - int(overall["subset_denominators"]["new_description"]))
        ),
    }


def frame_split(fixture: dict, decoded: dict, position: int) -> str:
    target = payload_bytes(fixture)
    raw = decoded["raw"]
    got = raw[position]
    neighbours = {
        target[offset] for offset in (position - 1, position + 1) if 0 <= offset < len(target)
    }
    copied = {target[offset] for offset in probe_mod.CONTENT_POSITIONS if offset < len(target)}
    if got in neighbours or got in copied:
        return "decode_eos_logic"
    return "model_generation"


def main() -> int:
    start_cpu = time.process_time()
    start_wall = time.perf_counter()
    payload = premise.build_payload()
    dev = premise.fixtures(payload, "dev")
    easy_pairs, new_pairs = pair_fixtures(dev)

    cells: dict[str, dict] = {}
    hybrid: list[str] = []
    for seed in SEEDS:
        for arm in ARMS:
            path = BUNDLE_DIR / f"{seed}-{arm}-step300.pt"
            candidate = load_routing_bundle(path)[0]
            name = f"{seed}-{arm}"
            tensors = parameter_tensors(candidate)
            nan_parameters = sum(int(torch.isnan(value).sum()) for value in tensors.values())
            if nan_parameters:
                hybrid.append(name)
                cells[name] = {
                    "seed": seed,
                    "arm": arm,
                    "skipped": True,
                    "nan_parameters": nan_parameters,
                    "d16_6": "D16-6: all-NaN cell, excluded from every pooled reading",
                }
                print(json.dumps({"cell": name, "skipped": True}), flush=True)
                continue
            probe = probe_mod.ProbeContext(candidate)
            probe.arm = arm
            rows = []
            for fixture in dev:
                decoded = decode_fixture(probe, fixture)
                verdict = classify(fixture, decoded)
                if verdict["bucket"] == "frame_byte_wrong":
                    verdict["frame_subtype"] = frame_split(fixture, decoded, verdict["position"])
                metrics = example_metrics(fixture, decoded)
                metrics["bucket"] = verdict["bucket"]
                metrics["frame_subtype"] = verdict.get("frame_subtype")
                metrics["position"] = verdict["position"]
                rows.append(metrics)
            swap_easy = [swap_pair(candidate, fixture, arm) for fixture in easy_pairs]
            swap_new = [swap_pair(candidate, fixture, arm) for fixture in new_pairs]
            cells[name] = {
                "seed": seed,
                "arm": arm,
                "skipped": False,
                "nan_parameters": 0,
                "full_192": aggregate(rows, None),
                "old_description_96": aggregate(rows, False),
                "new_description_96": aggregate(rows, True),
                "swap_easy": swap_summary(swap_easy),
                "swap_new": swap_summary(swap_new),
                "archived": archived_eval(seed, arm),
                "examples_detail": [
                    {
                        "world": row["world"],
                        "index": row["index"],
                        "new_description": row["new_description"],
                        "completed": row["completed"],
                        "bucket": row["bucket"],
                        "position": row["position"],
                        "frame_subtype": row["frame_subtype"],
                        "object_route": row["object"]["route_ok"],
                        "place_route": row["place"]["route_ok"],
                        "object_start": row["object"]["start_ok"],
                        "place_start": row["place"]["start_ok"],
                    }
                    for row in rows
                ],
            }
            print(
                json.dumps(
                    {
                        "cell": name,
                        "old96": round(cells[name]["old_description_96"]["completed"], 4),
                        "new96": round(cells[name]["new_description_96"]["completed"], 4),
                        "swap_easy_median_score": round(
                            cells[name]["swap_easy"]["median_score"], 6
                        ),
                        "swap_new_median_score": round(cells[name]["swap_new"]["median_score"], 6),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    healthy = [name for name in cells if not cells[name]["skipped"]]
    groups = arm_groups(cells, healthy)
    focus_swap = focus_view(cells, groups)
    verdict = decision_view(cells, groups, focus_swap, healthy)
    cpu = time.process_time() - start_cpu
    wall = time.perf_counter() - start_wall
    report = report_payload(
        cells, healthy, hybrid, groups, focus_swap, verdict, len(dev), cpu, wall
    )
    write_json(REPORT, report)
    write_json(MANIFEST, manifest_payload(report))
    print(
        json.dumps(
            {"report": str(REPORT), "cpu_seconds": round(cpu, 3), "branch": verdict["branch"]}
        )
    )
    return 0


STRATA = ("old_description_96", "new_description_96")
POOL_FIELDS = (
    "completed",
    "route_hit_object",
    "route_hit_place",
    "route_hit_both",
    "composite_start_hit_object",
    "composite_start_hit_place",
    "composite_start_hit_both",
    "content_byte_object",
    "content_byte_place",
    "conditional_byte_object",
    "conditional_byte_place",
)


def stratum_pool(cells: dict, members: list[str], stratum: str) -> dict:
    out = {
        field: statistics.fmean(cells[name][stratum][field] for name in members)
        for field in POOL_FIELDS
    }
    out["buckets_of_all"] = {
        bucket: statistics.fmean(cells[name][stratum]["buckets_of_all"][bucket] for name in members)
        for bucket in CLASSES
    }
    out["buckets_of_failures"] = {
        bucket: statistics.fmean(
            cells[name][stratum]["buckets_of_failures"][bucket] for name in members
        )
        for bucket in CLASSES
    }
    out["bucket_counts"] = {
        bucket: sum(cells[name][stratum]["bucket_counts"][bucket] for name in members)
        for bucket in CLASSES
    }
    out["pass_count"] = sum(cells[name][stratum]["pass"] for name in members)
    out["examples"] = sum(cells[name][stratum]["examples"] for name in members)
    return out


def arm_groups(cells: dict, healthy: list[str]) -> dict:
    groups: dict = {}
    for arm in ARMS:
        members = [name for name in healthy if name.endswith("-" + arm)]
        if not members:
            continue
        groups[arm] = {
            "cells": members,
            **{stratum: stratum_pool(cells, members, stratum) for stratum in STRATA},
        }
    return groups


def swap_pool(cells: dict, members: list[str], stratum: str) -> dict:
    out = {
        "pairs": sum(cells[name][stratum]["pairs"] for name in members),
        "pairs_score_ge_theta": sum(
            cells[name][stratum]["pairs_score_ge_theta"] for name in members
        ),
        "pairs_argmax_changed": sum(
            cells[name][stratum]["pairs_argmax_changed"] for name in members
        ),
        "pairs_repr_ge_theta": sum(cells[name][stratum]["pairs_repr_ge_theta"] for name in members),
        "zero_state_follow_a": sum(cells[name][stratum]["zero_state_follow_a"] for name in members),
        "zero_state_follow_b": sum(cells[name][stratum]["zero_state_follow_b"] for name in members),
        "pairs_decode_route_score_ge_theta": sum(
            cells[name][stratum]["pairs_decode_route_score_ge_theta"] for name in members
        ),
        "pairs_decode_route_argmax_changed": sum(
            cells[name][stratum]["pairs_decode_route_argmax_changed"] for name in members
        ),
        "decode_route_follow_a": sum(
            cells[name][stratum]["decode_route_follow_a"] for name in members
        ),
        "decode_route_follow_b": sum(
            cells[name][stratum]["decode_route_follow_b"] for name in members
        ),
        "order_sensitive_cells": [
            name for name in members if cells[name][stratum]["order_sensitive"]
        ],
    }
    for field in (
        "median_repr",
        "min_repr",
        "median_score",
        "min_score",
        "median_top_tv",
        "median_decode_route_score",
    ):
        out[field] = statistics.fmean(cells[name][stratum][field] for name in members)
    return out


def focus_view(cells: dict, groups: dict) -> dict:
    b_cells = groups["B_stop_lock"]["cells"]
    return {
        "B_cells": b_cells,
        "easy": swap_pool(cells, b_cells, "swap_easy"),
        "new": swap_pool(cells, b_cells, "swap_new"),
        "C_cells": groups.get("C_unlocked", {}).get("cells", []),
    }


def decision_view(cells: dict, groups: dict, focus_swap: dict, healthy: list[str]) -> dict:
    b_new = groups["B_stop_lock"]["new_description_96"]
    b_old = groups["B_stop_lock"]["old_description_96"]
    b_cells = groups["B_stop_lock"]["cells"]
    new_all = b_new["buckets_of_all"]
    new_failures = b_new["buckets_of_failures"]
    order_sensitive = all(
        name in focus_swap["easy"]["order_sensitive_cells"] for name in b_cells
    ) and all(name in focus_swap["new"]["order_sensitive_cells"] for name in b_cells)
    route_miss_dominant = new_failures["route_miss"] >= 0.5 and new_failures["route_miss"] == max(
        new_failures.values()
    )
    if order_sensitive and route_miss_dominant:
        branch = "combination_matching_capacity"
    elif not order_sensitive:
        branch = "ordered_encoding_not_wired"
    else:
        branch = "downstream_start_content"
    a_cells = groups.get("A_route_lock", {}).get("cells", [])
    return {
        "sentence_split": {
            name: {
                "old_description_96": cells[name]["old_description_96"]["completed"],
                "new_description_96": cells[name]["new_description_96"]["completed"],
                "full_192": (
                    cells[name]["old_description_96"]["completed"] * 96
                    + cells[name]["new_description_96"]["completed"] * 96
                )
                / 192,
            }
            for name in healthy
        },
        "B_pooled_new_description": {
            "examples": b_new["examples"],
            "pass": b_new["pass_count"],
            "failures": b_new["examples"] - b_new["pass_count"],
            "completed": b_new["completed"],
            "old_description_completed": b_old["completed"],
        },
        "order_sensitive": order_sensitive,
        "hard_argmax_saturated": bool(
            focus_swap["new"]["pairs_argmax_changed"] <= 0.1 * focus_swap["new"]["pairs"]
            and focus_swap["new"]["median_top_tv"] <= 0.1
        ),
        "new_description_route_miss_share_of_96": new_all["route_miss"],
        "new_description_route_miss_share_of_failures": new_failures["route_miss"],
        "new_description_content_byte_wrong_share_of_failures": new_failures["content_byte_wrong"],
        "A_arm_status": {
            "cells": a_cells,
            "reading": "0/192 in both seeds; 28031-A all-NaN (D16-6), 28032-A finite but emits a wrong first byte on every example (already registered A-arm collapse)",
            "used_in_decision": False,
        },
        "branch": branch,
    }


def report_payload(
    cells: dict,
    healthy: list[str],
    hybrid: list[str],
    groups: dict,
    focus_swap: dict,
    verdict: dict,
    examples: int,
    cpu: float,
    wall: float,
) -> dict:
    return {
        "schema": "taiji-package19-2f-new-description-wall-v1",
        "step": "N19.2f / new-description wall attribution (0 updates)",
        "date": "2026-11-04",
        "purpose": (
            "locate the 19.5 failure inside the 96 new-description examples and test whether the "
            "goal encoder is order-sensitive at all"
        ),
        "design": {
            "bundles": str(BUNDLE_DIR.relative_to(ROOT)),
            "bundles_are": "19.5 six step300 cells (no 19.2f training)",
            "examples_per_cell": examples,
            "strata": "96 old-description (indices 0-7) and 96 new-description (indices 8-15)",
            "decode": "free decode, pointer_hard, temperature 1.0, prefix=bytes(raw)",
            "sentence_gate": "exact 27-byte action and EOS",
            "route_null_rule": "argmax(a_hard) at the record count means null and is recorded as -1",
            "conditional_byte": "byte-weighted inside one entity, conditioned on that entity's start hit",
            "content_byte": "unconditional bytes of one entity over all its examples",
            "swap_witness": "goal byte multiset held fixed; the two description tokens are exchanged",
            "swap_metric": "17.3 T2 shape: relative L2 of pooled goal, max-abs record score diff, zero state",
            "decode_route_metric": "same swap at the object-start step (three bytes teacher-forced)",
            "theta_repr": THETA_REPR,
            "theta_score": THETA_SCORE,
            "pooling": "B and C arms are pooled separately; A is a control only",
            "saves": 0,
            "confirm_reads": 0,
            "adam_steps": 0,
        },
        "pooled_healthy_cells": healthy,
        "skipped_cells": hybrid,
        "by_cell": cells,
        "pooled_by_arm": groups,
        "focus_swap": focus_swap,
        "decision": verdict,
        "conclusions": list(CONCLUSIONS),
        "minimal_next_step": MINIMAL_NEXT_STEP,
        "cost": {
            "cpu_seconds": cpu,
            "wall_seconds": wall,
            "adam_updates": 0,
            "bundles_saved": 0,
            "confirm_reads": 0,
            "budget_cpu": 150.0,
            "over_budget": cpu > 150.0,
        },
    }


def manifest_payload(report: dict) -> dict:
    verdict = report["decision"]
    focus = report["focus_swap"]
    cost = report.get("cost", {})
    budget = cost.get("budget_cpu", 150.0)
    return {
        "schema": "taiji-package19-new-description-wall-manifest-v1",
        "step": "N19.2f new-description wall attribution",
        "date": "2026-11-04",
        "report": str(REPORT.relative_to(ROOT)),
        "cells": {
            "healthy": report["pooled_healthy_cells"],
            "skipped": report["skipped_cells"],
        },
        "B_strata": {
            "old_description_96": report["pooled_by_arm"]["B_stop_lock"]["old_description_96"][
                "completed"
            ],
            "new_description_96": report["pooled_by_arm"]["B_stop_lock"]["new_description_96"][
                "completed"
            ],
            "pass_new": report["pooled_by_arm"]["B_stop_lock"]["new_description_96"]["pass_count"],
            "fail_new": report["pooled_by_arm"]["B_stop_lock"]["new_description_96"]["examples"]
            - report["pooled_by_arm"]["B_stop_lock"]["new_description_96"]["pass_count"],
        },
        "swap": {
            "theta_repr": THETA_REPR,
            "theta_score": THETA_SCORE,
            "order_sensitive": verdict["order_sensitive"],
            "hard_argmax_saturated": verdict["hard_argmax_saturated"],
            "easy_median_score": focus["easy"]["median_score"],
            "new_median_score": focus["new"]["median_score"],
            "new_argmax_changed": focus["new"]["pairs_argmax_changed"],
        },
        "new_description_buckets_of_failures": report["pooled_by_arm"]["B_stop_lock"][
            "new_description_96"
        ]["buckets_of_failures"],
        "branch": verdict["branch"],
        "cost_cpu": cost.get("cpu_seconds", 0.0),
        "budget_cpu": budget,
        "over_budget": cost.get("over_budget", cost.get("cpu_seconds", 0.0) > budget),
        "adam_steps": 0,
        "bundles_saved": 0,
        "confirm_reads": 0,
    }


def reaggregate() -> int:
    """Recompute the pooled/decision sections from the stored per-cell readings."""
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    cells = report["by_cell"]
    healthy = [name for name in cells if not cells[name]["skipped"]]
    groups = arm_groups(cells, healthy)
    focus_swap = focus_view(cells, groups)
    verdict = decision_view(cells, groups, focus_swap, healthy)
    report["pooled_by_arm"] = groups
    report["focus_swap"] = focus_swap
    report["decision"] = verdict
    report["conclusions"] = list(CONCLUSIONS)
    report["minimal_next_step"] = MINIMAL_NEXT_STEP
    report.pop("pooled_strata", None)
    cost = report.setdefault("cost", {})
    cost.setdefault("budget_cpu", 150.0)
    cost.setdefault("over_budget", cost.get("cpu_seconds", 0.0) > cost["budget_cpu"])
    write_json(REPORT, report)
    write_json(MANIFEST, manifest_payload(report))
    print(json.dumps({"reaggregated": str(REPORT), "branch": verdict["branch"]}))
    return 0


if __name__ == "__main__":
    if "--reaggregate" in sys.argv:
        raise SystemExit(reaggregate())
    raise SystemExit(main())
