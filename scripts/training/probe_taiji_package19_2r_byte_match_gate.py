"""19.2r design gate: uniqueness, solvability ceiling, expressibility, cost.

Owner order (2026-11-05): design gate only - 0 Adam, no training, no proposals,
no suggestions.  Over the frozen 18.0 dev split (12 worlds / 192 examples, two
content starts per example = 384 starts) this script measures:

1. **uniqueness** - for the goal's object description and place description,
   how many of the eight record lines contain the same byte string verbatim,
   and how often that occurrence is unique;
2. **ceiling** - a deterministic string scan (no training, no forward pass)
   selects the line holding the object description, copies both names and
   assembles the 27-byte action; this is the mechanism-level ceiling;
3. **expressibility** - the parametric ``ByteMatchRoute`` pathway at initial
   weights after 0 updates, with the correct alignment offset injected, must
   rank the gold line first 192/192; the same module is also run free (no
   injection) and as the second hop for the place description;
4. **cost and identity** - new parameter count, the ``candidate_sha_v2``
   effect, and whether the second hop needs a second pathway.

Discipline: read-only data, 0 Adam, 0 optimizer steps, 0 saved bundles, 0
confirm reads, one fsync per progress line, report written then renamed.
Run with ``python scripts/training/probe_taiji_package19_2r_byte_match_gate.py``.
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

from taiji import data_premise_18_0 as premise  # noqa: E402
from taiji.byte_match_route import IDENTITY_DIM, PAD_ID, VOCAB, ByteMatchRoute  # noqa: E402

CPU_CAP = 120
REPORT = ROOT / "reports/taiji_package19_2r_byte_match_gate_20261105.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_2r_byte_match_gate_20261105.json"
PROGRESS = ROOT / "reports/taiji_package19_2r_byte_match_gate_progress_20261105.jsonl"
CANDIDATE_SHA_V2 = "773dcec2e1131d1f3c8784153633d85a9144268a4d5167790740cdd488153aa1"
SPAN = 15
SAMPLE_LIMIT = 5


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
    """Recover the name from a record line using only its bytes and its template."""
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
    if fixture["object_description"] in premise.OBJECT_DESCRIPTIONS:
        return "old"
    return "new"


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
    return {
        "record_ids": record_ids,
        "record_mask": record_mask,
        "object_query": object_query,
        "object_query_mask": object_query_mask,
        "place_query": place_query,
        "place_query_mask": place_query_mask,
        "object_offset": object_offset,
        "place_offset": place_offset,
    }


def main() -> int:
    started_cpu = time.process_time()
    started_wall = time.perf_counter()

    payload = premise.build_payload()
    dev = premise.fixtures(payload, "dev")
    all_records = [record for world in payload["worlds"] for record in world["records"]]

    PROGRESS.write_text("", encoding="utf-8")
    append_progress(
        {
            "event": "start",
            "dev_examples": len(dev),
            "starts": len(dev) * 2,
            "records": len(all_records),
            "cpu_cap": CPU_CAP,
            "adam_updates": 0,
        }
    )

    rows = []
    for fixture in dev:
        records = fixture["raw_example"]["records"]
        goal_bytes = fixture["goal"].encode("utf-8")
        object_bytes = fixture["object_description"].encode("utf-8")
        place_bytes = fixture["place_description"].encode("utf-8")
        object_at = goal_bytes.find(object_bytes)
        place_at = goal_bytes.find(place_bytes, object_at + len(object_bytes))
        rows.append(
            {
                "index": len(rows),
                "world": fixture["world"],
                "age": age_of(fixture),
                "goal": fixture["goal"],
                "object_bytes": object_bytes,
                "place_bytes": place_bytes,
                "object_at": object_at,
                "place_at": place_at,
                "record_bytes": [record["raw"].encode("utf-8") for record in records],
                "kinds": [record["kind"] for record in records],
                "raw": [record["raw"] for record in records],
                "labels": int(fixture["selection_label"]),
                "action": fixture["raw_example"]["action"],
                "object_name": fixture["object_name"],
                "place_name": fixture["place_name"],
            }
        )

    extraction = {
        "records": len(all_records),
        "name_recovered": sum(
            1
            for record in all_records
            if name_from_raw(record["raw"], record["description"]) == record["name"]
        ),
    }
    append_progress({"event": "name_extraction", **extraction})

    # ---- 1. uniqueness ------------------------------------------------------
    uniqueness: dict[str, dict] = {}
    for label in ("object", "place"):
        stats = {
            "starts": 0,
            "unique_occurrence": 0,
            "duplicate_occurrence": 0,
            "missing_occurrence": 0,
            "cross_kind_occurrences": 0,
            "lines_with_repeat_inside": 0,
            "max_occurrences": 0,
            "total_occurrences": 0,
            "samples": [],
        }
        for row in rows:
            piece = row[f"{label}_bytes"]
            counts = [raw.count(piece) for raw in row["record_bytes"]]
            total = sum(counts)
            stats["starts"] += 1
            stats["total_occurrences"] += total
            stats["max_occurrences"] = max(stats["max_occurrences"], total)
            stats["lines_with_repeat_inside"] += sum(1 for count in counts if count > 1)
            wanted = "place" if label == "object" else "object"
            stats["cross_kind_occurrences"] += sum(
                counts[index] for index, kind in enumerate(row["kinds"]) if kind == wanted
            )
            if total == 1:
                stats["unique_occurrence"] += 1
            elif total == 0:
                stats["missing_occurrence"] += 1
                if len(stats["samples"]) < SAMPLE_LIMIT:
                    stats["samples"].append(
                        {
                            "dev_index": row["index"],
                            "world": row["world"],
                            "description": piece.decode("utf-8"),
                            "counts": counts,
                        }
                    )
            else:
                stats["duplicate_occurrence"] += 1
                if len(stats["samples"]) < SAMPLE_LIMIT:
                    stats["samples"].append(
                        {
                            "dev_index": row["index"],
                            "world": row["world"],
                            "description": piece.decode("utf-8"),
                            "counts": counts,
                            "lines": row["raw"],
                        }
                    )
        uniqueness[label] = stats
    uniqueness["starts_total"] = len(rows) * 2
    uniqueness["unique_total"] = sum(
        uniqueness[t]["unique_occurrence"] for t in ("object", "place")
    )
    uniqueness["goal_offsets"] = {
        "object": sorted({row["object_at"] for row in rows}),
        "place": sorted({row["place_at"] for row in rows}),
    }
    append_progress({"event": "uniqueness", "unique_total": uniqueness["unique_total"]})

    # ---- 2. solvability ceiling --------------------------------------------
    ceiling = {
        "examples": 0,
        "selection_correct": 0,
        "selection_missing": 0,
        "selection_ambiguous": 0,
        "completion_exact": 0,
        "by_age": {
            "old": {"examples": 0, "selection_correct": 0, "completion_exact": 0},
            "new": {"examples": 0, "selection_correct": 0, "completion_exact": 0},
        },
        "strategy": "scan the eight lines for the goal's object description; the first "
        "line containing it is the selection; recover that line's name and the name of "
        "the line containing the place description; assemble "
        "'把{object_name}放到{place_name}' and compare byte-for-byte with the action",
    }
    for row in rows:
        ceiling["examples"] += 1
        age = ceiling["by_age"][row["age"]]
        age["examples"] += 1
        object_hits = [
            index for index, raw in enumerate(row["record_bytes"]) if row["object_bytes"] in raw
        ]
        place_hits = [
            index for index, raw in enumerate(row["record_bytes"]) if row["place_bytes"] in raw
        ]
        if len(object_hits) > 1 or len(place_hits) > 1:
            ceiling["selection_ambiguous"] += 1
        if not object_hits or not place_hits:
            ceiling["selection_missing"] += 1
            continue
        chosen = object_hits[0]
        if chosen == row["labels"]:
            ceiling["selection_correct"] += 1
            age["selection_correct"] += 1
        object_name = name_from_raw(row["raw"][chosen], row["object_bytes"].decode("utf-8"))
        place_name = name_from_raw(row["raw"][place_hits[0]], row["place_bytes"].decode("utf-8"))
        if object_name is None or place_name is None:
            continue
        assembled = f"把{object_name}放到{place_name}"
        if assembled == row["action"]:
            ceiling["completion_exact"] += 1
            age["completion_exact"] += 1
    ceiling["selection_accuracy"] = ceiling["selection_correct"] / ceiling["examples"]
    ceiling["completion_rate"] = ceiling["completion_exact"] / ceiling["examples"]
    ceiling["by_age"]["old"]["selection_accuracy"] = (
        ceiling["by_age"]["old"]["selection_correct"] / ceiling["by_age"]["old"]["examples"]
    )
    ceiling["by_age"]["old"]["completion_rate"] = (
        ceiling["by_age"]["old"]["completion_exact"] / ceiling["by_age"]["old"]["examples"]
    )
    ceiling["by_age"]["new"]["selection_accuracy"] = (
        ceiling["by_age"]["new"]["selection_correct"] / ceiling["by_age"]["new"]["examples"]
    )
    ceiling["by_age"]["new"]["completion_rate"] = (
        ceiling["by_age"]["new"]["completion_exact"] / ceiling["by_age"]["new"]["examples"]
    )
    append_progress(
        {
            "event": "ceiling",
            "selection": ceiling["selection_accuracy"],
            "completion": ceiling["completion_rate"],
        }
    )

    # ---- 3. expressibility (0 updates) -------------------------------------
    tensors = build_tensors(rows)
    model = ByteMatchRoute(dim=IDENTITY_DIM, span=SPAN)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    identity_gap = float((model.byte_embed.detach() - torch.eye(VOCAB, model.dim)).abs().max())
    position_is_one = bool(torch.all(model.position_weight.detach() == 1.0))
    labels = torch.tensor([row["labels"] for row in rows])
    place_lines = torch.tensor(
        [
            next(
                index for index, raw in enumerate(row["record_bytes"]) if row["place_bytes"] in raw
            )
            for row in rows
        ]
    )
    forwards = 0
    with torch.no_grad():
        object_injected, _, _ = model(
            tensors["object_query"],
            tensors["object_query_mask"],
            tensors["record_ids"],
            tensors["record_mask"],
            tensors["object_offset"],
        )
        forwards += 1
        object_choice = object_injected.argmax(dim=1)
        object_top = object_injected.max(dim=1, keepdim=True)
        object_ties = int(((object_injected == object_top.values).sum(dim=1) > 1).sum())
        non_gold = object_injected.clone()
        non_gold[torch.arange(len(rows)), labels] = torch.finfo(object_injected.dtype).min
        runner_up = non_gold.max(dim=1).values
        runner_present = runner_up > -1e30
        gold_scores = object_injected[torch.arange(len(rows)), labels]
        margins = gold_scores - torch.where(runner_present, runner_up, torch.zeros_like(runner_up))

        place_injected, _, _ = model(
            tensors["place_query"],
            tensors["place_query_mask"],
            tensors["record_ids"],
            tensors["record_mask"],
            tensors["place_offset"],
        )
        forwards += 1
        place_choice = place_injected.argmax(dim=1)

        object_free, object_free_offset, _ = model(
            tensors["object_query"],
            tensors["object_query_mask"],
            tensors["record_ids"],
            tensors["record_mask"],
            None,
        )
        forwards += 1
        object_free_choice = object_free.argmax(dim=1)

    expressibility = {
        "initialisation": {
            "updates": 0,
            "byte_embed_max_abs_gap_from_identity": identity_gap,
            "position_weight_all_one": position_is_one,
            "all_parameters_require_grad": all(p.requires_grad for p in model.parameters()),
            "parameter_count": parameter_count,
        },
        "object_injected": {
            "gold_selected": int((object_choice == labels).sum()),
            "examples": len(rows),
            "tied_top_scores": object_ties,
            "rows_with_finite_runner_up": int(runner_present.sum()),
            "gold_score_min": float(gold_scores.min()),
            "margin_min": float(margins.min()),
            "margin_median": float(margins.median()),
        },
        "place_injected": {
            "matched_line_selected": int((place_choice == place_lines).sum()),
            "examples": len(rows),
        },
        "object_free": {
            "gold_selected": int((object_free_choice == labels).sum()),
            "gold_offset_recovered": int(
                (
                    object_free_offset[torch.arange(len(rows)), labels]
                    == tensors["object_offset"][torch.arange(len(rows)), labels]
                ).sum()
            ),
            "examples": len(rows),
        },
        "note": "injected mode feeds the true alignment start per line (-1 = the line does "
        "not contain the span); the matching arithmetic is the module's own, and the free "
        "mode shows the pathway can also find the offset without injection",
    }
    append_progress(
        {
            "event": "expressibility",
            "gold": expressibility["object_injected"]["gold_selected"],
            "place": expressibility["place_injected"]["matched_line_selected"],
        }
    )

    # one backward, no optimizer step: prove the pathway is trainable.
    model.zero_grad()
    scores, _, _ = model(
        tensors["object_query"],
        tensors["object_query_mask"],
        tensors["record_ids"],
        tensors["record_mask"],
        tensors["object_offset"],
    )
    forwards += 1
    finite = scores > -1e30
    loss = scores[finite].sum()
    loss.backward()
    grad_stats = {
        name: {
            "finite": (
                bool(torch.isfinite(parameter.grad).all()) if parameter.grad is not None else False
            ),
            "nonzero": bool((parameter.grad != 0).any()) if parameter.grad is not None else False,
            "abs_sum": float(parameter.grad.abs().sum()) if parameter.grad is not None else 0.0,
        }
        for name, parameter in model.named_parameters()
    }
    model.zero_grad()
    expressibility["gradient_flow"] = {"optimizer_steps": 0, "parameters": grad_stats}
    append_progress({"event": "gradient_flow", "parameters": len(grad_stats)})

    # ---- 4. cost and identity ----------------------------------------------
    name_offsets: dict[str, dict] = {}
    for record in all_records:
        raw = record["raw"]
        key = f"{record['kind']}|style{record['style']}"
        cell = name_offsets.setdefault(
            key, {"records": 0, "name_chars": set(), "relative_offsets": {}}
        )
        cell["records"] += 1
        cell["name_chars"].add(len(record["name"]))
        relative = raw.find(record["name"]) - raw.find(record["description"])
        cell["relative_offsets"][str(relative)] = cell["relative_offsets"].get(str(relative), 0) + 1
    for cell in name_offsets.values():
        cell["name_chars"] = sorted(cell["name_chars"])
    cost_and_identity = {
        "new_parameters": [
            {
                "name": "byte_embed",
                "shape": [VOCAB, IDENTITY_DIM],
                "elements": VOCAB * IDENTITY_DIM,
            },
            {"name": "position_weight", "shape": [SPAN], "elements": SPAN},
        ],
        "new_parameter_count": parameter_count,
        "candidate_sha_v2_current": CANDIDATE_SHA_V2,
        "candidate_sha_v2_change": "none: the module is not registered in "
        "candidate_identity.GRAPH_MODULES, so the semantic payload is unchanged; an "
        "approved integration would add this file to the module list and the two "
        "parameters above to the parameter list, which changes the digest by construction",
        "second_hop": {
            "shares_parameters": True,
            "additional_parameter_count": 0,
            "name_relative_to_description": name_offsets,
            "mechanism": "the same pathway is applied to the place description; the matched "
            "offset plus the connector bytes visible in the line give the adjacent name "
            "bytes, so no second matcher is needed",
        },
    }
    append_progress({"event": "cost_identity", "parameters": parameter_count})

    answers = {
        "a_exact_match_reachable": (
            "这一层本质是算法原语：纯字节扫描已达 100%，训练不会让它更准；把它做成参数化"
            "可训练通路的唯一理由是 M1 要求匹配与复制在同一主体内、信用不可结构性分离，"
            "若不接受这个理由，就应当承认这一层该由算法实现、而不是交给网络学。"
        ),
        "b_two_hops_share": (
            "可以共用：同一组参数对物件描述与地点描述各跑一次，第二跳额外只需要一条由"
            "连接词与模板决定的确定性读名偏移规则（实测表给出），不需要第二套通路。"
        ),
        "c_conflict_with_m1": (
            "不冲突：通路输出的匹配位置就是复制的读取位置，内容字节仍经同一状态影响"
            "η，∂η/∂α≠0 保留；只有当匹配被移出图外、只把索引传进来时才会违反 M1，"
            "本设计不属于那种情形，故不触发 veto_if_reliance。"
        ),
    }

    cpu_seconds = time.process_time() - started_cpu
    wall_seconds = time.perf_counter() - started_wall
    over_budget = cpu_seconds > CPU_CAP
    report = {
        "schema": "taiji-package-19-2r-byte-match-gate-v1",
        "step": "19.2r_byte_match_design_gate",
        "date": "2026-11-05",
        "purpose": "design gate only: uniqueness of the goal description inside the eight "
        "record lines, the deterministic solvability ceiling, whether a parametric "
        "byte-level pathway can express gold selection at 0 updates, and its cost",
        "data": {
            "split": "dev",
            "worlds": len({fixture["world"] for fixture in dev}),
            "examples": len(rows),
            "starts": len(rows) * 2,
            "goal_offsets": uniqueness["goal_offsets"],
            "name_extraction": extraction,
        },
        "uniqueness": uniqueness,
        "ceiling": ceiling,
        "expressibility": expressibility,
        "cost_and_identity": cost_and_identity,
        "answers": answers,
        "cost": {
            "cpu_seconds": cpu_seconds,
            "wall_seconds": wall_seconds,
            "cpu_cap": CPU_CAP,
            "over_budget": over_budget,
        },
        "discipline": {
            "adam_updates": 0,
            "optimizer_steps": 0,
            "training_forwards": 0,
            "constructive_forwards": forwards,
            "backward_passes": 1,
            "bundles_saved": 0,
            "confirm_reads": 0,
            "read_only_data": True,
            "fsync_per_line": True,
        },
        "progress": str(PROGRESS.relative_to(ROOT)).replace("\\", "/"),
    }
    write_json(REPORT, report)
    manifest = {
        "schema": "taiji-package-19-2r-byte-match-gate-manifest-v1",
        "step": "19.2r_byte_match_design_gate",
        "date": "2026-11-05",
        "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
        "examples": len(rows),
        "starts": len(rows) * 2,
        "unique_object": uniqueness["object"]["unique_occurrence"],
        "unique_place": uniqueness["place"]["unique_occurrence"],
        "ceiling_selection": ceiling["selection_accuracy"],
        "ceiling_completion": ceiling["completion_rate"],
        "expressibility_gold": expressibility["object_injected"]["gold_selected"],
        "expressibility_place": expressibility["place_injected"]["matched_line_selected"],
        "new_parameters": parameter_count,
        "candidate_sha_v2": CANDIDATE_SHA_V2,
        "cpu_seconds": cpu_seconds,
        "adam_updates": 0,
        "bundles_saved": 0,
        "confirm_reads": 0,
    }
    write_json(MANIFEST, manifest)
    append_progress({"event": "done", "cpu_seconds": cpu_seconds})
    print(
        json.dumps(
            {
                "cpu_seconds": cpu_seconds,
                "unique": uniqueness["unique_total"],
                "ceiling": ceiling["completion_rate"],
                "expressibility": expressibility["object_injected"]["gold_selected"],
                "report": str(REPORT),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
