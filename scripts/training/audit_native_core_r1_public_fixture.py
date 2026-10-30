"""Build and validate the public R1 semantic fixture.

The fixture is public synthetic data for contract checks: same query with
different current values, same current with different history, the same query
text with different time intent, corrections, three versions, same-value
versions, distractors, unknown, ambiguity and a non-query receipt. It does not
read old sealed/query/holdout text and does not call a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji.joint_state_consumption_course import E1_ENTITIES  # noqa: E402

MANIFEST = ROOT / "plans/manifests/native_core_r1_public_fixture_v1.json"
REPORT = ROOT / "reports/taiji_native_core_r1_public_fixture_audit_20261030.json"
END_MARKER = "<|end|>"
PUBLIC_ENTITIES = ("雨伞", "背包", "水杯", "耳机", "台灯")
PUBLIC_ATTRIBUTES = {
    "颜色": ("红色", "蓝色", "绿色", "黄色"),
    "位置": ("桌上", "抽屉", "柜子", "包里"),
    "归属": ("我的", "你的", "他的", "她的"),
}


def _target(text: str) -> dict:
    raw = (text + END_MARKER).encode("utf-8")
    return {
        "target_text": text,
        "target_bytes_utf8": list(raw),
        "end_marker": END_MARKER,
        "first_byte": raw[0],
        "visible_prefix_at_t0": [],
        "supervision": "target_t_at_step_t",
        "forbidden": "target[1:]",
    }


def _history(entity: str, attribute: str, values: tuple[str, ...]) -> list[dict]:
    return [
        {
            "revision": revision,
            "value": value,
            "source_text": f"{entity}的{attribute}是{value}。",
            "source_input_id": f"r1-public:{entity}:{attribute}:{revision}",
            "valid_from": revision,
        }
        for revision, value in enumerate(values)
    ]


def _case(
    *,
    case_id: str,
    group_id: str,
    variant: str,
    category: str,
    entity: str,
    attribute: str,
    history: list[dict],
    query_text: str | None,
    time_intent: str | None,
    expected_answer_text: str,
    expected_state: dict | None,
    expected_abstain: bool = False,
    distractor: dict | None = None,
    note: str = "",
    scored: bool = True,
) -> dict:
    return {
        "fixture_id": case_id,
        "group_id": group_id,
        "variant": variant,
        "category": category,
        "entity": entity,
        "attribute": attribute,
        "history": history,
        "query_text": query_text,
        "time_intent": time_intent,
        "expected_answer_text": expected_answer_text,
        "expected_state": expected_state,
        "expected_abstain": expected_abstain,
        "distractor": distractor,
        "note": note,
        "scored": scored,
        "target": _target(expected_answer_text),
        "source": {
            "origin": "public_synthetic",
            "sealed": False,
            "owner": "public",
            "scope": "r1-public",
            "session": case_id,
        },
    }


def _cases() -> list[dict]:
    red, blue, green = "红色", "蓝色", "绿色"
    desk, drawer = "桌上", "抽屉"
    mine, yours = "我的", "你的"
    return [
        _case(
            case_id="same-query-current-a",
            group_id="same-query-current",
            variant="a",
            category="current_query",
            entity="雨伞",
            attribute="颜色",
            history=_history("雨伞", "颜色", (red, blue)),
            query_text="雨伞的颜色是什么？",
            time_intent="current",
            expected_answer_text="雨伞的颜色是蓝色。",
            expected_state={"revision": 1, "value": blue},
            note="same query text, current blue",
        ),
        _case(
            case_id="same-query-current-b",
            group_id="same-query-current",
            variant="b",
            category="current_query",
            entity="雨伞",
            attribute="颜色",
            history=_history("雨伞", "颜色", (blue, green)),
            query_text="雨伞的颜色是什么？",
            time_intent="current",
            expected_answer_text="雨伞的颜色是绿色。",
            expected_state={"revision": 1, "value": green},
            note="same query text, current green",
        ),
        _case(
            case_id="same-current-history-a",
            group_id="same-current-history",
            variant="a",
            category="current_query",
            entity="水杯",
            attribute="颜色",
            history=_history("水杯", "颜色", (red, blue)),
            query_text="水杯的颜色是什么？",
            time_intent="current",
            expected_answer_text="水杯的颜色是蓝色。",
            expected_state={"revision": 1, "value": blue},
            note="same current blue, previous red",
        ),
        _case(
            case_id="same-current-history-b",
            group_id="same-current-history",
            variant="b",
            category="current_query",
            entity="水杯",
            attribute="颜色",
            history=_history("水杯", "颜色", (green, blue)),
            query_text="水杯的颜色是什么？",
            time_intent="current",
            expected_answer_text="水杯的颜色是蓝色。",
            expected_state={"revision": 1, "value": blue},
            note="same current blue, previous green",
        ),
        _case(
            case_id="same-query-time-current",
            group_id="same-query-time",
            variant="current",
            category="current_query",
            entity="背包",
            attribute="颜色",
            history=_history("背包", "颜色", (red, blue)),
            query_text="背包的颜色是什么？",
            time_intent="current",
            expected_answer_text="背包的颜色是蓝色。",
            expected_state={"revision": 1, "value": blue},
            note="same surface query, current",
        ),
        _case(
            case_id="same-query-time-previous",
            group_id="same-query-time",
            variant="previous",
            category="previous_query",
            entity="背包",
            attribute="颜色",
            history=_history("背包", "颜色", (red, blue)),
            query_text="背包的颜色是什么？",
            time_intent="previous",
            expected_answer_text="之前是红色。",
            expected_state={"revision": 0, "value": red},
            note="same surface query, previous",
        ),
        _case(
            case_id="correction",
            group_id="correction",
            variant="single",
            category="current_query",
            entity="耳机",
            attribute="颜色",
            history=_history("耳机", "颜色", (red, blue)),
            query_text="耳机的颜色是什么？",
            time_intent="current",
            expected_answer_text="耳机的颜色是蓝色。",
            expected_state={"revision": 1, "value": blue},
            note="v0 red to v1 blue",
        ),
        _case(
            case_id="three-versions",
            group_id="three-versions",
            variant="single",
            category="previous_query",
            entity="背包",
            attribute="颜色",
            history=_history("背包", "颜色", (red, blue, green)),
            query_text="背包的颜色是什么？",
            time_intent="previous",
            expected_answer_text="之前是蓝色。",
            expected_state={"revision": 1, "value": blue},
            note="v0 red, v1 blue, v2 green; previous is v1, not v0",
        ),
        _case(
            case_id="same-value-version",
            group_id="same-value-version",
            variant="single",
            category="previous_query",
            entity="雨伞",
            attribute="颜色",
            history=_history("雨伞", "颜色", (blue, blue)),
            query_text="雨伞的颜色是什么？",
            time_intent="previous",
            expected_answer_text="之前是蓝色。",
            expected_state={"revision": 0, "value": blue},
            note="same value but previous revision 0, not a different-value dedup",
        ),
        _case(
            case_id="distractor",
            group_id="distractor",
            variant="single",
            category="current_query",
            entity="水杯",
            attribute="颜色",
            history=_history("水杯", "颜色", (red,)),
            query_text="水杯的颜色是什么？",
            time_intent="current",
            expected_answer_text="水杯的颜色是红色。",
            expected_state={"revision": 0, "value": red},
            distractor={
                "entity": "背包",
                "attribute": "颜色",
                "value": green,
                "source_text": "背包的颜色是绿色。",
            },
            note="another entity must not change the target answer",
        ),
        _case(
            case_id="location",
            group_id="location",
            variant="current",
            category="current_query",
            entity="水杯",
            attribute="位置",
            history=_history("水杯", "位置", (desk, drawer)),
            query_text="水杯的位置在哪里？",
            time_intent="current",
            expected_answer_text="水杯的位置是抽屉。",
            expected_state={"revision": 1, "value": drawer},
            note="non-colour attribute",
        ),
        _case(
            case_id="owner",
            group_id="owner",
            variant="current",
            category="current_query",
            entity="台灯",
            attribute="归属",
            history=_history("台灯", "归属", (mine, yours)),
            query_text="台灯归谁？",
            time_intent="current",
            expected_answer_text="台灯是我的。",
            expected_state={"revision": 1, "value": mine},
            note="non-colour attribute",
        ),
        _case(
            case_id="unknown-current",
            group_id="unknown-current",
            variant="single",
            category="unknown_query",
            entity="耳机",
            attribute="位置",
            history=[],
            query_text="耳机的位置在哪里？",
            time_intent="current",
            expected_answer_text="我不知道。",
            expected_state=None,
            expected_abstain=True,
            note="no legal candidate",
        ),
        _case(
            case_id="unknown-previous",
            group_id="unknown-previous",
            variant="single",
            category="unknown_query",
            entity="耳机",
            attribute="颜色",
            history=_history("耳机", "颜色", (red,)),
            query_text="耳机的颜色是什么？",
            time_intent="previous",
            expected_answer_text="我不知道。",
            expected_state=None,
            expected_abstain=True,
            note="no previous version; must not fall back to current",
        ),
        _case(
            case_id="ambiguous",
            group_id="ambiguous",
            variant="single",
            category="ambiguous_query",
            entity="雨伞",
            attribute="颜色",
            history=[],
            query_text="雨伞的颜色是什么？",
            time_intent="current",
            expected_answer_text="我不知道。",
            expected_state=None,
            expected_abstain=True,
            note="ambiguous input does not write a fact",
        ),
        _case(
            case_id="negation",
            group_id="negation",
            variant="single",
            category="ambiguous_query",
            entity="雨伞",
            attribute="颜色",
            history=[],
            query_text="雨伞的颜色是什么？",
            time_intent="current",
            expected_answer_text="我不知道。",
            expected_state=None,
            expected_abstain=True,
            note="negated input does not write a fact",
        ),
        _case(
            case_id="non-query",
            group_id="non-query",
            variant="single",
            category="non_query",
            entity="背包",
            attribute="颜色",
            history=_history("背包", "颜色", (red,)),
            query_text=None,
            time_intent=None,
            expected_answer_text="我记下了，背包的颜色是红色。",
            expected_state={"revision": 0, "value": red},
            note="receipt turn still counted in the fixed denominator",
        ),
    ]


def _canonical(case: dict) -> dict:
    payload = {key: value for key, value in case.items() if key != "source_sha256"}
    return payload


def _sha256_case(case: dict) -> str:
    return hashlib.sha256(
        json.dumps(_canonical(case), ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _manifest() -> dict:
    cases = _cases()
    for case in cases:
        for item in case["history"]:
            item["source_sha256"] = hashlib.sha256(item["source_text"].encode("utf-8")).hexdigest()
        case["source_sha256"] = _sha256_case(case)
    return {
        "format": "taiji-native-core-r1-public-fixture-v1",
        "version": 1,
        "fixture_id": "native-core-r1-public-fixture-20261030",
        "world": {
            "entities": list(PUBLIC_ENTITIES),
            "attributes": {name: list(values) for name, values in PUBLIC_ATTRIBUTES.items()},
            "sealed": False,
        },
        "byte_protocol": {
            "end_marker": END_MARKER,
            "supervision": "target_t_at_step_t",
            "first_byte_supervised": True,
            "forbidden": "target[1:]",
        },
        "cases": cases,
    }


def _check(condition: bool, label: str, details: list[str]) -> None:
    if not condition:
        details.append(label)


def _validate(manifest: dict) -> dict:
    cases = manifest["cases"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for case in cases:
        groups[case["group_id"]].append(case)
    issues: list[str] = []

    def group(name: str) -> list[dict]:
        rows = groups.get(name, [])
        _check(len(rows) >= 2, f"group {name} needs at least two cases", issues)
        return rows

    same_query = group("same-query-current")
    if len(same_query) == 2:
        a, b = same_query
        _check(a["query_text"] == b["query_text"], "same-query text differs", issues)
        _check(
            a["expected_answer_text"] != b["expected_answer_text"],
            "same-query answers do not differ",
            issues,
        )
        _check(
            a["expected_state"]["value"] != b["expected_state"]["value"],
            "same-query current values do not differ",
            issues,
        )

    same_history = group("same-current-history")
    if len(same_history) == 2:
        a, b = same_history
        _check(
            a["expected_answer_text"] == b["expected_answer_text"],
            "same-current answers do not match",
            issues,
        )
        _check(
            a["expected_state"]["value"] == b["expected_state"]["value"],
            "same-current values do not match",
            issues,
        )
        previous_a = a["history"][0]["value"]
        previous_b = b["history"][0]["value"]
        _check(previous_a != previous_b, "same-current histories do not differ", issues)

    same_time = group("same-query-time")
    if len(same_time) == 2:
        current, previous = same_time
        _check(current["query_text"] == previous["query_text"], "same-time query differs", issues)
        _check(
            current["time_intent"] == "current" and previous["time_intent"] == "previous",
            "same-time intents are not current/previous",
            issues,
        )
        _check(
            current["expected_answer_text"] != previous["expected_answer_text"],
            "same-time answers do not differ",
            issues,
        )

    correction = groups.get("correction", [])
    _check(len(correction) == 1, "correction case missing", issues)
    if correction:
        row = correction[0]
        _check(len(row["history"]) == 2, "correction needs v0/v1", issues)
        _check(
            row["expected_state"]["revision"] == 1, "correction current revision is not 1", issues
        )

    three = groups.get("three-versions", [])
    _check(len(three) == 1, "three-versions case missing", issues)
    if three:
        row = three[0]
        _check(len(row["history"]) == 3, "three-versions needs v0/v1/v2", issues)
        _check(row["expected_state"]["revision"] == 1, "three-versions previous is not v1", issues)
        _check(
            row["expected_state"]["value"] != row["history"][0]["value"],
            "three-versions previous falls back to v0",
            issues,
        )

    same_value = groups.get("same-value-version", [])
    _check(len(same_value) == 1, "same-value-version case missing", issues)
    if same_value:
        row = same_value[0]
        _check(
            row["history"][0]["value"] == row["history"][1]["value"],
            "same-value values differ",
            issues,
        )
        _check(
            row["expected_state"]["revision"] == 0, "same-value previous revision is not 0", issues
        )

    distractor = groups.get("distractor", [])
    _check(len(distractor) == 1, "distractor case missing", issues)
    if distractor:
        row = distractor[0]
        _check(row["distractor"] is not None, "distractor payload missing", issues)
        _check(
            row["distractor"]["entity"] != row["entity"],
            "distractor entity equals target entity",
            issues,
        )

    for name in ("unknown-current", "unknown-previous", "ambiguous", "negation"):
        rows = groups.get(name, [])
        _check(len(rows) == 1, f"{name} case missing", issues)
        if rows:
            _check(rows[0]["expected_abstain"], f"{name} is not marked abstain", issues)
            _check(
                "不知道" in rows[0]["expected_answer_text"],
                f"{name} does not use an explicit unknown answer",
                issues,
            )

    non_query = groups.get("non-query", [])
    _check(len(non_query) == 1, "non-query case missing", issues)
    if non_query:
        _check(non_query[0]["query_text"] is None, "non-query has a query text", issues)
        _check(non_query[0]["scored"], "non-query is not in the fixed denominator", issues)

    attributes = {case["attribute"] for case in cases}
    _check(attributes == set(PUBLIC_ATTRIBUTES), "fixture does not cover all attributes", issues)
    _check(
        not (set(PUBLIC_ENTITIES) & set(E1_ENTITIES)),
        "public fixture reuses old E1 entities",
        issues,
    )

    for case in cases:
        _check(case["source"]["sealed"] is False, f"{case['fixture_id']} is sealed", issues)
        _check(
            case["source"]["origin"] == "public_synthetic",
            f"{case['fixture_id']} origin differs",
            issues,
        )
        target = case["target"]
        raw = (case["expected_answer_text"] + END_MARKER).encode("utf-8")
        _check(
            target["target_bytes_utf8"] == list(raw),
            f"{case['fixture_id']} target bytes differ",
            issues,
        )
        _check(target["first_byte"] == raw[0], f"{case['fixture_id']} first byte differs", issues)
        _check(
            target["visible_prefix_at_t0"] == [], f"{case['fixture_id']} t0 prefix differs", issues
        )
        _check(
            target["forbidden"] == "target[1:]",
            f"{case['fixture_id']} forbidden slice differs",
            issues,
        )

    return {
        "status": "public_fixture_audit_passed" if not issues else "public_fixture_audit_failed",
        "cases": len(cases),
        "groups": {name: len(rows) for name, rows in sorted(groups.items())},
        "attributes": sorted(attributes),
        "issues": issues,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build and validate R1 public semantic fixture")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--report", type=Path, default=REPORT)
    args = parser.parse_args(argv)
    args.manifest = args.manifest.resolve()
    args.report = args.report.resolve()
    if args.report.exists():
        raise ValueError("R1 public fixture audit report is write-once")
    cpu_start = time.process_time()
    wall_start = time.monotonic()
    if args.manifest.exists():
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    else:
        manifest = _manifest()
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    validation = _validate(manifest)
    report = {
        "schema": "taiji-native-core-r1-public-fixture-audit-v1",
        "status": validation["status"],
        "manifest": str(args.manifest.relative_to(ROOT)),
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "cases": validation["cases"],
        "groups": validation["groups"],
        "attributes": validation["attributes"],
        "issues": validation["issues"],
        "byte_protocol": manifest["byte_protocol"],
        "model_calls": 0,
        "optimizer_updates": 0,
        "sealed_text_read": 0,
        "can_promote": False,
        "cpu_seconds": time.process_time() - cpu_start,
        "wall_seconds": time.monotonic() - wall_start,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": report["status"], "cases": report["cases"]}, ensure_ascii=False))
    return 0 if validation["status"] == "public_fixture_audit_passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
