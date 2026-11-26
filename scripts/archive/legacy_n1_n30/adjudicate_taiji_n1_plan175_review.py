"""Validate N175 human ratings; freeze complete ratings before opening the blind key."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "reports/taiji_n1_plan175_v3_review_20261026"
MARKDOWN = ROOT / "plans/reference/PLAN-N1-175_blind_review_20261026.md"
ROLLOUT = ROOT / "reports/taiji_n1_plan175_v3_train_rollout_20261026.json"
BLIND_SHA = "948ea4e7df2393e509b4d6fc3d91637e7345fd491f64322f8ad7cf9992961922"
ROLLOUT_SHA = "a009ddf27614fb7e5b040e690d5402a2eb23559d274a2a721fafbf3abc113b05"
KEY_GIT_BLOB = "c901d4058a51b9168a6cd4068fa8be1123de6562"
MARKDOWN_SHA = "d233aa6c5102e0dc0714c605a2280f40f3e996fd2051711f20e4903f202d572c"
PLACEHOLDERS = {"", "待填", "待填写", "待定", "—", "-"}
RATING_ROW = re.compile(r"^\| `([0-9a-f]{32})` \| ([^|]+) \| (.*) \|$")
REVIEWER_ROW = re.compile(r"^评阅者：(.+?)。已填评分：(\d+)/312。$")


def sha(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def parse_markdown(text: str) -> dict[str, Any]:
    canonical, ratings, reviewer = [], {}, None
    fence_width = 0
    for line in text.splitlines():
        fence = re.match(r"^(`{3,})(.*)$", line)
        if fence_width:
            canonical.append(line)
            if fence and len(fence[1]) >= fence_width and not fence[2].strip():
                fence_width = 0
            continue
        if fence:
            fence_width = len(fence[1])
            canonical.append(line)
            continue
        if line == "状态：人工判读进行中；已记录弹窗评分。":
            canonical.append("状态：待人工判读；评分与评阅责任人尚未填写。")
            continue
        person = REVIEWER_ROW.fullmatch(line)
        if person:
            if reviewer is not None:
                raise ValueError("duplicate reviewer declaration")
            reviewer = person[1].strip()
            canonical.append("评阅者：待填写。已填评分：0/312。")
            continue
        row = RATING_ROW.fullmatch(line)
        if row:
            identity, score, rationale = row[1], row[2].strip(), row[3].strip()
            if identity in ratings:
                raise ValueError("duplicate response rating")
            if score not in {"0", "1", "2", "不可判", "待填"}:
                raise ValueError("rating must be 0, 1, 2, 不可判 or 待填")
            if score != "待填" and rationale in PLACEHOLDERS:
                raise ValueError("each completed rating needs a rationale")
            ratings[identity] = {
                "score": int(score) if score in {"0", "1", "2"} else None,
                "not_evaluable": score == "不可判",
                "pending": score == "待填",
                "rationale": rationale,
            }
            canonical.append(f"| `{identity}` | 待填 | 待填 |")
            continue
        canonical.append(line)
    if fence_width:
        raise ValueError("unclosed source-text fence")
    return {
        "reviewer": reviewer,
        "ratings": ratings,
        "canonical_sha256": sha(("\n".join(canonical) + "\n").encode()),
    }


def validate_review(text: str, blind: dict, expected_markdown_sha: str) -> dict:
    result = parse_markdown(text)
    expected = {
        response["response_id"]: case["case_id"]
        for case in blind["cases"]
        for response in case["responses"]
    }
    if result["canonical_sha256"] != expected_markdown_sha:
        raise ValueError("immutable question, answer, order or rubric changed")
    if set(result["ratings"]) != set(expected):
        raise ValueError("rating identities differ from the frozen blind responses")
    for identity, rating in result["ratings"].items():
        rating["case_id"] = expected[identity]
    result["pending"] = sum(row["pending"] for row in result["ratings"].values())
    result["reviewer_set"] = result["reviewer"] not in PLACEHOLDERS | {None}
    result["complete"] = result["pending"] == 0 and result["reviewer_set"]
    return result


def require_complete(review: dict) -> None:
    if not review["complete"]:
        raise ValueError("human ratings/reviewer are incomplete; key remains sealed")


def summarize(review: dict, mapping: list[dict], rollout: dict, family_count: int) -> dict:
    require_complete(review)
    ratings = review["ratings"]
    if len(mapping) != len(ratings) or {x["response_id"] for x in mapping} != set(ratings):
        raise ValueError("blind key response identities differ")
    cells, seen = {}, set()
    for row in mapping:
        identity = row["response_id"]
        rating = ratings[identity]
        seed, condition = row["seed"], row["condition"]
        cell = (row["case_id"], seed, condition)
        if (
            rating["case_id"] != row["case_id"]
            or seed not in {169, 170, 171}
            or condition not in {"matched", "mismatched_prompt"}
            or cell in seen
        ):
            raise ValueError("blind key case/run/condition binding differs")
        seen.add(cell)
        cells.setdefault((seed, condition), []).append(rating)
    r_by_seed = {row["seed"]: row["rollout_gate_passed"] for row in rollout["runs"]}
    if set(r_by_seed) != {169, 170, 171}:
        raise ValueError("R report lacks fixed runs")
    runs = []
    for seed in (169, 170, 171):
        counts = {}
        for condition in ("matched", "mismatched_prompt"):
            rows = cells.get((seed, condition), [])
            if len(rows) * 6 != len(mapping) or len(rows) > family_count:
                raise ValueError("condition coverage differs from the frozen review face")
            counts[condition] = {
                "correct": sum(x["score"] == 2 for x in rows),
                "partial": sum(x["score"] == 1 for x in rows),
                "zero": sum(x["score"] == 0 for x in rows),
                "not_evaluable": sum(x["not_evaluable"] for x in rows),
                "excluded_no_replacement": family_count - len(rows),
                "denominator": family_count,
            }
        matched, wrong = counts["matched"]["correct"], counts["mismatched_prompt"]["correct"]
        s_passed = matched >= 16 and matched - wrong >= 8
        runs.append(
            {
                "seed": seed,
                "counts": counts,
                "matched_minus_mismatched_correct": matched - wrong,
                "R_passed": r_by_seed[seed],
                "S_passed": s_passed,
                "joint_passed": r_by_seed[seed] and s_passed,
            }
        )
    joint = sum(x["joint_passed"] for x in runs) >= 2
    status = "LIMITED_GAIN" if joint else "FAIL_NO_SEMANTIC_GAIN"
    if sum(x["R_passed"] for x in runs) < 2:
        status = "INCONCLUSIVE_ROLLOUT"
    return {"status": status, "runs": runs, "joint_passed": joint, "can_promote": False}


def write_once(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def finalize(review: dict, paths: dict[str, Path], expected_key_blob: str) -> dict:
    # Completion is checked before even opening the key or creating a freeze.
    require_complete(review)
    if paths["result"].exists():
        raise FileExistsError("adjudication already completed; preserve the original result")
    freeze = {"schema": "taiji-n175-human-rating-freeze-v1", **review}
    if paths["freeze"].exists():
        existing = json.loads(paths["freeze"].read_text(encoding="utf-8"))
        if existing != freeze:
            raise ValueError("ratings differ from the durable freeze")
    else:
        write_once(paths["freeze"], freeze)
    raw_key = paths["key"].read_bytes().replace(b"\r\n", b"\n")
    blob = b"blob " + str(len(raw_key)).encode() + b"\0" + raw_key
    if hashlib.sha1(blob, usedforsecurity=False).hexdigest() != expected_key_blob:
        raise ValueError("key differs from its fixed Git object identity")
    raw_rollout = paths["rollout"].read_bytes()
    if sha(raw_rollout) != ROLLOUT_SHA:
        raise ValueError("fixed R measurement differs")
    result = summarize(review, json.loads(raw_key)["responses"], json.loads(raw_rollout), 64)
    result.update(
        {
            "schema": "taiji-n175-human-adjudication-v1",
            "reviewer": review["reviewer"],
            "rating_freeze_sha256": sha(paths["freeze"].read_bytes()),
            "key_git_blob": expected_key_blob,
            "original_family_denominator": 64,
            "training_updates": 0,
            "new_generation": 0,
        }
    )
    write_once(paths["result"], result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", type=Path, default=MARKDOWN)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    started = time.process_time()
    blind_raw = (BUNDLE / "blind_review.json").read_bytes()
    if sha(blind_raw) != BLIND_SHA:
        raise ValueError("original blind review differs")
    blind = json.loads(blind_raw)
    if len(blind["cases"]) != 64 or sum(len(x["responses"]) for x in blind["cases"]) != 312:
        raise ValueError("fixed 64-family/312-response scope differs")
    review = validate_review(args.ratings.read_text(encoding="utf-8"), blind, MARKDOWN_SHA)
    review.update(
        {
            "blind_review_sha256": BLIND_SHA,
            "annotated_markdown_sha256": sha(args.ratings.read_bytes()),
            "adjudicator_source_sha256": sha(Path(__file__).read_bytes()),
        }
    )
    if args.finalize:
        result = finalize(
            review,
            {
                "freeze": BUNDLE / "human_rating_freeze.json",
                "result": BUNDLE / "human_adjudication.json",
                "key": BUNDLE / "blind_review_key.json",
                "rollout": ROLLOUT,
            },
            KEY_GIT_BLOB,
        )
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(
            json.dumps(
                {
                    "status": (
                        "ready_to_finalize" if review["complete"] else "awaiting_human_ratings"
                    ),
                    "filled": len(review["ratings"]) - review["pending"],
                    "pending": review["pending"],
                    "reviewer_set": review["reviewer_set"],
                    "blind_key_opened": False,
                    "process_cpu_seconds": time.process_time() - started,
                }
            )
        )


if __name__ == "__main__":
    main()
