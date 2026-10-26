"""Build a bounded F0-F2 curriculum prototype; this script never trains a model."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from taiji.language_alignment import LanguageEpisode

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/conversation/taiji_n1_plan176_foundation_20261026"
REPORT = ROOT / "reports/taiji_n1_plan176_foundation_corpus_20261026.json"
OBJECTS = ("木盒", "水杯", "书包", "皮球", "积木", "小车", "画笔", "雨伞")
COLORS = ("红色", "蓝色", "绿色", "黄色", "白色", "黑色", "紫色", "橙色")
PLACES = ("书桌上", "柜子里", "门边", "窗边", "架子上", "椅子上", "地板上", "抽屉里")
OWNERS = ("小林", "小周", "小陈", "小李")
SPLITS = ("train", "calibration", "sealed")


def pair_split(object_index: int, color_index: int) -> str:
    offset = (color_index - object_index) % len(COLORS)
    return "train" if offset < 5 else "calibration" if offset == 5 else "sealed"


def as_language_episode(row: dict) -> LanguageEpisode:
    # Oracle metadata is for labels/audits, never part of the model prompt.
    return LanguageEpisode(
        episode_id=row["episode_id"],
        family_id=row["family_id"],
        task_family=row["task_family"],
        split={"train": "train", "calibration": "dev", "sealed": "final"}[row["split"]],
        user_input=row["user_input"],
        response=row["response"],
        history=tuple(tuple(turn) for turn in row["history"]),
    )


def build_rows() -> list[dict]:
    rows = []
    for oi, obj in enumerate(OBJECTS):
        for ci, color in enumerate(COLORS):
            split = pair_split(oi, ci)
            for variant in range(4):
                family = f"n176:world:{oi}:{ci}:{variant}"
                di = (oi + variant + 1) % len(OBJECTS)
                distractor = OBJECTS[di]
                # Every distractor object/color fact is from the train pair pool.
                dc = (di + variant) % len(COLORS)
                initial = PLACES[(oi + variant) % len(PLACES)]
                final = PLACES[(oi + variant + 1) % len(PLACES)]
                other_place = PLACES[(oi + variant + 2) % len(PLACES)]
                old_owner = OWNERS[variant]
                new_owner = OWNERS[(variant + 1) % len(OWNERS)]
                facts = f"{obj}是{color}的。{distractor}是{COLORS[dc]}的。"
                # Held-out surface forms supplement, rather than replace, pair isolation.
                if split == "train":
                    color_question = f"{obj}是什么颜色？"
                    other_question = f"{distractor}是什么颜色？"
                elif split == "calibration":
                    color_question = f"请说出{obj}的颜色。"
                    other_question = f"请说出{distractor}的颜色。"
                else:
                    color_question = f"根据这些话，{obj}的颜色是？"
                    other_question = f"根据这些话，{distractor}的颜色是？"
                tasks = [
                    ("F0", facts + f"请照写：{color}。", color + "。", [], "copy", color),
                    ("F0", facts + f"请照写：{initial}。", initial + "。", [], "copy", initial),
                    ("F1", facts + color_question, color + "。", [], "primary_color", color),
                    (
                        "F1",
                        facts + other_question,
                        COLORS[dc] + "。",
                        [],
                        "distractor_color",
                        COLORS[dc],
                    ),
                    (
                        "F2",
                        facts + f"{new_owner}把{obj}移到了{final}。{distractor}还在{other_place}。"
                        f"{obj}现在在哪里？",
                        final + "。",
                        [[f"{obj}在{initial}。", "我记住了。"]],
                        "location_update",
                        final,
                    ),
                    (
                        "F2",
                        facts + f"更正一下，{obj}是{new_owner}的，不是{old_owner}的。"
                        f"{distractor}仍然是{old_owner}的。{obj}是谁的？",
                        new_owner + "的。",
                        [[f"{obj}是{old_owner}的。", "我记住了。"]],
                        "owner_correction",
                        new_owner + "的",
                    ),
                ]
                for index, (level, prompt, response, history, kind, answer) in enumerate(tasks):
                    rows.append(
                        {
                            "episode_id": f"{family}:q{index}",
                            "family_id": family,
                            "split": split,
                            "task_family": "foundation_closed_world",
                            "curriculum_level": level,
                            "user_input": prompt,
                            "response": response,
                            "history": history,
                            "metadata": {
                                "world_kind": "synthetic_closed_world_not_real_world_evidence",
                                "primary_pair": [obj, color],
                                "distractor_pair": [distractor, COLORS[dc]],
                                "query_kind": kind,
                                "answer_from_world": answer,
                                "initial_location": initial,
                                "final_location": final,
                                "initial_owner": old_owner,
                                "final_owner": new_owner,
                                "model_receives_oracle_metadata": False,
                                "unseen_claim": (
                                    "literal-copy form only, not semantic understanding"
                                    if level == "F0"
                                    else (
                                        "primary_object_color_pair_and_question_form"
                                        if level == "F1" and kind == "primary_color"
                                        else "scene_family; relation pairs may occur in train"
                                    )
                                ),
                            },
                        }
                    )
    return rows


def validate_rows(rows: list[dict]) -> dict:
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate episode identity")
    families, pairs, factual_pairs = {}, {}, {split: set() for split in SPLITS}
    counts = Counter()
    target_bytes = Counter()
    prompts = set()
    for row in rows:
        split, family = row["split"], row["family_id"]
        if split not in SPLITS or row["curriculum_level"] not in {"F0", "F1", "F2"}:
            raise ValueError("unknown split or curriculum level")
        if families.setdefault(family, split) != split:
            raise ValueError("world family crosses splits")
        pair = tuple(row["metadata"]["primary_pair"])
        if pairs.setdefault(pair, split) != split:
            raise ValueError("primary object/color combination crosses splits")
        if row["response"] != row["metadata"]["answer_from_world"] + "。":
            raise ValueError("target disagrees with oracle world")
        metadata = row["metadata"]
        expected = {
            "copy": row["user_input"].rsplit("请照写：", 1)[-1].removesuffix("。"),
            "primary_color": metadata["primary_pair"][1],
            "distractor_color": metadata["distractor_pair"][1],
            "location_update": metadata["final_location"],
            "owner_correction": metadata["final_owner"] + "的",
        }
        if (
            metadata["query_kind"] not in expected
            or metadata["answer_from_world"] != expected[metadata["query_kind"]]
        ):
            raise ValueError("oracle answer disagrees with queried world field")
        if row["metadata"]["model_receives_oracle_metadata"]:
            raise ValueError("oracle metadata must remain outside model input")
        factual_pairs[split].add(pair)
        factual_pairs[split].add(tuple(row["metadata"]["distractor_pair"]))
        episode = as_language_episode(row)
        if episode.prompt_bytes in prompts:
            raise ValueError("identical model prompt repeats within or across splits")
        prompts.add(episode.prompt_bytes)
        if len(episode.target_bytes) > 96 or len(episode.prompt_bytes) > 1024:
            raise ValueError("foundation byte limits exceeded")
        counts[(split, row["curriculum_level"])] += 1
        target_bytes[split] += len(episode.target_bytes)
    for pair, split in pairs.items():
        if split != "train" and pair in factual_pairs["train"]:
            raise ValueError("held-out primary pairing leaked through training distractors")
    return {
        "episodes": len(rows),
        "families": len(families),
        "primary_object_color_pairs": len(pairs),
        "unique_model_prompts": len(prompts),
        "split_counts": {
            split: {
                "episodes": sum(counts[(split, level)] for level in ("F0", "F1", "F2")),
                "families": sum(value == split for value in families.values()),
                "by_level": {level: counts[(split, level)] for level in ("F0", "F1", "F2")},
                "target_bytes_per_pass": target_bytes[split],
            }
            for split in SPLITS
        },
        "oracle_metadata_in_model_prompt": False,
        "primary_pair_isolation": "passed_including_distractor_facts",
    }


def main() -> None:
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("foundation prototype already exists; preserve it")
    rows = build_rows()
    summary = validate_rows(rows)
    staging = Path(tempfile.mkdtemp(prefix=".tmp-n176-foundation-", dir=OUTPUT.parent))
    try:
        hashes = {}
        for split in SPLITS:
            path = staging / f"{split}.jsonl"
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                for row in rows:
                    if row["split"] == split:
                        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = {
            "schema": "taiji-n176-foundation-corpus-prototype-v1",
            "status": "prototype_structural_checks_passed_no_training",
            "summary": summary,
            "split_sha256": hashes,
            "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "output_directory": str(OUTPUT.relative_to(ROOT)),
            "training_updates": 0,
            "model_queries": 0,
            "human_quality_review": "pending",
            "limitations": [
                "Finite synthetic vocabulary and templates; not open-domain language evidence",
                "F0 copying measures form, not understanding",
                "Only F1 primary-color queries claim isolated unseen object/color pairings",
                "F2 relation pairs can repeat across splits; only scene-family isolation is claimed",
                "Formal training budget and thresholds are not frozen by this prototype",
            ],
        }
        (staging / "manifest.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(staging, OUTPUT)
        with REPORT.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    except BaseException:
        # Retain the staged prototype as failure evidence rather than delete it.
        raise
    print(json.dumps({"status": report["status"], "summary": summary}))


if __name__ == "__main__":
    main()
