"""PLAN-N1-93：新未见题集 v6 的造面器与机检。

生成链不重抄——排除集复用 `build_taiji_n1_confirmation_v4.exclusions()`（它自带"三枚冻结源
未值"的检查），本件加两件：难度旋钮（每题 4 条观测，v4/v5 是 2 条）与 §6 的机检。
`SIGNED_DISTRIBUTION` 为 None 时**只许 `--check-draft`（内存造面并出版检查数，不落盘、不查询
任何端点）**；owner 签署 §1 的 shape 参数后由我填入常数并同轮提交，才允许 `--execute-signed`。

〔㊵-1030 的实测更正写在这里〕首版 `--check-draft` 把三条平凡策略打中 4／8／5（上限我当时拍的
2/40 根本低于这张面的机会水平），最佳常量方向 6/10。⇒ 判据不能手算，必须由件的形状导出：
现在位置类策略的上界是 ⌈可答题数／观测条数⌉＋1，常量方向上界是 ⌈transition 题数／4⌉＋1，
两者由 `audit()` 自己算并出版；同时造面器**按轮转放置答案行与目标方向**，把这个上界变成设计
约束而不是抽样运气。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_confirmation_v4 as v4  # noqa: E402

V5 = ROOT / "plans/manifests/n1_fresh_confirmation_v5_20261016.json"
OUT = ROOT / "plans/manifests/n1_fresh_confirmation_v6_20261017.json"
REPORT = ROOT / "reports/taiji_n1_v6_face_construction_checks_20261017.json"
DIRECTIONS = v4.DIRECTIONS
KINDS = v4.KINDS
SUBJECT_ROWS = {"lookup": 1, "revision": 2, "transition": 2, "unknown": 0}
POSITIONAL_MARGIN = 1
DIRECTION_MARGIN = 1

#: owner 签署 §1 的 shape 参数后替换成本件字典；None 即"未签"，造面器不得落盘。
SIGNED_DISTRIBUTION = None


class Refuse(Exception):
    """一条准入条件不成立：rc=2，不落盘、不查询。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusions():
    """v4 的三枚冻结源并上 v4/v5 两张已用面，得到 v6 必须避开的实体与材质词全集。"""
    names, words = v4.exclusions()
    for path in (v4.MANIFEST, V5):
        face = json.loads(path.read_text(encoding="utf-8"))
        names |= {name for episode in face["episodes"] for name in episode["entities"]}
        words.update(
            text.removesuffix(".").split()[-1]
            for episode in face["episodes"]
            for text in episode["observations"]
            if " uses " in text
        )
    sources = [*v4.SOURCES, v4.MANIFEST, V5]
    return names, words, [(path.resolve().as_posix(), sha(path)) for path in sources]


def draft_distribution() -> dict:
    return {
        "items": 40,
        "items_per_kind": 10,
        "observations_per_item": 4,
        "entity_length": 10,
        "answer_length": 6,
        "seed": 91,
        "schema": "n1-fresh-confirmation-v6",
        "id_prefix": "v6-confirm-",
    }


def build(params: dict):
    """按 shape 造 40 题；答案行与目标方向按轮转放置。返回 (episodes, 排除集, 来源指纹)。"""
    names, words, sources = exclusions()
    rng = random.Random(params["seed"])
    used_names = set(names)
    used_words = set(words)
    slots_per_item = params["observations_per_item"]

    def novel(length: int, pool: set) -> str:
        while True:
            value = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=length))
            if value not in pool:
                pool.add(value)
                return value

    seen = {kind: 0 for kind in KINDS}
    episodes = []
    for index in range(params["items"]):
        kind = KINDS[index % 4]
        own = seen[kind]
        seen[kind] += 1
        distractor_slots = slots_per_item - SUBJECT_ROWS[kind]
        require(distractor_slots >= 0, f"{kind} needs more observations than the face provides")
        subject = novel(params["entity_length"], used_names)
        distractors = [novel(params["entity_length"], used_names) for _ in range(distractor_slots)]
        materials = [novel(params["answer_length"], used_words) for _ in range(distractor_slots)]
        rows: list[str | None]
        if kind == "lookup":
            answer = novel(params["answer_length"], used_words)
            answer_rows = [f"{subject} uses {answer}."]
            question = f"material of {subject}?"
        elif kind == "revision":
            stale, answer = (novel(params["answer_length"], used_words) for _ in range(2))
            answer_rows = [f"{subject} uses {stale}.", f"{subject} now uses {answer}."]
            question = f"material of {subject}?"
        elif kind == "transition":
            answer = DIRECTIONS[own % len(DIRECTIONS)]
            turn = rng.choice((1, 3))
            start = (DIRECTIONS.index(answer) - turn) % len(DIRECTIONS)
            word = "right" if turn == 1 else "left"
            answer_rows = [f"{subject} faces {DIRECTIONS[start]}.", f"{subject} turns {word}."]
            question = f"direction of {subject}?"
        else:
            answer = "unknown"
            answer_rows = []
            question = f"material of {subject}?"
        rows = [None] * slots_per_item
        # 答案行占位轮转：位置类平凡策略的命中数因此被设计约束住，而不是靠抽样运气。
        for offset, row in enumerate(answer_rows):
            rows[(own + offset) % slots_per_item] = row
        filler = iter(
            [
                f"{name} uses {material}."
                for name, material in zip(distractors, materials, strict=True)
            ]
            if kind != "transition"
            else [
                f"{name} faces {DIRECTIONS[(own + number) % len(DIRECTIONS)]}."
                for number, name in enumerate(distractors)
            ]
        )
        for position, row in enumerate(rows):
            if row is None:
                rows[position] = next(filler)
        require(None not in rows and len(rows) == slots_per_item, f"{kind} face shape broke")
        episodes.append(
            {
                "id": f"{params['id_prefix']}{index:03d}",
                "split": "holdout",
                "kind": kind,
                "entities": [subject, *distractors],
                "observations": [str(row) for row in rows],
                "question": question,
                "answer": answer,
            }
        )
    return episodes, (names, words), sources


def answer_from(row: str) -> str | None:
    marker = " uses "
    if marker not in row:
        return None
    return row.split(marker, 1)[1].removesuffix(".")


def audit(episodes: list, params: dict, exclusions_pair) -> dict:
    """§6 机检逐条现取；机会水平由件的形状算出并出版，任一条超线即 accepted=false。"""
    names, words = exclusions_pair
    slots = params["observations_per_item"]
    per_kind = {kind: 0 for kind in KINDS}
    for episode in episodes:
        per_kind[episode["kind"]] += 1
    ids = [episode["id"] for episode in episodes]
    bearing = [
        episode
        for episode in episodes
        if any(answer_from(row) for row in episode["observations"])
        and episode["answer"] != "unknown"
    ]
    positional_ceiling = math.ceil(len(bearing) / slots) + POSITIONAL_MARGIN
    direction_ceiling = math.ceil(per_kind["transition"] / len(DIRECTIONS)) + DIRECTION_MARGIN

    def positional(strategy: str) -> int:
        hits = 0
        for episode in bearing:
            candidates = [answer_from(row) for row in episode["observations"] if answer_from(row)]
            if not candidates:
                continue
            if strategy == "oldest":
                guess = candidates[0]
            elif strategy == "newest":
                guess = candidates[-1]
            else:
                guess = max(set(candidates), key=candidates.count)
            hits += int(guess == episode["answer"])
        return hits

    transitions = [episode for episode in episodes if episode["kind"] == "transition"]

    def materials_of(episode: dict) -> list:
        words = []
        for row in episode["observations"]:
            word = answer_from(row)
            if word:
                words.append(word)
        return words

    repeated_material_items = sum(
        1 for episode in episodes if len(materials_of(episode)) != len(set(materials_of(episode)))
    )
    defined = [
        episode
        for episode in bearing
        if materials_of(episode)
        and max(materials_of(episode).count(word) for word in set(materials_of(episode))) > 1
    ]

    def most_frequent_hits(pool: list) -> int:
        total = 0
        for episode in pool:
            words = [word for row in episode["observations"] if (word := answer_from(row))]
            if not words:
                continue
            top = max(words, key=words.count)
            total += int(top == episode["answer"])
        return total

    constant_hits, constant_direction = max(
        (sum(int(episode["answer"] == direction) for episode in transitions), direction)
        for direction in DIRECTIONS
    )
    addressable = [episode for episode in episodes if episode["kind"] in ("lookup", "revision")]
    unique_source = sum(
        1
        for episode in addressable
        if sum(1 for row in episode["observations"] if f" uses {episode['answer']}." in row) == 1
    )
    orphan = [episode for episode in episodes if episode["kind"] == "unknown"]
    orphan_clean = sum(
        1
        for episode in orphan
        if not any(f" uses {episode['answer']}." in row for row in episode["observations"])
    )
    answer_positions = {kind: [0] * slots for kind in ("lookup", "revision")}
    for episode in addressable:
        for position, row in enumerate(episode["observations"]):
            if f" uses {episode['answer']}." in row:
                answer_positions[episode["kind"]][position] += 1
    overlaps = {
        "entities_with_prior_faces": len(
            {name for episode in episodes for name in episode["entities"]} & names
        ),
        "materials_with_prior_faces": len(
            {
                word
                for episode in episodes
                for row in episode["observations"]
                if (word := answer_from(row))
            }
            & words
        ),
    }
    checks = {
        "J93_1_class_balance": {
            "per_kind": per_kind,
            "ok": all(count == params["items_per_kind"] for count in per_kind.values()),
        },
        "J93_1_identity": {
            "ids_unique": len(set(ids)) == len(ids),
            "all_holdout": all(episode["split"] == "holdout" for episode in episodes),
            "observations_per_item": sorted({len(episode["observations"]) for episode in episodes}),
            "ok": (
                len(set(ids)) == len(ids)
                and all(episode["split"] == "holdout" for episode in episodes)
                and all(len(episode["observations"]) == slots for episode in episodes)
            ),
        },
        "J93_2_disjoint_from_prior_faces": {
            **overlaps,
            "ok": all(count == 0 for count in overlaps.values()),
        },
        "J93_3_trivial_strategies": {
            "answer_bearing_items": len(bearing),
            "positional_ceiling_from_shape": positional_ceiling,
            "oldest": positional("oldest"),
            "newest": positional("newest"),
            "items_with_repeated_material": repeated_material_items,
            "frequency_strategy_defined_items": len(defined),
            "most_frequent_hits_where_defined": most_frequent_hits(defined),
            "answer_position_counts": answer_positions,
            "ok": (
                positional("oldest") <= positional_ceiling
                and positional("newest") <= positional_ceiling
                and repeated_material_items == 0
                and len(defined) == 0
                and all(max(counts) - min(counts) <= 1 for counts in answer_positions.values())
            ),
        },
        "J93_3b_constant_direction": {
            "transition_items": len(transitions),
            "best_direction": constant_direction,
            "hits": constant_hits,
            "ceiling_from_shape": direction_ceiling,
            "ok": constant_hits <= direction_ceiling,
        },
        "J93_4_addressability": {
            "addressable_items": len(addressable),
            "unique_source_items": unique_source,
            "unknown_items": len(orphan),
            "unknown_without_answer_source": orphan_clean,
            "ok": unique_source == len(addressable) and orphan_clean == len(orphan),
        },
    }
    return {"checks": checks, "accepted": all(block["ok"] for block in checks.values())}


def check_draft() -> int:
    params = draft_distribution()
    episodes, pair, sources = build(params)
    result = audit(episodes, params, pair)
    print(
        json.dumps(
            {
                "mode": "check-draft",
                "written": False,
                "params": params,
                "exclusion_source_sha256": [digest for _path, digest in sources],
                **result,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["accepted"] else 1


def execute_signed(out_path: Path) -> int:
    require(SIGNED_DISTRIBUTION is not None, "PLAN-N1-93 §1 的 shape 参数尚未由 owner 签署")
    require(not out_path.exists(), f"v6 face already exists: {out_path}")
    require(not REPORT.exists(), f"v6 check report already exists: {REPORT}")
    params = dict(SIGNED_DISTRIBUTION)
    episodes, pair, sources = build(params)
    result = audit(episodes, params, pair)
    require(
        result["accepted"],
        f"machine checks failed: {json.dumps(result['checks'], sort_keys=True)[:400]}",
    )
    manifest = {
        "schema": params["schema"],
        "seed": params["seed"],
        "excluded_source_sha256": [digest for _path, digest in sources],
        "episodes": episodes,
    }
    out_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    REPORT.write_text(
        json.dumps(
            {
                "format": "taiji-n1-93-v6-face-construction-v1",
                "params": params,
                "face": str(out_path),
                "face_sha256": sha(out_path),
                "exclusion_sources": {path: digest for path, digest in sources},
                "model_loads": 0,
                "queries": 0,
                "consumes_confirmation_face": False,
                **result,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {"mode": "execute-signed", "face": str(out_path), "accepted": True}, sort_keys=True
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check-draft", action="store_true", help="内存造面并出版机检，不落盘")
    parser.add_argument(
        "--execute-signed", action="store_true", help="仅在 owner 签署 shape 参数后落盘"
    )
    parser.add_argument("--out-manifest", type=Path, default=OUT)
    arguments = parser.parse_args(argv)
    try:
        require(
            arguments.check_draft != arguments.execute_signed,
            "choose exactly one of --check-draft/--execute-signed",
        )
        if arguments.check_draft:
            return check_draft()
        return execute_signed(arguments.out_manifest)
    except Refuse as refusal:
        print(f"REFUSE: {refusal}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
