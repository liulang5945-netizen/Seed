"""PLAN-N3-05：为 DEBT-G49 的"零命中重建"验候选探针（只读语料，不改训练器）。

验收式住在件里（`plans/reference/PLAN-N3-05_holdout_probe_rebuild_prereg_20261008.md` §1），本脚本只是它的执行者：

* **A-1** 每个候选的 24 字节窗口（步长 4，取法＝现成的 `_probe_windows`，**不重造尺**）在真实语料里
  必须 `windows_found_in_corpus == 0`；
* **A-2** 候选字节数落在 64..512（旧探针 163 B）；
* **A-3** 形状同类＝事实句 ＋ 问答对（含 `问：`/`\\n答：`）＋ 指令句，三段都必须在场；
* **A-4** 同机连跑两趟，两趟都 0 命中，且**两趟的语料 sha 必须相同**（不同⇒这趟不算复现）。

选择规则**预先指定**：全部通过的候选里取**字节数最接近 163 B** 的那一枚（不是"我看着顺眼的那一枚"）；
无一通过 ⇒ 判 `no_zero_hit_probe_today` 并 rc=2，**不放宽 A-1**（放宽等于把"独立"重新变成命名）。

`--expect-legacy-hit` 那条是本仪器的判别力自检：同一把尺对**旧**探针必须报出非零命中（在库实测 5/35），
报不出来说明尺坏了，此时候选件的"0 命中"一文不值。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "scripts" / "training") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "training"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audit_taiji_n3a_data_face import (  # noqa: E402
    NGRAM_BYTES,
    _disjointness,
    _probe_windows,
)
from train_seed_corpus import DEFAULT_CORPUS, HOLDOUT_PROBE  # noqa: E402

LEGACY_TARGET_BYTES = 163
MIN_CANDIDATE_BYTES = 64
MAX_CANDIDATE_BYTES = 512

#: 三枚候选都是**为了零重合而新写的**（不是从任何语料里摘的）——这句就是 A-1 那个 0 的成因说明。
CANDIDATES: dict[str, str] = {
    "c1_tide": (
        "潮汐发电站把月球引力引起的潮位差转换成电能，一年里两次大潮的水量差别最明显。"
        "问：请把这句话缩短一点。\n答：好的：月亮拉扯海水，电站借这股力发电。"
        "请说明为什么铜适合做导线而玻璃不适合。"
    ),
    "c2_bridge": (
        "悬索桥的主缆承受整座桥面与车辆的重量，重量经由塔顶的鞍座传到地基，"
        "所以缆索的锈蚀检查要按季节重复做。问：桥为什么会晃。\n答：因为风与车流的能量被结构吸收了一部分，"
        "剩下的以振动形式回到桥面。请把下面这句话改得更短：金属在潮湿环境里更容易被氧化。"
    ),
    "c3_seed": (
        "种子在湿润而温暖的土壤里开始吸水，胚根先破壳向下生长，随后子叶被顶出土面接受光照，"
        "整个过程大约需要一个星期，温度低于零上五度时发芽几乎停滞。"
        "问：为什么把种子泡一夜更好发芽。\n答：泡过的种子吸足了水，酶的活动提前开始，破壳就更快。"
        "请用一句话说明根系受伤后植株为什么会矮小，并把这段关于光照的说明改写得更短。"
    ),
}

SHAPE_MARKERS = ("问：", "\n答：", "请")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _shape_ok(text: str) -> dict[str, bool]:
    return {marker: marker in text for marker in SHAPE_MARKERS}


def evaluate(corpus: list[Path], passes: int = 2) -> dict[str, Any]:
    """对每枚候选跑 `passes` 趟重合扫描，并按 §1 的四条验收式逐条记结果。"""

    corpus_sha = [_file_sha256(path) for path in corpus]
    rows: dict[str, dict[str, Any]] = {}
    for name, text in CANDIDATES.items():
        raw = text.encode("utf-8")
        windows = _probe_windows(raw)
        runs = [_disjointness(corpus, windows) for _ in range(max(1, passes))]
        zero_runs = [run["windows_found_in_corpus"] == 0 for run in runs]
        rows[name] = {
            "bytes": len(raw),
            "windows": len(windows),
            "window_bytes": NGRAM_BYTES,
            "a1_zero_hit_per_pass": zero_runs,
            "a1_hits_per_pass": [run["windows_found_in_corpus"] for run in runs],
            "a2_size_in_band": MIN_CANDIDATE_BYTES <= len(raw) <= MAX_CANDIDATE_BYTES,
            "a2_distance_to_legacy": abs(len(raw) - LEGACY_TARGET_BYTES),
            "a3_shape": _shape_ok(text),
            "a3_shape_ok": all(_shape_ok(text).values()),
            "a4_corpus_sha_stable": len(set(corpus_sha)) == len(corpus_sha),
            "pass_all": all(zero_runs)
            and MIN_CANDIDATE_BYTES <= len(raw) <= MAX_CANDIDATE_BYTES
            and all(_shape_ok(text).values()),
        }
    passing = {name: row for name, row in rows.items() if row["pass_all"]}
    chosen: str | None = None
    if passing:
        chosen = min(passing, key=lambda k: (passing[k]["a2_distance_to_legacy"], k))
    legacy = _disjointness(corpus, _probe_windows())
    return {
        "format": "taiji-n3-05-probe-verification-v1",
        "prereg": "plans/reference/PLAN-N3-05_holdout_probe_rebuild_prereg_20261008.md#1",
        "corpus": [str(path.relative_to(PROJECT_ROOT)) for path in corpus],
        "corpus_sha256": corpus_sha,
        "passes": max(1, passes),
        "legacy_probe": {
            "bytes": len(HOLDOUT_PROBE),
            "windows": len(_probe_windows()),
            "windows_found_in_corpus": legacy["windows_found_in_corpus"],
            "hit_rate": legacy["hit_rate"],
        },
        "candidates": rows,
        "candidates_passing": sorted(passing),
        "chosen": chosen,
        "selection_rule": "全部过 A-1..A-4 的候选里取字节数最接近 163 B 的那一枚；无一通过则判造不出",
        "verdict": "probe_selected" if chosen else "no_zero_hit_probe_today",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PLAN-N3-05 候选探针的零重合验收（只读）")
    parser.add_argument(
        "--corpus", nargs="*", default=[str(PROJECT_ROOT / name) for name in DEFAULT_CORPUS]
    )
    parser.add_argument("--passes", type=int, default=2, help="A-4 要求同机两趟都 0 命中")
    parser.add_argument("--out-report", default=None)
    args = parser.parse_args(argv)

    corpus = [Path(p) if Path(p).is_absolute() else PROJECT_ROOT / p for p in args.corpus]
    missing = [str(p) for p in corpus if not p.is_file()]
    if missing:
        print(json.dumps({"status": "refused", "missing_corpus": missing}, ensure_ascii=True))
        return 2
    payload = evaluate(corpus, passes=args.passes)
    #: 判别力自检（G-N5-3）：旧探针必须仍报非零命中；它报 0 说明这把尺今天量不动东西，
    #: 那么候选的 0 命中就一文不值 ⇒ 响亮拒绝，不出"候选可用"的结论。
    if payload["legacy_probe"]["windows_found_in_corpus"] == 0:
        payload["status"] = "refused_ruler_blind"
        print(json.dumps(payload, ensure_ascii=True)[:400])
        return 2
    payload["status"] = "ok"
    rc = 0 if payload["chosen"] else 2
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out_report:
        out = Path(args.out_report)
        out = out if out.is_absolute() else PROJECT_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8", newline="\n")
    #: stdout 走 ASCII 转义：拒绝与判级句里有"⇒"这类字形，GBK 终端会炸在 rc 之前（㊵-484⑤ 那一类）。
    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "chosen": payload["chosen"],
                "passing": payload["candidates_passing"],
                "legacy_hits": payload["legacy_probe"]["windows_found_in_corpus"],
                "per_candidate": {
                    k: [v["a1_hits_per_pass"], v["bytes"], v["a2_size_in_band"], v["a3_shape_ok"]]
                    for k, v in payload["candidates"].items()
                },
            },
            ensure_ascii=True,
        )
    )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
