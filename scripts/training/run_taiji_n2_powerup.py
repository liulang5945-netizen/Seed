"""N2 巩固通电驱动（PLAN-N2-01 §4bis／§4ter）。

两支，都不覆写 `DEFAULT_CHECKPOINT`：

* `--phase guard`＝G-N2-2 守卫臂。只按 `sleep_pass.run()` 的**现行默认**调一次，
  要求 `organs.ran` 为假且权重摘要不变——这条能红（若默认位哪天被翻开）。
* `--phase powerup`＝顺序 ①→④→⑤→⑥：母状态取两份（内存 mapping＋磁盘文件）、
  通电一次、候选档落盘、回退演示、新材料的"治疗 vs 对照（回退＝同一份母权重）"读数。

改权重半径由 owner 2026-10-08 批文冻结（09 §3.2 第 6 条），参数一律沿用 `run()` 默认。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MOTHER_PATH = PROJECT_ROOT / "checkpoints" / "seed_n2_mother_20261008.pt"
CANDIDATE_PATH = PROJECT_ROOT / "checkpoints" / "seed_n2_candidate_20261008.pt"
ROLLBACK_PATH = PROJECT_ROOT / "checkpoints" / "seed_n2_rollback_20261008.pt"
REPORT_DIR = PROJECT_ROOT / "reports"
GUARD_REPORT = REPORT_DIR / "taiji_n2_guard_default_20261008.json"
POWERUP_REPORT = REPORT_DIR / "taiji_n2_powerup_20261008.json"

#: 件内只留每条文本的摘要与短预览；全文留在内存里供配对读数用。
PREVIEW_CHARS = 90

#: 与 seed.judge.DEFAULT_WEIGHTS 同序（mean_surprise／mean_error_norm／mean_confidence／accuracy）。
SURPRISE_WEIGHT_INDEX = 0


def _write_report(path: Path, payload: dict[str, Any]) -> None:
    """落盘一份读数；中断不许留下半件（tmp + replace，与 CAP-0 runner 同法）。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _digest(model: Any) -> str:
    from taiji.internalization import content_digest

    return content_digest(model.checkpoint())


def _model_stats(model: Any) -> dict[str, Any]:
    """件自述：这枚权重摘要对应的 tick 与参数量。"""

    return {"tick": int(model.tick), "digest": _digest(model)}


def phase_guard() -> int:
    """G-N2-2：默认参数 pass 必须**不碰**器官，也不碰权重。"""

    from api.seed_runtime import DEFAULT_CHECKPOINT, SeedRuntime
    from seed_platform import sleep_pass

    started = time.time()
    runtime = SeedRuntime.load(DEFAULT_CHECKPOINT)
    before = _model_stats(runtime.model)
    parameters_before = int(runtime.model.parameter_count())

    report = sleep_pass.run(reason="n2-guard-default")

    after = _model_stats(runtime.model)
    organs = report.get("organs") or {}
    checks = {
        "organs_ran_false": organs.get("ran") is False,
        "organs_reason_is_not_requested": organs.get("reason") == "organs not requested",
        "weight_digest_unchanged": before["digest"] == after["digest"],
        "tick_unchanged": before["tick"] == after["tick"],
        "parameter_count_unchanged": parameters_before == int(runtime.model.parameter_count()),
    }
    payload = {
        "format": "taiji-n2-guard-default-v1",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "call": "sleep_pass.run(reason='n2-guard-default')  # 其余参数一律沿用 run() 默认",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "seconds": round(time.time() - started, 1),
        "digest_before": before["digest"],
        "digest_after": after["digest"],
        "tick_before": before["tick"],
        "tick_after": after["tick"],
        "parameter_count": parameters_before,
        "pass_id": report.get("pass_id"),
        "projection_records": (report.get("projection") or {}).get("records"),
        "by_source": (report.get("projection") or {}).get("by_source"),
        "spec_written": (report.get("spec") or {}).get("written"),
        "organs": organs,
        "checks": checks,
        "guard_pass": all(checks.values()),
    }
    _write_report(GUARD_REPORT, payload)
    print(f"guard_pass={payload['guard_pass']}")
    for key, value in checks.items():
        print(f"  {key}={value}")
    print(f"records={payload['projection_records']} by_source={payload['by_source']}")
    print(f"report -> {GUARD_REPORT}")
    return 0 if payload["guard_pass"] else 1


def _material_texts(report: dict[str, Any]) -> list[str]:
    """回读**本 pass 自己落盘的那枚语料**，作为 J-N2b 的材料（同一文件、同一行序）。"""

    from seed_platform import sleep_pass

    corpus = str((report.get("projection") or {}).get("corpus") or "")
    if not corpus:
        return []
    directory = str(sleep_pass.status().get("corpus_directory") or "")
    path = Path(directory) / Path(corpus).name
    if not path.is_file():
        raise RuntimeError(f"本 pass 自述的语料不在盘上，无法配对：{path}")
    texts: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        texts.append(str(json.loads(line)["text"]))
    return texts


def _preview(text: str) -> dict[str, Any]:
    return {
        "sha16": sha256(text.encode("utf-8")).hexdigest()[:16],
        "chars": len(text),
        "preview": text[:PREVIEW_CHARS],
    }


def _night_selection(model: Any, texts: list[str], k: int) -> dict[str, Any]:
    """现场重算夜文本与 G-N2-4 的调制前/后对照（只用现成仪器，不另写打分）。

    池子＝本 pass 落盘语料的**全部**记录——真实 pass 就是在整个池子上排序取最差 k 条，
    截断池子会选出另一批文本，配对读数就失去同一材料这条前提。
    """

    from seed.judge import DEFAULT_WEIGHTS, SeedJudge
    from seed.sleep import SeedSleepScheduler

    judge = SeedJudge(model)
    payload = [text.encode("utf-8") for text in texts if text.strip()]
    if not payload:
        raise RuntimeError("夜文本重算没有可打分的材料")

    scored = []
    for text in payload:
        report = judge.score(text)
        without_surprise = report["quality"] - (
            DEFAULT_WEIGHTS[SURPRISE_WEIGHT_INDEX] * report["mean_surprise"]
        )
        scored.append(
            {
                "text": text.decode("utf-8", "replace"),
                "bytes": len(text),
                "quality": report["quality"],
                "quality_without_surprise": without_surprise,
                "mean_surprise": report["mean_surprise"],
                "mean_error_norm": report["mean_error_norm"],
                "mean_confidence": report["mean_confidence"],
                "accuracy": report["accuracy"],
            }
        )

    after = sorted(scored, key=lambda item: item["quality"])
    before = sorted(scored, key=lambda item: item["quality_without_surprise"])
    selected_texts = [item["text"] for item in after[:k]]
    scheduler_selected = [
        text.decode("utf-8", "replace")
        for text in SeedSleepScheduler(model, judge).select_for_sleep(payload, k=k)
    ]
    return {
        "pool_size": len(scored),
        "k": int(k),
        "selected_texts": selected_texts,
        "selected_after_modulation": [_preview(item) for item in selected_texts],
        "selected_before_modulation": [_preview(item["text"]) for item in before[:k]],
        "scheduler_agrees_with_weighted_sort": scheduler_selected == selected_texts,
        "quality_mean": sum(item["quality"] for item in scored) / len(scored),
        "mean_surprise_mean": sum(item["mean_surprise"] for item in scored) / len(scored),
        "rows": [
            {
                "sha16": _preview(item["text"])["sha16"],
                "bytes": item["bytes"],
                "quality": item["quality"],
                "quality_without_surprise": item["quality_without_surprise"],
                "mean_surprise": item["mean_surprise"],
                "mean_error_norm": item["mean_error_norm"],
                "mean_confidence": item["mean_confidence"],
                "accuracy": item["accuracy"],
            }
            for item in scored
        ],
    }


def _readout(model: Any, texts: list[str]) -> dict[str, float]:
    """材料读数：对这组文本的 judge quality／accuracy／surprise 均值（learn=False 侧）。"""

    from seed.judge import SeedJudge

    judge = SeedJudge(model)
    rows = [judge.score(text.encode("utf-8")) for text in texts if text.strip()]
    if not rows:
        raise RuntimeError("材料读数为空")
    return {
        "items": float(len(rows)),
        "quality_mean": sum(row["quality"] for row in rows) / len(rows),
        "accuracy_mean": sum(row["accuracy"] for row in rows) / len(rows),
        "mean_surprise_mean": sum(row["mean_surprise"] for row in rows) / len(rows),
    }


def phase_powerup() -> int:
    """①母状态→④通电一次→候选档→⑥回退演示→治疗/对照配对材料读数。"""

    from api.seed_runtime import DEFAULT_CHECKPOINT, SeedRuntime
    from seed_platform import sleep_pass

    started = time.time()
    runtime = SeedRuntime.load(DEFAULT_CHECKPOINT)
    mother_state = runtime.model.checkpoint()
    mother_digest = _digest(runtime.model)
    mother_tick = int(runtime.model.tick)

    #: G-N2-3 的可测那半：母档独立落盘、独立 load、摘要必须逐位同。
    MOTHER_PATH.parent.mkdir(parents=True, exist_ok=True)
    runtime.save(MOTHER_PATH)
    mother_anchor = SeedRuntime.load(MOTHER_PATH)
    mother_anchor_digest = _digest(mother_anchor.model)

    pass_report = sleep_pass.run(runtime=runtime, reason="n2-powerup-1", organs=True, learn=True)
    organs = pass_report.get("organs") or {}
    treated_digest = _digest(runtime.model)
    treated_tick = int(runtime.model.tick)

    payload: dict[str, Any] = {
        "format": "taiji-n2-powerup-v1",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
        "loaded_checkpoint": str(DEFAULT_CHECKPOINT),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "mother_digest": mother_digest,
        "mother_tick": mother_tick,
        "mother_saved_path": str(MOTHER_PATH.relative_to(PROJECT_ROOT)),
        "mother_saved_bytes": MOTHER_PATH.stat().st_size,
        "mother_anchor_digest": mother_anchor_digest,
        "mother_anchor_matches_inmemory": mother_anchor_digest == mother_digest,
        "pass_id": pass_report.get("pass_id"),
        "projection_records": (pass_report.get("projection") or {}).get("records"),
        "by_source": (pass_report.get("projection") or {}).get("by_source"),
        "workbench_note": (pass_report.get("projection") or {}).get("workbench_note"),
        "spec_written": (pass_report.get("spec") or {}).get("written"),
        "organs": organs,
        "treated_digest": treated_digest,
        "treated_tick": treated_tick,
        "weight_changed_by_powerup": treated_digest != mother_digest,
        "duration_seconds": round(time.time() - started, 1),
    }

    material = _material_texts(pass_report)
    payload["material_records"] = len(material)
    if not organs.get("ran") or not organs.get("learn"):
        raise RuntimeError(f"通电没有真的改权重：organs={organs!r}——整件不判，见 §4ter 实测二")
    if not payload["weight_changed_by_powerup"]:
        raise RuntimeError("通电跑完但权重摘要未变——判据不成立，整件不判")

    runtime.save(CANDIDATE_PATH)
    payload["candidate_saved_path"] = str(CANDIDATE_PATH.relative_to(PROJECT_ROOT))
    payload["candidate_saved_bytes"] = CANDIDATE_PATH.stat().st_size

    #: G-N2-1 的回退演示：还原母状态后摘要必须逐位同，且还原后的磁盘往返也同。
    runtime.model.restore(mother_state)
    payload["restored_digest"] = _digest(runtime.model)
    payload["restore_matches_mother"] = payload["restored_digest"] == mother_digest
    runtime.save(ROLLBACK_PATH)
    rollback_runtime = SeedRuntime.load(ROLLBACK_PATH)
    payload["rollback_anchor_digest"] = _digest(rollback_runtime.model)
    payload["rollback_disk_matches_mother"] = payload["rollback_anchor_digest"] == mother_digest
    payload["rollback_saved_bytes"] = ROLLBACK_PATH.stat().st_size

    #: 材料读数在**回退（＝同一份母权重）与候选**两档上对同一批夜文本各取一次。
    k = int(organs.get("texts") or 0)
    if k <= 0:
        raise RuntimeError(f"通电没有交出夜文本条数：organs={organs!r}")
    night = _night_selection(rollback_runtime.model, material, k=k)
    payload["night_selection"] = night
    payload["pool_equals_material"] = night["pool_size"] == len([t for t in material if t.strip()])
    selected = night["selected_texts"]
    treated_runtime = SeedRuntime.load(CANDIDATE_PATH)
    payload["material_treated"] = _readout(treated_runtime.model, selected)
    payload["material_control"] = _readout(rollback_runtime.model, selected)
    payload["j_n2b_control_frozen"] = payload["material_control"]["quality_mean"]
    payload["j_n2b_value"] = payload["material_treated"]["quality_mean"]
    payload["j_n2b"] = payload["j_n2b_value"] > payload["j_n2b_control_frozen"]

    fail_closed = [
        "mother_anchor_matches_inmemory",
        "weight_changed_by_powerup",
        "restore_matches_mother",
        "rollback_disk_matches_mother",
        "night_selection",
    ]
    payload["fail_closed_pass"] = (
        bool(payload["mother_anchor_matches_inmemory"])
        and bool(payload["weight_changed_by_powerup"])
        and bool(payload["restore_matches_mother"])
        and bool(payload["rollback_disk_matches_mother"])
    )
    payload["hard_fields_present"] = all(key in payload for key in fail_closed)
    payload["seconds"] = round(time.time() - started, 1)
    #: 全文只用于配对读数，不进件（件里留摘要＋预览＋分布行）。
    payload["night_selection"].pop("selected_texts")

    _write_report(POWERUP_REPORT, payload)
    print(f"weight_changed={payload['weight_changed_by_powerup']}")
    print(f"restore_matches_mother={payload['restore_matches_mother']}")
    print(f"rollback_disk_matches_mother={payload['rollback_disk_matches_mother']}")
    print(f"night_pool={night['pool_size']} k={night['k']}")
    print(
        f"j_n2b={payload['j_n2b']} treated={payload['j_n2b_value']} control={payload['j_n2b_control_frozen']}"
    )
    print(f"fail_closed_pass={payload['fail_closed_pass']}")
    print(f"report -> {POWERUP_REPORT}")
    return 0 if payload["fail_closed_pass"] else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N2 巩固通电驱动（PLAN-N2-01 §4ter）")
    parser.add_argument("--phase", choices=("guard", "powerup"), required=True)
    args = parser.parse_args(argv)
    if args.phase == "guard":
        return phase_guard()
    return phase_powerup()


if __name__ == "__main__":
    raise SystemExit(main())
