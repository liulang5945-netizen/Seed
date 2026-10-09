"""PLAN-N2-02：第二次通电驱动（G-N2-5 双向装载守卫＋剂量同窗的 J-N2b'）。

与第一次（`run_taiji_n2_powerup.py --phase powerup`）的三处不同，都是批文要求的，不是换指标：

1. 顺序按 PLAN-N2-02 §4：母状态→G-N2-2 守卫臂→通电→G-N2-5→窗内两臂→还原→窗内对照；
2. **不产对齐件**（G-N2-6）：候选档读不回来就是 J-持久化 判不过，整件不判能力——第一次靠对齐件
   读出了能力，代价是结论被限定成"对齐之后的候选权重"，这次要红就要红；
3. 新收益的分子＝每条夜文本的**前 `max_symbols` 字节**（与经验预算同一式子），整篇那一列照旧取
   但只作旁证（§0(i) 结的就是这格不同窗）。

改权重半径＝owner 2026-10-08 第三次弹窗批（09 §3.2 第 6 条）。三枚档都走新名字，
`DEFAULT_CHECKPOINT` 与第一次的产物都不覆写。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_taiji_n2_powerup import (  # noqa: E402 - 复用第一次的取数件，不重抄生成链
    _digest,
    _material_texts,
    _night_selection,
    _readout,
    _write_report,
)

REPORT_DIR = PROJECT_ROOT / "reports"
MOTHER_PATH = PROJECT_ROOT / "checkpoints" / "seed_n2b_mother_20261008.pt"
CANDIDATE_PATH = PROJECT_ROOT / "checkpoints" / "seed_n2b_candidate_20261008.pt"
ROLLBACK_PATH = PROJECT_ROOT / "checkpoints" / "seed_n2b_rollback_20261008.pt"
POWERUP2_REPORT = REPORT_DIR / "taiji_n2b_powerup_20261008.json"
GUARD2_REPORT = REPORT_DIR / "taiji_n2b_guard_default_20261008.json"


def _window_rows(model: Any, texts: list[str], dose: int) -> dict[str, float]:
    """剂量同窗的材料读数：每条只取前 `dose` 个字节，`dose` 来自 pass 自己的自述。"""

    from seed.judge import SeedJudge

    judge = SeedJudge(model)
    rows = [judge.score(text.encode("utf-8")[:dose]) for text in texts if text.strip()]
    if not rows:
        raise RuntimeError("窗口读数为空")
    return {
        "items": float(len(rows)),
        "quality_mean": sum(row["quality"] for row in rows) / len(rows),
        "accuracy_mean": sum(row["accuracy"] for row in rows) / len(rows),
        "mean_surprise_mean": sum(row["mean_surprise"] for row in rows) / len(rows),
    }


def phase_guard2() -> int:
    """G-N2-2 在第二次里的重跑：默认参数 pass 不碰器官也不碰权重。"""

    from api.seed_runtime import DEFAULT_CHECKPOINT, SeedRuntime
    from seed_platform import sleep_pass

    runtime = SeedRuntime.load(DEFAULT_CHECKPOINT)
    before = _digest(runtime.model)
    report = sleep_pass.run(reason="n2b-guard-default")
    after = _digest(runtime.model)
    organs = report.get("organs") or {}
    checks = {
        "organs_ran_false": organs.get("ran") is False,
        "weight_digest_unchanged": before == after,
    }
    payload = {
        "format": "taiji-n2b-guard-v1",
        "prereg": "plans/reference/PLAN-N2-02_second_powerup_dosewindow_prereg_20261008.md",
        "digest_before": before,
        "digest_after": after,
        "organs": organs,
        "checks": checks,
        "guard_pass": all(bool(value) for value in checks.values()),
        "pass_id": report.get("pass_id"),
        "projection_records": (report.get("projection") or {}).get("records"),
    }
    _write_report(GUARD2_REPORT, payload)
    print(
        f"guard_pass={payload['guard_pass']} digest_unchanged={checks['weight_digest_unchanged']}"
    )
    return 0 if payload["guard_pass"] else 1


def phase_powerup2() -> int:
    """第二次通电本体＋双向装载守卫＋窗内配对读数（顺序即批文 §4）。"""

    from api.seed_runtime import DEFAULT_CHECKPOINT, SeedRuntime
    from seed_platform import sleep_pass

    started = time.time()
    payload: dict[str, Any] = {
        "format": "taiji-n2b-powerup-v1",
        "prereg": "plans/reference/PLAN-N2-02_second_powerup_dosewindow_prereg_20261008.md",
        "loaded_checkpoint": str(DEFAULT_CHECKPOINT),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)),
        "stage": "start",
        "errors": [],
    }
    code = 1
    try:
        runtime = SeedRuntime.load(DEFAULT_CHECKPOINT)
        mother_state = runtime.model.checkpoint()
        payload["mother_digest"] = _digest(runtime.model)
        payload["mother_tick"] = int(runtime.model.tick)
        MOTHER_PATH.parent.mkdir(parents=True, exist_ok=True)
        runtime.save(MOTHER_PATH)
        mother_anchor = SeedRuntime.load(MOTHER_PATH)
        payload["mother_saved_path"] = str(MOTHER_PATH.relative_to(PROJECT_ROOT))
        payload["mother_anchor_digest"] = _digest(mother_anchor.model)
        #: G-N2-5 的反向半支：没睡过觉的档 save→load 摘要必须逐位不变——
        #: 防"修法是靠覆写有效状态换取可装载"。
        payload["g_n2_5_mother_roundtrip_identical"] = (
            payload["mother_anchor_digest"] == payload["mother_digest"]
        )
        payload["stage"] = "mother_anchored"

        pass_report = sleep_pass.run(
            runtime=runtime, reason="n2-powerup-2", organs=True, learn=True
        )
        organs = pass_report.get("organs") or {}
        treated_digest = _digest(runtime.model)
        payload["stage"] = "pass_done"
        payload.update(
            {
                "pass_id": pass_report.get("pass_id"),
                "projection_records": (pass_report.get("projection") or {}).get("records"),
                "by_source": (pass_report.get("projection") or {}).get("by_source"),
                "organs": organs,
                "treated_digest": treated_digest,
                "weight_changed_by_powerup": treated_digest != payload["mother_digest"],
            }
        )
        if not organs.get("ran") or not organs.get("learn"):
            raise RuntimeError(f"通电没有真的改权重：organs={organs!r}")

        #: G-N2-5 正向半支：**这里不留任何对齐旁路**（G-N2-6），读不回来即 J-持久化 不过。
        runtime.save(CANDIDATE_PATH)
        payload["candidate_saved_path"] = str(CANDIDATE_PATH.relative_to(PROJECT_ROOT))
        payload["candidate_saved_bytes"] = CANDIDATE_PATH.stat().st_size
        try:
            reloaded = SeedRuntime.load(CANDIDATE_PATH)
            payload["candidate_digest_after_load"] = _digest(reloaded.model)
            payload["j_persistence"] = payload["candidate_digest_after_load"] == treated_digest
        except Exception as exc:  # noqa: BLE001 - 这条拒绝就是判据要的答案
            payload["candidate_digest_after_load"] = None
            payload["j_persistence"] = False
            payload["persistence_error"] = f"{type(exc).__name__}: {exc}"
        payload["stage"] = "persistence_checked"

        material = _material_texts(pass_report)
        dose = int(organs.get("max_symbols") or 64)
        k = int(organs.get("texts") or 0)
        if k <= 0 or not material:
            raise RuntimeError(
                f"通电没有交出材料或夜文本条数：organs={organs!r} material={len(material)}"
            )
        night = _night_selection(mother_anchor.model, material, k=k)
        selected = night["selected_texts"]
        payload["window_bytes"] = dose
        payload["night_pool_size"] = night["pool_size"]
        payload["night_k"] = night["k"]
        payload["scheduler_agrees_with_weighted_sort"] = night[
            "scheduler_agrees_with_weighted_sort"
        ]
        payload["surprise_modulation"] = {
            "selected_before_modulation": night["selected_before_modulation"],
            "selected_after_modulation": night["selected_after_modulation"],
            "quality_mean": night["quality_mean"],
            "mean_surprise_mean": night["mean_surprise_mean"],
        }
        payload["material_treated_window"] = _window_rows(runtime.model, selected, dose)
        payload["material_treated_whole"] = _readout(runtime.model, selected)
        payload["stage"] = "treated_read"

        runtime.model.restore(mother_state)
        payload["restore_matches_mother"] = _digest(runtime.model) == payload["mother_digest"]
        runtime.save(ROLLBACK_PATH)
        rollback_reload = SeedRuntime.load(ROLLBACK_PATH)
        payload["rollback_disk_matches_mother"] = (
            _digest(rollback_reload.model) == payload["mother_digest"]
        )
        payload["material_control_window"] = _window_rows(runtime.model, selected, dose)
        payload["material_control_whole"] = _readout(runtime.model, selected)
        payload["stage"] = "control_read"

        delta_quality = (
            payload["material_treated_window"]["quality_mean"]
            - payload["material_control_window"]["quality_mean"]
        )
        delta_accuracy = (
            payload["material_treated_window"]["accuracy_mean"]
            - payload["material_control_window"]["accuracy_mean"]
        )
        payload["j_n2b_prime_delta_quality"] = delta_quality
        payload["j_n2b_prime_delta_accuracy"] = delta_accuracy
        if delta_quality > 0 and delta_accuracy >= 0:
            payload["j_n2b_prime"] = "holds"
        elif delta_quality <= 0:
            payload["j_n2b_prime"] = "not_holds"
        else:
            payload["j_n2b_prime"] = "not_resolved"
        #: 整篇那一列只作旁证，不参与判级（§2 明写）。
        payload["whole_text_delta_quality"] = (
            payload["material_treated_whole"]["quality_mean"]
            - payload["material_control_whole"]["quality_mean"]
        )
        hard = (
            "g_n2_5_mother_roundtrip_identical",
            "weight_changed_by_powerup",
            "restore_matches_mother",
            "rollback_disk_matches_mother",
        )
        payload["fail_closed_pass"] = all(bool(payload[key]) for key in hard)
        payload["stage"] = "done"
        code = 0 if payload["fail_closed_pass"] else 1
    except Exception as exc:  # noqa: BLE001 - 失败也要把已取到的读数留在盘上
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        payload["seconds"] = round(time.time() - started, 1)
        _write_report(POWERUP2_REPORT, payload)
        print(f"stage={payload.get('stage')} errors={json.dumps(payload.get('errors'))}")
        for key in (
            "j_persistence",
            "weight_changed_by_powerup",
            "g_n2_5_mother_roundtrip_identical",
            "restore_matches_mother",
            "rollback_disk_matches_mother",
            "j_n2b_prime",
            "fail_closed_pass",
        ):
            print(f"  {key}={payload.get(key)}")
        print(f"report -> {POWERUP2_REPORT}")
    return code


def _apply_run_tag(tag: str) -> None:
    """把四枚落点名整体换一档（G-N2c-2：治疗档必须落**新名**，不许覆写 run-2 的证据）。

    默认不给 ⇒ 名字与今天逐字相同（旧命令与既有件不受影响）；给了 tag 就在词干后追加 `_<tag>`，
    母档／候选档／回滚档／两份报告一起换，免得只改一半造出"前后不同源"这种最难查的错。
    """

    if not tag:
        return
    global MOTHER_PATH, CANDIDATE_PATH, ROLLBACK_PATH, POWERUP2_REPORT, GUARD2_REPORT
    for name in (
        "MOTHER_PATH",
        "CANDIDATE_PATH",
        "ROLLBACK_PATH",
        "POWERUP2_REPORT",
        "GUARD2_REPORT",
    ):
        path = globals()[name]
        globals()[name] = path.with_name(f"{path.stem}_{tag}{path.suffix}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N2 第二次通电驱动（PLAN-N2-02）")
    parser.add_argument("--phase", choices=("guard2", "powerup2"), required=True)
    parser.add_argument(
        "--run-tag",
        default="",
        help="给落点名统一追加 _<tag> 后缀（G-N2c-2 要求治疗档落新名；默认空＝沿用今天的名字）",
    )
    args = parser.parse_args(argv)
    _apply_run_tag(args.run_tag)
    if args.phase == "guard2":
        return phase_guard2()
    return phase_powerup2()


if __name__ == "__main__":
    raise SystemExit(main())
