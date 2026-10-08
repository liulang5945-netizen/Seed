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
ALIGNED_PATH = PROJECT_ROOT / "checkpoints" / "seed_n2_candidate_aligned_20261008.pt"
REPORT_DIR = PROJECT_ROOT / "reports"
GUARD_REPORT = REPORT_DIR / "taiji_n2_guard_default_20261008.json"
POWERUP_REPORT = REPORT_DIR / "taiji_n2_powerup_20261008.json"
MATERIAL_REPORT = REPORT_DIR / "taiji_n2_material_20261008.json"

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
    payload: dict[str, Any] = {
        "format": "taiji-n2-powerup-v2",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
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

        #: G-N2-3 的可测那半：母档独立落盘、独立 load、摘要必须逐位同。
        MOTHER_PATH.parent.mkdir(parents=True, exist_ok=True)
        runtime.save(MOTHER_PATH)
        mother_anchor = SeedRuntime.load(MOTHER_PATH)
        payload["mother_saved_path"] = str(MOTHER_PATH.relative_to(PROJECT_ROOT))
        payload["mother_saved_bytes"] = MOTHER_PATH.stat().st_size
        payload["mother_anchor_digest"] = _digest(mother_anchor.model)
        payload["mother_anchor_matches_inmemory"] = (
            payload["mother_anchor_digest"] == payload["mother_digest"]
        )
        payload["stage"] = "mother_anchored"

        pass_report = sleep_pass.run(
            runtime=runtime, reason="n2-powerup-1", organs=True, learn=True
        )
        organs = pass_report.get("organs") or {}
        payload["stage"] = "pass_done"
        payload.update(
            {
                "pass_id": pass_report.get("pass_id"),
                "projection_records": (pass_report.get("projection") or {}).get("records"),
                "by_source": (pass_report.get("projection") or {}).get("by_source"),
                "workbench_note": (pass_report.get("projection") or {}).get("workbench_note"),
                "spec_written": (pass_report.get("spec") or {}).get("written"),
                "organs": organs,
                "treated_digest": _digest(runtime.model),
                "treated_tick": int(runtime.model.tick),
            }
        )
        payload["weight_changed_by_powerup"] = payload["treated_digest"] != payload["mother_digest"]
        if not organs.get("ran") or not organs.get("learn"):
            raise RuntimeError(f"通电没有真的改权重：organs={organs!r}——整件不判")
        if not payload["weight_changed_by_powerup"]:
            raise RuntimeError("通电跑完但权重摘要未变——判据不成立，整件不判")

        runtime.save(CANDIDATE_PATH)
        payload["candidate_saved_path"] = str(CANDIDATE_PATH.relative_to(PROJECT_ROOT))
        payload["candidate_saved_bytes"] = CANDIDATE_PATH.stat().st_size
        payload["stage"] = "candidate_saved"

        #: 夜文本用**母锚**（磁盘件可正常 load，且与内存母状态摘要同）重算；
        #: 治疗读数用**内存里这份刚改过权重、还没还原**的 runtime——候选档经
        #: `SeedRuntime.load` 会被 §12 同步守卫拒收（㊵-487 登记的产品缺陷），
        #: 所以治疗臂必须在还原之前取，否则整条治疗侧一个读数都留不下。
        material = _material_texts(pass_report)
        payload["material_records"] = len(material)
        k = int(organs.get("texts") or 0)
        if k <= 0:
            raise RuntimeError(f"通电没有交出夜文本条数：organs={organs!r}")
        night = _night_selection(mother_anchor.model, material, k=k)
        payload["night_selection"] = night
        payload["pool_equals_material"] = night["pool_size"] == len(
            [t for t in material if t.strip()]
        )
        selected = night["selected_texts"]
        payload["material_treated"] = _readout(runtime.model, selected)
        payload["stage"] = "treated_read"

        #: G-N2-1 的回退演示：还原母状态→摘要逐位同→落盘→对照读数。
        runtime.model.restore(mother_state)
        payload["restored_digest"] = _digest(runtime.model)
        payload["restore_matches_mother"] = payload["restored_digest"] == payload["mother_digest"]
        runtime.save(ROLLBACK_PATH)
        payload["rollback_saved_bytes"] = ROLLBACK_PATH.stat().st_size
        payload["material_control"] = _readout(runtime.model, selected)
        payload["stage"] = "rollback_read"

        #: 两档各自尝试一次独立磁盘 load：这条**不是仪器的失败，是被观测的产品行为**，
        #: 只记录、绝不让它中断取数。回退档应当为真，候选档当前应当为假。
        for label, path in (("rollback", ROLLBACK_PATH), ("candidate", CANDIDATE_PATH)):
            try:
                reloaded = SeedRuntime.load(path)
                payload[f"{label}_disk_digest"] = _digest(reloaded.model)
                payload[f"{label}_disk_loads"] = True
            except Exception as exc:  # noqa: BLE001 - 被观测的产品拒绝
                payload[f"{label}_disk_digest"] = None
                payload[f"{label}_disk_loads"] = False
                payload[f"{label}_disk_load_error"] = f"{type(exc).__name__}: {exc}"
        payload["rollback_disk_matches_mother"] = (
            payload.get("rollback_disk_digest") == payload["mother_digest"]
        )

        payload["j_n2b_control_frozen"] = payload["material_control"]["quality_mean"]
        payload["j_n2b_value"] = payload["material_treated"]["quality_mean"]
        payload["j_n2b"] = payload["j_n2b_value"] > payload["j_n2b_control_frozen"]
        payload["fail_closed_pass"] = all(
            bool(payload[key])
            for key in (
                "mother_anchor_matches_inmemory",
                "weight_changed_by_powerup",
                "restore_matches_mother",
            )
        )
        payload["scheduler_agrees_with_weighted_sort"] = night[
            "scheduler_agrees_with_weighted_sort"
        ]
        payload["stage"] = "done"
        code = 0 if payload["fail_closed_pass"] else 1
    except Exception as exc:  # noqa: BLE001 - 失败也要把已取到的读数留在盘上
        payload["errors"].append(f"{type(exc).__name__}: {exc}")
    finally:
        (payload.get("night_selection") or {}).pop("selected_texts", None)
        payload["seconds"] = round(time.time() - started, 1)
        _write_report(POWERUP_REPORT, payload)
        print(f"stage={payload.get('stage')} errors={json.dumps(payload.get('errors'))}")
        for key in (
            "weight_changed_by_powerup",
            "restore_matches_mother",
            "rollback_disk_loads",
            "candidate_disk_loads",
            "j_n2b",
            "fail_closed_pass",
        ):
            print(f"  {key}={payload.get(key)}")
        print(f"report -> {POWERUP_REPORT}")
    return code


def phase_postcheck() -> int:
    """只读复查已落盘三档：磁盘装载性、信封两半的 tick/episode、张量级改动幅度。

    存在理由：`--phase powerup` 的 v1 把候选档的磁盘 load 放在取数**之前**，守卫一拒收就
    整件无读数可留。三档都已在盘上，这里把该测的测完，**不重跑任何改权重**（批文只批一次）。
    """

    import torch

    from api.seed_runtime import SeedRuntime

    payload: dict[str, Any] = {
        "format": "taiji-n2-postcheck-v1",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
        "files": {},
    }
    for label, path in (
        ("mother", MOTHER_PATH),
        ("candidate", CANDIDATE_PATH),
        ("rollback", ROLLBACK_PATH),
    ):
        if not path.is_file():
            payload["files"][label] = {"absent": True}
            continue
        entry: dict[str, Any] = {"bytes": path.stat().st_size}
        raw = torch.load(path, map_location="cpu", weights_only=True)
        taiji = raw.get("taiji") or {}
        kernel_state = (taiji.get("kernel") or {}).get("state") or {}
        state = taiji.get("cognitive_state") or {}
        entry["metadata_tick"] = (raw.get("metadata") or {}).get("tick")
        entry["cognitive_state_tick"] = state.get("tick")
        entry["cognitive_state_episode_id"] = state.get("episode_id")
        entry["kernel_state_tick"] = kernel_state.get("tick")
        entry["kernel_state_episode_id"] = kernel_state.get("episode_id")
        try:
            reloaded = SeedRuntime.load(path)
            entry["disk_load_ok"] = True
            entry["digest"] = _digest(reloaded.model)
        except Exception as exc:  # noqa: BLE001 - 被观测的产品拒绝，不是仪器失败
            entry["disk_load_ok"] = False
            entry["digest"] = None
            entry["disk_load_error"] = f"{type(exc).__name__}: {exc}"
        payload["files"][label] = entry

    def tensors(path: Path) -> dict[str, Any]:
        found: dict[str, Any] = {}

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}.{key}")
            elif torch.is_tensor(node):
                found[prefix] = node

        walk(torch.load(path, map_location="cpu", weights_only=True), "")
        return found

    base = tensors(MOTHER_PATH)
    deltas: dict[str, Any] = {}
    for label in ("candidate", "rollback"):
        other = tensors(PROJECT_ROOT / f"checkpoints/seed_n2_{label}_20261008.pt")
        shared = sorted(set(base) & set(other))
        changed = 0
        max_abs = 0.0
        for key in shared:
            a, b = base[key], other[key]
            if a.shape != b.shape or not torch.equal(a, b):
                changed += 1
                if a.shape == b.shape and a.numel():
                    max_abs = max(max_abs, float((a.float() - b.float()).abs().max()))
        deltas[label] = {
            "tensors_compared": len(shared),
            "tensors_only_in_one": len(set(base) ^ set(other)),
            "tensors_changed": changed,
            "changed_ratio": round(changed / max(1, len(shared)), 6),
            "max_abs_element_delta": max_abs,
        }
    payload["tensor_delta"] = deltas

    out = REPORT_DIR / "taiji_n2_postcheck_20261008.json"
    _write_report(out, payload)
    for label, entry in payload["files"].items():
        print(
            json.dumps(
                {
                    "label": label,
                    "disk_load_ok": entry.get("disk_load_ok"),
                    "cognitive_state_tick": entry.get("cognitive_state_tick"),
                    "kernel_state_tick": entry.get("kernel_state_tick"),
                    "episode": entry.get("cognitive_state_episode_id"),
                    "digest": (entry.get("digest") or "")[:16],
                    "error": (entry.get("disk_load_error") or "")[:60],
                },
                ensure_ascii=True,
            )
        )
    print(f"tensor_delta={json.dumps(deltas, ensure_ascii=True)}")
    print(f"report -> {out}")
    return 0


def phase_align() -> int:
    """把候选档的信封两半**对齐**成一份新件，并证明对齐只动计数器、不动任何张量。

    为什么走这条路而不是去放宽仪器里的守卫：`restore_native` 末尾那条守卫比的是**同一份
    信封自己的两半**（`cognitive_state` 对 `kernel.state`，[taiji/adapter.py:12631-12632]）。
    通电把 runtime 留在它自己的睡眠回合里（cognitive_state＝sleep-experience/tick 66，
    kernel＝tick 92），于是**产品自己存的档产品自己装不回来**。这里不修产品、不改仪器，
    只产出一份"如果落盘前先收束回合会得到什么"的候选件：只把 `cognitive_state` 的
    tick/episode 补成 kernel 那半的值，张量一字不动——这条正反对照由本相自己实测并落件。
    """

    import torch

    from api.seed_runtime import SeedRuntime
    from seed.persistence import atomic_save

    payload: dict[str, Any] = {
        "format": "taiji-n2-align-v1",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
        "guard_source": "taiji/adapter.py:12631-12632",
    }
    raw = torch.load(CANDIDATE_PATH, map_location="cpu", weights_only=True)
    aligned = dict(raw)
    taiji = dict(raw["taiji"])
    state = dict(taiji["cognitive_state"])
    kernel_state = (taiji.get("kernel") or {}).get("state") or {}
    state["tick"] = kernel_state.get("tick", state.get("tick"))
    state["episode_id"] = kernel_state.get("episode_id", state.get("episode_id"))
    taiji["cognitive_state"] = state
    aligned["taiji"] = taiji
    payload["cognitive_state_tick_before"] = raw["taiji"]["cognitive_state"].get("tick")
    payload["cognitive_state_episode_before"] = raw["taiji"]["cognitive_state"].get("episode_id")
    payload["cognitive_state_tick_after"] = state["tick"]
    payload["cognitive_state_episode_after"] = state["episode_id"]
    payload["kernel_state_tick"] = kernel_state.get("tick")
    payload["kernel_state_episode"] = kernel_state.get("episode_id")

    ALIGNED_PATH.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(aligned, ALIGNED_PATH)
    payload["aligned_saved_path"] = str(ALIGNED_PATH.relative_to(PROJECT_ROOT))
    payload["aligned_saved_bytes"] = ALIGNED_PATH.stat().st_size

    #: 正反对照两支都要走：未对齐的候选档**应当**被拒收，对齐件**应当**收下且张量逐位同。
    try:
        SeedRuntime.load(CANDIDATE_PATH)
        payload["candidate_still_refuses_load"] = False
    except Exception as exc:  # noqa: BLE001 - 这条拒绝就是要观测的
        payload["candidate_still_refuses_load"] = True
        payload["candidate_refusal"] = f"{type(exc).__name__}: {exc}"
    reloaded = SeedRuntime.load(ALIGNED_PATH)
    payload["aligned_loads"] = True
    payload["aligned_digest"] = _digest(reloaded.model)

    def tensor_map(path: Path) -> dict[str, Any]:
        found: dict[str, Any] = {}

        def walk(node: Any, prefix: str) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{prefix}.{key}")
            elif torch.is_tensor(node):
                found[prefix] = node

        walk(torch.load(path, map_location="cpu", weights_only=True), "")
        return found

    base, other = tensor_map(CANDIDATE_PATH), tensor_map(ALIGNED_PATH)
    shared = sorted(set(base) & set(other))
    payload["tensor_check"] = {
        "tensors_compared": len(shared),
        "tensors_only_in_one": len(set(base) ^ set(other)),
        "tensors_differing": sum(
            1
            for key in shared
            if base[key].shape != other[key].shape or not torch.equal(base[key], other[key])
        ),
    }
    payload["align_pass"] = bool(
        payload["candidate_still_refuses_load"]
        and payload["aligned_loads"]
        and payload["tensor_check"]["tensors_differing"] == 0
        and payload["tensor_check"]["tensors_only_in_one"] == 0
    )

    out = REPORT_DIR / "taiji_n2_align_20261008.json"
    _write_report(out, payload)
    print(
        json.dumps(
            {
                "candidate_still_refuses_load": payload["candidate_still_refuses_load"],
                "aligned_loads": payload["aligned_loads"],
                "aligned_digest": payload["aligned_digest"][:16],
                "tensors_differing": payload["tensor_check"]["tensors_differing"],
                "align_pass": payload["align_pass"],
            },
            ensure_ascii=True,
        )
    )
    print(f"report -> {out}")
    return 0 if payload["align_pass"] else 1


def phase_material() -> int:
    """J-N2b 的配对材料读数：治疗＝对齐件（候选权重），对照＝回退件（母权重）。

    夜文本用**母权重**在全池上重算（与 pass 自己的 `select_for_sleep` 同池同序同函数），
    两臂读的是**同一批文本、同一台仪器**（`SeedJudge.score`），只换权重那一份。
    """

    from api.seed_runtime import SeedRuntime
    from seed_platform import sleep_pass

    started = time.time()
    last = json.loads(
        (Path(sleep_pass._consolidation_dir()) / "last_report.json").read_text(encoding="utf-8")
    )
    payload: dict[str, Any] = {
        "format": "taiji-n2-material-v1",
        "prereg": "plans/reference/PLAN-N2-01_consolidation_powerup_prereg_20261007.md#4ter",
        "pass_id": last.get("pass_id"),
        "pass_reason": last.get("reason"),
    }
    if last.get("reason") != "n2-powerup-1":
        raise RuntimeError(f"last_report 不是那次通电：reason={last.get('reason')!r}")
    organs = last.get("organs") or {}
    material = _material_texts(last)
    k = int(organs.get("texts") or 0)
    if k <= 0 or not material:
        raise RuntimeError(
            f"通电自述缺材料或夜文本条数：organs={organs!r} material={len(material)}"
        )

    control_runtime = SeedRuntime.load(ROLLBACK_PATH)
    treated_runtime = SeedRuntime.load(ALIGNED_PATH)
    night = _night_selection(control_runtime.model, material, k=k)
    selected = night["selected_texts"]
    payload["material_records"] = len(material)
    payload["night_pool_size"] = night["pool_size"]
    payload["night_k"] = night["k"]
    payload["scheduler_agrees_with_weighted_sort"] = night["scheduler_agrees_with_weighted_sort"]
    payload["selected"] = night["selected_after_modulation"]
    payload["surprise_modulation"] = {
        "selected_before_modulation": night["selected_before_modulation"],
        "quality_mean": night["quality_mean"],
        "mean_surprise_mean": night["mean_surprise_mean"],
        "rows": night["rows"],
    }
    payload["material_treated"] = _readout(treated_runtime.model, selected)
    payload["material_control"] = _readout(control_runtime.model, selected)
    payload["j_n2b_control_frozen"] = payload["material_control"]["quality_mean"]
    payload["j_n2b_value"] = payload["material_treated"]["quality_mean"]
    payload["j_n2b"] = payload["j_n2b_value"] > payload["j_n2b_control_frozen"]
    payload["treated_digest"] = _digest(treated_runtime.model)
    payload["control_digest"] = _digest(control_runtime.model)
    payload["seconds"] = round(time.time() - started, 1)

    _write_report(MATERIAL_REPORT, payload)
    print(
        json.dumps(
            {
                "night_k": payload["night_k"],
                "pool": payload["night_pool_size"],
                "treated": payload["j_n2b_value"],
                "control": payload["j_n2b_control_frozen"],
                "j_n2b": payload["j_n2b"],
                "treated_digest": payload["treated_digest"][:16],
                "control_digest": payload["control_digest"][:16],
            },
            ensure_ascii=True,
        )
    )
    print(f"report -> {MATERIAL_REPORT}")
    return 0


def phase_dose() -> int:
    """诊断臂（**事后加，不参与任何判据**）：把材料读数收在 pass 真正经验过的剂量窗内。

    通电那支 pass 的经验预算是 `max_symbols=64`（`run()` 现行默认），而 J-N2b 的读数用
    `SeedJudge.score()` 打的是**整篇**——一条几千字节的交互里，模型这轮只活过前 64 字节。
    ⇒ 判据按原判读（不成立）入册，本臂只回答一件事：改动落在剂量窗内时，窗内的数怎么走。
    """

    from api.seed_runtime import SeedRuntime
    from seed.judge import SeedJudge
    from seed_platform import sleep_pass

    last = json.loads(
        (Path(sleep_pass._consolidation_dir()) / "last_report.json").read_text(encoding="utf-8")
    )
    organs = last.get("organs") or {}
    dose = int(organs.get("max_symbols") or 64)
    material_record = json.loads(Path(MATERIAL_REPORT).read_text(encoding="utf-8"))
    wanted = [str(item["sha16"]) for item in material_record["selected"]]
    #: 件里只留了每条文本的 sha16 与短预览（不把整段语料抄进判读件），
    #: 这里按**同一式子**（sha256(text)[:16]）从本 pass 落盘的语料里把全文取回来。
    by_digest = {}
    for text in _material_texts(last):
        by_digest.setdefault(sha256(text.encode("utf-8")).hexdigest()[:16], text)
    texts = [by_digest[key] for key in wanted if key in by_digest]
    if len(texts) != len(wanted):
        raise RuntimeError(f"剂量臂取不回全部选中材料：{len(texts)}/{len(wanted)}")
    payload: dict[str, Any] = {
        "format": "taiji-n2-dose-diagnostic-v1",
        "kind": "diagnostic_post_hoc",
        "note": "本臂在 J-N2b 判读**之后**加，不参与判据；只回答剂量窗内的数怎么走。",
        "dose_symbols": dose,
        "selected_texts": len(texts),
    }
    for label, path in (("treated", ALIGNED_PATH), ("control", ROLLBACK_PATH)):
        model = SeedRuntime.load(path).model
        judge = SeedJudge(model)
        rows = [judge.score(text.encode("utf-8")[:dose]) for text in texts if text.strip()]
        payload[label] = {
            "items": float(len(rows)),
            "quality_mean": sum(row["quality"] for row in rows) / len(rows),
            "accuracy_mean": sum(row["accuracy"] for row in rows) / len(rows),
            "mean_surprise_mean": sum(row["mean_surprise"] for row in rows) / len(rows),
        }
    payload["window_holds"] = bool(
        payload["treated"]["quality_mean"] > payload["control"]["quality_mean"]
    )
    out = REPORT_DIR / "taiji_n2_dose_diagnostic_20261008.json"
    _write_report(out, payload)
    print(json.dumps(payload, ensure_ascii=True)[:520])
    print(f"report -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="N2 巩固通电驱动（PLAN-N2-01 §4ter）")
    parser.add_argument(
        "--phase",
        choices=("guard", "powerup", "postcheck", "align", "material", "dose"),
        required=True,
    )
    args = parser.parse_args(argv)
    if args.phase == "guard":
        return phase_guard()
    if args.phase == "postcheck":
        return phase_postcheck()
    if args.phase == "align":
        return phase_align()
    if args.phase == "material":
        return phase_material()
    if args.phase == "dose":
        return phase_dose()
    return phase_powerup()


if __name__ == "__main__":
    raise SystemExit(main())
