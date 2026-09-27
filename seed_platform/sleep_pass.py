"""The native sleep pass: the data ring's producer on the Seed-native line.

The legacy line produced three artifacts per sleep — an improvement report, a
training-data spec, and a corpus — from ``neuroplex.life.sleep_engine``.  The
native line produced none, so the native recursive loop had no way out of the
input rings.  This module is the native counterpart.

It deliberately **does not re-implement consolidation**: the organs already
exist and are tested (the intrinsic episodic field with its endogenous replay,
the content-addressed episodic store, the semantic and procedural learners, and
``seed.sleep.SeedSleepScheduler``, which is judge-driven).  What was missing was
the product-level pass around them, so a pass does four things:

1. **analyse** the native rings — the strategy/task/constraint records this
   runtime appends per turn, plus the memory journal the harness reports into —
   and state only what those records support;
2. **project** what deserves rehearsal into a corpus the native trainer can
   select by name: a constraint seed becomes one ``问：…\\n答：…`` record (the
   corpus's own dialogue shape, which is the only channel a constraint has), and
   a journal interaction replays as it was recorded;
3. **specify** the data ring's product — ``next_training_data_spec.json`` under
   an explicit readiness gate — naming the corpus file, so the spec has a
   consumer the trainer already accepts;
4. **optionally sleep the substrate** through ``SeedSleepScheduler``, which is
   the brain's own consolidation.  Off by default: a pass that changes weights is
   a training decision, not housekeeping.

Everything the pass cannot measure is reported as a note rather than guessed at.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from seed_platform import memory_store, turn_records
from seed_platform.evolution_adapters import (
    WORKBENCH_ANSWER_LEAD,
    WorkbenchCapabilityAdapter,
    workbench_training_record,
)
from seed_platform.paths import get_external_path

logger = logging.getLogger("SeedPlatform.SleepPass")

_LOCK = threading.Lock()
_RUNNING = False
_DIR_NAME = os.path.join("data", "consolidation")
_CORPUS_DIR = os.path.join("data", "consolidated")
_STATE_FILE = "state.json"
_SPEC_FILE = "next_training_data_spec.json"
_REPORT_FILE = "last_report.json"

# The question a constraint seed is projected under. The corpus holds dialogue,
# so a constraint has to arrive as an answer to something; this wording is the
# pass's own and is recorded in every manifest.
CONSTRAINT_QUESTION = "你有哪些必须遵守的行为约束？"

# Readiness gate. Legacy required two of four legacy metrics; the native line has
# no loss plateau to read (the native trainer never feeds sleep a metric), so the
# gate reads only what the native rings actually hold.
MIN_NEW_ENTRIES = 20
MIN_TASKS_FOR_FAILURE_RATE = 20
FAILURE_RATE_THRESHOLD = 0.4
MIN_TOOL_CALLS_FOR_RATE = 3
TOOL_SUCCESS_RATE_THRESHOLD = 0.5

_MAX_RECORDS = 20000
_MAX_PROJECTED = 5000
_MAX_TEXT_CHARS = 4000


def _consolidation_dir() -> str:
    """Return (and create) the directory the pass keeps its artifacts in.

    It sits inside the external ``data`` tree like the corpus dir: the default
    external root is the project root, so a state directory beside it would land
    in the tracked tree and trip the repository structure guard.
    """

    return get_external_path(_DIR_NAME)


def _corpus_dir() -> str:
    """Return (and create) the directory projected corpora live in.

    It sits inside the external ``data`` tree on purpose: the native trainer
    selects any ``.jsonl`` under that tree by name, so a produced corpus needs no
    registration step.
    """

    return get_external_path(_CORPUS_DIR)


def _digest(text: str) -> str:
    """Return the identity of one projectable text."""

    return sha256(str(text).encode("utf-8")).hexdigest()


def _load_state() -> dict[str, Any]:
    """Read the pass state, or start one."""

    path = os.path.join(_consolidation_dir(), _STATE_FILE)
    if not os.path.exists(path):
        return {"passes": 0, "last_pass_at": 0.0, "projected": [], "last_corpus": ""}
    try:
        with open(path, encoding="utf-8") as handle:
            state = json.load(handle)
    except Exception as e:
        logger.debug("【sleep_pass._load_state】读取失败（按空状态继续）: %s", e)
        return {"passes": 0, "last_pass_at": 0.0, "projected": [], "last_corpus": ""}
    if not isinstance(state, dict):
        return {"passes": 0, "last_pass_at": 0.0, "projected": [], "last_corpus": ""}
    projected = state.get("projected")
    state["projected"] = [str(item) for item in projected] if isinstance(projected, list) else []
    return state


def _save_state(state: dict[str, Any]) -> None:
    """Write the pass state, keeping the projected-digest list bounded."""

    state = dict(state)
    state["projected"] = [str(item) for item in state.get("projected") or []][-_MAX_PROJECTED:]
    path = os.path.join(_consolidation_dir(), _STATE_FILE)
    os.makedirs(_consolidation_dir(), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def _clip(value: Any) -> str:
    """Clip one projected text to the corpus budget."""

    return str(value or "")[:_MAX_TEXT_CHARS]


def analyze(state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return what the native rings hold, counted honestly.

    Public because the Life surface and the report both read it; it never writes.
    """

    state = state if state is not None else _load_state()
    projected = set(state.get("projected") or [])

    strategies = turn_records.strategies()
    tasks = turn_records.tasks()
    seeds = turn_records.constraints()
    journal = memory_store.status()

    by_type: dict[str, int] = {}
    tools: dict[str, dict[str, int]] = {}
    for record in strategies:
        kind = str(record.get("strategy_type") or "unknown")
        by_type[kind] = by_type.get(kind, 0) + 1
        if kind != "tool_choice":
            continue
        name = str(record.get("strategy_content") or "")
        bucket = tools.setdefault(name, {"calls": 0, "successes": 0})
        bucket["calls"] += 1
        if record.get("success"):
            bucket["successes"] += 1

    succeeded = sum(1 for record in tasks if record.get("success"))
    unprojected_seeds = [
        seed for seed in seeds if _digest(_clip(seed.get("text"))) not in projected
    ]
    entries = memory_store.entries(kind="interaction")
    unprojected_entries = [
        entry for entry in entries if _digest(_clip(entry.get("text"))) not in projected
    ]
    return {
        "strategies": {"total": len(strategies), "by_type": by_type, "tools": tools},
        "tasks": {
            "total": len(tasks),
            "succeeded": succeeded,
            "failed": len(tasks) - succeeded,
            "failure_rate": round((len(tasks) - succeeded) / len(tasks), 4) if tasks else 0.0,
        },
        "constraints": {
            "total": len(seeds),
            "unprojected": len(unprojected_seeds),
        },
        "memory": {
            "available": bool(journal.get("available")),
            "entries": int(journal.get("entries") or 0),
            "by_kind": journal.get("by_kind") or {},
            "interactions": len(entries),
            "unprojected_interactions": len(unprojected_entries),
            "sessions": int(journal.get("sessions") or 0),
            "last_recorded_at": float(journal.get("last_recorded_at") or 0.0),
        },
    }


def weaknesses_of(metrics: dict[str, Any]) -> list[str]:
    """Return the weaknesses these numbers support, and nothing else.

    Every line carries the count it rests on, so a reader can tell a measured
    weakness from a threshold that happened to be crossed.
    """

    out: list[str] = []
    tasks = metrics.get("tasks") or {}
    total = int(tasks.get("total") or 0)
    failed = int(tasks.get("failed") or 0)
    rate = float(tasks.get("failure_rate") or 0.0)
    if total >= MIN_TASKS_FOR_FAILURE_RATE and rate >= FAILURE_RATE_THRESHOLD:
        out.append(f"任务失败率 {rate:.0%}（{failed}/{total} 回合未产出可读回答）")
    for name, bucket in sorted((metrics.get("strategies") or {}).get("tools", {}).items()):
        calls = int(bucket.get("calls") or 0)
        successes = int(bucket.get("successes") or 0)
        if calls >= MIN_TOOL_CALLS_FOR_RATE and successes / calls < TOOL_SUCCESS_RATE_THRESHOLD:
            out.append(f"工具 {name} 成功率 {successes / calls:.0%}（{successes}/{calls} 次）")
    constraints = metrics.get("constraints") or {}
    if int(constraints.get("unprojected") or 0) > 0:
        out.append(f"{constraints['unprojected']} 条约束种子尚未进入语料")
    memory = metrics.get("memory") or {}
    if int(memory.get("interactions") or 0) == 0:
        out.append("记忆日志里没有 interaction 条目：没有可复盘的回合")
    return out


def notes_of(metrics: dict[str, Any]) -> list[str]:
    """Return what this pass cannot measure, stated instead of guessed."""

    tasks = metrics.get("tasks") or {}
    return [
        "本 pass 不报 loss 或平台期弱点：原生训练器不经睡眠喂指标，读不到这类证据。",
        "记忆条目与任务结果无法按回合相联：journal 记录 turn，而 chat 请求不带回合号，"
        "故投影只按「未中止」筛，不按成败筛（待 C2 式载荷补 turn 后可细化）。",
        f"任务记录共 {int(tasks.get('total') or 0)} 条（n≥{MIN_TASKS_FOR_FAILURE_RATE} 才报失败率）。",
    ]


def ready_reason(metrics: dict[str, Any]) -> str:
    """Return why the data ring is ready to specify a next training set, or ``""``.

    The gate is explicit and reads only native numbers: new constraints to
    internalise, a large enough body of new experience, or a failure rate above
    the threshold with enough tasks behind it.
    """

    constraints = metrics.get("constraints") or {}
    if int(constraints.get("unprojected") or 0) > 0:
        return f"{constraints['unprojected']} constraint seed(s) not yet in the corpus"
    memory = metrics.get("memory") or {}
    if int(memory.get("unprojected_interactions") or 0) >= MIN_NEW_ENTRIES:
        return f"{memory['unprojected_interactions']} unprojected interactions are waiting for rehearsal"
    tasks = metrics.get("tasks") or {}
    total = int(tasks.get("total") or 0)
    if (
        total >= MIN_TASKS_FOR_FAILURE_RATE
        and float(tasks.get("failure_rate") or 0.0) >= FAILURE_RATE_THRESHOLD
    ):
        return f"task failure rate {tasks['failure_rate']:.2f} over {total} tasks"
    return ""


def project(
    state: dict[str, Any],
    *,
    max_records: int,
    pass_id: str,
    workbench_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the rehearsal corpus this pass can justify, and its manifest.

    Constraints come first (they are the ones that cannot reach the model any
    other way), then the journal's interactions, then the workbench capability
    snapshot (C6 P1: one 问/答 record per declared capability, no-prose
    template).  Nothing already projected is projected again: re-presenting the
    same text every pass would grow the corpus without adding information, so
    the corpus is "what is new since last sleep".

    A pass with nothing new writes no corpus at all: an empty corpus is not a
    dataset, and writing one would both pollute the trainer's file list and
    truncate a corpus an earlier pass had produced.
    """

    projected = set(state.get("projected") or [])
    records: list[dict[str, str]] = []
    digests: list[str] = []

    seeds = [
        seed
        for seed in turn_records.constraints()
        if _digest(_clip(seed.get("text"))) not in projected
    ]
    for seed in reversed(seeds):  # newest first, so a capped corpus keeps the freshest constraints
        if len(records) >= max_records:
            break
        text = _clip(seed.get("text"))
        if not text.strip():
            continue
        records.append({"text": f"问：{CONSTRAINT_QUESTION}\n答：{text}"})
        digests.append(_digest(text))

    aborted = 0
    for entry in reversed(memory_store.entries(kind="interaction")):
        if len(records) >= max_records:
            break
        text = _clip(entry.get("text"))
        if not text.strip():
            continue
        digest = _digest(text)
        if digest in projected or digest in digests:
            continue
        if bool((entry.get("metadata") or {}).get("aborted")):
            aborted += 1
            digests.append(digest)
            continue
        records.append({"text": text})
        digests.append(digest)

    # C6 P1: the workbench capability snapshot joins the rehearsal corpus.  A
    # snapshot projects at most once (by snapshot_id) — the capabilities are the
    # runtime's own declaration, so re-projecting them every pass would only
    # repeat the model's environment section back to itself.
    workbench_capabilities = 0
    workbench_note = ""
    workbench_snapshot_id = str(workbench_snapshot.get("snapshot_id") or "").strip() if isinstance(
        workbench_snapshot, dict
    ) else ""
    if workbench_snapshot_id:
        if workbench_snapshot_id == state.get("workbench_snapshot_id"):
            workbench_note = f"snapshot {workbench_snapshot_id} already projected"
        else:
            try:
                capability_projection = WorkbenchCapabilityAdapter().project(workbench_snapshot)
            except (TypeError, ValueError) as exc:
                workbench_note = f"workbench snapshot rejected: {exc}"
            else:
                for unit in capability_projection.corpus:
                    if len(records) >= max_records:
                        break
                    record = workbench_training_record(dict(unit.content))
                    digest = _digest(record["text"])
                    if digest in projected or digest in digests:
                        # Already written by an earlier capped pass.
                        continue
                    records.append(record)
                    digests.append(digest)
                    workbench_capabilities += 1
                if len(records) < max_records:
                    # Everything the snapshot declares is now on record (written
                    # here or by an earlier capped pass): roll the id forward so
                    # future passes skip re-projection entirely.
                    state["workbench_snapshot_id"] = workbench_snapshot_id
                workbench_note = (
                    f"projected {workbench_capabilities}/"
                    f"{len(capability_projection.corpus)} capabilities from "
                    f"snapshot {workbench_snapshot_id}"
                )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    corpus_name = f"corpus-{stamp}-{pass_id}.jsonl"
    relative = os.path.join("consolidated", corpus_name).replace("\\", "/")
    by_source = {
        "constraints": sum(1 for record in records if CONSTRAINT_QUESTION in record["text"]),
        "interactions": sum(1 for record in records if CONSTRAINT_QUESTION not in record["text"] and WORKBENCH_ANSWER_LEAD not in record["text"]),
        "workbench_capabilities": sum(1 for record in records if WORKBENCH_ANSWER_LEAD in record["text"]),
    }
    if not records:
        # Nothing new: roll the skips forward, and leave every earlier corpus alone.
        state["projected"] = [*(state.get("projected") or []), *digests]
        return {
            "corpus": "",
            "corpus_path": "",
            "manifest": "",
            "records": 0,
            "by_source": by_source,
            "skipped": {"aborted_interactions": aborted},
            "workbench_note": workbench_note,
            "digests": digests,
        }
    os.makedirs(_corpus_dir(), exist_ok=True)
    corpus_path = os.path.join(_corpus_dir(), corpus_name)
    with open(corpus_path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    manifest = {
        "corpus": relative,
        "generated_at": time.time(),
        "records": len(records),
        "by_source": by_source,
        "skipped": {"aborted_interactions": aborted, "already_projected": len(projected)},
        "constraint_question": CONSTRAINT_QUESTION,
        "workbench_snapshot_id": workbench_snapshot_id,
        "workbench_note": workbench_note,
        "policy": (
            "constraints are projected as one 问/答 record each; interactions replay verbatim; "
            "workbench capabilities project once per snapshot via the no-prose template; "
            "already-projected text and aborted interactions are skipped"
        ),
    }
    manifest_path = os.path.join(_consolidation_dir(), f"manifest-{stamp}-{pass_id}.json")
    os.makedirs(_consolidation_dir(), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    # Roll the projected digests forward only for what this pass wrote or chose
    # to skip, and keep the newest digests last so the bound drops the oldest.
    state["projected"] = [*(state.get("projected") or []), *digests]
    state["last_corpus"] = relative
    return {
        "corpus": relative,
        "corpus_path": corpus_path,
        "manifest": manifest_path,
        "records": len(records),
        "by_source": manifest["by_source"],
        "skipped": {"aborted_interactions": aborted},
        "workbench_note": workbench_note,
        "digests": digests,
    }


def _spec_payload(
    metrics: dict[str, Any], projection: dict[str, Any], reason: str
) -> dict[str, Any]:
    """Compose the data ring's product: the legacy keys, filled with native facts."""

    datasets = [projection["corpus"]] if projection.get("records") else []
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "reason": reason,
        "metrics": metrics,
        "weaknesses": weaknesses_of(metrics),
        "training_recommendations": [],
        "datasets": datasets,
        "native": True,
        "generator": "seed_platform.sleep_pass",
    }


def _add_recommendations(spec: dict[str, Any], projection: dict[str, Any]) -> None:
    """Fill the recommendation list from what this pass actually produced."""

    recommendations: list[dict[str, Any]] = []
    by_source = projection.get("by_source") or {}
    datasets = spec.get("datasets") or []
    if by_source.get("constraints"):
        recommendations.append(
            {
                "kind": "internalise_constraints",
                "datasets": datasets,
                "rationale": f"{by_source['constraints']} constraint seed(s) projected as 问/答 records",
            }
        )
    if by_source.get("interactions"):
        recommendations.append(
            {
                "kind": "rehearse_interactions",
                "datasets": datasets,
                "rationale": f"{by_source['interactions']} recorded interaction(s) re-presented verbatim",
            }
        )
    spec["training_recommendations"] = recommendations


def sleep_organs(
    runtime: Any,
    *,
    texts: list[str],
    cycles_per_text: int,
    learn: bool,
    max_symbols: int,
    max_texts: int,
) -> dict[str, Any]:
    """Hand the night's texts to the substrate's own sleep scheduler.

    The scheduler is the brain's consolidation (judge-driven selection, then
    endogenous replay); this function only supplies the texts and records what it
    did.  With ``learn=False`` no weight changes, which is why the pass defaults
    to not reaching here at all.
    """

    if runtime is None:
        return {"ran": False, "reason": "no runtime attached"}
    seed = getattr(runtime, "model", None)
    if seed is None:
        return {"ran": False, "reason": "no native model attached"}
    required = (
        "substrate",
        "consolidate",
        "observe",
        "act",
        "settle_action",
        "snapshot",
        "reset_dynamics",
    )
    missing = [name for name in required if not hasattr(seed, name)]
    if missing:
        return {
            "ran": False,
            "reason": f"attached model is not a seed.Seed (missing {', '.join(missing)})",
        }
    payload = [text for text in texts if text.strip()]
    if not payload:
        return {"ran": False, "reason": "no texts to sleep on"}
    try:
        from seed.judge import SeedJudge
        from seed.sleep import SeedSleepScheduler

        scheduler = SeedSleepScheduler(seed, SeedJudge(seed))
        selected = scheduler.select_for_sleep(
            [text.encode("utf-8") for text in payload], k=int(max_texts)
        )
        stats = scheduler.night(
            selected,
            cycles_per_text=int(cycles_per_text),
            learn=bool(learn),
            max_symbols=int(max_symbols),
        )
    except Exception as e:  # pragma: no cover - depends on a live substrate
        logger.warning("【sleep_pass.sleep_organs】睡眠期巩固失败（非致命）: %s", e)
        return {"ran": False, "reason": f"{type(e).__name__}: {e}"}
    return {
        "ran": True,
        "learn": bool(learn),
        "cycles_per_text": int(cycles_per_text),
        "max_symbols": int(max_symbols),
        "texts": len(selected),
        "stats": stats,
    }


def run(
    *,
    reason: str = "manual",
    runtime: Any = None,
    organs: bool = False,
    learn: bool = False,
    cycles_per_text: int = 1,
    max_symbols: int = 64,
    max_texts: int = 8,
    max_records: int = 200,
) -> dict[str, Any]:
    """Run one pass and return its report; write the report, spec, and corpus.

    ``organs`` decides whether the substrate sleeps at all, and ``learn`` decides
    whether that sleep may change weights.  Both default off: housekeeping never
    trains a brain by accident.
    """

    global _RUNNING
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("a sleep pass is already running")
    try:
        _RUNNING = True
        started = time.time()
        pass_id = uuid.uuid4().hex[:12]
        state = _load_state()
        metrics = analyze(state)
        # C6 P1: the workbench capability snapshot rides the pass when the
        # runtime can hand it over; anything the pass cannot measure is a note,
        # never a guess.
        workbench_snapshot = None
        workbench_fetch_note = ""
        try:
            workbench_snapshot = runtime.workbench_environment.capability_snapshot.to_payload()
        except Exception as exc:  # noqa: BLE001 - reported, never guessed
            workbench_fetch_note = f"workbench snapshot unavailable: {exc}"
        projection = project(
            state,
            max_records=max(1, int(max_records)),
            pass_id=pass_id,
            workbench_snapshot=workbench_snapshot,
        )
        reason_ready = ready_reason(metrics)
        spec = _spec_payload(metrics, projection, reason_ready or f"not ready: {reason}")
        _add_recommendations(spec, projection)
        spec_path = os.path.join(_consolidation_dir(), _SPEC_FILE)
        spec_written = bool(reason_ready) and bool(spec.get("datasets"))
        if spec_written:
            os.makedirs(_consolidation_dir(), exist_ok=True)
            with open(spec_path, "w", encoding="utf-8") as handle:
                json.dump(spec, handle, ensure_ascii=False, indent=2)

        night_texts: list[str] = []
        if organs:
            night_texts = [str(record["text"]) for record in _projected_records(projection)]
        organs_report = (
            sleep_organs(
                runtime,
                texts=night_texts,
                cycles_per_text=cycles_per_text,
                learn=learn,
                max_symbols=max_symbols,
                max_texts=max_texts,
            )
            if organs
            else {"ran": False, "reason": "organs not requested"}
        )

        state["passes"] = int(state.get("passes") or 0) + 1
        state["last_pass_at"] = time.time()
        _save_state(state)

        report = {
            "pass_id": pass_id,
            "reason": reason,
            "started_at": started,
            "finished_at": time.time(),
            "duration_ms": round((time.time() - started) * 1000, 1),
            "analyze": metrics,
            "weaknesses": weaknesses_of(metrics),
            "notes": notes_of(metrics),
            "projection": {
                "corpus": projection["corpus"],
                "records": projection["records"],
                "by_source": projection["by_source"],
                "skipped": projection["skipped"],
                "manifest": projection["manifest"],
                "workbench_note": projection.get("workbench_note") or workbench_fetch_note,
            },
            "spec": {
                "written": spec_written,
                "path": spec_path,
                "reason": reason_ready or f"not ready: {reason}",
                "datasets": spec.get("datasets") or [],
            },
            "organs": organs_report,
            "passes": state["passes"],
        }
        with open(
            os.path.join(_consolidation_dir(), _REPORT_FILE), "w", encoding="utf-8"
        ) as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)
        return report
    finally:
        _RUNNING = False
        _LOCK.release()


def _projected_records(projection: dict[str, Any]) -> list[dict[str, str]]:
    """Read back the records this pass just wrote, for the optional organ sleep."""

    path = str(projection.get("corpus_path") or "")
    if not path or not os.path.exists(path):
        return []
    texts: list[dict[str, str]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                texts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return texts


def status() -> dict[str, Any]:
    """Return what the last pass did, what the data ring holds, and what the journal holds.

    The memory journal is the pass's own input, so it rides here rather than
    making the panel poll two endpoints; both counts are read live from disk.
    """

    state = _load_state()
    spec_path = os.path.join(_consolidation_dir(), _SPEC_FILE)
    spec: dict[str, Any] | None = None
    if os.path.exists(spec_path):
        try:
            with open(spec_path, encoding="utf-8") as handle:
                spec = json.load(handle)
        except Exception as e:
            logger.debug("【sleep_pass.status】读取 spec 失败: %s", e)
    report: dict[str, Any] | None = None
    report_path = os.path.join(_consolidation_dir(), _REPORT_FILE)
    if os.path.exists(report_path):
        try:
            with open(report_path, encoding="utf-8") as handle:
                report = json.load(handle)
        except Exception as e:
            logger.debug("【sleep_pass.status】读取报告失败: %s", e)
    return {
        "available": True,
        "directory": _consolidation_dir(),
        "corpus_directory": _corpus_dir(),
        "passes": int(state.get("passes") or 0),
        "last_pass_at": float(state.get("last_pass_at") or 0.0),
        "last_corpus": str(state.get("last_corpus") or ""),
        "projected_digests": len(state.get("projected") or []),
        "running": _RUNNING,
        "spec": spec,
        "last_report": report,
        "journal": memory_store.status(),
    }
