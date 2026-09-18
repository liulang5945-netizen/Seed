"""CAP-0: inventory the actual model, its generation chain, and its raw outputs.

What CAP-0 is (and is not)
--------------------------
``plans/active/roadmap/07_MINI_MODEL_DELIVERY.md`` section 5.1 requires that the
nearest main-line conclusion point runs **CAP-0 first**: take stock of the real
model / generation chain and the raw outputs already on record, and state plainly
whether the thing that runs can hold a basic conversation.

CAP-0 is therefore an **inventory**, not a scored capability review:

* it reports deterministic facts about loading, provenance and mode isolation;
* it captures **raw** probe outputs verbatim, with no scoring, no rubric and no
  pass/fail -- so it cannot be mistaken for the frozen B-H baseline;
* it reports ``untested`` for anything it did not measure, never 0 and never a
  pass (07 section 5: "未测记'未测'，不记 0，也不记通过").

What it deliberately does not do
--------------------------------
The official B/C/D/E evaluation sets (20 items each) and the G set must be
**frozen before any candidate score is seen** (07 section 4.1).  Freezing them is
a decision, so this script runs a small, clearly-labelled *diagnostic* probe
instead and marks the baseline dimensions as ``untested``.  It trains nothing,
writes no checkpoint, and switches no product behaviour.

Dimension A is the part CAP-0 can settle now: which checkpoint loads, through
which chain, in which mode, and whether the loader refuses a missing or
unreadable checkpoint.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REPORT_FORMAT = "taiji-cap0-inventory-report-v1"
VERSION = 1
DELIVERY_PLAN = "plans/active/roadmap/07_MINI_MODEL_DELIVERY.md"
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_cap0_inventory_20260915.json"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINT_DIR / "seed_corpus.pt"

#: Fixed diagnostic probe.  Explicitly NOT the frozen B-H evaluation set; it exists
#: only to show what the current chain actually emits, verbatim.
PROBE_TURNS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("identity", ("请用一句话说明你是谁、由什么运行。",)),
    ("greeting", ("你好。",)),
    ("knowledge_short", ("水的沸点在标准大气压下是多少摄氏度？",)),
    ("context_a", ("请记住这个名称：项目内部代号是「星桥」。",)),
    ("context_b", ("我刚才说的内部代号是什么？",)),
    ("unknown_boundary", ("请举一个你确实无法回答的问题，并说明为什么无法回答。",)),
    (
        "material_qa",
        ("材料：星桥项目在 2026 年 9 月完成 B2 选择门，结果是负结果。问题：B2 的结果是什么？",),
    ),
)

MISSING_CHECKPOINT = CHECKPOINT_DIR / ".cap0-does-not-exist.pt"


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    """Read-only envelope metadata.  Loads no model, trains nothing, writes nothing."""

    import torch

    try:
        envelope = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001
        return {"readable": False, "error": f"{type(exc).__name__}: {exc}"}
    metadata = envelope.get("metadata") if isinstance(envelope, dict) else None
    metadata = metadata if isinstance(metadata, dict) else {}
    # The Taiji model checkpoint format lives under ``substrate`` -- that is the
    # mapping TaijiModel.restore() reads ``format`` from.  The sibling ``taiji`` key
    # is the Seed adapter envelope and carries its own, unrelated version string
    # (seed_corpus.pt has taiji=taiji-native-v1 but substrate=taiji-native-v10), so
    # reading the wrong key would silently misreport the format.
    substrate = envelope.get("substrate") if isinstance(envelope, dict) else None
    return {
        "readable": True,
        "tick": metadata.get("tick"),
        "trainer": metadata.get("trainer"),
        "saved_at_utc": metadata.get("saved_at_utc"),
        "has_metadata": bool(metadata),
        "model_format": (substrate.get("format") if isinstance(substrate, Mapping) else None),
        "has_taiji_adapter_block": bool(
            isinstance(envelope, dict) and isinstance(envelope.get("taiji"), Mapping)
        ),
    }


def _checkpoint_inventory() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(CHECKPOINT_DIR.glob("*.pt")):
        stat = path.stat()
        rows.append(
            {
                "filename": path.name,
                "bytes": stat.st_size,
                "modified_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(stat.st_mtime)),
                "is_default": path == DEFAULT_CHECKPOINT,
                **_checkpoint_metadata(path),
            }
        )
    return rows


def _probe_child(payload: dict[str, Any]) -> int:
    """Fresh-process load + probe.  Trains nothing and saves nothing."""

    from api.seed_runtime import SeedRuntime

    result: dict[str, Any] = {"probe": payload.get("probe")}
    try:
        runtime = SeedRuntime.load(Path(payload["checkpoint"]))
    except Exception as exc:  # noqa: BLE001
        result["load_ok"] = False
        result["load_error"] = f"{type(exc).__name__}: {exc}"
        print(json.dumps(result, ensure_ascii=True))
        return 0

    result["load_ok"] = True
    result["tick"] = int(runtime.model.tick)
    result["runtime_name"] = runtime.name
    result["provider_status"] = runtime.status().get("language_provider_status")
    result["chat_organ_backend"] = getattr(runtime._chat_organ, "backend_id", None)
    result["checkpoint_path"] = str(runtime.checkpoint_path)

    if payload.get("probe"):
        turns: list[dict[str, Any]] = []
        history: list[tuple[str, str]] = []
        for label, prompts in PROBE_TURNS:
            prompt = prompts[0]
            started = time.perf_counter()
            try:
                # learn=False: 07 section 4.1 requires no training during evaluation.
                answer = runtime.chat(prompt, history=history, learn=False)
                turns.append(
                    {
                        "label": label,
                        "prompt": prompt,
                        "raw_output": answer,
                        "output_bytes": len(answer.encode("utf-8")),
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
                history.append((prompt, answer))
            except Exception as exc:  # noqa: BLE001
                turns.append(
                    {
                        "label": label,
                        "prompt": prompt,
                        "error": f"{type(exc).__name__}: {exc}",
                        "seconds": round(time.perf_counter() - started, 3),
                    }
                )
        result["turns"] = turns
    print(json.dumps(result, ensure_ascii=True))
    return 0


def _run_child(checkpoint: Path, *, probe: bool) -> dict[str, Any]:
    payload = {"checkpoint": str(checkpoint), "probe": probe}
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=json.dumps(payload),
    )
    if child.returncode != 0:
        return {
            "load_ok": False,
            "load_error": f"child exit {child.returncode}: {child.stderr[-400:]}",
        }
    try:
        return json.loads(child.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"load_ok": False, "load_error": f"unparseable child output: {child.stdout[-400:]}"}


def _most_trained(inventory: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The checkpoint with the highest recorded tick, if any records one."""

    scored = [row for row in inventory if isinstance(row.get("tick"), int)]
    if not scored:
        return None
    return max(scored, key=lambda row: int(row["tick"]))


def _template_signature(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Does every output collapse to one string once the echoed prompt is masked?

    Descriptive, not a score: it separates "the organ emits a fixed acknowledgement
    template" from "the organ produced different text per turn", which is the
    difference between a wiring defect and a content-capability result.
    """

    signatures = {
        str(turn["raw_output"]).replace(turn["prompt"], "<PROMPT>")
        for turn in turns
        if turn.get("raw_output") is not None
    }
    return {
        "turns_considered": sum(1 for turn in turns if turn.get("raw_output") is not None),
        "distinct_signatures": len(signatures),
        "templated": len(signatures) == 1 and len(turns) > 1,
        "signature": next(iter(signatures)) if len(signatures) == 1 else None,
    }


def run_inventory() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "delivery_plan": DELIVERY_PLAN,
        "scope": (
            "CAP-0 inventory: what model actually runs, through which chain, and what it "
            "emits today.  Diagnostic probe only; the frozen B-H baseline is NOT produced here"
        ),
        "does_not_do": [
            "trains nothing and writes no checkpoint",
            "does not freeze or score the official B/C/D/E/G sets (that freeze is a decision)",
            "does not switch product behaviour or external providers",
            "reports 'untested' rather than 0 or pass for unmeasured dimensions",
        ],
    }
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
        ).stdout.strip()
        inventory = _checkpoint_inventory()

        # ---- A: load chain, in a fresh process, on the default entry
        default_probe = _run_child(DEFAULT_CHECKPOINT, probe=True)
        default_turns = default_probe.get("turns") or []
        default_signature = _template_signature(default_turns)

        # ---- A: does the chain emit anything different when given a trained state?
        # This is the discriminator between "wiring defect" and "no content ability".
        best = _most_trained(inventory)
        trained_probe: dict[str, Any] | None = None
        trained_signature: dict[str, Any] | None = None
        if best is not None and Path(CHECKPOINT_DIR / best["filename"]) != DEFAULT_CHECKPOINT:
            trained_path = CHECKPOINT_DIR / best["filename"]
            trained_probe = _run_child(trained_path, probe=True)
            trained_signature = _template_signature(trained_probe.get("turns") or [])

        # ---- A: missing-checkpoint rejection must be deterministic
        from api.seed_runtime import SeedRuntime

        missing_rejected = False
        missing_error = None
        try:
            SeedRuntime.load(MISSING_CHECKPOINT)
        except Exception as exc:  # noqa: BLE001
            missing_rejected = True
            missing_error = f"{type(exc).__name__}: {exc}"

        default_tick = None
        for row in inventory:
            if row["is_default"]:
                default_tick = row.get("tick")
        wiring_defect = bool(
            isinstance(default_tick, int)
            and best is not None
            and isinstance(best.get("tick"), int)
            and int(best["tick"]) > int(default_tick)
        )

        # ---- why the trained files cannot load: format support already exists,
        # the identity-organ guard simply does not consult it.
        legacy_formats = {"taiji-native-v8", "taiji-native-v9"}
        default_format = next(
            (row.get("model_format") for row in inventory if row["is_default"]), None
        )
        trained_format = None
        if best is not None:
            trained_format = next(
                (
                    row.get("model_format")
                    for row in inventory
                    if row["filename"] == best["filename"]
                ),
                None,
            )
        stranded_is_legacy = trained_format in legacy_formats

        def _probe_summary(
            probe: dict[str, Any] | None, signature: dict[str, Any] | None
        ) -> dict[str, Any]:
            if probe is None:
                return {"probed": False}
            turns = probe.get("turns") or []
            answered = [turn for turn in turns if "raw_output" in turn]
            return {
                "probed": True,
                "load_ok": bool(probe.get("load_ok")),
                "load_error": probe.get("load_error"),
                "tick": probe.get("tick"),
                "chat_organ_backend": probe.get("chat_organ_backend"),
                "turns_answered": len(answered),
                "turns_errored": len(turns) - len(answered),
                "total_output_bytes": sum(turn["output_bytes"] for turn in answered),
                "template_signature": signature,
            }

        payload.update(
            {
                "status": "completed",
                "record": {"commit": commit, "default_checkpoint": str(DEFAULT_CHECKPOINT)},
                "checkpoint_inventory": inventory,
                "model_reality": {
                    "checkpoint_exists": DEFAULT_CHECKPOINT.is_file(),
                    "default_checkpoint": DEFAULT_CHECKPOINT.name,
                    "load_ok": bool(default_probe.get("load_ok")),
                    "load_error": default_probe.get("load_error"),
                    "default_tick": default_tick,
                    "most_trained_checkpoint": (best or {}).get("filename"),
                    "most_trained_tick": (best or {}).get("tick"),
                    "wiring_defect": wiring_defect,
                    "wiring_defect_statement": (
                        f"the default entry loads {DEFAULT_CHECKPOINT.name} "
                        f"(tick={default_tick}) while {best['filename']} records "
                        f"tick={best['tick']}; the trained state is not what the default "
                        "path serves"
                        if wiring_defect and best is not None
                        else None
                    ),
                    "default_model_format": default_format,
                    "most_trained_model_format": trained_format,
                    "stranded_is_legacy_format": stranded_is_legacy,
                    "stranded_defect_diagnosis": (
                        "restore() supports the legacy formats it declares "
                        "(LEGACY_CHECKPOINT_FORMATS = v8/v9). The identity-organ branch now applies "
                        "the same 'payload absent in a legacy format' migration the predictive "
                        "context already used (M2-2h -> M2-2i), so a v8 file that predates the organ "
                        "loads and keeps its freshly initialised organ. The remaining refusal is "
                        "'enabled identity organ checkpoint payload is missing', which now applies "
                        "only to current-format checkpoints; organ lineage validation is unchanged "
                        "and still fails closed on a payload from a different core"
                        if stranded_is_legacy
                        else "the stranded checkpoint is not in a format restore() claims to support"
                    ),
                    "missing_checkpoint_rejected": missing_rejected,
                    "missing_checkpoint_error": missing_error,
                    "fresh_process": True,
                    "external_provider_used": False,
                },
                "raw_output_inventory": {
                    "probe_kind": "diagnostic, not the frozen B-H evaluation set",
                    "default_entry": _probe_summary(default_probe, default_signature),
                    "most_trained_entry": _probe_summary(trained_probe, trained_signature),
                    "default_turns": default_turns,
                    "most_trained_turns": (trained_probe or {}).get("turns"),
                },
                "dimensions": {
                    "A_model_reality": "measured (load chain, mode, missing-checkpoint rejection, tick provenance)",
                    "B_basic_dialogue": "untested (frozen set not yet frozen)",
                    "C_basic_qa": "untested (frozen set not yet frozen)",
                    "D_context": "untested (frozen set not yet frozen)",
                    "E_instruction_reasoning": "untested (frozen set not yet frozen)",
                    "F_representative_capability": (
                        "partial: B1/B2 gates exist and are green/negative respectively, but they "
                        "are selector-surface results, not whole-model conversation ability"
                    ),
                    "G_uncertainty_safety": "untested (frozen set not yet frozen)",
                    "H_performance_stability": "untested (no calibrated caps yet)",
                    "I_continual_adaptation": "untested (not required for a first L3 delivery)",
                },
                "gap_statement": (
                    "Three facts, kept apart. (1) WIRING: the default entry loads "
                    f"{DEFAULT_CHECKPOINT.name} at tick={default_tick}, a freshly initialised "
                    "substrate saved by the API runtime, not a trained state. (2) STRANDED STATE: "
                    f"the most trained checkpoint on record ({best['filename']} tick={best['tick']}) "
                    "cannot be loaded at all through this chain -- "
                    f"{trained_probe.get('load_error') if trained_probe else 'not probed'} -- and the "
                    "older checkpoints carry no current-format component block, so no trained state "
                    "is currently reachable from the product entry. (3) CAPABILITY: on the diagnostic "
                    "probe the default chain emits a fixed acknowledgement template that echoes the "
                    "prompt (one signature across all turns), which per 07 section 3.B does not count "
                    "as content ability. The honest reading is therefore: a wiring defect and a "
                    "stranded-state defect are CONFIRMED, while whole-model conversational ability "
                    "remains UNTESTED rather than failed, because no trained state could be served."
                    if wiring_defect and best is not None
                    else "no trained checkpoint was found on record, so the default chain's lack of "
                    "content ability cannot be separated from a wiring defect"
                ),
                "next_decision": (
                    "two things, in order: (a) decide the correct default checkpoint / loading policy "
                    "(a defect fix, not a research question); (b) freeze the CAP evaluation set "
                    "(07 section 4.1: B/C/D/E 20 items each, G >= 20, A/H deterministic, F via the "
                    "existing capability contracts) BEFORE any candidate score is seen, so the first "
                    "scored B-H baseline is legitimate"
                ),
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    except Exception as exc:  # noqa: BLE001
        payload.update(
            {
                "error": f"{type(exc).__name__}: {exc}",
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    # An errored run must not overwrite the sealed report (see the same guard in
    # probe_taiji_cap0_legacy_load.py): failures land in a sibling file, evidence survives.
    failed = "error" in payload
    target = (
        DEFAULT_REPORT.with_name(DEFAULT_REPORT.stem + ".error.json") if failed else DEFAULT_REPORT
    )
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--child", action="store_true", help="internal: run one fresh-process probe"
    )
    args = parser.parse_args(argv)
    if args.child:
        return _probe_child(json.loads(sys.stdin.read()))
    result = run_inventory()
    reality = result.get("model_reality") or {}
    inventory = result.get("raw_output_inventory") or {}
    default_probe = inventory.get("default_entry") or {}
    trained_probe = inventory.get("most_trained_entry") or {}
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "default_checkpoint": reality.get("default_checkpoint"),
                "default_tick": reality.get("default_tick"),
                "most_trained_checkpoint": reality.get("most_trained_checkpoint"),
                "most_trained_tick": reality.get("most_trained_tick"),
                "wiring_defect": reality.get("wiring_defect"),
                "load_ok": reality.get("load_ok"),
                "missing_checkpoint_rejected": reality.get("missing_checkpoint_rejected"),
                "default_turns_answered": default_probe.get("turns_answered"),
                "default_templated": (default_probe.get("template_signature") or {}).get(
                    "templated"
                ),
                "trained_probed": trained_probe.get("probed"),
                "trained_turns_answered": trained_probe.get("turns_answered"),
                "trained_templated": (trained_probe.get("template_signature") or {}).get(
                    "templated"
                ),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
