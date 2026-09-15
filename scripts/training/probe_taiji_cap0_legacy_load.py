"""CAP-0 counterfactual: if the legacy guard were applied, would the trained
checkpoint load -- and what would it emit?

Why this probe exists
---------------------
CAP-0 measured two separate defects (report ``taiji_cap0_inventory_20260915.json``):
the default entry serves an untrained ``tick=2`` substrate, and the trained
``seed_beta.pt`` (tick 16,000,000) cannot load at all because the identity-organ
branch in ``taiji/model.py`` refuses *any* checkpoint lacking that payload -- including
``taiji-native-v8`` files that predate the organ, even though ``restore()`` explicitly
supports that legacy format elsewhere.

Deciding how to fix it needs one fact that is not yet on record: **does relaxing that
one guard actually recover the trained state, and does that state produce readable
text?**  This probe answers exactly that, and nothing else.

How it stays a measurement, not a change
----------------------------------------
The repository source is never edited.  In a short-lived subprocess the probe wraps
``Taiji.restore`` in memory so that a legacy checkpoint with no organ payload runs the
tolerant branch (the organ is attached fresh afterwards), which is precisely the
semantics the proposed migration would have.  The wrapper is process-local, nothing is
written to any checkpoint, and no training happens.

The probe is deliberately explicit about its own limits: a successful load here does
not mean the fix is safe, and readable output here would not by itself justify it --
that is a refuse-vs-migrate policy decision, not a code question.
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

REPORT_FORMAT = "taiji-cap0-legacy-load-probe-report-v1"
VERSION = 1
DEFAULT_REPORT = PROJECT_ROOT / "reports" / "taiji_cap0_legacy_load_probe_20260915.json"
TARGET_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_beta.pt"
CONTROL_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "seed_corpus.pt"

PROBE_PROMPTS: tuple[str, ...] = (
    "请用一句话说明你是谁、由什么运行。",
    "你好。",
    "水的沸点在标准大气压下是多少摄氏度？",
)


def _install_legacy_guard() -> dict[str, Any]:
    """Process-local simulation of the proposed legacy guard.  Source untouched."""

    from taiji.model import Taiji

    original = Taiji.restore
    legacy_formats = frozenset(Taiji.LEGACY_CHECKPOINT_FORMATS)
    applied: list[str] = []

    def patched(self: Any, checkpoint: Mapping[str, Any]) -> None:
        is_legacy = str(checkpoint.get("format", "")) in legacy_formats
        payload = checkpoint.get("identity_organ")
        if is_legacy and not isinstance(payload, Mapping) and self.identity_organ is not None:
            # The proposed migration: a pre-organ format simply has no organ state,
            # so run the tolerant branch and attach the fresh organ afterwards.
            fresh = self.identity_organ
            self.identity_organ = None
            try:
                original(self, checkpoint)
            finally:
                self.identity_organ = fresh
            applied.append(str(checkpoint.get("format", "")))
            return
        original(self, checkpoint)

    Taiji.restore = patched
    return {
        "patched": True,
        "legacy_formats": sorted(legacy_formats),
        "applied_marker": applied,
    }


def _run_probe(checkpoint: Path, *, apply_guard: bool) -> dict[str, Any]:
    """One fresh-process load (+ optional chat probe) under a chosen guard policy."""

    started = time.perf_counter()
    result: dict[str, Any] = {
        "checkpoint": checkpoint.name,
        "guard_relaxed": apply_guard,
        "source_edited": False,
    }
    guard_info: dict[str, Any] = {}
    if apply_guard:
        guard_info = _install_legacy_guard()
        result["legacy_guard"] = guard_info

    from api.seed_runtime import SeedRuntime

    try:
        runtime = SeedRuntime.load(checkpoint)
    except Exception as exc:  # noqa: BLE001
        result.update(
            {
                "load_ok": False,
                "load_error": f"{type(exc).__name__}: {exc}",
                "seconds": round(time.perf_counter() - started, 3),
            }
        )
        return result

    result.update(
        {
            "load_ok": True,
            "tick": int(runtime.model.tick),
            "chat_organ_backend": getattr(runtime._chat_organ, "backend_id", None),
        }
    )
    if apply_guard:
        # Confirm the wrapper actually took the tolerant branch for this file.
        result["legacy_guard_applied"] = bool(guard_info.get("applied_marker"))

    turns: list[dict[str, Any]] = []
    for prompt in PROBE_PROMPTS:
        try:
            # learn=False: 07 section 4.1 requires no training during evaluation.
            answer = runtime.chat(prompt, learn=False)
            turns.append(
                {
                    "prompt": prompt,
                    "raw_output": answer,
                    "output_bytes": len(answer.encode("utf-8")),
                    "contains_replacement_char": "\ufffd" in answer,
                }
            )
        except Exception as exc:  # noqa: BLE001
            turns.append({"prompt": prompt, "error": f"{type(exc).__name__}: {exc}"})
    result["turns"] = turns
    answered = [turn for turn in turns if "raw_output" in turn]
    signatures = {str(turn["raw_output"]).replace(turn["prompt"], "<PROMPT>") for turn in answered}
    result["output_summary"] = {
        "turns_answered": len(answered),
        "distinct_signatures": len(signatures),
        "templated": len(signatures) == 1 and len(answered) > 1,
        "signature": next(iter(signatures)) if len(signatures) == 1 else None,
    }
    result["seconds"] = round(time.perf_counter() - started, 3)
    return result


def _child(payload: dict[str, Any]) -> int:
    outcome = _run_probe(Path(payload["checkpoint"]), apply_guard=bool(payload["apply_guard"]))
    print(json.dumps(outcome, ensure_ascii=False))
    return 0


def _spawn(checkpoint: Path, *, apply_guard: bool) -> dict[str, Any]:
    child = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        input=json.dumps({"checkpoint": str(checkpoint), "apply_guard": apply_guard}),
    )
    if child.returncode != 0:
        return {
            "checkpoint": checkpoint.name,
            "guard_relaxed": apply_guard,
            "load_ok": False,
            "load_error": f"child exit {child.returncode}: {child.stderr[-500:]}",
        }
    try:
        return json.loads(child.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {
            "checkpoint": checkpoint.name,
            "guard_relaxed": apply_guard,
            "load_ok": False,
            "load_error": f"unparseable child output: {child.stdout[-500:]}",
        }


def run_probe() -> dict[str, Any]:
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "format": REPORT_FORMAT,
        "version": VERSION,
        "status": "failed",
        "scope": (
            "counterfactual measurement of the proposed legacy guard: does the trained "
            "checkpoint load, and what does it emit?  No source file is edited"
        ),
        "does_not_do": [
            "edits no repository source; the guard is wrapped in a short-lived process",
            "writes no checkpoint and trains nothing",
            "does not decide the refuse-vs-migrate policy question",
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
        arms = {
            "trained_current_guard": _spawn(TARGET_CHECKPOINT, apply_guard=False),
            "trained_relaxed_guard": _spawn(TARGET_CHECKPOINT, apply_guard=True),
            "default_control_current_guard": _spawn(CONTROL_CHECKPOINT, apply_guard=False),
        }
        trained_relaxed = arms["trained_relaxed_guard"]
        recovered = bool(trained_relaxed.get("load_ok"))
        summary = trained_relaxed.get("output_summary") or {}
        readable_recovered = bool(
            recovered and summary.get("turns_answered") and not summary.get("templated")
        )
        payload.update(
            {
                "status": "completed",
                "record": {
                    "commit": commit,
                    "target_checkpoint": TARGET_CHECKPOINT.name,
                    "control_checkpoint": CONTROL_CHECKPOINT.name,
                },
                "arms": arms,
                "verdict": {
                    "trained_state_recovered_by_relaxing_guard": recovered,
                    "recovered_tick": trained_relaxed.get("tick"),
                    "recovered_output_templated": summary.get("templated"),
                    "recovered_output_is_non_template": readable_recovered,
                    "reading": (
                        "relaxing the guard recovers the trained state AND its output is no "
                        "longer the fixed template, so the fix has measurable effect"
                        if readable_recovered
                        else (
                            "relaxing the guard recovers the trained state, but its output is "
                            "still the fixed template: the load defect and the language gap are "
                            "separate problems and fixing the guard alone will not produce "
                            "conversation"
                            if recovered
                            else "relaxing the guard does not recover the trained state; the "
                            "blocker is deeper than the identity-organ branch"
                        )
                    ),
                },
                "limits": (
                    "a successful load under a relaxed guard does not make the fix safe, and "
                    "readable output would not by itself justify it: the refuse-vs-migrate "
                    "question is a policy decision. This probe also cannot tell whether the "
                    "recovered state was trained for language at all -- only a frozen "
                    "evaluation set can."
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
    temporary = DEFAULT_REPORT.with_suffix(DEFAULT_REPORT.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(DEFAULT_REPORT)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help="internal: one fresh-process arm")
    args = parser.parse_args(argv)
    if args.child:
        return _child(json.loads(sys.stdin.read()))
    result = run_probe()
    verdict = result.get("verdict") or {}
    arms = result.get("arms") or {}
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "trained_current_guard_load_ok": (arms.get("trained_current_guard") or {}).get(
                    "load_ok"
                ),
                "trained_relaxed_guard_load_ok": (arms.get("trained_relaxed_guard") or {}).get(
                    "load_ok"
                ),
                "recovered_tick": verdict.get("recovered_tick"),
                "recovered_output_is_non_template": verdict.get("recovered_output_is_non_template"),
                "error": result.get("error"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
