"""G2b: replace G2's relative-margin clause with an absolute floor on ``m_spec``.

P3 showed the gate cannot see the crowding the SPEC §5 talks about: on 8/16 six
admitted writes took ``m_spec`` from 0.068631 to 0.033074 (the first alone cost
-0.01623, eight times the tolerance), while the quantity G2 actually compares,
``m_gate = 1 - nearest``, stayed at 0.857136 for all six.  A relative tolerance
of 0.002 on a quantity that lives near 0.857 can essentially never fire.

This probe keeps every other clause untouched and swaps only the judging
quantity: ``G2b`` reads ``m_spec`` *after* the write and refuses when it falls
below an absolute floor.  ``m_gate`` is still measured, as a reading only.

The three floors are stated before the first number is read, each with its
source.  No floor is reverse-engineered to land just before a chosen write.

  F0  0.06663  the SPEC §5 P3-A number, reused as an *absolute* floor.
              Source: 0.06863 - 0.002.  Expected to be unusable: 0.06863 is the
              baseline, so any write that appends a genuinely new direction
              lands far below it (P3's first append: 0.052401).  It is included
              to show that the registered number cannot simply be re-used.
  F1  0.02000  the collapse line already present in the growth design (S-B):
              "nearest and second nearest are effectively tied".  Source: an
              existing constant, independent of any P3 reading.
  F2  0.03432  half of the F1 baseline 0.06863, under the stated rule "the read
              margin may be halved, not quartered".  Source: the F1 constant and
              that rule only.  Disclosure: it happens to fall between the 5th and
              6th P3 readings; that coincidence is reported, not relied on --
              the usability of a floor is decided by the three sanity cases, not
              by where it lands on the trajectory.

A floor is **usable** only if, in the same run, all three P1b legacy writes keep
their direction:

  S1  replace  ``policy="replace"`` on an occupied slot, new value -- admitted
                (P1b: admitted, residual 0.000207)
  S2  new      ``policy="append"`` with a key whose (token, position) is not in
                the store -- admitted (P1b: admitted, overlap 0.283)
  S3  copy     ``policy="append"`` with a byte-identical key -- refused
                (P1b: refused, ``commit_rejected_no_direction``)

Any inversion disqualifies that floor; the three are reported per floor, never
only the main result.

Rungs, per floor:
  low  8/16 six sequential appends, m_spec trajectory, the first write that
       crosses the floor and its exit code (or "no crossing" if none does).
  high 64/128 one append: P3 read baseline 0.019988, refused as
       ``commit_rejected_no_direction``.  Reported again here so the question
       "does the absolute floor change its exit?" is answered, not assumed.
  fill 112/128: h = 0.117188 < 0.125, so ``commit_rejected_no_capacity`` must
       still win, and must not be confused with the new floor.

Zero training, zero optimizer steps, zero Adam, zero confirm, zero model
forward.  The fixture is imported from the P3 probe, so the geometry is
identical to P1/P2/P1b/P3.  No ``python -m black`` anywhere; the runner refuses
to start while that process exists (D16-13), holds a single-instance lock, and
the launcher aborts after 60 s without a new reading.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import sys
import time
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_gate as gg  # noqa: E402
from taiji import growth_runtime as gr  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

REPORT = ROOT / "reports/taiji_gate_g2b_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_g2b_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_g2b_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_g2b_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_g2b_20261108.lock"
P3_RUNNER = ROOT / "scripts/training/probe_taiji_growth_gate_p3_20261108.py"
GATE_FILE = ROOT / "taiji/growth_gate.py"

CPU_CAP = 100.0
CPU_GUARD = 92.0
LOW = {"name": "low", "occupied": 8, "slots": 16}
HIGH = {"name": "high", "occupied": 64, "slots": 128}
FILL = {"name": "fill", "occupied": 112, "slots": 128}
LOW_SEQUENCE = 6
F1_BASELINE = 0.06863

FLOORS: dict[str, dict[str, Any]] = {
    "F0_spec_floor_as_absolute": {
        "value": 0.06663,
        "source": "SPEC §5 P3-A: 0.06863 - 0.002, reused as an absolute floor",
        "expectation": "unusable: the baseline itself is 0.06863, so any new direction lands below it",
    },
    "F1_collapse_line": {
        "value": 0.02000,
        "source": "growth design S-B collapse line (existing constant, independent of P3)",
        "expectation": "no crossing on 8/16 (min 0.033074); 64/128 baseline 0.019988 is already below it",
    },
    "F2_half_baseline": {
        "value": 0.03432,
        "source": "0.5 x F1 baseline 0.06863, rule 'the read margin may be halved, not quartered'",
        "expectation": "crossed by the 5th P3 append (0.033074); disclosed coincidence, not the basis of the verdict",
    },
}

CLAUSE_ORDER = (
    gg.EXIT_RETENTION,
    gg.EXIT_SEPARABILITY,
    gg.EXIT_INFLATION,
    gg.EXIT_NO_CAPACITY,
    gg.EXIT_NO_DIRECTION,
    gg.EXIT_NO_GAIN,
)

START_CPU = time.process_time()
START_WALL = time.time()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def append_progress(record: dict) -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def reset_run_logs() -> None:
    PROGRESS.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS.write_text("", encoding="utf-8")
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    for stale in JOURNAL_DIR.glob("*.jsonl"):
        stale.write_text("", encoding="utf-8")
    leftover = REPORT.with_suffix(REPORT.suffix + ".tmp")
    if leftover.exists():
        leftover.unlink()


def acquire_single_instance() -> str:
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + 5.0
    while True:
        try:
            handle = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            age = time.time() - LOCK.stat().st_mtime
            if age > 600.0:
                LOCK.unlink(missing_ok=True)
                continue
            if time.time() >= deadline:
                raise SystemExit(
                    f"refusing to start: another g2b run holds {LOCK.name} (age {age:.0f}s)"
                ) from None
            time.sleep(0.2)
            continue
        os.write(handle, f"{os.getpid()}\n".encode())
        os.close(handle)
        return LOCK.name


def release_single_instance() -> None:
    LOCK.unlink(missing_ok=True)


def journal_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def load_p3():
    spec = importlib.util.spec_from_file_location("p3_probe", P3_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {P3_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def decide(
    p3,
    state: gg.GateState,
    write: gg.WriteCell,
    floor: float,
    tag: str,
) -> tuple[dict[str, Any], gg.GateState]:
    """One submission under G2b: evaluate, swap the G2 clause, then commit.

    Everything except the G2 quantity is the gate's own code path: the same
    snapshot, the same shadow, the same G1/G3/G4/direction clauses and the same
    ``commit`` that publishes or restores bit-for-bit and writes one quarantine
    row.  Only the separability clause changes its question, from "did the
    median 1 - nearest drop by more than 0.002" to "is m_spec below the floor".
    """
    queries = p3.battery(state, int(state.occupied))
    snap = gg.snapshot(state)
    shadow, apply_receipt = gg.shadow_apply(state, write)
    slot = apply_receipt.get("slot")
    cert = gg.evaluate(state, shadow, queries, write, written_slot=slot)

    spec_before = p3.spec_margin(state)
    spec_shadow = p3.spec_margin(shadow)
    relative_failed = gg.EXIT_SEPARABILITY in cert.failed
    floor_failed = spec_shadow < floor

    kept = [code for code in cert.failed if code != gg.EXIT_SEPARABILITY]
    if floor_failed:
        kept.append(gg.EXIT_SEPARABILITY)
    ordered = [code for code in CLAUSE_ORDER if code in kept]
    passed = not ordered
    exit_code = gg.EXIT_ACCEPTED if passed else ordered[0]

    certificate = dc_replace(
        cert,
        passed=passed,
        exit_code=exit_code,
        failed=tuple(ordered),
        detail={
            **cert.detail,
            "g2b_floor": floor,
            "g2b_margin_quantity": "m_spec (occupied-key nearest - second, self excluded)",
            "g2b_spec_margin_after": spec_shadow,
            "g2b_floor_failed": floor_failed,
            "relative_g2_failed": relative_failed,
            "relative_g2_dropped_in_favour_of_floor": relative_failed and not floor_failed,
        },
    )
    run = gg.GateRun(
        before_snapshot=snap,
        before=state,
        shadow=shadow,
        certificate=certificate,
        write=write,
        apply_receipt=apply_receipt,
    )
    quarantine_path = JOURNAL_DIR / f"{tag}_quarantine.jsonl"
    log_path = JOURNAL_DIR / f"{tag}_gate_log.jsonl"
    quarantine_path.write_text("", encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    receipt = gg.commit(run, quarantine_path=quarantine_path, log_path=log_path)

    hr = gr.write_headroom(int(state.occupied), int(state.slots), 1)
    key = write.key
    record = {
        "tag": tag,
        "policy": write.policy,
        "floor": floor,
        "occupancy_before": int(state.occupied),
        "slots": int(state.slots),
        "headroom_before": round(float(hr["h_slots"]), 6),
        "capacity_sufficient": bool(hr["sufficient"]),
        "dropped_writes": int(bool(hr["spills"])),
        "spec_margin_before": round(spec_before, 6),
        "spec_margin_after_forced": round(spec_shadow, 6),
        "spec_margin_after_state": round(p3.spec_margin(receipt.state), 6),
        "median_margin_before_m_gate": round(cert.g2_margin_before, 6),
        "median_margin_after_forced_m_gate": round(cert.g2_margin_after, 6),
        "floor_failed": bool(floor_failed),
        "relative_g2_failed": bool(relative_failed),
        "band_verdict": cert.direction_verdict,
        "direction_overlap": round(cert.direction_overlap, 6),
        "direction_separation": round(cert.direction_separation, 6),
        "g4_value_residual": (
            None if cert.g4_value_residual is None else round(float(cert.g4_value_residual), 6)
        ),
        "value_residual_if_forced": round(
            p3.relative_residual(p3.read_value(shadow, key), write.value), 6
        ),
        "published": bool(receipt.published),
        "exit_code": receipt.exit_code,
        "failed": list(certificate.failed),
        "sha_before": receipt.sha_before,
        "sha_after": receipt.sha_after,
        "bit_identical": bool(receipt.bit_identical),
        "evictions": int(receipt.evictions),
        "quarantine_id": receipt.quarantine_id,
        "quarantine_rows": journal_rows(quarantine_path),
        "occupancy_after": int(receipt.state.occupied),
        "headroom_after": round(
            (int(receipt.state.slots) - int(receipt.state.occupied))
            / max(int(receipt.state.slots), 1),
            6,
        ),
    }
    append_progress(record)
    return record, receipt.state


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("g2b probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    reset_run_logs()

    # pre-registration lands before the first reading
    write_json(
        MANIFEST,
        {
            "format": "taiji-gate-g2b-preregistration-v1",
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "written_before_reading": True,
            "question": (
                "if G2's judging quantity becomes m_spec with an absolute floor, does the gate "
                "refuse the write that crowds the store while keeping the three P1b legacy "
                "verdicts (replace admit, new admit, copy refuse) intact?"
            ),
            "criterion_change": (
                "G2: median(1 - nearest) must not drop by more than 0.002  ->  "
                "m_spec (occupied-key nearest - second, self excluded) must stay >= floor"
            ),
            "floors": FLOORS,
            "sanity_cases_required_per_floor": {
                "S1_replace_admitted": "policy=replace on an occupied slot must be published",
                "S2_new_admitted": "append of a key not in the store must be published",
                "S3_copy_refused": "append of a byte-identical key must be refused",
            },
            "rungs": {
                "low_8_16": "six sequential appends; report m_spec trajectory and the first crossing",
                "high_64_128": "one append; report whether the exit changes vs P3's no_direction",
                "fill_112_128": "one append; no_capacity must still win (h = 0.117188 < 0.125)",
            },
            "untouched": [
                "G1 retention",
                "G3 inflation and capacity",
                "G4 value residual",
                "direction band",
                "every threshold except the G2 quantity itself",
                "fixture, rungs, battery",
            ],
            "discipline": {
                "training_steps": 0,
                "optimizer_steps": 0,
                "adam_steps": 0,
                "confirm_steps": 0,
                "banned_formatter": "python -m black (process_guard at entry)",
                "single_instance_lock": True,
                "watchdog": "abort after 60 s without a new reading (launcher)",
                "cpu_cap_seconds": CPU_CAP,
            },
        },
    )

    p3 = load_p3()
    p3.f1 = p3.load_f1()
    p3.base0, p3.tokens0, p3.contexts0 = p3.f1.fixture()
    payloads = p3.f1.value_payloads(FILL["occupied"] + LOW_SEQUENCE)
    legacy = gr.EncoderConfig(mode="mean")
    parents = np.stack([gr.encode_read_key(p3.contexts0[i], legacy) for i in range(4)])
    centroid = gr.store_centroid(parents)
    encoder = gr.EncoderConfig(mode="localized_centered", center=True)
    tokens = p3.new_tokens(LOW_SEQUENCE)

    def fresh(rung: dict) -> gg.GateState:
        return p3.build_state(
            rung["occupied"],
            rung["slots"],
            p3.build_contexts(rung["occupied"]),
            payloads,
            encoder,
            centroid,
        )

    def sequence(rung: dict, count: int, floor: float, tag_prefix: str) -> dict[str, Any]:
        state = fresh(rung)
        base_spec = p3.spec_margin(state)
        rows: list[dict[str, Any]] = []
        stop_reason = "completed"
        for index in range(count):
            key = gr.encode_read_key(
                p3.f1.paraphrase(p3.base0, p3.tokens0[tokens[index]], p3.CORRECTION_POSITION),
                encoder,
                centroid,
            )
            tag = f"{tag_prefix}_{index}"
            write = gg.WriteCell(
                key=key,
                value=payloads[FILL["occupied"] + index],
                family="new_direction",
                policy="append",
            )
            record, state_after = decide(p3, state, write, floor, tag)
            record["sequence_index"] = index
            record["write_token"] = int(tokens[index])
            rows.append(record)
            if not record["published"]:
                stop_reason = f"refused_at_{index}:{record['exit_code']}"
                break
            state = state_after
        return {
            "rung": rung,
            "baseline_spec_margin": round(base_spec, 6),
            "steps_attempted": len(rows),
            "stop_reason": stop_reason,
            "first_refused_index": next(
                (row["sequence_index"] for row in rows if not row["published"]), None
            ),
            "trajectory_spec_margin": [row["spec_margin_before"] for row in rows]
            + ([rows[-1]["spec_margin_after_state"]] if rows else []),
            "evictions_total": int(sum(row["evictions"] for row in rows)),
            "dropped_writes_total": int(sum(row["dropped_writes"] for row in rows)),
            "trajectory": rows,
        }

    def sanity(floor: float, suffix: str) -> dict[str, Any]:
        base = fresh(LOW)
        slot = 3
        stored_key = base.keys[slot].copy()
        cases = {
            "S1_replace": gg.WriteCell(
                key=stored_key,
                value=payloads[FILL["occupied"] + LOW_SEQUENCE - 1],
                family="supersede",
                policy="replace",
            ),
            "S2_new": gg.WriteCell(
                key=gr.encode_read_key(
                    p3.f1.paraphrase(p3.base0, p3.tokens0[tokens[0]], p3.CORRECTION_POSITION),
                    encoder,
                    centroid,
                ),
                value=payloads[FILL["occupied"]],
                family="new_direction",
                policy="append",
            ),
            "S3_copy": gg.WriteCell(
                key=stored_key,
                value=base.values[slot].copy(),
                family="copy",
                policy="append",
            ),
        }
        out: dict[str, Any] = {}
        for name, write in cases.items():
            record, _ = decide(p3, fresh(LOW), write, floor, f"{suffix}_{name}")
            out[name] = {
                "published": record["published"],
                "exit_code": record["exit_code"],
                "band_verdict": record["band_verdict"],
                "direction_overlap": record["direction_overlap"],
                "value_residual": record["value_residual_if_forced"],
                "spec_margin_after_forced": record["spec_margin_after_forced"],
                "floor_failed": record["floor_failed"],
            }
        return out

    results: dict[str, Any] = {}
    for name, spec in FLOORS.items():
        floor = float(spec["value"])
        cases = sanity(floor, name)
        low = sequence(LOW, LOW_SEQUENCE, floor, f"{name}_low")
        high = sequence(HIGH, 1, floor, f"{name}_high")
        fill = sequence(FILL, 1, floor, f"{name}_fill")
        high_row = high["trajectory"][0]
        fill_row = fill["trajectory"][0]
        checks = {
            "S1_replace_admitted": cases["S1_replace"]["published"],
            "S2_new_admitted": cases["S2_new"]["published"],
            "S3_copy_refused": not cases["S3_copy"]["published"],
            "fill_refused_on_capacity": not fill_row["published"]
            and fill_row["exit_code"] == gg.EXIT_NO_CAPACITY,
            "low_refusals_use_separability": all(
                row["exit_code"] == gg.EXIT_SEPARABILITY
                for row in low["trajectory"]
                if not row["published"]
            ),
            "no_eviction": low["evictions_total"]
            + high["evictions_total"]
            + fill["evictions_total"]
            == 0,
            "dropped_writes_zero": low["dropped_writes_total"]
            + high["dropped_writes_total"]
            + fill["dropped_writes_total"]
            == 0,
            "quarantine_exact": all(
                row["quarantine_rows"] == (0 if row["published"] else 1)
                for run in (low, high, fill)
                for row in run["trajectory"]
            ),
            "refusals_bit_identical": all(
                row["bit_identical"]
                for run in (low, high, fill)
                for row in run["trajectory"]
                if not row["published"]
            ),
        }
        usable = (
            checks["S1_replace_admitted"]
            and checks["S2_new_admitted"]
            and checks["S3_copy_refused"]
        )
        results[name] = {
            "floor": floor,
            "source": spec["source"],
            "expectation": spec["expectation"],
            "sanity": cases,
            "checks": checks,
            "usable": usable,
            "low": low,
            "high": {
                "baseline_spec_margin": high["baseline_spec_margin"],
                "published": high_row["published"],
                "exit_code": high_row["exit_code"],
                "changed_vs_p3_no_direction": high_row["exit_code"] != gg.EXIT_NO_DIRECTION,
                "spec_margin_after_forced": high_row["spec_margin_after_forced"],
                "floor_failed": high_row["floor_failed"],
                "relative_g2_failed": high_row["relative_g2_failed"],
                "band_verdict": high_row["band_verdict"],
            },
            "fill": {
                "published": fill_row["published"],
                "exit_code": fill_row["exit_code"],
                "failed": fill_row["failed"],
                "headroom_after": fill_row["headroom_after"],
                "floor_failed": fill_row["floor_failed"],
            },
        }

    usable_floors = [name for name, row in results.items() if row["usable"]]
    if usable_floors:
        decision = "g2b_usable:" + ",".join(usable_floors)
        decision_zh = (
            "以下绝对下限同时满足三件事：拒掉跨线的 crowding 写入、不误杀 replace 与真新条目、"
            "复制仍被拒 —— " + "、".join(usable_floors)
        )
    else:
        decision = "g2b_no_usable_floor"
        decision_zh = (
            "三个候选下限都至少翻转了一条 P1b 遗判（replace/新条目/复制），故都不采用；"
            "不得挑一个最接近的凑通过"
        )

    payload: dict[str, Any] = {
        "format": "taiji-gate-g2b-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "question": (
            "with G2's quantity replaced by m_spec and an absolute floor, does the gate refuse "
            "the crowding write while keeping replace/new admitted and copy refused?"
        ),
        "criterion_change": (
            "G2 := m_spec(shadow) >= floor; the old relative clause "
            "(median 1 - nearest must not drop by 0.002) is measured and reported but no longer "
            "decides"
        ),
        "clause_order": list(CLAUSE_ORDER),
        "floors": FLOORS,
        "implementation_corrections": [
            {
                "when": "before the first reading; the crashed run wrote the manifest only",
                "what": (
                    "the journal directory was never created because the port of the P3 harness "
                    "dropped its reset call; the first attempt died on FileNotFoundError while "
                    "writing the quarantine file, before any decide() returned"
                ),
                "effect_on_decision": (
                    "none: no reading was produced, and no floor, criterion, threshold, rung or "
                    "sanity case was changed"
                ),
                "criteria_changed": False,
                "thresholds_changed": False,
            }
        ],
        "results": results,
        "usable_floors": usable_floors,
        "decision": decision,
        "decision_zh": decision_zh,
        "discipline": {
            "evictions": 0,
            "dropped_writes": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_steps": 0,
            "model_forward": 0,
            "frozen_files_touched": 0,
        },
        "isolation": {
            "runner_sha256": file_sha(Path(__file__).resolve()),
            "gate_sha256": file_sha(GATE_FILE),
            "runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "process_guard": process_guard,
            "formatter_path": "scripts/training/black_check_gate.py (in-process); python -m black banned",
            "banned_formatter_invoked_by_this_runner": False,
            "single_instance_lock": lock_name,
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 6),
            "wall_seconds": round(time.time() - START_WALL, 6),
            "cpu_cap": CPU_CAP,
            "readings_written": sum(
                1 for line in PROGRESS.read_text(encoding="utf-8").splitlines() if line.strip()
            ),
            "watchdog_rule": "abort when no new reading lands for 60 s (external launcher)",
        },
    }
    write_json(REPORT, payload)

    summary = {
        "decision": decision,
        "usable_floors": usable_floors,
        "per_floor": {
            name: {
                "usable": row["usable"],
                "sanity": {key: val["published"] for key, val in row["sanity"].items()},
                "low_stop": row["low"]["stop_reason"],
                "low_spec": row["low"]["trajectory_spec_margin"],
                "high_exit": row["high"]["exit_code"],
                "high_changed": row["high"]["changed_vs_p3_no_direction"],
                "fill_exit": row["fill"]["exit_code"],
            }
            for name, row in results.items()
        },
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if usable_floors else 1


if __name__ == "__main__":
    raise SystemExit(main())
