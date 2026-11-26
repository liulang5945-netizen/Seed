"""P3: with the gate live, does the read margin hold at low occupancy, and is a
write refused at high occupancy instead of an existing slot being evicted?

Two fixtures, same encoder and rung geometry as P1/P2/P1b:

low   8/16   six sequential appends of genuinely new entries (tokens 20, 22..26
             at position 5, i.e. combinations that do not already exist in the
             store); after six the headroom is exactly 2/16 = 0.125, still legal.
high  64/128 the fourth generation baseline.  Its read margin sits on the
             collapse line (0.02), so any further write that lowers the margin
             must be refused.
fill  112/128 the only place where the slot arithmetic itself forbids a write:
             h = (128-112)/128 = 0.125, so the next append would leave 15/128 =
             0.1172 < 0.125.  Built directly, because the capacity clause is
             about slot arithmetic, not about how the slots were filled.

Registered before the first number is read:

Two margins are reported because they are different quantities and the
registration names one of them:
  m_spec  the SPEC §5 / F1 quantity: median over occupied keys of
          (nearest - second) after excluding self, i.e. read_margin_stats.
          The 0.06863 baseline is this quantity.
  m_gate  what G2 actually compares: median over the battery of
          (1 - nearest_other), the same formula as growth_gate._margin.
A1 is judged on m_spec, because that is the quantity the registered floor
0.06863 - 0.002 belongs to.  Judging A1 on m_gate would compare 0.857 against
0.06663 and pass for the wrong reason.

A1  low rung: (a) every post-write m_spec is at or above the literal SPEC §5
    P3-A floor 0.06863 - 0.002 = 0.06663, and (b) no single write lowers m_spec
    by more than 0.002.  (a) is the cumulative reading, (b) the per-write one.
A2  low rung: every attempted write is not a copy (band != growth_is_copy).
A3  low rung: all six writes are admitted and the sequence completes.
B1  high rung: the write is refused.  If it is published, B1 is "not triggered"
    and is reported as unconfirmed -- never as a pass.
B2  high rung exit is `commit_rejected_no_capacity`, as the task instruction
    requires.  Pre-registered note: at 64 occupied of 128 slots the headroom is
    0.5, four times the reserve, so the capacity clause cannot fire; this
    criterion is expected to be false and is reported as "instruction prediction
    not confirmed", not as a gate defect.  It is listed anyway so the reading
    cannot be quietly dropped.
B3  high rung, the SPEC §5 P3-B prediction: if the forced write lowers m_spec by
    >= 0.002, that write must be refused with `commit_rejected_separability`.
    If the forced drop is below 0.002 there is no prediction to test, and B3 is
    "not triggered" (unconfirmed).  If the drop is >= 0.002 and the gate still
    passes, P3-B is falsified -- that is the SPEC's own negative condition.
C1  the fill arm (112/128) is refused and `commit_rejected_no_capacity` is among
    the failing clauses.  The exit code itself is the first failing clause and is
    reported as-is; requiring it to be first would over-specify the ordering of
    two clauses that may both be true.
D1  no eviction anywhere; a published write leaves no quarantine row and a
    refused write leaves exactly one.
D2  every refusal is bit-identical (sha_after == sha_before).
E   for every write, what a forced write would have done (margin, headroom,
    bytes) is computed and reported, never published.

Zero training, zero optimizer steps, zero Adam, zero confirm, zero model
forward.  numpy only; the fixture is imported from the F1 runner.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_gate as gg  # noqa: E402
from taiji import growth_runtime as gr  # noqa: E402
from taiji.process_guard import assert_no_black_processes  # noqa: E402

REPORT = ROOT / "reports/taiji_growth_gate_p3_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_growth_gate_p3_20261108.json"
PROGRESS = ROOT / "reports/taiji_growth_gate_p3_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_growth_gate_p3_20261108_journal"
LOCK = ROOT / "reports/taiji_growth_gate_p3_20261108.lock"
F1_RUNNER = ROOT / "scripts/training/probe_taiji_growth_f1_correction_20261108.py"
GATE_FILE = ROOT / "taiji/growth_gate.py"
CLEANUP = ROOT / "reports/taiji_growth_gate_p3_black_hang_cleanup_20261108.json"

CPU_CAP = 150.0
CPU_GUARD = 140.0
LOW = {"name": "low", "occupied": 8, "slots": 16}
HIGH = {"name": "high", "occupied": 64, "slots": 128}
FILL = {"name": "fill", "occupied": 112, "slots": 128}
MARGIN_TOLERANCE = 0.002  # the SPEC's epsilon_m, unchanged
COLLAPSE_LINE = 0.02  # growth design S-B collapse line, unchanged
F1_LOW_BASELINE = 0.06863  # F1 measurement on 8/16, the constant SPEC §5 uses
P3_LOW_FLOOR = F1_LOW_BASELINE - MARGIN_TOLERANCE  # 0.06663
LOW_SEQUENCE = 6
FIRST_NEW_TOKEN = 20
CORRECTION_POSITION = 5
POSITION_COUNT = 8

CRITERIA = {
    "A1_low_margin_floors": (
        f"on 8/16, measured on m_spec (the read_margin_stats quantity the 0.06863 baseline "
        f"belongs to): (a) every post-write m_spec >= {P3_LOW_FLOOR:.5f} "
        f"({F1_LOW_BASELINE} - {MARGIN_TOLERANCE}), the literal SPEC §5 P3-A floor; (b) no single "
        f"write lowers m_spec by more than {MARGIN_TOLERANCE}"
    ),
    "A2_low_not_a_copy": "on 8/16 every attempted write has band != growth_is_copy",
    "A3_low_all_six_admitted": "on 8/16 all six writes are published and the sequence completes",
    "B1_high_refused": "on 64/128 the write is refused; published means B1 is unconfirmed",
    "B2_high_exit_is_no_capacity": (
        "on 64/128 the exit is commit_rejected_no_capacity (task instruction).  Headroom is 0.5 "
        "there, so this cannot fire; reported as unconfirmed, not as a gate defect"
    ),
    "B3_high_separability_rule": (
        "SPEC §5 P3-B: if the forced write lowers m_spec by >= 0.002 it must be refused with "
        "commit_rejected_separability; if the forced drop is < 0.002 the prediction is not "
        "triggered (unconfirmed); if the drop is >= 0.002 and the gate passes, P3-B is falsified"
    ),
    "C1_fill_arm_refuses_on_capacity": (
        "a 112/128 state (h = 0.125) is refused and commit_rejected_no_capacity is among the "
        "failing clauses (the exit code is the first failing clause and is reported as-is)"
    ),
    "D1_no_eviction_quarantine_exact": (
        "evictions 0 everywhere; a published write leaves 0 quarantine rows and a refused write "
        "leaves exactly 1"
    ),
    "D2_refusal_is_bit_identical": "every refusal has sha_after == sha_before",
    "E_counterfactual_reported": (
        "for every write, the forced-write margin, headroom and bytes are computed and reported, "
        "nothing is published"
    ),
    "unchanged": "no threshold, rung, battery or question is retuned after the first reading",
}

START_CPU = time.process_time()
START_WALL = time.time()


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def engineering_waste() -> dict[str, Any]:
    """D16-13 recurrence, disclosed in this round's cost section.

    These are formatter processes that spun on this machine while P3 was being
    prepared.  They are charged to the disclosure, never to an experiment
    budget: nothing in this probe, in the gate or in any training run benefits
    from them.  This probe itself never calls the banned formatter, and refuses
    to start if one is alive (``assert_no_black_processes``).
    """
    if not CLEANUP.exists():
        return {"available": False, "source": str(CLEANUP.relative_to(ROOT))}
    data = json.loads(CLEANUP.read_text(encoding="utf-8"))
    return {
        "available": True,
        "source": str(CLEANUP.relative_to(ROOT)).replace("\\", "/"),
        "kind": "engineering_waste_not_an_experiment_cost",
        "black_processes_killed": data.get("killed_total"),
        "black_processes_matched": data.get("black_count_before"),
        "black_processes_after": data.get("black_count_after"),
        "cpu_seconds_reclaimed": data.get("cpu_seconds_reclaimed"),
        "free_mb_before": data.get("free_mb_before"),
        "free_mb_after": data.get("free_mb_after"),
        "earliest_start": min(
            (row.get("start") or "" for row in data.get("victims", [])), default=""
        ),
        "charged_to_experiment_budget": False,
    }


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
    # a stale write-then-rename temp from an interrupted run must not survive
    leftover = REPORT.with_suffix(REPORT.suffix + ".tmp")
    if leftover.exists():
        leftover.unlink()


def acquire_single_instance() -> str:
    """One process at a time: two concurrent runs share the journal and report.

    The earlier D1 negative was produced by two runs overlapping on the same
    quarantine journals (2 rows where 1 was expected); this makes that state
    unreachable rather than merely unlikely.  It changes no criterion.
    """
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
                    f"refusing to start: another p3 run holds {LOCK.name} (age {age:.0f}s)"
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


def load_f1():
    spec = importlib.util.spec_from_file_location("f1_probe", F1_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {F1_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(state: gg.GateState) -> tuple[np.ndarray, np.ndarray]:
    rows = int(state.occupied)
    return state.keys[:rows], state.values[:rows]


def margin(state: gg.GateState, key: np.ndarray) -> float:
    keys, _ = read_rows(state)
    unit = key / max(float(np.linalg.norm(key)), 1e-12)
    cosine = (keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)) @ unit
    top = np.sort(cosine)[::-1]
    return float(top[0] - top[1]) if top.size > 1 else 0.0


def median_margin(state: gg.GateState, queries: list[gg.BatteryQuery]) -> float:
    return float(np.median([margin(state, row.key) for row in queries]))


def spec_margin(state: gg.GateState) -> float:
    """The SPEC §5 / F1 quantity, identical to growth_design.read_margin_stats.

    Over the occupied keys only, self excluded: median of (nearest - second).
    This is the quantity the registered 0.06863 baseline belongs to; it is *not*
    the gate's G2 margin (``1 - nearest``), which is near 0.857 on the same state.
    """
    keys, _ = read_rows(state)
    count = int(keys.shape[0])
    if count < 3:
        return 0.0
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    ordered = np.sort(gram, axis=1)[:, ::-1]
    return float(np.median(ordered[:, 0] - ordered[:, 1]))


def read_value(state: gg.GateState, key: np.ndarray) -> np.ndarray:
    keys, values = read_rows(state)
    unit = key / max(float(np.linalg.norm(key)), 1e-12)
    cosine = (keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)) @ unit
    logits = cosine / gg.GATE_TEMPERATURE
    weights = np.exp(logits - logits.max())
    weights /= weights.sum()
    return (weights[:, None] * values).sum(axis=0)


def relative_residual(read: np.ndarray, target: np.ndarray) -> float:
    return float(np.linalg.norm(read - target) / max(float(np.linalg.norm(target)), 1e-12))


def build_contexts(occupied: int) -> list[np.ndarray]:
    """First 64 slots are the F1 fixture verbatim; beyond that, paraphrases.

    The 8/16 and 64/128 rungs therefore start from exactly the contexts F1
    measured 0.06863 / 0.01999 on, and only the 112 slot arm needs synthetic
    contexts (the fixture has 64 sentences).
    """
    out = [contexts0[index] for index in range(min(occupied, len(contexts0)))]
    for index in range(len(out), occupied):
        out.append(
            f1.paraphrase(
                base0,
                tokens0[index % len(tokens0)],
                CORRECTION_POSITION + index // len(tokens0),
            )
        )
    return out


def build_state(occupied: int, slots: int, contexts, payloads, encoder, centroid) -> gg.GateState:
    keys = np.zeros((slots, f1.DIM), dtype=np.float64)
    values = np.zeros((slots, int(np.prod(f1.VALUE_SHAPE))), dtype=np.float64)
    for slot in range(occupied):
        keys[slot] = gr.encode_read_key(contexts[slot], encoder, centroid)
        values[slot] = payloads[slot]
    return gg.GateState(
        keys=keys,
        values=values,
        age=np.array(list(range(occupied)) + [-1] * (slots - occupied)),
        hits=np.zeros(slots, dtype=np.int64),
        occupied=occupied,
        metric_version=0,
    )


def battery(state: gg.GateState, occupied: int) -> list[gg.BatteryQuery]:
    return [
        gg.BatteryQuery(
            name=f"held_out_rule_{index}",
            key=state.keys[index].copy(),
            expect_slot=index,
            family=f"rule_{index}",
        )
        for index in range(1, occupied)
    ]


def new_tokens(count: int) -> list[int]:
    """Tokens whose (token, position) combination is not already in the store."""
    pool = [
        token
        for token in range(FIRST_NEW_TOKEN, len(tokens0))
        if token % POSITION_COUNT != CORRECTION_POSITION
    ]
    return pool[:count]


def headroom(state: gg.GateState, pending: int) -> dict[str, Any]:
    return gr.write_headroom(int(state.occupied), int(state.slots), pending)


def step(
    state: gg.GateState,
    policy: str,
    key: np.ndarray,
    value: np.ndarray,
    family: str,
    quarantine_path: Path,
    tag: str,
) -> tuple[dict[str, Any], gg.GateState]:
    queries = battery(state, int(state.occupied))
    before_margin = median_margin(state, queries)
    write = gg.WriteCell(key=key, value=value, family=family, policy=policy)
    quarantine_path.write_text("", encoding="utf-8")
    log_path = JOURNAL_DIR / f"{tag}_gate_log.jsonl"
    log_path.write_text("", encoding="utf-8")
    receipt, run = gg.run_gate(
        state, write, queries, quarantine_path=quarantine_path, log_path=log_path
    )
    cert = run.certificate
    forced = run.shadow
    forced_margin = median_margin(forced, queries)
    forced_headroom = (int(forced.slots) - int(forced.occupied)) / max(int(forced.slots), 1)
    occupied_after = int(receipt.state.occupied)
    margin_after_state = cert.g2_margin_after if receipt.published else before_margin
    spec_before = spec_margin(state)
    spec_after = spec_margin(receipt.state)
    spec_forced = spec_margin(forced)
    hr = headroom(state, 1)
    record = {
        "tag": tag,
        "policy": policy,
        "occupancy_before": int(state.occupied),
        "slots": int(state.slots),
        "headroom_before": round(hr["h_slots"], 6),
        "capacity_sufficient": bool(hr["sufficient"]),
        "dropped_writes": int(bool(hr["spills"])),
        "band_verdict": cert.direction_verdict,
        "direction_overlap": round(cert.direction_overlap, 6),
        "direction_separation": round(cert.direction_separation, 6),
        "margin_before": round(before_margin, 6),
        "margin_after_published": round(cert.g2_margin_after, 6),
        "margin_after_state": round(margin_after_state, 6),
        "margin_delta": round(margin_after_state - before_margin, 6),
        "spec_margin_before": round(spec_before, 6),
        "spec_margin_after": round(spec_after, 6),
        "spec_margin_delta": round(spec_after - spec_before, 6),
        "spec_margin_if_forced": round(spec_forced, 6),
        "spec_margin_delta_if_forced": round(spec_forced - spec_before, 6),
        "published": receipt.published,
        "exit_code": receipt.exit_code,
        "failed": list(cert.failed),
        "g1_before": cert.g1_before,
        "g1_after": cert.g1_after,
        "g4_value_residual": (
            None if cert.g4_value_residual is None else round(float(cert.g4_value_residual), 6)
        ),
        "written_slot": cert.written_slot,
        "sha_before": receipt.sha_before,
        "sha_after": receipt.sha_after,
        "bit_identical": receipt.bit_identical,
        "evictions": receipt.evictions,
        "quarantine_id": receipt.quarantine_id,
        "quarantine_rows": journal_rows(quarantine_path),
        "quarantine_missing": receipt.quarantine_id is None and not receipt.published,
        "occupancy_after": occupied_after,
        "headroom_after": round(
            (int(receipt.state.slots) - occupied_after) / max(int(receipt.state.slots), 1), 6
        ),
        "value_residual_if_forced": round(relative_residual(read_value(forced, key), value), 6),
        "counterfactual_if_forced": {
            "published": False,
            "occupancy": int(forced.occupied),
            "headroom": round(forced_headroom, 6),
            "margin": round(forced_margin, 6),
            "margin_delta_vs_before": round(forced_margin - before_margin, 6),
            "bytes_delta": (gg._state_floats(forced) - gg._state_floats(state))
            * gg.GATE_FLOAT_BYTES,
        },
    }
    append_progress(record)
    return record, receipt.state


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    # D16-13: never start while a banned formatter process is alive
    process_guard = assert_no_black_processes("p3 probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)

    global f1, base0, tokens0, contexts0, payloads, encoder, centroid

    reset_run_logs()
    f1 = load_f1()
    base0, tokens0, contexts0 = f1.fixture()
    # the widest state is the 112/128 arm; the six write values follow it
    payloads = f1.value_payloads(FILL["occupied"] + LOW_SEQUENCE)
    legacy = gr.EncoderConfig(mode="mean")
    parents = np.stack([gr.encode_read_key(contexts0[i], legacy) for i in range(4)])
    centroid = gr.store_centroid(parents)
    encoder = gr.EncoderConfig(mode="localized_centered", center=True)
    tokens = new_tokens(LOW_SEQUENCE)

    def sequence(rung: dict, count: int, tag_prefix: str) -> dict[str, Any]:
        state = build_state(
            rung["occupied"],
            rung["slots"],
            build_contexts(rung["occupied"]),
            payloads,
            encoder,
            centroid,
        )
        queries = battery(state, rung["occupied"])
        base_margin = median_margin(state, queries)
        base_spec = spec_margin(state)
        rows = []
        stop_reason = "completed"
        for index in range(count):
            key = gr.encode_read_key(
                f1.paraphrase(base0, tokens0[tokens[index]], CORRECTION_POSITION),
                encoder,
                centroid,
            )
            tag = f"{tag_prefix}_{index}"
            record, state_after = step(
                state,
                "append",
                key,
                payloads[FILL["occupied"] + index],
                "new_direction",
                JOURNAL_DIR / f"{tag}_quarantine.jsonl",
                tag,
            )
            record["sequence_index"] = index
            record["write_token"] = int(tokens[index])
            rows.append(record)
            if not record["published"]:
                stop_reason = f"refused_at_{index}:{record['exit_code']}"
                break
            state = state_after
        return {
            "rung": rung,
            "baseline_margin": round(base_margin, 6),
            "baseline_spec_margin": round(base_spec, 6),
            "steps_attempted": len(rows),
            "stop_reason": stop_reason,
            "dropped_writes_total": int(sum(row["dropped_writes"] for row in rows)),
            "evictions_total": int(sum(row["evictions"] for row in rows)),
            "trajectory": rows,
        }

    low = sequence(LOW, LOW_SEQUENCE, "low")
    high = sequence(HIGH, 1, "high")
    fill = sequence(FILL, 1, "fill")

    low_rows = low["trajectory"]
    high_row = high["trajectory"][0]
    fill_row = fill["trajectory"][0]

    low_abs_min = min(
        [row["margin_before"] for row in low_rows] + [row["margin_after_state"] for row in low_rows]
    )
    low_spec_min = min(row["spec_margin_after"] for row in low_rows)
    low_spec_worst_drop = min(row["spec_margin_delta"] for row in low_rows)
    high_forced_drop = high_row["spec_margin_delta_if_forced"]
    b3_triggered = -high_forced_drop >= MARGIN_TOLERANCE

    def quarantine_exact(run: dict) -> bool:
        return all(
            row["evictions"] == 0 and row["quarantine_rows"] == (0 if row["published"] else 1)
            for row in run["trajectory"]
        )

    checks = {
        "A1": low_spec_min >= P3_LOW_FLOOR and low_spec_worst_drop >= -MARGIN_TOLERANCE,
        "A2": all(row["band_verdict"] != "growth_is_copy" for row in low_rows),
        "A3": all(row["published"] for row in low_rows) and low["stop_reason"] == "completed",
        "B1": not high_row["published"],
        "B2": high_row["exit_code"] == gg.EXIT_NO_CAPACITY,
        "B3": b3_triggered
        and not high_row["published"]
        and high_row["exit_code"] == gg.EXIT_SEPARABILITY,
        "C1": not fill_row["published"] and gg.EXIT_NO_CAPACITY in fill_row["failed"],
        "D1": quarantine_exact(low) and quarantine_exact(high) and quarantine_exact(fill),
        "D2": all(
            row["bit_identical"]
            for run in (low, high, fill)
            for row in run["trajectory"]
            if not row["published"]
        ),
        "E": all(
            "counterfactual_if_forced" in row
            for run in (low, high, fill)
            for row in run["trajectory"]
        ),
    }

    status = {
        name: ("passed" if checks[name] else "falsified")
        for name in ("A1", "A2", "A3", "C1", "D1", "D2", "E")
    }
    status["B1"] = "passed" if checks["B1"] else "unconfirmed"
    status["B2"] = "passed" if checks["B2"] else "unconfirmed"
    status["B3"] = "passed" if checks["B3"] else ("falsified" if b3_triggered else "unconfirmed")

    falsified = [name for name, value in status.items() if value == "falsified"]
    unconfirmed = [name for name, value in status.items() if value == "unconfirmed"]
    if not falsified and not unconfirmed:
        decision = "p3_passed"
    elif falsified:
        decision = "p3_falsified_" + "_".join(falsified)
    else:
        decision = "p3_unconfirmed_" + "_".join(unconfirmed)

    if decision == "p3_passed":
        decision_zh = (
            "低档余量守住、六次写入全准入、容量臂以 no_capacity 拒写、高档被拒：预注册全部成立"
        )
    elif falsified:
        decision_zh = (
            "预注册的低档/容量/纪律判据未全部成立，见 falsified 列表与逐次读数；"
            "不得为通过而放宽任何阈值"
        )
    else:
        decision_zh = (
            "低档与容量臂符合预注册；但高档没有被拒写，按预注册记为『该预测未被证实』，不算通过"
        )

    payload: dict[str, Any] = {
        "format": "taiji-growth-gate-p3-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criteria_written_before_reading": CRITERIA,
        "question": (
            "with the gate live, does the read margin hold on 8/16, and is a write refused at high "
            "occupancy instead of evicting an existing slot?"
        ),
        "implementation_corrections": [
            {
                "when": "before the first reading was produced",
                "what": (
                    "the payload table was sized TOTAL_SENTENCES+16=80, but the 112/128 fill arm needs "
                    "112 stored values plus six write values; it is now sized FILL.occupied+LOW_SEQUENCE"
                ),
                "effect_on_decision": (
                    "none: the crashed run wrote no report, so no reading was observed, and no "
                    "criterion, threshold, rung, battery or question was changed"
                ),
                "criteria_changed": False,
                "thresholds_changed": False,
            },
            {
                "when": "before this report's readings; a previous overlapping run is discarded",
                "what": (
                    "a single-instance lock was added (reports/taiji_growth_gate_p3_20261108.lock, "
                    "O_EXCL).  Two runs of this probe had overlapped on the same quarantine journals "
                    "and report path, which is the only reason the earlier reading showed 2 quarantine "
                    "rows where 1 was expected (D1 false).  The earlier reading is superseded, not "
                    "relaxed; every criterion and threshold is byte-identical."
                ),
                "effect_on_decision": (
                    "removes an environment artifact (concurrent duplicate journal writes); the "
                    "substantive readings (A1 low-rung margin decay, high-rung refusal reason) are "
                    "re-measured from scratch under the lock"
                ),
                "criteria_changed": False,
                "thresholds_changed": False,
            },
        ],
        "preregistration_conflict": {
            "instruction": "64/128 must refuse with commit_rejected_no_capacity",
            "spec": "SPEC §5 P3-B expects commit_rejected_separability at the 64/128 gen-4 baseline",
            "arithmetic": "at 64 occupied of 128 slots h = 0.5 >= 0.125, so the capacity clause cannot fire",
            "resolution": (
                "both readings are reported; the capacity refusal is also tested where it can occur "
                "(112/128, h = 0.125).  B2 is reported as unconfirmed, not as a gate defect, and "
                "nothing is relaxed"
            ),
        },
        "margin_quantities": {
            "m_spec": (
                "median over occupied keys of (nearest - second), self excluded; the quantity "
                "growth_design.read_margin_stats reports and the 0.06863 baseline belongs to. "
                "A1 and B3 are judged on this one."
            ),
            "m_gate": (
                "median over the retention battery of (1 - nearest_other); what growth_gate._margin "
                "compares between before and after.  Reported for traceability, not used for A1."
            ),
            "why_two": (
                "they differ by construction (0.0686 vs 0.857 on the same 8-key state); judging A1 on "
                "m_gate would compare 0.857 against 0.06663 and pass for the wrong reason"
            ),
        },
        "thresholds": {
            "margin_tolerance": MARGIN_TOLERANCE,
            "collapse_line": COLLAPSE_LINE,
            "p3_low_floor_literal": round(P3_LOW_FLOOR, 6),
            "f1_low_baseline": F1_LOW_BASELINE,
            "gate_g2_max_margin_drop": gg.GATE_G2_MAX_MARGIN_DROP,
            "gate_max_state_bytes": gg.GATE_MAX_STATE_BYTES,
            "gate_min_slot_headroom": gg.GATE_MIN_SLOT_HEADROOM,
            "gate_max_value_residual": gg.GATE_MAX_VALUE_RESIDUAL,
        },
        "checks": checks,
        "status": status,
        "falsified": falsified,
        "unconfirmed": unconfirmed,
        "b3_premise": {
            "triggered": b3_triggered,
            "forced_spec_margin_drop": high_forced_drop,
            "rule": "predict a separability refusal only when the forced drop >= 0.002",
        },
        "runs": {"low": low, "high": high, "fill": fill},
        "discipline": {
            "evictions_total": (
                low["evictions_total"] + high["evictions_total"] + fill["evictions_total"]
            ),
            "dropped_writes_total": (
                low["dropped_writes_total"]
                + high["dropped_writes_total"]
                + fill["dropped_writes_total"]
            ),
            "capacity_sufficient_at_high": high_row["capacity_sufficient"],
            "capacity_sufficient_at_fill": fill_row["capacity_sufficient"],
            "note": (
                "dropped_writes counts writes the slot arithmetic would have had to drop "
                "(write_headroom.spills); it is 0 everywhere because nothing was evicted or "
                "silently discarded -- the fill arm is refused by the reserve, not by overflow"
            ),
        },
        "readings": {
            "low": {
                "baseline_margin_gate": low["baseline_margin"],
                "baseline_margin_spec": low["baseline_spec_margin"],
                "floor_literal": round(P3_LOW_FLOOR, 6),
                "spec_margin_min": round(low_spec_min, 6),
                "spec_margin_worst_delta": round(low_spec_worst_drop, 6),
                "margin_min_overall_gate": round(low_abs_min, 6),
                "steps_attempted": low["steps_attempted"],
                "stop_reason": low["stop_reason"],
                "steps": [
                    {
                        "step": row["sequence_index"],
                        "write_token": row["write_token"],
                        "band": row["band_verdict"],
                        "overlap": row["direction_overlap"],
                        "spec_margin_before": row["spec_margin_before"],
                        "spec_margin_after": row["spec_margin_after"],
                        "spec_margin_delta": row["spec_margin_delta"],
                        "gate_margin_before": row["margin_before"],
                        "gate_margin_after": row["margin_after_state"],
                        "gate_margin_delta": row["margin_delta"],
                        "headroom_after": row["headroom_after"],
                        "published": row["published"],
                        "exit_code": row["exit_code"],
                        "failed": row["failed"],
                    }
                    for row in low_rows
                ],
            },
            "high": {
                "baseline_margin_gate": high["baseline_margin"],
                "baseline_margin_spec": high["baseline_spec_margin"],
                "occupancy": high["rung"]["occupied"],
                "slots": high["rung"]["slots"],
                "headroom_before": high_row["headroom_before"],
                "published": high_row["published"],
                "exit_code": high_row["exit_code"],
                "failed": high_row["failed"],
                "band": high_row["band_verdict"],
                "overlap": high_row["direction_overlap"],
                "spec_margin_before": high_row["spec_margin_before"],
                "spec_margin_after": high_row["spec_margin_after"],
                "spec_margin_if_forced": high_row["spec_margin_if_forced"],
                "spec_margin_delta_if_forced": high_row["spec_margin_delta_if_forced"],
                "gate_margin_before": high_row["margin_before"],
                "gate_margin_forced": high_row["counterfactual_if_forced"]["margin"],
                "gate_margin_delta_if_forced": high_row["counterfactual_if_forced"][
                    "margin_delta_vs_before"
                ],
                "forced_headroom": high_row["counterfactual_if_forced"]["headroom"],
            },
            "fill": {
                "occupancy": fill["rung"]["occupied"],
                "slots": fill["rung"]["slots"],
                "headroom_before": fill_row["headroom_before"],
                "published": fill_row["published"],
                "exit_code": fill_row["exit_code"],
                "failed": fill_row["failed"],
                "spec_margin_delta_if_forced": fill_row["spec_margin_delta_if_forced"],
                "forced_headroom": fill_row["counterfactual_if_forced"]["headroom"],
                "gate_margin_delta_if_forced": fill_row["counterfactual_if_forced"][
                    "margin_delta_vs_before"
                ],
            },
        },
        "decision": decision,
        "decision_zh": decision_zh,
        "isolation": {
            "runner_sha256": file_sha(Path(__file__).resolve()),
            "gate_sha256": file_sha(GATE_FILE),
            "runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "gate_imports_torch": "import torch" in GATE_FILE.read_text(encoding="utf-8"),
            "process_guard": process_guard,
            "formatter_path": "scripts/training/black_check_gate.py (in-process); python -m black banned",
            "banned_formatter_invoked_by_this_runner": False,
            "single_instance_lock": lock_name,
            "frozen_files_touched": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_steps": 0,
            "model_forward": 0,
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 6),
            "wall_seconds": round(time.time() - START_WALL, 6),
            "cpu_cap": CPU_CAP,
            "readings_written": sum(len(run["trajectory"]) for run in (low, high, fill)),
            "watchdog_rule": "abort when no new reading lands for 60 s (external launcher)",
            "engineering_waste_this_round": engineering_waste(),
        },
    }

    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "format": "taiji-growth-gate-p3-manifest-v1",
            "generated_at": payload["generated_at"],
            "report": str(REPORT.relative_to(ROOT)).replace("\\", "/"),
            "report_sha256": file_sha(REPORT),
            "runner": str(Path(__file__).resolve().relative_to(ROOT)).replace("\\", "/"),
            "runner_sha256": file_sha(Path(__file__).resolve()),
            "gate_sha256": file_sha(GATE_FILE),
            "decision": decision,
            "checks": checks,
            "criteria_sha256": hashlib.sha256(
                json.dumps(CRITERIA, sort_keys=True, ensure_ascii=False).encode("utf-8")
            ).hexdigest(),
            "cpu_seconds": payload["cost"]["cpu_seconds"],
        },
    )

    print(json.dumps({"decision": decision, "checks": checks}, ensure_ascii=False, indent=2))
    print("status", status)
    print(
        "low ",
        [
            (
                row["sequence_index"],
                row["write_token"],
                row["published"],
                row["spec_margin_before"],
                row["spec_margin_after"],
                row["margin_after_state"],
            )
            for row in low_rows
        ],
    )
    print(
        "high",
        high_row["published"],
        high_row["exit_code"],
        high_row["failed"],
        high_row["spec_margin_before"],
        high_row["spec_margin_if_forced"],
        high_row["spec_margin_delta_if_forced"],
    )
    print(
        "fill",
        fill_row["published"],
        fill_row["exit_code"],
        fill_row["failed"],
        fill_row["headroom_before"],
    )
    if time.process_time() - START_CPU > CPU_CAP:
        raise SystemExit("cpu cap exceeded")
    return 0 if not falsified else 1


if __name__ == "__main__":
    raise SystemExit(main())
