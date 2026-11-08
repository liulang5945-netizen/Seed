"""G2c: measure the ``m_spec`` reference curve, then decide the criterion's shape.

G2b replaced G2's relative tolerance with an absolute floor on ``m_spec`` and
found both usable floors graded the wrong way round: 0.02 never fires at low
occupancy, 0.03432 refuses a write while the margin is still healthy.  The
diagnosis was not "the number is wrong" but "the shape is wrong": ``m_spec``
falls from 0.0686 at 8 keys to 0.0200 at 64 keys, so any constant floor is
either permissive at 8 or prohibitive at 64.

This probe is measurement first, judgement second, in one process:

  phase 1  read-only curve.  Five rungs on the same encoder and the same
           fixture (8/16, 16/32, 32/64, 64/128, 112/128).  Per rung: median,
           minimum and p10 of the occupied-key ``(nearest - second)`` margin,
           plus effective rank, mean |cos| and width.  No gate, no write, no
           training: only ``build_state`` followed by reads.
  phase 2  forms and constants.  Four forms are stated with their theory and
           their constants are computed from phase 1 only, then written to the
           pre-registration manifest.  The manifest is flushed and fsynced
           before any legacy case is decided; its sha256 is recorded in the
           report.  Judgement uses only those landed forms.
  phase 3  legacy re-check.  Each form is run through the three P1b cases on
           8/16 -- replace admit, new admit, copy refuse -- through the gate's
           own ``snapshot``/``shadow_apply``/``evaluate``/``commit`` path, with
           only the G2 clause swapped for the form's floor.

Forms, all fixed before the case verdicts:

  A  m >= c_A / sqrt(m)        occupancy only; c_A anchored at the published
                               64-key reading 0.019988 (F1/F3), never at a case.
  B  m >= c_B / eff_rank       rank instead of row count; c_B anchored the same
                               way, at the measured effective rank of the
                               64-key state.
  C  m >= c_C * baseline(m)    normalised by what the same number of isotropic
                               random unit keys gives (seeded, deterministic);
                               c_C anchored the same way.
  D  m >= c_D * m ** -alpha    one global power law, c_D and alpha fitted by
                               least squares over all five phase-1 points.
                               Included because A fixes alpha = 0.5 without
                               evidence; alpha is the quantity of interest.

A form *fits* when its largest relative residual over all five rungs is at most
20%.  A form is *adoptable* when it fits **and** all three legacy directions
hold.  Pre-registered fallback: if no form fits all five rungs, the criterion's
shape is undetermined and the answer is "measure more rungs", never a piecewise
fit or a pick of the least bad form.

One deliberate cross-reference, no new writes: G2b's measured 8/16 trajectory
(0.068631 -> 0.052401 on the first append) is compared against the phase-1
curve at 16.  If the equilibrium curve at 16 sits near the trajectory value,
the curve describes states; if it sits well above, the curve describes states
but not the path, and no occupancy floor can cover both.

Zero training, zero optimizer steps, zero Adam, zero confirm, zero model
forward.  numpy only.  No ``python -m black`` anywhere (D16-13): the runner
refuses to start while one is alive, holds a single-instance lock, and a
watchdog aborts the process when no new reading lands for 60 s.
"""

from __future__ import annotations

import atexit
import hashlib
import importlib.util
import json
import os
import sys
import threading
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

REPORT = ROOT / "reports/taiji_gate_g2c_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_gate_g2c_20261108.json"
PROGRESS = ROOT / "reports/taiji_gate_g2c_20261108_progress.jsonl"
JOURNAL_DIR = ROOT / "reports/taiji_gate_g2c_20261108_journal"
LOCK = ROOT / "reports/taiji_gate_g2c_20261108.lock"
WATCHDOG_NOTE = ROOT / "reports/taiji_gate_g2c_20261108_watchdog.json"
P3_RUNNER = ROOT / "scripts/training/probe_taiji_growth_gate_p3_20261108.py"
GATE_FILE = ROOT / "taiji/growth_gate.py"

CPU_CAP = 100.0
CPU_GUARD = 90.0
WATCHDOG_SECONDS = 60.0
FIT_TOL = 0.20
RANDOM_SEED = 20261108
PUBLISHED_64 = 0.019988
G2B_FIRST_APPEND = 0.052401
WIDTH = 128  # f1.DIM; asserted against the built state in main()

# First run of this probe, kept on the record because rung 5 (112) turned out to
# be unmeasurable on this fixture.  The first run's criteria, forms and constants
# are identical to the ones below; only the validity instrumentation is new.
SUPERSEDED = {
    "report_sha256": "5e5502a223babc05e7a4cfbc17aed428882743ee7e377397fd194bd05592c6a4",
    "manifest_sha256": "339f56892f882624a1356d920d8e1b5af0dc3c3aa624567d72861136879ecb25",
    "decision": "g2c_shape_undetermined:measure_more_rungs",
}

RUNGS = (
    {"name": "r8", "occupied": 8, "slots": 16},
    {"name": "r16", "occupied": 16, "slots": 32},
    {"name": "r32", "occupied": 32, "slots": 64},
    {"name": "r64", "occupied": 64, "slots": 128},
    {"name": "r112", "occupied": 112, "slots": 128},
)
CASE_RUNG = {"name": "low", "occupied": 8, "slots": 16}

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
    with tmp.open("r+b") as handle:
        os.fsync(handle.fileno())
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
    for leftover in (REPORT.with_suffix(REPORT.suffix + ".tmp"), WATCHDOG_NOTE):
        if leftover.exists():
            leftover.unlink()
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)


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
                    f"refusing to start: another g2c run holds {LOCK.name} (age {age:.0f}s)"
                ) from None
            time.sleep(0.2)
            continue
        os.write(handle, f"{os.getpid()}\n".encode())
        os.close(handle)
        return LOCK.name


def release_single_instance() -> None:
    LOCK.unlink(missing_ok=True)


def start_watchdog() -> None:
    """Abort the process when no new reading lands for WATCHDOG_SECONDS."""

    def loop() -> None:
        while True:
            time.sleep(2.0)
            try:
                age = time.time() - PROGRESS.stat().st_mtime
            except OSError:
                continue
            if age > WATCHDOG_SECONDS:
                write_json(
                    WATCHDOG_NOTE,
                    {
                        "fired": True,
                        "seconds_without_new_reading": round(age, 3),
                        "limit_seconds": WATCHDOG_SECONDS,
                        "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    },
                )
                LOCK.unlink(missing_ok=True)
                os._exit(3)

    threading.Thread(target=loop, daemon=True).start()


def cpu_check(stage: str) -> None:
    used = time.process_time() - START_CPU
    if used > CPU_CAP:
        write_json(
            WATCHDOG_NOTE,
            {
                "cpu_cap_tripped": True,
                "stage": stage,
                "cpu_seconds": round(used, 3),
                "cap_seconds": CPU_CAP,
            },
        )
        raise SystemExit(f"cpu cap {CPU_CAP}s tripped during {stage}")


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


# --- pure geometry -----------------------------------------------------------


def occupied_keys(state: gg.GateState) -> np.ndarray:
    return state.keys[: int(state.occupied)]


def key_margins(state: gg.GateState) -> np.ndarray:
    """Per-key (nearest - second) cosine, self excluded, over occupied rows."""
    keys = occupied_keys(state)
    count = int(keys.shape[0])
    if count < 3:
        return np.zeros(0, dtype=np.float64)
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    ordered = np.sort(gram, axis=1)[:, ::-1]
    return ordered[:, 0] - ordered[:, 1]


def effective_rank(state: gg.GateState) -> float:
    """Participation ratio of the occupied-key Gram spectrum (same as the gate)."""
    keys = occupied_keys(state)
    count = int(keys.shape[0])
    if count < 2:
        return 0.0
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    gram = normalized @ normalized.T
    eigenvalues = np.clip(np.linalg.eigvalsh(gram), 0.0, None)
    total = float(eigenvalues.sum())
    if total <= 0.0:
        return 0.0
    return float(total * total / float((eigenvalues**2).sum()))


def top_share(state: gg.GateState) -> float:
    keys = occupied_keys(state)
    if int(keys.shape[0]) < 2:
        return 0.0
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    eigenvalues = np.clip(np.linalg.eigvalsh(normalized @ normalized.T), 0.0, None)
    total = float(eigenvalues.sum())
    return float(eigenvalues.max() / total) if total > 0.0 else 0.0


def mean_abs_cos(state: gg.GateState) -> float:
    keys = occupied_keys(state)
    count = int(keys.shape[0])
    if count < 2:
        return 0.0
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    gram = np.abs(normalized @ normalized.T)
    np.fill_diagonal(gram, 0.0)
    return float(gram.sum() / (count * (count - 1)))


def random_baseline(count: int) -> dict[str, float]:
    """Isotropic control: the same number of random unit keys in the same width.

    Seeded and deterministic, one draw.  Reported as a control value, not as a
    distribution: the fixture margin is itself a single deterministic number.
    """
    rng = np.random.default_rng(RANDOM_SEED + int(count))
    keys = rng.normal(size=(int(count), WIDTH))
    normalized = keys / np.clip(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12, None)
    gram = normalized @ normalized.T
    np.fill_diagonal(gram, -np.inf)
    ordered = np.sort(gram, axis=1)[:, ::-1]
    margins = ordered[:, 0] - ordered[:, 1]
    return {
        "median_margin": float(np.median(margins)),
        "min_margin": float(np.min(margins)),
    }


def measure(state: gg.GateState, rung: dict, contexts: list[np.ndarray]) -> dict[str, Any]:
    margins = key_margins(state)
    rank = effective_rank(state)
    keys = occupied_keys(state)
    unique_contexts = len({np.asarray(item, dtype=np.float64).tobytes() for item in contexts})
    unique_keys = len({row.tobytes() for row in keys})
    occupied = int(state.occupied)
    return {
        "rung": rung["name"],
        "occupied": occupied,
        "slots": int(state.slots),
        "width": int(state.keys.shape[1]),
        "median_margin": round(float(np.median(margins)), 6),
        "min_margin": round(float(np.min(margins)), 6),
        "p10_margin": round(float(np.percentile(margins, 10)), 6),
        "argmin_key": int(np.argmin(margins)),
        "effective_rank": round(rank, 6),
        "effective_rank_over_occupied": round(rank / max(occupied, 1), 6),
        "mean_abs_cos": round(mean_abs_cos(state), 6),
        "top_share": round(top_share(state), 6),
        "random_baseline_median": round(random_baseline(occupied)["median_margin"], 6),
        "unique_contexts": unique_contexts,
        "unique_keys": unique_keys,
        "duplicate_keys": occupied - unique_keys,
        "rows_with_margin_above_half": int((margins > 0.5).sum()),
        "measurable": unique_contexts == occupied and occupied == unique_keys,
    }


# --- forms -------------------------------------------------------------------


def floor_of(form: dict, shadow: gg.GateState) -> float:
    name = form["name"]
    if name == "B_effective_rank":
        rank = max(effective_rank(shadow), 1e-9)
        return float(form["constant"]) / rank
    if name == "C_random_baseline":
        return float(form["constant"]) * random_baseline(int(shadow.occupied))["median_margin"]
    if name == "D_fitted_power_law":
        return float(form["constant"]) * int(shadow.occupied) ** (-float(form["exponent"]))
    return float(form["constant"]) / float(np.sqrt(int(shadow.occupied)))


def predict(form: dict, row: dict) -> float:
    name = form["name"]
    if name == "B_effective_rank":
        return float(form["constant"]) / max(float(row["effective_rank"]), 1e-9)
    if name == "C_random_baseline":
        return float(form["constant"]) * float(row["random_baseline_median"])
    if name == "D_fitted_power_law":
        return float(form["constant"]) * int(row["occupied"]) ** (-float(form["exponent"]))
    return float(form["constant"]) / float(np.sqrt(int(row["occupied"])))


def build_forms(curve: list[dict]) -> list[dict]:
    by_occupied = {int(row["occupied"]): row for row in curve}
    r64 = by_occupied[64]
    anchor_rank = max(float(r64["effective_rank"]), 1e-9)
    anchor_base = max(float(r64["random_baseline_median"]), 1e-12)

    xs = np.log(np.array([float(row["occupied"]) for row in curve]))
    ys = np.log(np.array([float(row["median_margin"]) for row in curve]))
    x_mean = float(xs.mean())
    y_mean = float(ys.mean())
    exponent = -float(((xs - x_mean) * (ys - y_mean)).sum() / ((xs - x_mean) ** 2).sum())
    log_c = y_mean + exponent * x_mean

    forms = [
        {
            "name": "A_occupancy_sqrt",
            "formula": "median_margin >= c / sqrt(m)",
            "theory": (
                "occupancy only: each appended key adds (m-1) pairwise constraints, and the "
                "leading correction for a cloud that is isotropic inside its own span is "
                "m^-1/2.  If the measured decay is far steeper than this, the exponent is the "
                "thing that is wrong, not the constant."
            ),
            "constant": PUBLISHED_64 * float(np.sqrt(64.0)),
            "constant_source": (
                "anchor: the published 64-key reading 0.019988 (F1/F3), scaled by sqrt(64); "
                "no phase-1 or case reading enters the constant"
            ),
        },
        {
            "name": "B_effective_rank",
            "formula": "median_margin >= c / effective_rank(shadow)",
            "theory": (
                "rank instead of rows: if the binding quantity is the number of directions the "
                "store actually spans, the gap should track the participation ratio.  B equals "
                "A whenever effective_rank == m; they separate exactly where the store stops "
                "gaining dimensions."
            ),
            "constant": PUBLISHED_64 * anchor_rank,
            "constant_source": (
                "anchor: 0.019988 x the phase-1 effective rank of the 64-key state; the state "
                "is a pure-geometry measurement, no case reading enters it"
            ),
        },
        {
            "name": "C_random_baseline",
            "formula": "median_margin >= c * random_baseline_median(m)",
            "theory": (
                "same-size isotropic control: divide out how crowded m random unit keys in this "
                "width would already be, so the criterion asks whether the store is worse than "
                "an unstructured cloud of the same size rather than setting a raw level.  The "
                "control is seeded and deterministic."
            ),
            "constant": PUBLISHED_64 / anchor_base,
            "constant_source": (
                "anchor: 0.019988 / the phase-1 isotropic baseline median at 64 keys; the "
                "baseline is a seeded random draw, no case reading enters it"
            ),
        },
        {
            "name": "D_fitted_power_law",
            "formula": "median_margin >= c * m ** -alpha",
            "theory": (
                "one global power law over all five rungs.  A fixes alpha = 0.5 without "
                "evidence; D measures it.  A single exponent is a shape claim, not a piecewise "
                "fit: if the five points are not one line in log-log, D must fail its residual "
                "test rather than be segmented."
            ),
            "constant": float(np.exp(log_c)),
            "exponent": round(exponent, 6),
            "constant_source": "least squares over the five phase-1 points, log-log",
        },
    ]

    for form in forms:
        residuals = []
        for row in curve:
            predicted = predict(form, row)
            observed = float(row["median_margin"])
            residuals.append(
                {
                    "occupied": int(row["occupied"]),
                    "observed": round(observed, 6),
                    "predicted": round(predicted, 6),
                    "relative_residual": round(abs(predicted - observed) / max(observed, 1e-12), 6),
                }
            )
        worst = max(item["relative_residual"] for item in residuals)
        form["residuals"] = residuals
        form["max_relative_residual"] = round(worst, 6)
        form["fits"] = worst <= FIT_TOL
    return forms


# --- gate path, G2 clause swapped for the form's floor -----------------------


def decide(
    p3,
    state: gg.GateState,
    write: gg.WriteCell,
    form: dict,
    tag: str,
) -> tuple[dict[str, Any], gg.GateState]:
    queries = p3.battery(state, int(state.occupied))
    snap = gg.snapshot(state)
    shadow, apply_receipt = gg.shadow_apply(state, write)
    slot = apply_receipt.get("slot")
    cert = gg.evaluate(state, shadow, queries, write, written_slot=slot)

    spec_before = p3.spec_margin(state)
    spec_shadow = p3.spec_margin(shadow)
    floor = floor_of(form, shadow)
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
            "g2c_form": form["name"],
            "g2c_formula": form["formula"],
            "g2c_floor": floor,
            "g2c_margin_quantity": "m_spec (occupied-key nearest - second, self excluded)",
            "g2c_spec_margin_after": spec_shadow,
            "g2c_floor_failed": floor_failed,
            "relative_g2_failed": relative_failed,
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

    record = {
        "tag": tag,
        "form": form["name"],
        "policy": write.policy,
        "occupancy_before": int(state.occupied),
        "occupancy_after_forced": int(shadow.occupied),
        "slots": int(state.slots),
        "floor": round(float(floor), 6),
        "spec_margin_before": round(spec_before, 6),
        "spec_margin_after_forced": round(spec_shadow, 6),
        "spec_margin_after_state": round(p3.spec_margin(receipt.state), 6),
        "floor_failed": bool(floor_failed),
        "relative_g2_failed": bool(relative_failed),
        "band_verdict": cert.direction_verdict,
        "direction_overlap": round(cert.direction_overlap, 6),
        "direction_separation": round(cert.direction_separation, 6),
        "g4_value_residual": (
            None if cert.g4_value_residual is None else round(float(cert.g4_value_residual), 6)
        ),
        "value_residual_if_forced": round(
            p3.relative_residual(p3.read_value(shadow, write.key), write.value), 6
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
        "occupancy_after_state": int(receipt.state.occupied),
    }
    append_progress(record)
    return record, receipt.state


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    process_guard = assert_no_black_processes("g2c probe")
    lock_name = acquire_single_instance()
    atexit.register(release_single_instance)
    reset_run_logs()
    start_watchdog()
    append_progress({"phase": "start", "runner_sha256": file_sha(Path(__file__).resolve())})
    runner_sha_at_start = file_sha(Path(__file__).resolve())

    p3 = load_p3()
    p3.f1 = p3.load_f1()
    if int(p3.f1.DIM) != WIDTH:
        raise SystemExit(f"fixture width {p3.f1.DIM} != registered control width {WIDTH}")
    p3.base0, p3.tokens0, p3.contexts0 = p3.f1.fixture()
    payloads = p3.f1.value_payloads(112 + 1)
    legacy = gr.EncoderConfig(mode="mean")
    parents = np.stack([gr.encode_read_key(p3.contexts0[i], legacy) for i in range(4)])
    centroid = gr.store_centroid(parents)
    encoder = gr.EncoderConfig(mode="localized_centered", center=True)

    def fresh(rung: dict) -> gg.GateState:
        return p3.build_state(
            rung["occupied"],
            rung["slots"],
            p3.build_contexts(rung["occupied"]),
            payloads,
            encoder,
            centroid,
        )

    # --- phase 1: read-only curve, one reading per rung ----------------------
    curve: list[dict[str, Any]] = []
    for rung in RUNGS:
        cpu_check(f"curve {rung['name']}")
        contexts = p3.build_contexts(rung["occupied"])
        row = measure(fresh(rung), rung, contexts)
        append_progress({"phase": "curve", **row})
        curve.append(row)

    published_match = {
        "published_8_key": 0.068631,
        "measured_8_key": float(curve[0]["median_margin"]),
        "published_64_key": PUBLISHED_64,
        "measured_64_key": float(curve[3]["median_margin"]),
    }
    published_match["delta_8"] = round(
        published_match["measured_8_key"] - published_match["published_8_key"], 6
    )
    published_match["delta_64"] = round(
        published_match["measured_64_key"] - published_match["published_64_key"], 6
    )

    # --- phase 2: forms and constants, landed before any case verdict --------
    forms = build_forms(curve)
    invalid_rungs = [row["rung"] for row in curve if not row["measurable"]]
    curve_validity = {
        "rule": (
            "a rung is measurable only if the fixture supplies `occupied` distinct contexts and "
            "no two occupied keys are byte-identical; the rule is mechanical and was not tuned to "
            "any margin value"
        ),
        "all_rungs_measurable": not invalid_rungs,
        "invalid_rungs": invalid_rungs,
        "mechanism_at_112": (
            "the fixture holds 64 sentences; build_contexts extends beyond 64 by reusing tokens "
            "0..47 at a fixed position, which yields 6 exactly duplicate contexts and 48 "
            "same-token-different-position near twins.  96 of 112 rows then have nearest ~0.96-1.0 "
            "and second ~0.22, so the median (0.742992) describes twin pairing, not crowding.  "
            "This rung was inherited from the P3 fill arm, which only ever tested capacity and "
            "never read a margin."
        ),
        "consequence": (
            "the registered five-rung fit test cannot be completed on this fixture; the fallback "
            "applies, and the next measurement needs a fixture with at least 112 distinct sentences"
        ),
    }
    curve_vs_path = {
        "g2b_first_append_after_8": G2B_FIRST_APPEND,
        "curve_median_at_16": float(curve[1]["median_margin"]),
        "delta": round(float(curve[1]["median_margin"]) - G2B_FIRST_APPEND, 6),
        "reading": (
            "if the equilibrium 16-key margin is well above the post-append 9-key margin, the "
            "curve describes states, not the path, and no occupancy floor covers both"
        ),
    }
    write_json(
        MANIFEST,
        {
            "format": "taiji-gate-g2c-preregistration-v1",
            "written_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "written_before_any_case_verdict": True,
            "sequence": "phase 1 curve (read-only) -> this manifest -> phase 3 case verdicts",
            "question": (
                "is the decay of m_spec with occupancy a shape that can be written as one "
                "normalised floor across all five rungs, and does that floor keep replace/append "
                "admitted and copy refused?"
            ),
            "curve": curve,
            "curve_validity": curve_validity,
            "published_points": published_match,
            "curve_vs_path": curve_vs_path,
            "forms": forms,
            "fit_rule": {
                "fits_when": f"max relative residual over all five rungs <= {FIT_TOL}",
                "five_rung_test_completed": not invalid_rungs,
                "no_piecewise_fit": (
                    "a form is tested once over all five rungs; if it fails, it fails.  Segmenting "
                    "the curve, or picking the least bad form, is not allowed."
                ),
                "fallback": (
                    "if no form fits, the criterion's shape is undetermined and the only permitted "
                    "next step is to measure more rungs"
                ),
            },
            "case_rule": {
                "rung": "8/16, same encoder and fixture",
                "S1_replace": "policy=replace on an occupied slot must remain published",
                "S2_new": "append of a key not already stored must remain published",
                "S3_copy": "append of a byte-identical key must remain refused",
                "any_inversion": "that form is unusable, whatever its residual",
            },
            "discipline": {
                "training_steps": 0,
                "optimizer_steps": 0,
                "adam_steps": 0,
                "confirm_steps": 0,
                "banned_formatter": "python -m black (process_guard at entry)",
                "single_instance_lock": True,
                "watchdog": f"abort after {WATCHDOG_SECONDS:.0f} s without a new reading",
                "cpu_cap_seconds": CPU_CAP,
            },
        },
    )
    manifest_sha = file_sha(MANIFEST)

    # --- phase 3: legacy re-check per form ----------------------------------
    tokens = p3.new_tokens(1)
    legacy_reference = {
        "S1_replace": {"published": True, "source": "P1b admitted, residual 0.000207"},
        "S2_new": {"published": True, "source": "P1b admitted, overlap 0.283"},
        "S3_copy": {"published": False, "source": "P1b refused (no_direction)"},
    }

    def sanity(form: dict) -> dict[str, Any]:
        base = fresh(CASE_RUNG)
        slot = 3
        stored_key = base.keys[slot].copy()
        cases = {
            "S1_replace": gg.WriteCell(
                key=stored_key,
                value=payloads[111],
                family="supersede",
                policy="replace",
            ),
            "S2_new": gg.WriteCell(
                key=gr.encode_read_key(
                    p3.f1.paraphrase(p3.base0, p3.tokens0[tokens[0]], p3.CORRECTION_POSITION),
                    encoder,
                    centroid,
                ),
                value=payloads[112],
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
            cpu_check(f"case {form['name']} {name}")
            record, _ = decide(p3, fresh(CASE_RUNG), write, form, f"{form['name']}_{name}")
            out[name] = {
                "published": record["published"],
                "exit_code": record["exit_code"],
                "failed": record["failed"],
                "floor": record["floor"],
                "spec_margin_before": record["spec_margin_before"],
                "spec_margin_after_forced": record["spec_margin_after_forced"],
                "floor_failed": record["floor_failed"],
                "band_verdict": record["band_verdict"],
                "direction_overlap": record["direction_overlap"],
                "value_residual": record["value_residual_if_forced"],
                "evictions": record["evictions"],
                "quarantine_rows": record["quarantine_rows"],
                "bit_identical": record["bit_identical"],
                "expected_published": legacy_reference[name]["published"],
                "direction_ok": record["published"] == legacy_reference[name]["published"],
            }
        return out

    for form in forms:
        form["cases"] = sanity(form)
        form["case_directions_ok"] = all(row["direction_ok"] for row in form["cases"].values())
        form["adoptable"] = bool(form["fits"] and form["case_directions_ok"])
        measurable = [row for row in form["residuals"] if row["occupied"] != 112]
        form["diagnostic_measurable_rungs"] = {
            "max_relative_residual": round(
                max(item["relative_residual"] for item in measurable), 6
            ),
            "source": (
                "diagnostic only: the registered constant is reused without refitting and rung "
                "112 is excluded because it is not measurable on this fixture; this does not "
                "change the five-rung verdict"
            ),
        }

    # --- divergence between forms, at measured occupancies only -------------
    divergence_rows = []
    for row in curve:
        entry = {"occupied": int(row["occupied"])}
        for form in forms:
            entry[form["name"]] = round(predict(form, row), 6)
        divergence_rows.append(entry)
    names = [form["name"] for form in forms]
    pairs = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            gaps = [
                {
                    "occupied": row["occupied"],
                    "relative_gap": round(
                        abs(row[left] - row[right]) / max(min(row[left], row[right]), 1e-12), 6
                    ),
                }
                for row in divergence_rows
            ]
            worst = max(gaps, key=lambda item: item["relative_gap"])
            pairs.append(
                {
                    "left": left,
                    "right": right,
                    "max_relative_gap": worst["relative_gap"],
                    "at_occupied": worst["occupied"],
                    "gaps": gaps,
                }
            )

    fit_forms = [form["name"] for form in forms if form["fits"]]
    case_ok_forms = [form["name"] for form in forms if form["case_directions_ok"]]
    adoptable = [form["name"] for form in forms if form["adoptable"]]
    worst_measurable = {
        form["name"]: form["diagnostic_measurable_rungs"]["max_relative_residual"] for form in forms
    }
    if not fit_forms:
        decision = "g2c_shape_undetermined:measure_more_rungs"
        if invalid_rungs:
            decision_reason = (
                "the registered five-rung fit test cannot be completed: rung 112 is not measurable "
                "on this fixture (only 64 distinct sentences).  On the four measurable rungs the "
                f"worst registered residuals are {worst_measurable}, so dropping the invalid rung "
                "does not rescue any form either.  The next measurement needs a fixture that can "
                "supply 112 distinct sentences; no form is adopted."
            )
        else:
            decision_reason = (
                "all five rungs are measurable and no single form is within the 20% residual band "
                "across them; the shape is undetermined and the only permitted next step is more "
                "rungs, never a piecewise fit"
            )
        decision_zh = (
            "五档测试没能完成（第 5 档 112 在本夹具上不可测），且四个可测档上没有任何单一形式"
            "能进 20% 残差带；判据形状未定，按预注册只能去测更多档并换更宽的夹具，"
            "不得分段硬拟合、也不得挑最接近的一个"
        )
    elif adoptable:
        decision = "g2c_adoptable:" + ",".join(adoptable)
        decision_reason = (
            "these forms both fit the five rungs within the registered residual band and keep "
            "replace/new admitted and copy refused: " + ", ".join(adoptable)
        )
        decision_zh = (
            "以下形式同时满足“五档内吻合”与“三条遗判不翻转”："
            + "、".join(adoptable)
            + "；若有多个，分歧点见 divergence，不做二选一"
        )
    else:
        decision = "g2c_fits_but_misfires:" + ",".join(fit_forms)
        decision_reason = (
            "these forms fit the curve but invert at least one P1b direction, so they are unusable: "
            + ", ".join(fit_forms)
        )
        decision_zh = (
            "有形式能拟合曲线，但至少翻转了一条 P1b 遗判（replace/新条目/复制），"
            "故该形式不可用：" + "、".join(fit_forms)
        )

    payload: dict[str, Any] = {
        "format": "taiji-gate-g2c-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "manifest": str(MANIFEST.relative_to(ROOT)).replace("\\", "/"),
        "manifest_sha256": manifest_sha,
        "question": (
            "what shape does m_spec follow with occupancy, and can one normalised floor across "
            "all five rungs replace the absolute floor without inverting the legacy verdicts?"
        ),
        "phase_order": [
            "phase 1 curve (read-only, five rungs)",
            "phase 2 forms and constants -> manifest (sha above)",
            "phase 3 legacy re-check per form",
        ],
        "curve": curve,
        "curve_validity": curve_validity,
        "published_points": published_match,
        "curve_vs_path": curve_vs_path,
        "attribution": {
            "width": (
                "width is 128 at every rung (the encoder is fixed), so width cannot explain the "
                "decay inside this fixture; it is a constant, not a variable here"
            ),
            "occupancy_vs_rank": (
                "effective_rank / occupied is reported per rung.  Where that ratio stays near 1, "
                "rank and occupancy are the same variable and A vs B cannot be separated; where "
                "it falls, the store stops gaining dimensions and the two forms must diverge"
            ),
            "loglog_slope_occupancy": [
                {
                    "left_occupied": curve[index]["occupied"],
                    "right_occupied": curve[index + 1]["occupied"],
                    "slope": round(
                        float(
                            np.log(curve[index + 1]["median_margin"])
                            - np.log(curve[index]["median_margin"])
                        )
                        / float(
                            np.log(curve[index + 1]["occupied"]) - np.log(curve[index]["occupied"])
                        ),
                        6,
                    ),
                    "note": "slope -0.5 means the sqrt form; -1 means inverse occupancy",
                }
                for index in range(len(curve) - 1)
            ],
        },
        "forms": forms,
        "divergence": pairs,
        "fit_forms": fit_forms,
        "case_ok_forms": case_ok_forms,
        "adoptable_forms": adoptable,
        "decision": decision,
        "decision_reason": decision_reason,
        "decision_zh": decision_zh,
        "superseded_run": {
            **SUPERSEDED,
            "why": (
                "first run landed the same five readings, forms, constants and decision, but had "
                "no per-rung validity instrumentation, so the 112 rung's twin-pair artefact was "
                "not on the record; this run adds the validity check only"
            ),
        },
        "implementation_corrections": [
            {
                "what": "`os.fsync` on a read-only handle fails on Windows (EBADF) in the first run",
                "when": "phase 2 manifest write, before any case verdict and before the report",
                "effect_on_judgement": "none; the manifest is now fsynced on an r+b handle",
            },
            {
                "what": (
                    "rung 112 was used as a curve point without a validity check; its contexts are "
                    "not 112 distinct sentences"
                ),
                "when": "discovered after the first run's phase 1, before this run's cases",
                "criteria_changed": False,
                "thresholds_changed": False,
                "forms_changed": False,
                "constants_changed": False,
                "decision_changed": False,
                "effect_on_judgement": (
                    "the verdict stays `shape_undetermined:measure_more_rungs`; the reason becomes "
                    "explicit: the five-rung test cannot be completed on this fixture"
                ),
            },
        ],
        "discipline": {
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_steps": 0,
            "model_forward": 0,
            "writes_in_phase_1": 0,
            "gate_commits_in_phase_1": 0,
            "case_writes_attempted": sum(len(form["cases"]) for form in forms),
            "case_writes_published": sum(
                1 for form in forms for row in form["cases"].values() if row["published"]
            ),
            "frozen_files_touched": 0,
        },
        "isolation": {
            "runner_sha256_at_start": runner_sha_at_start,
            "runner_sha256_at_end": file_sha(Path(__file__).resolve()),
            "manifest_sha256": manifest_sha,
            "gate_sha256": file_sha(GATE_FILE),
            "runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "process_guard": process_guard,
            "formatter_path": "scripts/training/black_check_gate.py (in-process); python -m black banned",
            "banned_formatter_invoked_by_this_runner": False,
            "single_instance_lock": lock_name,
            "watchdog_fired": WATCHDOG_NOTE.exists(),
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 6),
            "wall_seconds": round(time.time() - START_WALL, 6),
            "cpu_cap": CPU_CAP,
            "readings_written": sum(
                1 for line in PROGRESS.read_text(encoding="utf-8").splitlines() if line.strip()
            ),
        },
    }
    write_json(REPORT, payload)

    print(
        json.dumps(
            {
                "decision": decision,
                "curve": [
                    {
                        "occupied": row["occupied"],
                        "median": row["median_margin"],
                        "min": row["min_margin"],
                        "eff_rank": row["effective_rank"],
                        "measurable": row["measurable"],
                    }
                    for row in curve
                ],
                "forms": {
                    form["name"]: {
                        "max_relative_residual": form["max_relative_residual"],
                        "fits": form["fits"],
                        "cases": {key: val["published"] for key, val in form["cases"].items()},
                        "adoptable": form["adoptable"],
                    }
                    for form in forms
                },
                "published_points": published_match,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if decision.startswith("g2c_adoptable"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
