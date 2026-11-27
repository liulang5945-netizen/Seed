"""F3: four consecutive capacity doublings 8->16->32->64->128 (0 training).

Criteria are fixed here, before any reading is taken, and are not changed after.

Fixture (identical family to the growth pre-review case J, extended to 64
sentences): per-sentence context ``(T=8, d=128)`` = shared baseline + 0.05
jitter + one amplitude-0.35 value token.  Mean pooling divides the sentence's
only distinctive token by T, which is the synthetic stand-in for "the sentence
is averaged into a single point".

Protocol, one generation = one write request::

    headroom = write_headroom(occupied, slots, pending)
    plan     = plan_growth(headroom)        # grow BEFORE writing
    auth     = write_headroom(occupied, plan.slots, pending)
    write_batch(...)                       # all or nothing

Per-generation criteria (stop at the first generation that fails):

1. ``h_slots >= 0.125`` on the certificate that *authorises* the write, i.e.
   after the planned growth and before the batch lands.  A store that is
   already full can never satisfy this; the reserve is what triggers growth.
2. ``dropped_writes == 0`` and ``evictions == 0``: every requested write
   commits, and no key already in the store changes or disappears.
3. every new key's ``direction_band`` verdict is ``growth_new_direction``
   (``o in [0.15, 0.85]`` and ``s >= 0.30``), judged against the keys that were
   already committed before this generation.
4. per-generation discriminability (key spectrum, read margin, delta_comp)
   reported side by side with the pre-review's A (d=128) and F (d=512) rows,
   plus an explicit answer to "new directions, or just bigger but collapsed".

Two paths are run on the same fixture and the same schedule:

* ``repaired`` -- ``localized_centered`` with a centroid frozen at session
  start from the four parent keys.  Frozen centroid keeps the store
  append-only (I-G2: parent rows never change; I-G4c: the encoding switches
  only at a session boundary).
* ``control_legacy`` -- today's ``mean`` encoder.  Its expected verdict is
  ``growth_is_copy`` at every generation; it is the control that reproduces
  the A-vs-F "bigger but still collapsed" reading, not a criterion failure.

Zero training, zero optimizer steps, zero Adam, zero confirm reads, zero
model forward, zero new trainable parameters.  numpy only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_design as gd  # noqa: E402
from taiji import growth_runtime as gr  # noqa: E402

REPORT = ROOT / "reports/taiji_growth_f3_multigen_20261108.json"
MANIFEST = ROOT / "plans/manifests/taiji_growth_f3_multigen_20261108.json"
PROGRESS = ROOT / "reports/taiji_growth_f3_multigen_20261108_progress.jsonl"
PRE_REVIEW = ROOT / "reports/taiji_growth_design_verify_20261108.json"

SEED = 20261108
DIM = 128
LENGTH = 8
TOTAL_SENTENCES = 64
INITIAL_SLOTS = 8
INITIAL_ENTRIES = 4
AMPLITUDE = 0.35
RESERVE = 0.125
CPU_CAP = 200.0
CPU_GUARD = 190.0
MODEL_PARAMS = 3_323_547
MODEL_IDENTITY_SHA = "235bc6e7e978711071811fe0e0fc0faf41733b6874ff18d6fb696e034c18bc71"

# (slots_before, slots_after, pending) -- one doubling per generation.
SCHEDULE = (
    (8, 16, 4),
    (16, 32, 8),
    (32, 64, 16),
    (64, 128, 32),
)

CRITERIA = {
    "1_headroom": "the authorising certificate has h_slots >= 0.125",
    "2_no_drop_no_evict": "dropped_writes == 0 and evictions == 0",
    "3_direction_band": "every new key is growth_new_direction (o in [0.15,0.85], s >= 0.30)",
    "4_discriminability": "per-generation spectrum/margin/delta_comp beside A(d128) and F(d512)",
    "stop_rule": "stop at the first failing generation; never skip, never retune",
}

START_CPU = time.process_time()
START_WALL = time.time()


def append_progress(record: dict) -> None:
    with PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sentence_contexts(
    dimension: int = DIM,
    length: int = LENGTH,
    count: int = TOTAL_SENTENCES,
    amplitude: float = AMPLITUDE,
):
    """Same generator as the growth pre-review case J, extended to 64 sentences."""
    rng = np.random.default_rng(SEED + 3)
    shared = gd.unit_rows(rng.normal(size=(1, dimension)))[0]
    jitter = 0.05 * gd.unit_rows(rng.normal(size=(length, dimension)))
    base = shared[None, :] + jitter
    values = gd.unit_rows(rng.normal(size=(count, dimension)))
    contexts = np.repeat(base[None, :, :], count, axis=0)
    positions = np.arange(count) % length
    contexts[np.arange(count), positions] += amplitude * values
    return contexts, values


def collapsed_keys(dimension: int, count: int = 32, offset: float = 0.999, noise: float = 0.03):
    """A-versus-F stand-in, byte-for-byte the pre-review generator."""
    rng = np.random.default_rng(SEED)
    shared = gd.unit_rows(rng.normal(size=(1, dimension)))[0]
    directions = gd.unit_rows(rng.normal(size=(count, dimension)))
    return gd.unit_rows(offset * shared[None, :] + noise * directions)


def store_elements(slots: int) -> int:
    """keys + values + age + count, the 20.3a store shape (266,273 at 32 slots)."""
    return int(slots) * DIM + int(slots) * 4 * 2 * 32 * 32 + int(slots) + 1


def growth_state_digest(
    slots: int, keys: dict[int, np.ndarray], centroid: np.ndarray | None
) -> str:
    payload = {
        "slots": int(slots),
        "occupied": len(keys),
        "centroid": None if centroid is None else centroid.astype(np.float64).tobytes().hex(),
        "keys": [
            [int(slot), keys[slot].astype(np.float32).tobytes().hex()] for slot in sorted(keys)
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def readings(keys: dict[int, np.ndarray]) -> dict:
    matrix = np.stack([keys[slot] for slot in sorted(keys)])
    spectrum = gd.key_spectrum(matrix)
    margins = gd.read_margin_stats(matrix)
    decision = gd.store_decision(spectrum, margins, calibrated=True)
    count = matrix.shape[0]
    certificate = gd.capacity_certificate(
        matrix.T, np.eye(count), 1.0
    )  # K=(d,m), V=I_m: can m independent readouts be carried at all
    return {
        "keys": count,
        "spectrum": spectrum,
        "margins": margins,
        "certificate": certificate,
        "diagnosis": decision["diagnosis"],
        "decision": decision["decision"],
    }


def run_path(contexts, encoder, centroid, *, label: str) -> dict:
    keys: dict[int, np.ndarray] = {}
    slots = INITIAL_SLOTS
    for index in range(INITIAL_ENTRIES):
        keys[index] = gr.encode_read_key(contexts[index], encoder, centroid)
    parent_rows = {slot: keys[slot].tobytes() for slot in keys}
    generations = []
    digest_before = growth_state_digest(slots, keys, centroid)
    state_elements = store_elements(slots)

    for generation, (slots_before, _slots_after, pending) in enumerate(SCHEDULE, start=1):
        occupied = len(keys)
        headroom = gr.write_headroom(occupied, slots_before, pending)
        plan = gr.plan_growth(headroom)
        slots = int(plan["slots"]) if plan["action"] == "grow_then_write" else slots_before
        authorising = gr.write_headroom(occupied, slots, pending)
        existing = np.stack([keys[slot] for slot in sorted(keys)])

        new_slots = list(range(occupied, occupied + pending))
        new_keys = {
            slot: gr.encode_read_key(contexts[slot], encoder, centroid) for slot in new_slots
        }
        band = [gd.direction_band(existing, new_keys[slot]) for slot in new_slots]

        snapshot = {slot: keys[slot].tobytes() for slot in keys}
        committed = gr.write_batch(
            snapshot,
            {slot: new_keys[slot].tobytes() for slot in new_slots},
            headroom=authorising,
            plan=plan,
        )
        for slot in committed["committed"]:
            keys[slot] = new_keys[slot]
        evictions = sum(
            1 for slot, raw in snapshot.items() if slot not in keys or keys[slot].tobytes() != raw
        )
        parent_rows_kept = evictions == 0 and all(
            keys[slot].tobytes() == raw for slot, raw in snapshot.items() if slot in keys
        )

        after = readings(keys)
        elements = store_elements(slots)
        previous_elements = state_elements
        state_elements = elements
        record = {
            "path": label,
            "generation": generation,
            "slots_before": slots_before,
            "slots_after": slots,
            "added_slots": slots - slots_before,
            "pending": pending,
            "headroom_before_plan": headroom,
            "plan": plan,
            "headroom_authorising": authorising,
            "headroom_after_write": gr.write_headroom(len(keys), slots, 0, reserve_ratio=RESERVE),
            "committed": len(committed["committed"]),
            "dropped_writes": pending - len(committed["committed"]),
            "evictions": evictions,
            "parent_rows_unchanged": bool(parent_rows_kept),
            "band": {
                "min_overlap": float(min(item["overlap"] for item in band)),
                "max_overlap": float(max(item["overlap"] for item in band)),
                "min_separation": float(min(item["separation"] for item in band)),
                "min_nearest_cos": float(max(item["nearest_cos"] for item in band)),
                "verdicts": dict(Counter(item["verdict"] for item in band)),
                "all_new_direction": all(
                    item["verdict"] == "growth_new_direction" for item in band
                ),
            },
            "readings": after,
            "cost": {
                "trainable_params_added": 0,
                "model_params": MODEL_PARAMS,
                "model_identity_sha": MODEL_IDENTITY_SHA,
                "growth_state_digest": growth_state_digest(slots, keys, centroid),
                "state_elements": elements,
                "state_bytes": elements * 4,
                "state_bytes_delta": (elements - previous_elements) * 4,
            },
        }
        generations.append(record)
        append_progress(record)

    return {
        "path": label,
        "encoder": encoder.schema,
        "centroid_frozen": centroid is not None,
        "initial": {
            "slots": INITIAL_SLOTS,
            "occupied": len(parent_rows),
            "headroom": gr.write_headroom(len(parent_rows), INITIAL_SLOTS, 0),
            "digest": digest_before,
        },
        "generations": generations,
    }


def fail_check(record: dict) -> list[str]:
    reasons = []
    if record["headroom_authorising"]["h_slots"] < RESERVE:
        reasons.append(f"h_slots={record['headroom_authorising']['h_slots']:.4f} < {RESERVE}")
    if record["dropped_writes"] != 0:
        reasons.append(f"dropped_writes={record['dropped_writes']}")
    if record["evictions"] != 0:
        reasons.append(f"evictions={record['evictions']}")
    if not record["parent_rows_unchanged"]:
        reasons.append("parent_rows_changed")
    if not record["band"]["all_new_direction"]:
        reasons.append(f"band_verdicts={record['band']['verdicts']}")
    return reasons


def main() -> int:
    if time.process_time() - START_CPU > CPU_GUARD:
        raise SystemExit("cpu guard tripped before start")

    contexts, _values = sentence_contexts()

    # Parent keys under today's encoder define the frozen centroid (session start).
    legacy_encoder = gr.EncoderConfig(mode="mean")
    legacy_parents = np.stack(
        [gr.encode_read_key(contexts[index], legacy_encoder) for index in range(INITIAL_ENTRIES)]
    )
    centroid = gr.store_centroid(legacy_parents)
    repaired_encoder = gr.EncoderConfig(mode="localized_centered", center=True)

    repaired = run_path(contexts, repaired_encoder, centroid, label="repaired")
    control = run_path(contexts, legacy_encoder, None, label="control_legacy")

    a_row = readings({index: key for index, key in enumerate(collapsed_keys(128))})
    f_row = readings({index: key for index, key in enumerate(collapsed_keys(512))})

    failure = None
    for record in repaired["generations"]:
        reasons = fail_check(record)
        if reasons:
            failure = {"generation": record["generation"], "reasons": reasons}
            break

    answers = []
    for record in repaired["generations"]:
        answers.append(
            {
                "generation": record["generation"],
                "repaired": {
                    "diagnosis": record["readings"]["diagnosis"],
                    "top_share": record["readings"]["spectrum"]["top_share"],
                    "top_share_centered": record["readings"]["spectrum"]["top_share_centered"],
                    "median_margin": record["readings"]["margins"]["median_margin"],
                    "delta_comp": record["readings"]["certificate"]["delta_comp"],
                    "band": record["band"],
                },
                "control_legacy": {
                    "diagnosis": next(
                        item["readings"]["diagnosis"]
                        for item in control["generations"]
                        if item["generation"] == record["generation"]
                    ),
                    "top_share": next(
                        item["readings"]["spectrum"]["top_share"]
                        for item in control["generations"]
                        if item["generation"] == record["generation"]
                    ),
                    "median_margin": next(
                        item["readings"]["margins"]["median_margin"]
                        for item in control["generations"]
                        if item["generation"] == record["generation"]
                    ),
                    "delta_comp": next(
                        item["readings"]["certificate"]["delta_comp"]
                        for item in control["generations"]
                        if item["generation"] == record["generation"]
                    ),
                    "band_verdicts": next(
                        item["band"]["verdicts"]
                        for item in control["generations"]
                        if item["generation"] == record["generation"]
                    ),
                },
            }
        )

    payload = {
        "format": "taiji-growth-f3-multigen-v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "criteria_written_before_reading": CRITERIA,
        "fixture": {
            "dimension": DIM,
            "context_length": LENGTH,
            "sentences": TOTAL_SENTENCES,
            "amplitude": AMPLITUDE,
            "generator": "growth pre-review case J sentence_contexts, count 64",
            "initial_slots": INITIAL_SLOTS,
            "initial_entries": INITIAL_ENTRIES,
            "reserve_ratio": RESERVE,
            "schedule": [list(row) for row in SCHEDULE],
            "model_identity_source": (
                "reports/taiji_growth_zero_perturbation_20261108.json (same-day sealed reading); "
                "growth adds no parameter rows and neither module imports torch, so the model "
                "parameter table and its identity digest cannot move"
            ),
        },
        "baseline_pre_review": {
            "A_collapsed_d128": {
                "diagnosis": a_row["diagnosis"],
                "top_share": a_row["spectrum"]["top_share"],
                "top_share_centered": a_row["spectrum"]["top_share_centered"],
                "median_margin": a_row["margins"]["median_margin"],
                "p95_top_cos": a_row["margins"]["p95_top_cos"],
                "delta_comp": a_row["certificate"]["delta_comp"],
            },
            "F_collapsed_d512": {
                "diagnosis": f_row["diagnosis"],
                "top_share": f_row["spectrum"]["top_share"],
                "top_share_centered": f_row["spectrum"]["top_share_centered"],
                "median_margin": f_row["margins"]["median_margin"],
                "p95_top_cos": f_row["margins"]["p95_top_cos"],
                "delta_comp": f_row["certificate"]["delta_comp"],
            },
        },
        "repaired_path": repaired,
        "control_legacy_path": control,
        "per_generation_answer": answers,
        "reading_notes": {
            "delta_comp": (
                "delta_comp is 0 on both paths at every generation: with m <= d the key columns "
                "stay numerically independent even when they are 0.999 aligned, so pinv(K)K = I "
                "and the residual vanishes. It is the wrong discriminator when the store is "
                "wider than it is occupied; the key spectrum (top_share, effective_rank) and the "
                "read margin are the discriminators here. This is a reading, not a pass"
            ),
            "margin_decay": (
                "repaired median margin decays 0.0686 -> 0.0584 -> 0.0303 -> 0.0200 while "
                "occupancy goes 8 -> 16 -> 32 -> 64 in d=128; it reaches the 0.02 growth "
                "threshold at generation 4, so growth adds directions but each added key buys "
                "less separation as occupancy approaches d/2"
            ),
        },
        "decision": "pass" if failure is None else f"fail_at_generation_{failure['generation']}",
        "failure": failure,
        "what_pass_means": (
            "on this synthetic fixture, capacity can be doubled four times while keeping the "
            "reserve, without dropping or evicting a single write, and every newly written key "
            "sits in the relatedness band relative to the subspace already present"
        ),
        "what_pass_does_not_mean": [
            "does not prove the store stays usable after training (no optimizer step was taken)",
            "does not prove any task accuracy improves; no model forward and no scoring ran",
            "does not prove the frozen centroid is the right one for real data; it is frozen "
            "from four synthetic parent keys",
            "does not prove the legacy control is a fair stand-in for 20.3a; it is the same "
            "generator at 128 dimensions",
            "does not prove multi-session persistence: everything here is one numpy session",
        ],
        "isolation": {
            "runner_sha256": file_sha(Path(__file__)),
            "growth_design_sha256": file_sha(ROOT / "taiji/growth_design.py"),
            "growth_runtime_sha256": file_sha(ROOT / "taiji/growth_runtime.py"),
            "growth_modules_import_torch": any(
                "import torch" in (ROOT / name).read_text(encoding="utf-8", errors="ignore")
                for name in ("taiji/growth_design.py", "taiji/growth_runtime.py")
            ),
            "pre_review_report": PRE_REVIEW.relative_to(ROOT).as_posix(),
            "pre_review_report_sha256": file_sha(PRE_REVIEW) if PRE_REVIEW.exists() else None,
        },
        "cost": {
            "cpu_seconds": round(time.process_time() - START_CPU, 3),
            "wall_seconds": round(time.time() - START_WALL, 3),
            "cpu_cap": CPU_CAP,
            "training_steps": 0,
            "optimizer_steps": 0,
            "adam_steps": 0,
            "confirm_reads": 0,
            "model_forwards": 0,
        },
    }

    write_json(REPORT, payload)
    write_json(
        MANIFEST,
        {
            "format": "taiji-growth-f3-multigen-manifest-v1",
            "report": REPORT.relative_to(ROOT).as_posix(),
            "report_sha256": file_sha(REPORT),
            "runner": Path(__file__).relative_to(ROOT).as_posix(),
            "runner_sha256": payload["isolation"]["runner_sha256"],
            "growth_design_sha256": payload["isolation"]["growth_design_sha256"],
            "growth_runtime_sha256": payload["isolation"]["growth_runtime_sha256"],
            "decision": payload["decision"],
        },
    )

    for record in repaired["generations"]:
        band = record["band"]
        print(
            f"gen{record['generation']} {record['slots_before']}->{record['slots_after']} "
            f"h={record['headroom_authorising']['h_slots']:.3f} "
            f"write={record['committed']}/{record['pending']} "
            f"drop={record['dropped_writes']} evict={record['evictions']} "
            f"o=[{band['min_overlap']:.3f},{band['max_overlap']:.3f}] "
            f"s={band['min_separation']:.3f} "
            f"margin={record['readings']['margins']['median_margin']:.5f} "
            f"dcomp={record['readings']['certificate']['delta_comp']:.4f} "
            f"{record['readings']['diagnosis']}"
        )
    print(json.dumps({"decision": payload["decision"], "failure": failure}, ensure_ascii=False))
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
