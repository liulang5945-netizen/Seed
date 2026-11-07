"""Constructive pre-review of the growth decision rule (0 training, numpy only).

Goal, exactly as the owner asked: show that the trigger fires on a known
key-space collapse and does *not* fire when capacity is sufficient, and that
"capacity short" and "structurally inseparable" are told apart.  Fixtures are
synthetic and seeded; the only real-world reading replayed is the recorded
20.3a aggregate (top cosine 0.99766, margin 0.00835).

No torch, no model, no checkpoint, no data set, no optimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from taiji import growth_design as gd  # noqa: E402
from taiji import growth_runtime as gr  # noqa: E402

REPORT = ROOT / "reports/taiji_growth_design_verify_20261108.json"
SEED = 20261108
MARGIN_MIN = gd.THRESHOLDS["margin"]
FROZEN_MANIFEST = ROOT / "plans/manifests/taiji_frozen_readset_20261108.json"


def collapsed_keys(dimension: int, count: int = 32, offset: float = 0.999, noise: float = 0.03):
    """Mean-pooled-context stand-in: one shared direction plus a little noise."""
    rng = np.random.default_rng(SEED)
    shared = gd.unit_rows(rng.normal(size=(1, dimension)))[0]
    directions = gd.unit_rows(rng.normal(size=(count, dimension)))
    return gd.unit_rows(offset * shared[None, :] + noise * directions)


def separable_keys(dimension: int, count: int = 32):
    rng = np.random.default_rng(SEED + 1)
    return gd.unit_rows(rng.normal(size=(count, dimension)))


def capacity_case(*, rho_target: float = 6.0, inconsistent: float = 0.0):
    """Rank-deficient key set (m > rank) so an inconsistent target is possible."""
    rng = np.random.default_rng(SEED + 2)
    K = rng.normal(size=(16, 4)) @ rng.normal(size=(4, 8))
    X0 = rng.normal(size=(16, 4))
    pinv = np.linalg.pinv(K)
    X0 *= rho_target / np.linalg.norm(X0.T @ K @ pinv)
    V = X0.T @ K
    if inconsistent:
        junk = rng.normal(size=V.shape)
        junk -= junk @ pinv @ K
        junk *= inconsistent / np.linalg.norm(junk)
        V = V + junk
    return K, V


def sentence_contexts(
    dimension: int = 128,
    length: int = 8,
    count: int = 32,
    amplitude: float = 0.35,
):
    """Per-sentence context (T,d): shared per-position baseline + one value token.

    Mean pooling divides the sentence-specific token by ``T``; this is the
    synthetic analogue of "the sentence is averaged into one point".
    """
    rng = np.random.default_rng(SEED + 3)
    shared = gd.unit_rows(rng.normal(size=(1, dimension)))[0]
    jitter = 0.05 * gd.unit_rows(rng.normal(size=(length, dimension)))
    base = shared[None, :] + jitter
    values = gd.unit_rows(rng.normal(size=(count, dimension)))
    contexts = np.repeat(base[None, :, :], count, axis=0)
    positions = np.arange(count) % length
    contexts[np.arange(count), positions] += amplitude * values
    return contexts, values


def encoder_variants(contexts: np.ndarray) -> dict[str, dict]:
    configs = {
        "mean_legacy": gr.EncoderConfig(mode="mean"),
        "mean_centered": gr.EncoderConfig(mode="mean", center=True),
        "localized": gr.EncoderConfig(mode="localized_centered"),
        "localized_centered": gr.EncoderConfig(mode="localized_centered", center=True),
    }
    readings = {}
    for name, cfg in configs.items():
        plain = np.stack(
            [
                gr.encode_read_key(
                    context, gr.EncoderConfig(mode=cfg.mode, tau=cfg.tau, top_k=cfg.top_k)
                )
                for context in contexts
            ]
        )
        centroid = gr.store_centroid(plain) if cfg.center else None
        keys = np.stack([gr.encode_read_key(context, cfg, centroid) for context in contexts])
        margin = gd.read_margin_stats(keys)
        spectrum = gd.key_spectrum(keys)
        readings[name] = {
            "median_margin": margin["median_margin"],
            "p95_top_cos": margin["p95_top_cos"],
            "top_share": spectrum["top_share"],
            "top_share_centered": spectrum["top_share_centered"],
        }
    return readings


def frozen_modules() -> set[str]:
    if not FROZEN_MANIFEST.exists():
        return set()
    payload = json.loads(FROZEN_MANIFEST.read_text(encoding="utf-8"))
    entries = payload.get("files") or payload.get("readset") or []
    names = set()
    for entry in entries:
        path = entry.get("path") if isinstance(entry, dict) else entry
        if path:
            names.add(str(path).replace("\\", "/"))
    return names


def case(name: str, purpose: str, decision: dict, expected: dict) -> dict:
    checks = {key: decision.get(key) == value for key, value in expected.items()}
    return {
        "name": name,
        "purpose": purpose,
        "decision": decision["decision"],
        "diagnosis": decision["diagnosis"],
        "expected": expected,
        "checks": checks,
        "passed": all(checks.values()),
        "readings": decision.get("evidence", {}),
    }


def main() -> int:
    started, started_cpu = time.time(), time.process_time()
    cases = []

    spectrum_a = gd.key_spectrum(collapsed_keys(128))
    margins_a = gd.read_margin_stats(collapsed_keys(128))
    decision_a = gd.store_decision(spectrum_a, margins_a, calibrated=True)
    cases.append(
        case(
            "A_shared_offset_collapse_d128",
            "known collapse: one shared direction dominates the key space",
            decision_a,
            {"collapse_detected": True, "decision": "representation_repair"},
        )
    )

    spectrum_b = gd.key_spectrum(separable_keys(128))
    margins_b = gd.read_margin_stats(separable_keys(128))
    decision_b = gd.store_decision(spectrum_b, margins_b, calibrated=True)
    cases.append(
        case(
            "B_separable_d128",
            "known sufficient case: random near-orthogonal keys",
            decision_b,
            {"collapse_detected": False, "decision": "no_growth"},
        )
    )

    spectrum_f = gd.key_spectrum(collapsed_keys(512))
    margins_f = gd.read_margin_stats(collapsed_keys(512))
    decision_f = gd.store_decision(spectrum_f, margins_f, calibrated=True)
    cases.append(
        case(
            "F_same_collapse_d512",
            "dimension growth does not change an angular collapse",
            decision_f,
            {"collapse_detected": True, "decision": decision_a["decision"]},
        )
    )

    replay = {
        "p95_top_cos": 0.9976629252533259,
        "median_margin": 0.008352757617859585,
    }
    decision_g = gd.store_decision(None, replay, calibrated=True)
    cases.append(
        case(
            "G_replay_20_3a_recorded_reading",
            "recorded 20.3a aggregate: collapse detected, attribution withheld",
            decision_g,
            {"collapse_detected": True, "decision": "no_growth"},
        )
    )

    K, V = capacity_case()
    cert_c = gd.capacity_certificate(K, V, 4.0)
    decision_c = gd.operator_decision(cert_c)
    cases.append(
        case(
            "C_capacity_short_consistent",
            "consistent target, ball too small",
            decision_c,
            {"decision": "capacity_growth"},
        )
    )

    K_d, V_d = capacity_case(inconsistent=0.5)
    cert_d = gd.capacity_certificate(K_d, V_d, 4.0)
    decision_d = gd.operator_decision(cert_d)
    cases.append(
        case(
            "D_structurally_inseparable",
            "target outside the key row space: growth cannot help",
            decision_d,
            {"decision": "structural_repair"},
        )
    )

    cert_e = gd.capacity_certificate(K, V, 10.0)
    decision_e = gd.operator_decision(cert_e)
    cases.append(
        case(
            "E_capacity_sufficient",
            "same target, ball large enough",
            decision_e,
            {"decision": "no_growth"},
        )
    )

    allocation = gd.allocate(
        {"object": [0.0, 0.50, 0.52, 0.53], "record": [0.0, 0.20, 0.25, 0.26]},
        {"object": 3, "record": 3},
        r_max=4,
    )
    cases.append(
        {
            "name": "H_allocation_not_even",
            "purpose": "max-min allocation never splits units evenly by default",
            "decision": json.dumps(allocation, sort_keys=True),
            "diagnosis": None,
            "expected": {},
            "checks": {},
            "passed": allocation.get("object") != allocation.get("record"),
            "readings": {"allocation": allocation},
        }
    )

    contexts, _ = sentence_contexts()
    variants = encoder_variants(contexts)
    encoder_checks = {
        "mean_legacy_collapsed": variants["mean_legacy"]["median_margin"] < MARGIN_MIN,
        "localized_centered_separates": variants["localized_centered"]["median_margin"]
        >= MARGIN_MIN,
        "localized_centered_beats_legacy": variants["localized_centered"]["median_margin"]
        > variants["mean_legacy"]["median_margin"],
    }
    cases.append(
        {
            "name": "J_encoder_localized_centered",
            "purpose": "the encoder keeps per-sentence differences instead of averaging them",
            "decision": (
                "encoding_repair" if all(encoder_checks.values()) else "encoding_insufficient"
            ),
            "diagnosis": "localized_centered" if all(encoder_checks.values()) else "centering_only",
            "expected": {},
            "checks": encoder_checks,
            "passed": all(encoder_checks.values()),
            "readings": variants,
        }
    )

    band_old = gd.unit_rows(np.random.default_rng(SEED + 4).normal(size=(8, 128)))
    band_basis = np.linalg.svd(band_old, full_matrices=False)[2][:8]
    spread = gd.unit_rows(band_old.sum(axis=0, keepdims=True))[0]
    perp = np.random.default_rng(SEED + 5).normal(size=128)
    perp -= (band_basis @ perp) @ band_basis
    perp = gd.unit_rows(perp[None, :])[0]
    band_copy = gd.direction_band(band_old, band_old[0])
    band_isolated = gd.direction_band(band_old, perp)
    band_new = gd.direction_band(band_old, 0.55 * spread + math.sqrt(1 - 0.55**2) * perp)
    band_checks = {
        "copy_verdict": band_copy["verdict"] == "growth_is_copy",
        "isolated_verdict": band_isolated["verdict"] == "growth_isolated",
        "new_verdict": band_new["verdict"] == "growth_new_direction",
        "copy_not_capacity": band_copy["counts_as_capacity"] is False,
        "isolated_not_capacity": band_isolated["counts_as_capacity"] is False,
    }
    cases.append(
        {
            "name": "I_direction_band",
            "purpose": "partly overlapping but distinguishable: copy and isolated both rejected",
            "decision": "band_ok" if all(band_checks.values()) else "band_mismatch",
            "diagnosis": (band_copy["verdict"], band_isolated["verdict"], band_new["verdict"]),
            "expected": {},
            "checks": band_checks,
            "passed": all(band_checks.values()),
            "readings": {"copy": band_copy, "isolated": band_isolated, "new": band_new},
        }
    )

    headroom_short = gr.write_headroom(30, 32, 3)
    plan_short = gr.plan_growth(headroom_short)
    batch = {30: b"k30", 31: b"k31", 32: b"k32"}
    write_short = gr.write_batch({}, batch, headroom=headroom_short, plan=plan_short)
    headroom_ok = gr.write_headroom(10, 32, 2)
    plan_ok = gr.plan_growth(headroom_ok)
    write_ok = gr.write_batch({}, {10: b"k10", 11: b"k11"}, headroom=headroom_ok, plan=plan_ok)
    capped = gr.plan_growth(headroom_short, slot_cap=32)
    write_capped = gr.write_batch({}, batch, headroom=headroom_short, plan=capped)
    big_batch = {slot: b"v" for slot in range(40)}
    failed_growth = {
        "action": "grow_then_write",
        "slots": 32,
        "added": 0,
        "reason": "simulated_growth_failure",
        "needed": 70,
    }
    write_failed = gr.write_batch({}, big_batch, headroom=headroom_short, plan=failed_growth)
    capacity_checks = {
        "short_grows_first": plan_short["action"] == "grow_then_write"
        and plan_short["slots"] == 64,
        "short_write_commits": len(write_short["committed"]) == 3,
        "ok_writes_now": plan_ok["action"] == "write_now" and len(write_ok["committed"]) == 2,
        "capped_rejects": capped["action"] == "reject_write" and write_capped["committed"] == [],
        "rollback_keeps_store": write_failed["rolled_back"] is True
        and write_failed["store"] == {}
        and write_failed["committed"] == [],
    }
    cases.append(
        {
            "name": "K_capacity_first_protocol",
            "purpose": "grow before writing; a capped or overfull batch writes nothing",
            "decision": (
                "capacity_first_ok" if all(capacity_checks.values()) else "protocol_mismatch"
            ),
            "diagnosis": (plan_short["action"], plan_ok["action"], capped["action"]),
            "expected": {},
            "checks": capacity_checks,
            "passed": all(capacity_checks.values()),
            "readings": {
                "headroom_short": headroom_short,
                "plan_short": plan_short,
                "headroom_ok": headroom_ok,
                "capped": capped,
                "rollback": {
                    key: write_failed[key] for key in ("committed", "reason", "rolled_back")
                },
            },
        }
    )

    frozen = frozen_modules()
    mentions = []
    runner_mentions = []
    for path in sorted(frozen):
        target = ROOT / path
        if target.suffix != ".py" or not target.exists():
            continue
        text = target.read_text(encoding="utf-8", errors="ignore")
        if "growth_runtime" in text or "growth_design" in text:
            mentions.append(path)
    runner_path = ROOT / "scripts/training/run_taiji_training_window_v2_20261108.py"
    if runner_path.exists():
        runner_text = runner_path.read_text(encoding="utf-8", errors="ignore")
        runner_mentions = [
            name for name in ("growth_runtime", "growth_design") if name in runner_text
        ]
    isolation_checks = {
        "frozen_set_has_25": len(frozen) == 25,
        "no_frozen_file_imports_it": mentions == [],
        "runner_does_not_import_it": runner_mentions == [],
    }
    cases.append(
        {
            "name": "L_import_isolation",
            "purpose": "the new modules stay outside the running window's import closure",
            "decision": "isolated" if all(isolation_checks.values()) else "leaked",
            "diagnosis": (len(frozen), mentions, runner_mentions),
            "expected": {},
            "checks": isolation_checks,
            "passed": all(isolation_checks.values()),
            "readings": {
                "frozen_files": len(frozen),
                "mentions": mentions,
                "runner_mentions": runner_mentions,
            },
        }
    )

    payload = {
        "format": "taiji-growth-design-verify-v2",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "thresholds": gd.THRESHOLDS,
        "cases": cases,
        "all_expected_met": all(item["passed"] for item in cases),
        "slot_growth_bytes_32_to_64": gd.slot_growth_bytes(32),
        "cpu_seconds": round(time.process_time() - started_cpu, 3),
        "wall_seconds": round(time.time() - started, 3),
        "boundary": (
            "synthetic fixtures only; the 20.3a replay uses its recorded aggregate "
            "cosine/margin, not a replayed key tensor, so attribution stays withheld"
        ),
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    tmp = REPORT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(REPORT)
    for item in cases:
        print(
            f"{item['name']:<34} passed={item['passed']!s:<5} "
            f"decision={item['decision']:<22} diagnosis={item['diagnosis']}"
        )
    print(
        json.dumps(
            {
                "all_expected_met": payload["all_expected_met"],
                "cpu_seconds": payload["cpu_seconds"],
                "wall_seconds": payload["wall_seconds"],
                "report": REPORT.relative_to(ROOT).as_posix(),
            },
            ensure_ascii=False,
        )
    )
    return 0 if payload["all_expected_met"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
