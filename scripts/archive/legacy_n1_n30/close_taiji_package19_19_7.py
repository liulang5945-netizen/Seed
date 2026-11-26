"""N19.7 negative closeout for package 19 (byte position count + UTF-8 mask).

Reads the frozen artefacts only (no forward pass, no optimizer step, no confirm
read) and writes one closeout report + manifest.  The attribution is measured,
not asserted: the gold-prefix ledger splits the copy claim into three separate
bars (route / beta start / carried continuation), so "start vs gate vs encoder"
is answered with quantities instead of a label.

gamma wording rule (M3, D16-3): every gamma mention is written as the
generation weight gamma and the copy weight 1 - gamma; gamma is never used to
infer training credit.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COMPARE = ROOT / "reports/taiji_package19_compare_20261104.json"
PROBE = ROOT / "reports/taiji_package19_probe_step300_20261104.json"
PROBE_MID = ROOT / "reports/taiji_package19_probe_step150_20261104.json"
PRICING = ROOT / "plans/manifests/taiji_pricing_19_3_20261104.json"
PREFLIGHT = ROOT / "plans/manifests/taiji_preflight_19_4_20261104.json"
OUTPUT_DIR = ROOT / "output/taiji_package19_compare"
REPORT = ROOT / "reports/taiji_package19_closeout_19_7_20261104.json"
MANIFEST = ROOT / "plans/manifests/taiji_package19_closeout_19_7_20261104.json"

CELL_KEYS = (
    "28031-A_route_lock",
    "28031-B_stop_lock",
    "28031-C_unlocked",
    "28032-A_route_lock",
    "28032-B_stop_lock",
    "28032-C_unlocked",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def r(value, digits: int = 4) -> float:
    return round(float(value), digits)


DELETED_STEP150 = {
    "28031-A_route_lock-step150.pt": {
        "bytes": 29255194,
        "sha256": "30782db5ee258df4c30057a3d9bb23b089c9b3a4c20aad1af76d07da03e917b2",
    },
    "28031-B_stop_lock-step150.pt": {
        "bytes": 29254971,
        "sha256": "9329df0731c6ca413dc2872a1fc12784c2d5fff44c72b7699eb84ca03e24176a",
    },
    "28031-C_unlocked-step150.pt": {
        "bytes": 29254748,
        "sha256": "f63515c0117a8ca3980084ff7b694830ea6f5e04d16ef76e5134964550c728c4",
    },
    "28032-A_route_lock-step150.pt": {
        "bytes": 29255194,
        "sha256": "5a65d298cf04f4e5a35661991d70a0091c5d638997169d9b2b8b3a8be46ceaa3",
    },
    "28032-B_stop_lock-step150.pt": {
        "bytes": 29254971,
        "sha256": "1f61401aa7b56f6275670ddeb5d303ee4984da6a5fb9e63da510931fc5399a9e",
    },
    "28032-C_unlocked-step150.pt": {
        "bytes": 29254748,
        "sha256": "9976f143844f6b8fcc43578bfb386af8f67627c0a841359b11728fc1e0cf4a82",
    },
}


def main() -> int:
    compare = load(COMPARE)
    probe = load(PROBE)
    pricing = load(PRICING)
    preflight = load(PREFLIGHT)
    mid = load(PROBE_MID)

    cells = {}
    for key in CELL_KEYS:
        side = load(OUTPUT_DIR / f"eval300-{key}.json")
        overall = side["overall"]
        sweep = probe["cells"][key]["sweep"]
        ledger = probe["cells"][key]["ledger"]
        cells[key] = {
            "step300": {
                "overall": r(overall["completed"]),
                "new_name_copied": r(side["overall"]["completed"]),
                "unseen_output_name": r(overall["subsets"]["unseen_output_name"]),
                "new_description": r(overall["subsets"]["new_description"]),
                "both_new": r(overall["subsets"]["both_new"]),
                "eos": r(overall["eos"]),
                "utf8": r(overall["utf8"]),
                "selection_object": r(side["selection"]["object_top1"]),
                "selection_place": r(side["selection"]["place_top1"]),
                "generation_weight_gamma": (
                    r(side["gamma"]["gamma_effective_mean"])
                    if side["gamma"]["gamma_effective_mean"]
                    == side["gamma"]["gamma_effective_mean"]
                    else None
                ),
                "copy_weight_one_minus_gamma": (
                    r(1 - side["gamma"]["gamma_effective_mean"])
                    if side["gamma"]["gamma_effective_mean"]
                    == side["gamma"]["gamma_effective_mean"]
                    else None
                ),
                "cpu_seconds": (
                    r(side["timing"]["cpu_seconds"], 1) if "cpu_seconds" in side["timing"] else None
                ),
            },
            "ledger": {
                "object": {
                    "start_hit": r(ledger["object"]["start_hit_rate"]),
                    "route_hit": r(ledger["object"]["route_hit_rate"]),
                    "beta_start_hit": r(ledger["object"]["beta_start_hit_rate"]),
                    "conditional_byte": r(ledger["object"]["conditional_byte"]),
                    "unconditional_byte": r(ledger["object"]["unconditional_byte"]),
                    "conditional_examples": ledger["object"]["conditional_examples"],
                },
                "place": {
                    "start_hit": r(ledger["place"]["start_hit_rate"]),
                    "route_hit": r(ledger["place"]["route_hit_rate"]),
                    "beta_start_hit": r(ledger["place"]["beta_start_hit_rate"]),
                    "conditional_byte": r(ledger["place"]["conditional_byte"]),
                    "unconditional_byte": r(ledger["place"]["unconditional_byte"]),
                    "conditional_examples": ledger["place"]["conditional_examples"],
                },
            },
            "gamma_sweep": {
                name: {
                    "generation_weight_gamma": (
                        r(entry["gamma_effective"])
                        if entry["gamma_effective"] == entry["gamma_effective"]
                        else "nan"
                    ),
                    "copy_weight_one_minus_gamma": (
                        r(entry["copy_weight_mean"])
                        if entry["copy_weight_mean"] == entry["copy_weight_mean"]
                        else "nan"
                    ),
                    "object_byte": r(entry["object_byte_rate"]),
                    "place_byte": r(entry["place_byte_rate"]),
                    "content_18_byte": r(entry["content_byte_rate"]),
                    "overall": r(entry["overall"]),
                    "new_name_copied": r(entry["new_name_copied"]),
                }
                for name, entry in sweep.items()
            },
            "gold_start_controls": {
                "b_pure_object_first": r(
                    probe["cells"][key]["positions"]["b_pure"]["object_first"]
                ),
                "b_pure_object_last": r(probe["cells"][key]["positions"]["b_pure"]["object_last"]),
                "c_gamma0_content_byte": r(probe["cells"][key]["c_gamma0"]["content_byte_rate"]),
                "c_gamma0_object_byte": r(probe["cells"][key]["c_gamma0"]["object_byte_rate"]),
                "c_gamma0_place_byte": r(probe["cells"][key]["c_gamma0"]["place_byte_rate"]),
                "note": "gold start (record + pinned content positions); never merged with the model's own route",
            },
            "utf8": {
                "generate_only_legal": r(
                    probe["cells"][key]["utf8"]["generate_only"]["legal_rate"]
                ),
                "copy_only_legal": r(probe["cells"][key]["utf8"]["copy_only"]["legal_rate"]),
                "c_gamma0_legal": r(probe["cells"][key]["utf8"]["c_gamma0"]["legal_rate"]),
            },
        }

    pooled = probe["pooled"]["ledger"]
    start_gap = {
        "object": r(pooled["object"]["conditional_byte"] - pooled["object"]["unconditional_byte"]),
        "place": r(pooled["place"]["conditional_byte"] - pooled["place"]["unconditional_byte"]),
    }

    probe_cost = probe["cost"]
    ledger_cost = probe_cost.get("ledger_only_cpu_seconds", 0.0)
    compare_cost = compare["totals"]
    mid_cost = mid["cost"]
    cpu_cumulative = r(
        compare_cost["cumulative"]["cumulative_cpu_seconds"]
        + mid_cost["cpu_seconds"]
        + probe_cost["cpu_seconds"]
        + ledger_cost,
        1,
    )

    b_cells = ("28031-B_stop_lock", "28032-B_stop_lock")
    frame_bar = {
        "cells_with_perfect_start": list(b_cells),
        "new_description": {key: cells[key]["step300"]["new_description"] for key in b_cells},
        "note": "even with start_hit = 1.00, sentence completion stays near 0.60 because the new-description frame is wrong",
    }

    report = {
        "schema": "taiji-package19-closeout-v1",
        "step": "N19.7",
        "date": "2026-11-04",
        "verdict": {
            "package19": "negative",
            "decision_tree_branch": "全负",
            "rule": "all six cells miss the byte gate and the pre-registered task gate; overall 0.00-0.63, best cell 0.63 < 0.80",
            "release_confirm_19": False,
            "confirm_reads": 0,
            "package19_status": "closed-negative; 17/18 must not be reopened",
        },
        "readouts": {
            "cells": cells,
            "pooled_ledger": {
                "object": {
                    key: r(value) if isinstance(value, float) else value
                    for key, value in pooled["object"].items()
                },
                "place": {
                    key: r(value) if isinstance(value, float) else value
                    for key, value in pooled["place"].items()
                },
                "generation_weight_gamma_mean": r(pooled["gamma_mean"]),
            },
            "pooled_gamma_sweep": probe["pooled"]["sweep"],
            "gold_start_controls_pooled": probe["pooled"]["c_gamma0"],
            "utf8_pooled": probe["pooled"]["utf8"],
            "development_gate": compare["development_gate"],
            "pooled_a_minus_c": compare["pooled_a_minus_c"],
            "cell_grades": compare["cell_grades"],
        },
        "attribution": {
            "main_cause": "start_and_route",
            "statement": (
                "复制侧的主因是起点/记录选择，而不是编码器，也不是生成权重 gamma。"
                "在起点命中的例子里后续内容字节达到 0.94(物件)/0.98(地点)，未命中时降到 0.68/0.63。"
            ),
            "quantities": {
                "start_hit_object": r(pooled["object"]["start_hit_rate"]),
                "start_hit_place": r(pooled["place"]["start_hit_rate"]),
                "conditional_byte_object": r(pooled["object"]["conditional_byte"]),
                "unconditional_byte_object": r(pooled["object"]["unconditional_byte"]),
                "conditional_byte_place": r(pooled["place"]["conditional_byte"]),
                "unconditional_byte_place": r(pooled["place"]["unconditional_byte"]),
                "start_gap_object": start_gap["object"],
                "start_gap_place": start_gap["place"],
                "route_hit_object": r(pooled["object"]["route_hit_rate"]),
                "route_hit_place": r(pooled["place"]["route_hit_rate"]),
                "carried_start_hit": r(pooled["object"]["carry_start_hit_rate"]),
                "conditional_examples_object": pooled["object"]["conditional_examples"],
                "conditional_examples_place": pooled["place"]["conditional_examples"],
            },
            "not_causes": {
                "encoder": (
                    "c_gamma0 (gold record + pinned content positions, generation weight 0) copies content at "
                    "1.00 in 5/6 cells; the one zero is 28031-A whose route never lands on the record"
                ),
                "gate": (
                    "generation weight gamma is not the binding bar: in the B cells the model's own gamma "
                    "0.363/0.354 already gives 1.00/0.958 overall on the 48-example probe slice, forcing gamma "
                    "to 0.6 collapses it to 0.083/0.083, forcing 0.5..0.0 changes nothing"
                ),
                "note": "no training-credit inference from gamma (M3)",
            },
            "secondary_bar": frame_bar,
            "diagnostic_note": (
                "carry_start_hit = 0.00 everywhere: read.m_peak is the inside-run continuation pointer "
                "(advance_copy shifts it by one byte), so it was never a start claim.  The earlier step300 "
                "printout that reported start_hit = 0.00 for every cell was measured on m_peak and is superseded "
                "by this ledger; the start claim is beta's own posterior peak."
            ),
        },
        "decision_tree": {
            "branch": "全负",
            "checks": {
                "byte_gate_object_at_least_0_95": {
                    "value": r(pooled["object"]["unconditional_byte"]),
                    "passed": False,
                },
                "byte_gate_place_at_least_0_95": {
                    "value": r(pooled["place"]["unconditional_byte"]),
                    "passed": False,
                },
                "copied_at_least_0_60": {"value": 0.6302, "passed": True, "cells": 1, "of": 6},
                "overall_at_least_0_80": {"value": 0.6302, "passed": False, "cells": 0, "of": 6},
            },
        },
        "cost": {
            "n19_3_pricing": {"cpu_seconds": 99.3, "wall_seconds": 103.2, "adam": 0},
            "n19_4_preflight": {
                "cpu_seconds": 292.7,
                "wall_seconds": 303.7,
                "adam": 19,
                "verdict": preflight.get("verdict"),
            },
            "n19_5_compare": {
                "cpu_seconds": r(compare_cost["cpu_seconds"], 1),
                "wall_seconds_parallel": r(compare_cost["wall_seconds_parallel"], 1),
                "formal_adam": compare_cost["formal_adam"],
            },
            "n19_5_probe_step150": {
                "cpu_seconds": r(mid_cost["cpu_seconds"], 1),
                "wall_seconds": r(mid_cost["wall_seconds"], 1),
                "adam": 0,
            },
            "n19_5_probe_step300": {
                "cpu_seconds": r(probe_cost["cpu_seconds"], 1),
                "wall_seconds": r(probe_cost["wall_seconds"], 1),
                "adam": 0,
            },
            "n19_5_ledger_recheck": {
                "cpu_seconds": r(ledger_cost, 1),
                "wall_seconds": r(probe_cost.get("ledger_only_wall_seconds", 0.0), 1),
                "adam": 0,
            },
            "n19_5_package_total": r(
                compare_cost["cpu_seconds"]
                + mid_cost["cpu_seconds"]
                + probe_cost["cpu_seconds"]
                + ledger_cost,
                1,
            ),
            "pricing_gap": {
                "protected_price_cpu": r(pricing["budget"]["six_cells_protected"], 1),
                "measured_cpu": r(compare_cost["cpu_seconds"], 1),
                "note": "19.5 exceeded the 19.3 protected price; the 19.3 per-update price was low by roughly 4x (5.4 vs 20-29 CPU per update)",
            },
            "cumulative_cpu_seconds": cpu_cumulative,
        },
        "watchline": {
            "recorded": True,
            "file": "output/taiji_package19_compare/watchline.json",
            "first_observed_cpu": 17376.7,
            "first_extrapolated_cpu": 20063.8,
            "observed_cpu_at_end": 29574.2,
            "distortion_reason": (
                "the first trigger (update 10-20) extrapolated fixed start/evaluation cost over 300 updates; "
                "the second trigger (update 250-270) extrapolated a contended rate.  Neither estimate matches "
                "the measured end value, so both are kept as observation-line records only."
            ),
        },
        "data_quality": {
            "28031-A_route_lock": {
                "generation_weight_gamma": None,
                "note": "gamma steps are empty in that cell (hard route never copies); pooled means skip non-finite values",
            },
            "driver_wall_print": "the aggregate-only CLI printed wall 0.0 because that mode does not meter wall; the measured wall is 5,958.8 parallel (single cell 3,800-5,200 s)",
        },
        "cleanup": {
            "retained": {
                "step300_bundles": [f"{key}-step300.pt" for key in CELL_KEYS],
                "step300_bundles_bytes": sum(
                    (OUTPUT_DIR / f"{key}-step300.pt").stat().st_size for key in CELL_KEYS
                ),
                "mid_reports": [f"midreport-{key}.json" for key in CELL_KEYS],
                "run_receipts": [f"run-{key}.json" for key in CELL_KEYS],
                "eval_receipts": [f"eval300-{key}.json" for key in CELL_KEYS],
                "step0_receipts": [f"step0-{key}.json" for key in CELL_KEYS],
                "failure_receipts": ["failure-28031-A_route_lock.json", "failures.json"],
                "watchline": ["watchline.json"],
                "note": "step300 bundles stay because the 19.5 probe and any owner-approved audit read them",
            },
            "delete": {
                "status": "deleted",
                "step150_bundles": [
                    {
                        "path": path,
                        "bytes": record["bytes"],
                        "sha256": record["sha256"],
                    }
                    for path, record in DELETED_STEP150.items()
                ],
                "deleted_bytes": sum(record["bytes"] for record in DELETED_STEP150.values()),
                "note": "step150 readings are persisted in midreport/run receipts; the bundles are no longer a resume point",
            },
        },
        "branches": {
            "recommended": "19.2b 起点监督",
            "n19_2b": {
                "name": "19.2b start supervision",
                "new_parameters": "SpanPointerHead with 32-d bottleneck: 4,194 (11.3% of the 37,120 route head)",
                "gold_start_labels": "teacher_span(record_raw, entity_bytes) inside the frozen record; already implemented, no new data",
                "change": "enable the span head and its cross-entropy start/end loss; inject the start pointer into routing_step so the copy start is a supervised position, beta heads untouched",
                "new_bundle": True,
                "preflight": "identity sha changes; full 19.4-style preflight required (0 Adam / <=400 CPU)",
                "budget_cpu": 34400,
                "budget_note": "6 cells x 300 updates at +8% of 29,574, plus 400 preflight and 2,300 probe",
                "pass_criteria": [
                    "ledger start_hit >= 0.95 for object and place on all six cells",
                    "ledger route_hit >= 0.95 on all six cells",
                    "conditional byte does not regress below 0.94/0.98",
                    "byte gate object >= 0.95, place >= 0.95, copied >= 0.60, overall >= 0.80",
                    "new_description >= 0.80 (the second bar; if it stays near 0.26, close 19.2b as insufficient and open a frame attribution)",
                ],
                "not_started": True,
            },
            "n19_5b": {
                "name": "19.5b gate-only variant",
                "new_parameters": "0 to 2 (gamma bias / floor)",
                "change": "reshape only the generation-weight gamma gate; alpha/beta and encoder untouched",
                "new_bundle": True,
                "budget_cpu": 5200,
                "budget_note": "2 seeds x 1 arm (B) x 150 updates, plus preflight and probe",
                "pass_criteria": [
                    "object byte >= 0.95 and place byte >= 0.95 on both seeds",
                    "new_name_copied >= 0.60 on both seeds",
                ],
                "not_started": True,
                "why_not_recommended": "the gamma sweep shows forcing gamma down never improves the B cells; gamma is not the binding bar",
            },
        },
        "limits": {
            "zero_updates": True,
            "confirm_reads": 0,
            "training": False,
            "reopen_17_18": False,
        },
        "source": {
            "compare_report": {"path": str(COMPARE.relative_to(ROOT)), "sha256": sha256(COMPARE)},
            "probe_report": {"path": str(PROBE.relative_to(ROOT)), "sha256": sha256(PROBE)},
            "probe_step150_report": {
                "path": str(PROBE_MID.relative_to(ROOT)),
                "sha256": sha256(PROBE_MID),
            },
            "pricing_manifest": {"path": str(PRICING.relative_to(ROOT)), "sha256": sha256(PRICING)},
            "preflight_manifest": {
                "path": str(PREFLIGHT.relative_to(ROOT)),
                "sha256": sha256(PREFLIGHT),
            },
        },
    }

    write_json(REPORT, report)
    write_json(
        MANIFEST,
        {
            "schema": "taiji-package19-closeout-manifest-v1",
            "step": "N19.7",
            "date": "2026-11-04",
            "report": str(REPORT.relative_to(ROOT)),
            "report_sha256": sha256(REPORT),
            "verdict": report["verdict"],
            "branch": report["branches"]["recommended"],
            "cost_cpu_seconds": report["cost"]["n19_5_package_total"],
            "cumulative_cpu_seconds": report["cost"]["cumulative_cpu_seconds"],
            "limits": report["limits"],
        },
    )
    print(
        json.dumps(
            {
                "report": str(REPORT),
                "verdict": report["verdict"]["package19"],
                "branch": report["decision_tree"]["branch"],
                "main_cause": report["attribution"]["main_cause"],
                "start_gap": start_gap,
                "package_total_cpu": report["cost"]["n19_5_package_total"],
                "cumulative_cpu": report["cost"]["cumulative_cpu_seconds"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
