"""Build and source-bind the frozen mid-dose transition schedule for N1-165."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import build_taiji_n1_plan163_transition_template_schedule as n163_builder
from scripts.training import run_taiji_n1_plan162b_question_surface_matched_continuation as n162

PLAN = ROOT / "plans/reference/PLAN-N1-165_transition_template_mid_dose_calibration_20261020.md"
PLAN_SHA256 = "62ab1d2568f4141be239dff8e15b2ca6d7d27614c6d7a651e10490a61e636f8c"
REBINDING_PLAN_SHA256 = n163_builder.REBINDING_PLAN_SHA256
PLAN163 = n163_builder.PLAN
PLAN163A = n163_builder.REBINDING_PLAN
MANIFEST163 = n163_builder.MANIFEST
READINESS163 = n163_builder.READINESS
RESULT163 = ROOT / "reports/taiji_n1_plan163_transition_template_rebalanced_result_20261020.json"
N163_RESULT = RESULT163
PROGRESS163 = ROOT / "reports/taiji_n1_plan163_transition_template_rebalanced_progress_20261020.jsonl"
PLAN164 = ROOT / "plans/reference/PLAN-N1-164_plan163_itemwise_pair_attribution_20261020.md"
REPORT164 = ROOT / "reports/taiji_n1_plan163_itemwise_attribution_v2_20261020.json"
AUDITOR164 = ROOT / "scripts/training/audit_taiji_n1_plan163_itemwise_attribution.py"
BUILDER163 = Path(n163_builder.__file__).resolve()
RUNNER163 = ROOT / "scripts/training/run_taiji_n1_plan163_transition_template_rebalanced_continuation.py"
BUILDER = Path(__file__).resolve()
RUNNER = ROOT / "scripts/training/run_taiji_n1_plan165_transition_template_mid_dose.py"
MANIFEST = ROOT / "plans/manifests/n1_plan165_transition_template_mid_dose_schedule_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan165_transition_template_mid_dose_readiness_20261020.json"

SEEDS = (11, 12, 13)
PARENT_SEEDS = SEEDS
PARENT_ARM = n163_builder.PARENT_ARM
N163_CONTROL_ARM = n163_builder.CONTROL_ARM
N163_CANDIDATE_ARM = n163_builder.CANDIDATE_ARM
PRIOR_RESULT = n163_builder.PRIOR_RESULT
PRIOR_RESULT_SHA256 = n163_builder.PRIOR_RESULT_SHA256
PREFERRED_TEMPLATE = n163_builder.PREFERRED_TEMPLATE
REPLACED_TEMPLATE = n163_builder.BAD_TEMPLATE
CANDIDATE_ARM = "preferred_turn_mid_dose"
SLOTS_PER_PASS = 9_216
UPDATES_PER_MODEL = 18_432
GATE_QUERIES_PER_ENDPOINT = 960
LENGTH_QUERIES_PER_ENDPOINT = 288
TOTAL_UPDATES = len(SEEDS) * UPDATES_PER_MODEL
TOTAL_SCORE_QUERIES = len(SEEDS) * (GATE_QUERIES_PER_ENDPOINT + LENGTH_QUERIES_PER_ENDPOINT)
CPU_LIMIT_SECONDS = 900.0
N163_MANIFEST_SHA256 = "fc19b065bde3a2f0cec31c908669f022e0d2ac4152138d1d14395679b1eaa732"
N163_READINESS_SHA256 = "2ed909796d9afea1630eb7da8f10df75bae56a5387f4e9ade8b17bcc431001fa"
N163_RESULT_SHA256 = "942883debd24b21559b8c27f7b2a54945fd22bffeaf7a39e3fef1da1c7669861"
N163_PROGRESS_SHA256 = "32456df6730e8e5591dd5d94fa80040cf3a130a659b9c06d1c2bba017a6d5862"
N164_REPORT_SHA256 = "1f4da75db8b5b0dc4c5fd583d8c60c789f842ea0a64163bd9c67f4b6220471b9"
N164_AUDITOR_SHA256 = "5f3dce6b0704964632d80d19efbf31a80093c74128d721128fe820a8956be279"
N164_PLAN_SHA256 = "e7dc103973a75a12a2adefd2408272e66b7613681c985022ad9abd2fa850dae9"
F1_RESULT_SHA256 = n163_builder.F1_RESULT_SHA256
G2_RESULT_SHA256 = n163_builder.G2_RESULT_SHA256


class Refuse(RuntimeError):
    """N1-165 frozen source or schedule contract is not met."""


def sha256(path: Path) -> str:
    return n162.sha256(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"N1-165 write-once output exists: {path}")
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _parent_order(value: str) -> int:
    try:
        return int(str(value).rsplit("-", 1)[1])
    except (IndexError, ValueError) as exc:
        raise Refuse(f"unexpected N1-165 parent id: {value}") from exc


def derive_mid_dose_rows(base_rows: list[dict]) -> tuple[list[dict], dict]:
    if len(base_rows) != SLOTS_PER_PASS:
        raise Refuse(f"N1-165 base schedule length differs: {len(base_rows)}")
    rows = [{**row, "n165_candidate_template_id": row["augmented_template_id"]} for row in base_rows]

    replaced_groups: dict[tuple[str, str, int], list[int]] = defaultdict(list)
    for index, row in enumerate(base_rows):
        if row["kind"] == "transition" and row["augmented_template_id"] == REPLACED_TEMPLATE:
            group = (str(row["parent_id"]), str(row["route"]), int(row["context_length"]))
            replaced_groups[group].append(index)

    groups_by_route_length: dict[tuple[str, int], list[tuple[str, str, int]]] = defaultdict(list)
    for group, indexes in replaced_groups.items():
        if len(indexes) != 2:
            raise Refuse(f"N1-165 expected two retained-template slots in {group}; got {len(indexes)}")
        groups_by_route_length[(group[1], group[2])].append(group)

    replaced_slots = 0
    retained_slots = 0
    balance_receipts = {}
    for (route, length), groups in sorted(groups_by_route_length.items()):
        ordered = sorted(groups, key=lambda group: _parent_order(group[0]))
        if len(ordered) != 48:
            raise Refuse(f"N1-165 expected 48 balanced groups for {route}/{length}; got {len(ordered)}")
        for group_rank, group in enumerate(ordered):
            indexes = sorted(replaced_groups[group], key=lambda i: int(base_rows[i]["step"]))
            keep_index = group_rank % 2
            for occurrence, row_index in enumerate(indexes):
                if occurrence == keep_index:
                    retained_slots += 1
                    rows[row_index]["n165_candidate_template_id"] = REPLACED_TEMPLATE
                else:
                    replaced_slots += 1
                    rows[row_index]["n165_candidate_template_id"] = PREFERRED_TEMPLATE
        balance_receipts[f"{route}/{length}"] = {
            "groups": len(ordered),
            "retained_variant_slots": len(ordered),
            "replaced_variant_slots": len(ordered),
            "first_parent": ordered[0][0],
            "last_parent": ordered[-1][0],
        }

    if replaced_slots != 288 or retained_slots != 288 or len(replaced_groups) != 288:
        raise Refuse(
            f"N1-165 mid-dose transition replacement differs: replaced={replaced_slots}, retained={retained_slots}, groups={len(replaced_groups)}"
        )

    templates_by_length: dict[int, Counter[str]] = defaultdict(Counter)
    context_counts = Counter()
    for base, row in zip(base_rows, rows, strict=True):
        for key in (
            "step", "parent_id", "route", "kind", "context_length", "item_id", "item_key",
            "second_id", "second_key", "augmented_template_id",
        ):
            if row.get(key) != base.get(key):
                raise Refuse(f"N1-165 changed a frozen base schedule field: {base.get('step')}/{key}")
        if row["kind"] == "transition":
            templates_by_length[int(row["context_length"])][row["n165_candidate_template_id"]] += 1
        context_counts[int(row["context_length"])] += 1

    expected = {
        8: Counter({"original": 384, PREFERRED_TEMPLATE: 288, REPLACED_TEMPLATE: 96}),
        16: Counter({"original": 384, PREFERRED_TEMPLATE: 288, REPLACED_TEMPLATE: 96}),
        32: Counter({"original": 384, PREFERRED_TEMPLATE: 288, REPLACED_TEMPLATE: 96}),
    }
    if dict(templates_by_length) != expected:
        raise Refuse(f"N1-165 transition template counts differ: {dict(templates_by_length)}")
    if context_counts != Counter({8: 3_456, 16: 2_880, 32: 2_880}):
        raise Refuse(f"N1-165 context length balance changed: {dict(context_counts)}")
    return rows, {
        "updates_per_model_per_pass": len(rows),
        "mid_dose_replaced_slots": replaced_slots,
        "mid_dose_retained_variant_slots": retained_slots,
        "balanced_parent_route_length_groups": balance_receipts,
        "transition_template_counts_by_length": {
            str(length): dict(sorted(counts.items()))
            for length, counts in sorted(templates_by_length.items())
        },
        "context_length_counts": {str(k): v for k, v in sorted(context_counts.items())},
    }


def build() -> dict:
    started = time.process_time()
    fixed_hashes = {
        PLAN.resolve(): PLAN_SHA256,
        PLAN163.resolve(): n163_builder.PLAN_SHA256,
        PLAN163A.resolve(): n163_builder.REBINDING_PLAN_SHA256,
        MANIFEST163.resolve(): N163_MANIFEST_SHA256,
        READINESS163.resolve(): N163_READINESS_SHA256,
        RESULT163.resolve(): N163_RESULT_SHA256,
        PLAN164.resolve(): N164_PLAN_SHA256,
        REPORT164.resolve(): N164_REPORT_SHA256,
        AUDITOR164.resolve(): N164_AUDITOR_SHA256,
        BUILDER163.resolve(): sha256(BUILDER163),
        RUNNER163.resolve(): sha256(RUNNER163),
        BUILDER.resolve(): sha256(BUILDER),
        RUNNER.resolve(): sha256(RUNNER),
        PROGRESS163.resolve(): N163_PROGRESS_SHA256,
    }
    for path, expected in fixed_hashes.items():
        if not path.is_file() or sha256(path) != expected:
            raise Refuse(f"N1-165 frozen source SHA differs: {path}")
    if MANIFEST.exists() or READINESS.exists():
        raise FileExistsError("N1-165 schedule/readiness outputs are write-once")

    previous_manifest = read_json(MANIFEST163)
    previous_readiness = read_json(READINESS163)
    previous_result = read_json(RESULT163)
    attribution = read_json(REPORT164)
    if (
        previous_manifest.get("plan_sha256") != n163_builder.PLAN_SHA256
        or previous_readiness.get("status") != "ready"
        or previous_readiness.get("source_hashes_unchanged") is not True
        or previous_result.get("status") != "complete"
        or previous_result.get("complete") is not True
        or previous_result.get("source_hashes_unchanged") is not True
        or previous_result.get("can_promote") is not False
        or attribution.get("status") != "complete"
        or attribution.get("endpoint_gate_recomputed_matches_result") is not True
        or attribution.get("result_sha256") != N163_RESULT_SHA256
        or attribution.get("progress_sha256") != N163_PROGRESS_SHA256
    ):
        raise Refuse("N1-163 result/readiness or N1-164 attribution is not complete and bound")
    if not n162.n150_builder.verify_hashes(previous_result.get("source_hashes", {})):
        raise Refuse("N1-163 historical source/checkpoint hashes no longer verify")

    n163_progress_sha = sha256(PROGRESS163)
    if n163_progress_sha != N163_PROGRESS_SHA256:
        raise Refuse("N1-163 progress file changed after its read-only attribution")

    parents = {}
    for seed in SEEDS:
        old_parent = [row for row in previous_result["models"] if row["seed"] == seed and row["arm"] == n163_builder.CANDIDATE_ARM]
        control = [row for row in previous_result["models"] if row["seed"] == seed and row["arm"] == n163_builder.CONTROL_ARM]
        if len(old_parent) != 1 or len(control) != 1:
            raise Refuse(f"N1-165 requires one N1-163 control and full-dose anchor for seed{seed}")
        if old_parent[0]["parent_checkpoint_sha256"] != control[0]["parent_checkpoint_sha256"]:
            raise Refuse(f"N1-163 dose anchors do not share the same parent for seed{seed}")
        parent_matches = [
            row for row in read_json(n163_builder.PRIOR_RESULT)["models"]
            if row.get("seed") == seed and row.get("arm") == PARENT_ARM
        ]
        if len(parent_matches) != 1 or parent_matches[0]["final_checkpoint_sha256"] != old_parent[0]["parent_checkpoint_sha256"]:
            raise Refuse(f"N1-163 parent checkpoint lineage differs for seed{seed}")
        parent = parent_matches[0]
        parent_path = Path(parent["final_checkpoint_path"]).resolve()
        restore = parent.get("final_independent_restore", {})
        if (
            not parent_path.is_file()
            or sha256(parent_path) != parent["final_checkpoint_sha256"]
            or restore.get("complete") is not True
            or restore.get("equal") is not True
        ):
            raise Refuse(f"N1-162B full parent bundle is not ready for seed{seed}")
        parents[str(seed)] = {
            "path": str(parent_path),
            "sha256": parent["final_checkpoint_sha256"],
            "fingerprint": parent["final_bundle_fingerprint"],
            "history_count": parent["final_history_count"],
            "optimizer_step_values": parent["optimizer_step_values"],
            "restore": restore,
        }

    anchor_endpoints = {}
    for seed in SEEDS:
        anchor_endpoints[str(seed)] = {}
        for arm in (n163_builder.CONTROL_ARM, n163_builder.CANDIDATE_ARM):
            matches = [row for row in previous_result["models"] if row.get("seed") == seed and row.get("arm") == arm]
            if len(matches) != 1:
                raise Refuse(f"N1-163 dose anchor missing or ambiguous: seed{seed}/{arm}")
            anchor = matches[0]
            anchor_path = Path(anchor["final_checkpoint_path"]).resolve()
            restore = anchor.get("final_independent_restore", {})
            if (
                not anchor_path.is_file()
                or sha256(anchor_path) != anchor["final_checkpoint_sha256"]
                or restore.get("complete") is not True
                or restore.get("equal") is not True
            ):
                raise Refuse(f"N1-163 dose anchor full bundle changed: seed{seed}/{arm}")
            anchor_endpoints[str(seed)][arm] = {
                "path": str(anchor_path),
                "sha256": anchor["final_checkpoint_sha256"],
                "fingerprint": anchor["final_bundle_fingerprint"],
                "training_sufficient": anchor["training_sufficient"],
                "original_train_gate": anchor["original_train_gate"],
            }

    base_rows = previous_manifest.get("rows", [])
    if len(base_rows) != SLOTS_PER_PASS:
        raise Refuse("N1-163 source schedule row count differs")
    rows, checks = derive_mid_dose_rows(base_rows)
    rows_sha = n162.builder.canonical_sha(rows)
    source_hashes = dict(previous_readiness["source_hashes"])
    source_hashes.update({str(path): digest for path, digest in fixed_hashes.items()})
    for path, parent in parents.items():
        source_hashes[parent["path"]] = parent["sha256"]
    for seed_anchors in anchor_endpoints.values():
        for anchor in seed_anchors.values():
            source_hashes[anchor["path"]] = anchor["sha256"]

    manifest = {
        "schema": "taiji-n1-plan165-transition-template-mid-dose-schedule-v1",
        "plan_sha256": PLAN_SHA256,
        "n163_plan_sha256": n163_builder.PLAN_SHA256,
        "n163a_plan_sha256": n163_builder.REBINDING_PLAN_SHA256,
        "readiness_rebinding_addendum_sha256": n163_builder.REBINDING_PLAN_SHA256,
        "n163_manifest_sha256": N163_MANIFEST_SHA256,
        "n163_readiness_sha256": N163_READINESS_SHA256,
        "n163_result_sha256": N163_RESULT_SHA256,
        "n163_progress_sha256": N163_PROGRESS_SHA256,
        "n164_report_sha256": N164_REPORT_SHA256,
        "base_schedule_sha256": previous_manifest["base_schedule_sha256"],
        "rows_sha256": rows_sha,
        "seeds": list(SEEDS),
        "parent_arm": PARENT_ARM,
        "arms": [CANDIDATE_ARM],
        "updates_per_model_per_pass": SLOTS_PER_PASS,
        "continuation_passes": 2,
        "updates_per_endpoint": UPDATES_PER_MODEL,
        "total_updates": TOTAL_UPDATES,
        "gate_queries_per_endpoint": GATE_QUERIES_PER_ENDPOINT,
        "length_queries_per_endpoint": LENGTH_QUERIES_PER_ENDPOINT,
        "train_score_queries": TOTAL_SCORE_QUERIES,
        "parent_endpoints": parents,
        "n163_dose_anchors": anchor_endpoints,
        "checks": checks,
        "source_hashes": dict(sorted(source_hashes.items())),
        "can_promote": False,
        "rows": rows,
    }
    if not n162.n150_builder.verify_hashes(source_hashes):
        raise Refuse("N1-165 source/checkpoint hashes changed during schedule construction")
    write_new(MANIFEST, manifest)
    manifest_hash = sha256(MANIFEST)
    source_hashes[str(MANIFEST.resolve())] = manifest_hash
    if not n162.n150_builder.verify_hashes(source_hashes):
        raise Refuse("N1-165 source/checkpoint hashes changed after manifest write")

    cpu = time.process_time() - started
    if cpu > CPU_LIMIT_SECONDS:
        raise TimeoutError(f"N1-165 schedule builder CPU exceeded: {cpu:.3f}/{CPU_LIMIT_SECONDS:.3f}")
    readiness = {
        "schema": "taiji-n1-plan165-transition-template-mid-dose-readiness-v1",
        "status": "ready",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "readiness_rebinding_addendum_sha256": n163_builder.REBINDING_PLAN_SHA256,
        "manifest_sha256": manifest_hash,
        "n163_result_sha256": N163_RESULT_SHA256,
        "n164_report_sha256": N164_REPORT_SHA256,
        "parent_endpoints": parents,
        "n163_dose_anchors": anchor_endpoints,
        "source_hash_count": len(source_hashes),
        "source_hashes": dict(sorted(source_hashes.items())),
        "source_hashes_unchanged": True,
        "checkpoint_loads": 0,
        "model_loads": 0,
        "parameter_updates": 0,
        "model_queries": 0,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "checks": checks,
        "can_promote": False,
        "summary_zh": "N1-165中间剂量schedule/source readiness通过：六个parent/route/length分类的288个平衡块均为4 original/3 preferred/1保留变体；三seed父bundle与N1-163锚点逐SHA匹配。0模型/检查点加载、0更新/查询。",
    }
    write_new(READINESS, readiness)
    return readiness


def main() -> int:
    try:
        report = build()
    except Exception as exc:
        print(f"N1-165 schedule refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"status": report["status"], "source_hash_count": report["source_hash_count"], "cpu_seconds": report["cpu_seconds"], "checks": report["checks"], "can_promote": report["can_promote"]}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
