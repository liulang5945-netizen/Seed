"""Build the frozen matched transition-template schedule for N1-163."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import run_taiji_n1_plan162b_question_surface_matched_continuation as n162

PLAN = ROOT / "plans/reference/PLAN-N1-163_transition_template_rebalanced_matched_continuation_20261020.md"
PLAN_SHA256 = "c13eb8b2bb318f95de72e3813a669aa12a6d7321751a48a455d2a69769d65459"
REBINDING_PLAN = ROOT / "plans/reference/PLAN-N1-163A_readiness_source_rebinding_20261020.md"
REBINDING_PLAN_SHA256 = "40a68ccc9ff1fc057d4b88afea7c48a9129d12ce013f1b7caba09ccdd74deefc"
PRIOR_RESULT = ROOT / "reports/taiji_n1_plan162b_question_surface_result_20261020.json"
PRIOR_RESULT_SHA256 = "19dfaeb2086abaa503ee5150dba3bbfc9756d0892e8b6b54222fbfc4ecba2c31"
F1_RESULT = ROOT / "reports/taiji_n1_plan162f_transcript_integrity_20261020.json"
F1_RESULT_SHA256 = "e90341d6d8e93de7f8027322c10dc1b61b6ea2caa81531ae0b3f6915bcdebbc7"
G2_RESULT = ROOT / "reports/taiji_n1_plan162g2_mixed_context_thread_scaling_20261020.json"
G2_RESULT_SHA256 = "c9fc96c19b4a6aa9c6207018e36f6069eaa77f9400a1ba244c03f78dc9c162b1"
F1_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-162F1_execution_closeout_20261020.md"
G2_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-162G2_execution_closeout_20261020.md"
N162B_RUNNER = Path(n162.__file__).resolve()
BUILDER = ROOT / "scripts/training/build_taiji_n1_plan163_transition_template_schedule.py"
RUNNER = ROOT / "scripts/training/run_taiji_n1_plan163_transition_template_rebalanced_continuation.py"
TEST = ROOT / "tests/taiji_native/test_n1_plan163_transition_template_rebalanced_contract.py"
BASE_SCHEDULE = ROOT / "plans/manifests/n1_plan162_v34_question_surface_schedule_20261020.json"
SUPERSEDED_MANIFEST = ROOT / "plans/manifests/n1_plan163_transition_template_rebalanced_schedule_20261020.json"
SUPERSEDED_MANIFEST_SHA256 = "231d98e8a40457a30933d930940b868f4ce6615f316520065562bcd843b6371f"
SUPERSEDED_READINESS = ROOT / "reports/taiji_n1_plan163_transition_template_rebalanced_readiness_20261020.json"
SUPERSEDED_READINESS_SHA256 = "e43db7afd228a16f5eaf00f9046627816fb536324f36a3b462ec8eb128932813"
SUPERSEDED_BUILDER_SHA256 = "39e7944651fe71c8b9056677f85455e628ad35fd650218e21f36c287379b350d"
MANIFEST = ROOT / "plans/manifests/n1_plan163a_transition_template_rebalanced_schedule_20261020.json"
READINESS = ROOT / "reports/taiji_n1_plan163a_transition_template_rebalanced_readiness_20261020.json"
PARENT_SEEDS = (11, 12, 13)
PARENT_ARM = "surface_augmented"
CONTROL_ARM = "surface_replay_control"
CANDIDATE_ARM = "preferred_turn_replacement"
BAD_TEMPLATE = "direction_after_turning"
PREFERRED_TEMPLATE = "direction_after_turn"
BUILD_CPU_LIMIT_SECONDS = 900.0


class Refuse(RuntimeError):
    """N1-163来源、schedule或来源门不满足。"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"N1-163 write-once output exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


def derive_arm_rows(base_rows: list[dict]) -> list[dict]:
    rows = []
    for base in base_rows:
        template = str(base["augmented_template_id"])
        candidate = template
        if base["kind"] == "transition" and template == BAD_TEMPLATE:
            candidate = PREFERRED_TEMPLATE
        rows.append({
            **base,
            "n163_control_template_id": template,
            "n163_candidate_template_id": candidate,
        })
    return rows


def template_counts(rows: list[dict], key: str) -> dict[str, int]:
    counter = Counter((str(row["kind"]), int(row["context_length"]), str(row[key])) for row in rows)
    return {f"{kind}/{length}/{template}": count for (kind, length, template), count in sorted(counter.items())}


def validate_derived_rows(base_rows: list[dict], rows: list[dict]) -> dict:
    if len(base_rows) != len(rows) or len(rows) != 9_216:
        raise Refuse("N1-163 schedule denominator differs from N1-162B")
    replacements = 0
    for base, row in zip(base_rows, rows, strict=True):
        for key in ("step", "parent_id", "route", "kind", "context_length", "item_id", "item_key", "second_id", "second_key"):
            if row.get(key) != base.get(key):
                raise Refuse(f"N1-163 changed a frozen schedule field at step {base.get('step')}: {key}")
        if row["n163_control_template_id"] != base["augmented_template_id"]:
            raise Refuse("N1-163 control question assignment differs from N1-162B")
        expected_candidate = base["augmented_template_id"]
        if base["kind"] == "transition" and expected_candidate == BAD_TEMPLATE:
            expected_candidate = PREFERRED_TEMPLATE
            replacements += 1
        if row["n163_candidate_template_id"] != expected_candidate:
            raise Refuse(f"N1-163 candidate changed an unregistered template at step {base.get('step')}")
    if replacements != 576:
        raise Refuse(f"N1-163 must replace exactly 576 transition slots, got {replacements}")

    parent_counts = Counter(str(row["parent_id"]) for row in rows)
    route_counts = Counter((str(row["parent_id"]), str(row["route"])) for row in rows)
    if len(parent_counts) != 192 or set(parent_counts.values()) != {48} or set(route_counts.values()) != {24}:
        raise Refuse("N1-163 per-parent/raw-hard schedule balance differs")
    control = template_counts(rows, "n163_control_template_id")
    candidate = template_counts(rows, "n163_candidate_template_id")
    for length in (8, 16, 32):
        expected_control = {
            f"transition/{length}/original": 384,
            f"transition/{length}/direction_after_turn": 192,
            f"transition/{length}/direction_after_turning": 192,
        }
        expected_candidate = {
            f"transition/{length}/original": 384,
            f"transition/{length}/direction_after_turn": 384,
        }
        for key, expected in expected_control.items():
            if control.get(key) != expected:
                raise Refuse(f"N1-163 control transition ratio differs: {key}={control.get(key)}")
        for key, expected in expected_candidate.items():
            if candidate.get(key) != expected:
                raise Refuse(f"N1-163 candidate transition ratio differs: {key}={candidate.get(key)}")
        if candidate.get(f"transition/{length}/direction_after_turning", 0) != 0:
            raise Refuse("N1-163 candidate retains the replaced transition template")

    lengths = Counter(int(row["context_length"]) for row in rows)
    if lengths != Counter({8: 3_456, 16: 2_880, 32: 2_880}):
        raise Refuse("N1-163 context length denominator differs")
    return {
        "updates_per_model": len(rows),
        "parent_count": 192,
        "replaced_transition_slots": replacements,
        "context_length_counts": {str(k): v for k, v in sorted(lengths.items())},
        "control_template_counts": control,
        "candidate_template_counts": candidate,
    }


def build() -> dict:
    started = time.process_time()
    if sha256(PLAN) != PLAN_SHA256:
        raise Refuse("PLAN-N1-163 SHA mismatch")
    if sha256(REBINDING_PLAN) != REBINDING_PLAN_SHA256:
        raise Refuse("PLAN-N1-163A source rebinding addendum SHA mismatch")
    if sha256(SUPERSEDED_MANIFEST) != SUPERSEDED_MANIFEST_SHA256 or sha256(SUPERSEDED_READINESS) != SUPERSEDED_READINESS_SHA256:
        raise Refuse("superseded N1-163 schedule/readiness SHA mismatch")
    if sha256(PRIOR_RESULT) != PRIOR_RESULT_SHA256:
        raise Refuse("N1-162B result SHA mismatch")
    if sha256(F1_RESULT) != F1_RESULT_SHA256 or sha256(G2_RESULT) != G2_RESULT_SHA256:
        raise Refuse("N1-162F1/G2 evidence SHA mismatch")
    if any(path.exists() for path in (MANIFEST, READINESS)):
        raise FileExistsError("N1-163 manifest/readiness outputs are write-once")

    prior = read_json(PRIOR_RESULT)
    if prior.get("complete") is not True or prior.get("training_updates") != 55_296 or prior.get("source_hashes_unchanged") is not True:
        raise Refuse("N1-162B parent result is incomplete or drifted")
    f1 = read_json(F1_RESULT)
    g2 = read_json(G2_RESULT)
    if f1.get("complete") is not True or g2.get("complete") is not True or g2.get("stable_recommendation_thread_count") != 1:
        raise Refuse("N1-162F1/G2 evidence is incomplete or contradicts single-thread plan")
    superseded_manifest = read_json(SUPERSEDED_MANIFEST)
    superseded_readiness = read_json(SUPERSEDED_READINESS)
    if (
        superseded_manifest.get("plan_sha256") != PLAN_SHA256
        or superseded_readiness.get("status") != "ready"
        or superseded_readiness.get("complete") is not True
        or superseded_readiness.get("source_hashes_unchanged") is not True
        or superseded_readiness.get("checkpoint_loads") != 0
        or superseded_readiness.get("parameter_updates") != 0
        or superseded_readiness.get("model_queries") != 0
    ):
        raise Refuse("superseded N1-163 readiness does not prove a read-only schedule build")

    readiness, zero_preflight, runtime, sources = n162._verify_readiness()
    base_schedule = n162.builder.build_schedule(runtime["n155_schedule"])
    n162.builder.validate_schedule(read_json(BASE_SCHEDULE))
    rows = derive_arm_rows(base_schedule["rows"])
    checks = validate_derived_rows(base_schedule["rows"], rows)
    if (
        n162.builder.canonical_sha(base_schedule["rows"]) != superseded_manifest.get("base_schedule_sha256")
        or n162.builder.canonical_sha(rows) != superseded_manifest.get("rows_sha256")
        or rows != superseded_manifest.get("rows")
    ):
        raise Refuse("N1-163A rebuilt schedule differs from the preserved prior manifest")

    parent_endpoints = {}
    source_hashes = dict(sources)
    for seed in PARENT_SEEDS:
        matches = [row for row in prior["models"] if row.get("seed") == seed and row.get("arm") == PARENT_ARM]
        if len(matches) != 1:
            raise Refuse(f"N1-163 parent endpoint ambiguous: seed{seed}/{PARENT_ARM}")
        parent = matches[0]
        restore = parent.get("final_independent_restore", {})
        path = Path(parent["final_checkpoint_path"]).resolve()
        if (
            not path.is_file()
            or sha256(path) != parent.get("final_checkpoint_sha256")
            or restore.get("complete") is not True
            or restore.get("equal") is not True
            or parent.get("final_history_count") != 79_872
            or parent.get("optimizer_step_values") != [61_440]
        ):
            raise Refuse(f"N1-163 parent full bundle is not ready: seed{seed}")
        source_hashes[str(path)] = parent["final_checkpoint_sha256"]
        parent_endpoints[str(seed)] = {
            "path": str(path),
            "sha256": parent["final_checkpoint_sha256"],
            "fingerprint": parent["final_bundle_fingerprint"],
            "history_count": parent["final_history_count"],
            "optimizer_step_values": parent["optimizer_step_values"],
            "restore": restore,
        }

    fixed_sources = {
        str(PLAN.resolve()): PLAN_SHA256,
        str(REBINDING_PLAN.resolve()): REBINDING_PLAN_SHA256,
        str(SUPERSEDED_MANIFEST.resolve()): SUPERSEDED_MANIFEST_SHA256,
        str(SUPERSEDED_READINESS.resolve()): SUPERSEDED_READINESS_SHA256,
        str(PRIOR_RESULT.resolve()): PRIOR_RESULT_SHA256,
        str(F1_RESULT.resolve()): F1_RESULT_SHA256,
        str(G2_RESULT.resolve()): G2_RESULT_SHA256,
        str(F1_CLOSEOUT.resolve()): sha256(F1_CLOSEOUT),
        str(G2_CLOSEOUT.resolve()): sha256(G2_CLOSEOUT),
        str(N162B_RUNNER): sha256(N162B_RUNNER),
        str(BUILDER.resolve()): sha256(BUILDER),
        str(RUNNER.resolve()): sha256(RUNNER),
        str(TEST.resolve()): sha256(TEST),
        str(BASE_SCHEDULE.resolve()): sha256(BASE_SCHEDULE),
    }
    source_hashes.update(fixed_sources)
    if not n162.n150_builder.verify_hashes(source_hashes):
        raise Refuse("N1-163 source/checkpoint SHA changed during schedule build")

    manifest = {
        "schema": "taiji-n1-plan163-transition-template-rebalanced-schedule-v1",
        "plan_sha256": PLAN_SHA256,
        "readiness_rebinding_addendum_sha256": REBINDING_PLAN_SHA256,
        "supersedes_manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
        "supersedes_readiness_sha256": SUPERSEDED_READINESS_SHA256,
        "parent_result_sha256": PRIOR_RESULT_SHA256,
        "f1_evidence_sha256": F1_RESULT_SHA256,
        "g2_evidence_sha256": G2_RESULT_SHA256,
        "base_schedule_sha256": n162.builder.canonical_sha(base_schedule["rows"]),
        "rows_sha256": n162.builder.canonical_sha(rows),
        "seeds": list(PARENT_SEEDS),
        "parent_arm": PARENT_ARM,
        "arms": [CONTROL_ARM, CANDIDATE_ARM],
        "updates_per_model": 9_216,
        "continuation_passes": 2,
        "updates_per_endpoint": 18_432,
        "total_updates": 110_592,
        "gate_queries_per_endpoint": 960,
        "length_queries_per_endpoint": 288,
        "train_score_queries": 7_488,
        "parent_endpoints": parent_endpoints,
        "checks": checks,
        "rows": rows,
        "can_promote": False,
    }
    write_new(MANIFEST, manifest)
    source_hashes[str(MANIFEST.resolve())] = sha256(MANIFEST)
    if not n162.n150_builder.verify_hashes(source_hashes):
        raise Refuse("N1-163 source/SHA changed after manifest write")
    cpu = time.process_time() - started
    if cpu > BUILD_CPU_LIMIT_SECONDS:
        raise TimeoutError(f"N1-163 schedule build CPU exceeded: {cpu:.3f}/{BUILD_CPU_LIMIT_SECONDS:.3f}")
    report = {
        "schema": "taiji-n1-plan163-transition-template-readiness-v1",
        "status": "ready",
        "complete": True,
        "plan_sha256": PLAN_SHA256,
        "readiness_rebinding_addendum_sha256": REBINDING_PLAN_SHA256,
        "superseded_builder_sha256": SUPERSEDED_BUILDER_SHA256,
        "supersedes_manifest_sha256": SUPERSEDED_MANIFEST_SHA256,
        "supersedes_readiness_sha256": SUPERSEDED_READINESS_SHA256,
        "n162b_result_sha256": PRIOR_RESULT_SHA256,
        "manifest_sha256": sha256(MANIFEST),
        "parent_endpoints": parent_endpoints,
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
        "cpu_limit_seconds": BUILD_CPU_LIMIT_SECONDS,
        "checks": checks,
        "zero_preflight_sha256": sha256(n162.ZERO_PREFLIGHT),
        "can_promote": False,
        "summary_zh": f"N1-163A来源重绑完成：新旧schedule逐行相同；三枚N1-162B surface_augmented parent完整bundle及N1-162B零态前置核验通过；候选只替换576个transition模板槽，0模型/检查点加载、0更新/查询，CPU{cpu:.3f}秒。",
    }
    write_new(READINESS, report)
    return report


def main() -> int:
    try:
        report = build()
    except Exception as exc:
        print(f"N1-163 schedule refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({key: report[key] for key in ("status", "source_hash_count", "cpu_seconds", "checks", "can_promote")}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
