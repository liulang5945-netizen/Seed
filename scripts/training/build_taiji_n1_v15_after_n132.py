"""为已批准的PLAN-N1-133构建v15新确认面并完成零查询readiness。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.training import (  # noqa: E402
    build_taiji_n1_v14_after_n123 as v14_builder,
)
from scripts.training import (
    eval_taiji_n1_v11_after_n120 as n120_eval,
)
from scripts.training import (
    run_taiji_n1_expanded_normalized_rwl_length_balanced as n120,
)
from scripts.training import (
    run_taiji_n1_plan132_binding_supervision_calibration as n132,
)
from scripts.training import seal_ast  # noqa: E402

PLAN = ROOT / "plans/reference/PLAN-N1-133_v15_binding_supervision_confirmation_draft_20261019.md"
PLAN_SHA256 = "f885ee89e485fc042d9394bd0659714d118134625853afa4bbfb64643d5ba6f2"
SOURCE_UNION_PRECHECK = ROOT / "reports/taiji_n1_plan133_source_union_precheck_v2_20261019.json"
SOURCE_UNION_PRECHECK_SHA256 = "0423908cfd613d5819c49a4d7e0424e0a4a0ce199e414791e648e7adb940a333"
ENDPOINT_PRECHECK = (
    ROOT / "reports/taiji_n1_plan133_pre_authorization_endpoint_integrity_20261019.json"
)
ENDPOINT_PRECHECK_SHA256 = "c3cbb771017180fe76730670d6dba9b6d8cbd0e8c3241527571d657e1113162d"

N124_PLAN = (
    ROOT / "plans/reference/PLAN-N1-124_v14_full_context_evidence_position_probe_draft_20261019.md"
)
N124_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-124_execution_closeout_20261019.md"
N124_SOURCE_EXCLUSIONS = ROOT / "plans/manifests/n1_v14_after_n123_source_exclusions_20261019.json"
N124_READINESS = ROOT / "reports/taiji_n1_v14_after_n123_confirmation_readiness_20261019.json"
N124_FACE = ROOT / "plans/manifests/n1_hard_neighbor_v14_confirmation_seed114_20261019.json"
N124_FACE_ATTEMPT = N124_FACE.with_name(N124_FACE.stem + "_attempt.json")
N124_FACE_CHECKS = ROOT / "reports/taiji_n1_v14_after_n123_face_checks_20261019.json"
N124_RANGE = ROOT / "reports/taiji_n1_v14_after_n123_range_20261019.json"
N124_RANGE_ATTEMPT = ROOT / "reports/taiji_n1_v14_after_n123_range_20261019_attempt.json"
N124_CONFIRM = ROOT / "reports/taiji_n1_v14_after_n123_confirmation_20261019.json"
N124_CONFIRM_ATTEMPT = ROOT / "reports/taiji_n1_v14_after_n123_confirmation_20261019_attempt.json"
N124_ITEMWISE = ROOT / "reports/taiji_n1_v14_after_n123_itemwise_attribution_v2_20261019.json"
N13_N14_ATTRIBUTION = ROOT / "reports/taiji_n1_v13_v14_cross_face_position_effect_v2_20261019.json"
EXISTING_BINDING_AUDIT = ROOT / "reports/taiji_n1_existing_binding_error_audit_20261019.json"

N120_PLAN = (
    ROOT / "plans/reference/PLAN-N1-120_expanded_normalized_rwl_radius_training_draft_20261019.md"
)
N120_RESULT = ROOT / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019.json"
N120_REVALIDATION = (
    ROOT
    / "reports/taiji_n1_expanded_normalized_rwl_length_balanced_20261019_receipt_revalidation.json"
)
N132_PLAN = (
    ROOT
    / "plans/reference/PLAN-N1-132_trainable_question_binding_supervision_calibration_draft_20261019.md"
)
N132_CLOSEOUT = ROOT / "plans/reference/PLAN-N1-132_execution_closeout_20261019.md"
N132_RESULT = ROOT / "reports/taiji_n1_plan132_binding_supervision_calibration_result_20261019.json"
N132_RECONCILIATION = ROOT / "reports/taiji_n1_plan132_provenance_reconciliation_20261019.json"

INITIAL_SOURCE_EXCLUSIONS = (
    ROOT / "plans/manifests/n1_v15_after_n132_source_exclusions_20261019.json"
)
INITIAL_SOURCE_EXCLUSIONS_SHA256 = (
    "066676483c72e23c7e287107523e71431d40590b1288a51caf17e248783fe5a4"
)
INITIAL_READINESS = ROOT / "reports/taiji_n1_v15_after_n132_confirmation_readiness_20261019.json"
INITIAL_READINESS_SHA256 = "31d5b07d7076de343ddb73cf1d09620ceb577b7cb8312a430e12fa1394777140"
INITIAL_READINESS_ATTEMPT = (
    ROOT / "reports/taiji_n1_v15_after_n132_confirmation_readiness_attempt_20261019.json"
)
INITIAL_READINESS_ATTEMPT_SHA256 = (
    "539808df918e7a9ab2fa05c3d17b8352be14d99f33aad8b33db96603ebc7222d"
)
SECOND_SOURCE_EXCLUSIONS = (
    ROOT / "plans/manifests/n1_v15_after_n132_source_exclusions_revalidation_20261019.json"
)
SECOND_SOURCE_EXCLUSIONS_SHA256 = "acb8e92248735287cce3710e4f3576cfcebb21a0fae0c808c0e6d5940f708645"
SECOND_READINESS = (
    ROOT / "reports/taiji_n1_v15_after_n132_confirmation_readiness_revalidation_20261019.json"
)
SECOND_READINESS_SHA256 = "f51c96d651e1f66f7420cc70cace315a33b8ac69f17ca4f26a2a6dec3d1d79a3"
SECOND_READINESS_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_v15_after_n132_confirmation_readiness_revalidation_attempt_20261019.json"
)
SECOND_READINESS_ATTEMPT_SHA256 = "9cc9c518506870bae5a108be077807247fc6bd6b06d3204abb97311b0325094d"
SOURCE_EXCLUSIONS = (
    ROOT / "plans/manifests/n1_v15_after_n132_source_exclusions_revalidation2_20261019.json"
)
READINESS = (
    ROOT / "reports/taiji_n1_v15_after_n132_confirmation_readiness_revalidation2_20261019.json"
)
READINESS_ATTEMPT = (
    ROOT
    / "reports/taiji_n1_v15_after_n132_confirmation_readiness_revalidation2_attempt_20261019.json"
)
FACE = ROOT / "plans/manifests/n1_hard_neighbor_v15_confirmation_seed115_20261019.json"
FACE_ATTEMPT = FACE.with_name(FACE.stem + "_attempt.json")
FACE_CHECKS = ROOT / "reports/taiji_n1_v15_after_n132_face_checks_20261019.json"
RANGE_REPORT = ROOT / "reports/taiji_n1_v15_after_n132_range_20261019.json"
RANGE_ATTEMPT = ROOT / "reports/taiji_n1_v15_after_n132_range_20261019_attempt.json"
CONFIRM_REPORT = ROOT / "reports/taiji_n1_v15_after_n132_confirmation_20261019.json"
CONFIRM_ATTEMPT = ROOT / "reports/taiji_n1_v15_after_n132_confirmation_20261019_attempt.json"

SEED = 115
ITEMS = 40
ITEMS_PER_KIND = 10
OBSERVATIONS = 16
DISTANCE = 1
KINDS = ("lookup", "revision", "transition", "unknown")
EXPECTED_EXCLUDED_ENTITIES = 7308
EXPECTED_EXCLUDED_WORDS = 5808
CPU_LIMIT_SECONDS = 1200
TARGET_STARTS = {
    "lookup": (0, 4, 8, 12, 15),
    "revision": (0, 4, 7, 11, 14),
    "transition": (0, 4, 7, 11, 14),
    "unknown": (),
}

WRITE_ONCE_OUTPUTS = (
    SOURCE_EXCLUSIONS,
    READINESS,
    READINESS_ATTEMPT,
    FACE,
    FACE_ATTEMPT,
    FACE_CHECKS,
    RANGE_REPORT,
    RANGE_ATTEMPT,
    CONFIRM_REPORT,
    CONFIRM_ATTEMPT,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_new(path: Path, value: dict) -> None:
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()


def _add_source(source_hashes: dict[str, str], path: Path, expected: str | None = None) -> None:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"冻结来源缺失：{path}")
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"冻结来源哈希改变：{path}")
    prior = source_hashes.get(str(path))
    if prior is not None and prior != actual:
        raise ValueError(f"来源哈希冲突：{path}")
    source_hashes[str(path)] = actual


def _prior_sources_and_union() -> tuple[set[str], set[str], dict[str, str], dict]:
    if sha256(PLAN) != PLAN_SHA256:
        raise ValueError("PLAN-N1-133冻结稿哈希不匹配")
    precheck = read_json(SOURCE_UNION_PRECHECK)
    if (
        sha256(SOURCE_UNION_PRECHECK) != SOURCE_UNION_PRECHECK_SHA256
        or precheck.get("plan_sha256") != PLAN_SHA256
        or precheck.get("status") != "complete"
        or precheck.get("expected_v15_exclusion_union")
        != {"entity_names": EXPECTED_EXCLUDED_ENTITIES, "word_values": EXPECTED_EXCLUDED_WORDS}
        or precheck.get("new_face_generated") is not False
        or precheck.get("new_model_queries") != 0
    ):
        raise ValueError("v15来源并集预检报告与冻结计划不符")
    endpoint_precheck = read_json(ENDPOINT_PRECHECK)
    if (
        sha256(ENDPOINT_PRECHECK) != ENDPOINT_PRECHECK_SHA256
        or endpoint_precheck.get("status") != "complete"
        or endpoint_precheck.get("endpoints_hash_verified") != 12
        or endpoint_precheck.get("new_face_generated") is not False
        or endpoint_precheck.get("new_model_queries") != 0
    ):
        raise ValueError("v15终点授权前哈希核验失效")

    old = read_json(N124_SOURCE_EXCLUSIONS)
    face = read_json(N124_FACE)
    checks = read_json(N124_FACE_CHECKS)
    range_report = read_json(N124_RANGE)
    confirmation = read_json(N124_CONFIRM)
    if (
        old.get("schema") != "n1-plan124-v14-source-exclusions-v1"
        or old.get("status") != "ready_for_approved_execution"
        or old.get("entity_count") != 6678
        or old.get("word_value_count") != 5328
        or face.get("schema") != "n1-hard-neighbor-v14-confirmation-v1"
        or face.get("seed") != 114
        or face.get("items") != 40
        or face.get("observations_per_item") != 16
        or face.get("name_neighbor_distance") != 1
        or checks.get("accepted") is not True
        or not seal_ast.receipt_binding_matches(checks.get("face_sha256"), N124_FACE)
        or range_report.get("complete") is not True
        or range_report.get("model_queries") != 180
        or range_report.get("confirmation_queries") != 0
        or range_report.get("range_gate", {}).get("range_check_passed") is not True
        or confirmation.get("complete") is not True
        or confirmation.get("confirmation_queries") != 360
        or confirmation.get("total_model_queries") != 540
        or confirmation.get("parameter_updates") != 0
        or confirmation.get("dev_queries") != 0
        or confirmation.get("can_promote") is not False
    ):
        raise ValueError("N1-124的已消费v14面或执行回执不完整")

    excluded_names = set(old.get("excluded_entities", []))
    excluded_words = set(old.get("excluded_word_values", []))
    face_names, face_materials, face_answers = v14_builder._face_entities_and_words(face)
    v14_answer_overlap = len(excluded_words & face_answers)
    if len(face_names) != 630 or len(face_materials) != 480 or len(face_answers) != 25:
        raise ValueError("v14面计数与结案来源账本不一致")
    if excluded_names & face_names or excluded_words & face_materials:
        raise ValueError("已消费v14面实体名或观测材料与旧排除集重叠")
    if len(excluded_words & face_answers) != 5:
        raise ValueError("v14面通用答案词与已排除词值的重叠不符")
    excluded_names.update(face_names)
    excluded_words.update(face_materials)
    excluded_words.update(face_answers)
    if (
        len(excluded_names) != EXPECTED_EXCLUDED_ENTITIES
        or len(excluded_words) != EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError(
            f"v15来源并集数量改变：entities={len(excluded_names)}, words={len(excluded_words)}"
        )

    initial_manifest = read_json(INITIAL_SOURCE_EXCLUSIONS)
    initial_readiness = read_json(INITIAL_READINESS)
    initial_attempt = read_json(INITIAL_READINESS_ATTEMPT)
    initial_errors = initial_readiness.get("errors", [])
    if (
        sha256(INITIAL_SOURCE_EXCLUSIONS) != INITIAL_SOURCE_EXCLUSIONS_SHA256
        or sha256(INITIAL_READINESS) != INITIAL_READINESS_SHA256
        or sha256(INITIAL_READINESS_ATTEMPT) != INITIAL_READINESS_ATTEMPT_SHA256
        or initial_manifest.get("entity_count") != EXPECTED_EXCLUDED_ENTITIES
        or initial_manifest.get("word_value_count") != EXPECTED_EXCLUDED_WORDS
        or set(initial_manifest.get("excluded_entities", [])) != excluded_names
        or set(initial_manifest.get("excluded_word_values", [])) != excluded_words
        or initial_readiness.get("status") != "stopped"
        or initial_readiness.get("complete") is not False
        or initial_readiness.get("face_generated") is not False
        or initial_readiness.get("model_queries") != 0
        or initial_readiness.get("parameter_updates") != 0
        or initial_attempt.get("status") != "started"
        or initial_attempt.get("plan_sha256") != PLAN_SHA256
        or not any("s11_cut_final.pt" in error for error in initial_errors)
    ):
        raise ValueError("初次readiness停止回执不符合已记录的Adam状态数校验问题")

    second_manifest = read_json(SECOND_SOURCE_EXCLUSIONS)
    second_readiness = read_json(SECOND_READINESS)
    second_attempt = read_json(SECOND_READINESS_ATTEMPT)
    second_errors = second_readiness.get("errors", [])
    if (
        sha256(SECOND_SOURCE_EXCLUSIONS) != SECOND_SOURCE_EXCLUSIONS_SHA256
        or sha256(SECOND_READINESS) != SECOND_READINESS_SHA256
        or sha256(SECOND_READINESS_ATTEMPT) != SECOND_READINESS_ATTEMPT_SHA256
        or second_manifest.get("schema") != "n1-plan133-v15-source-exclusions-revalidation-v1"
        or second_manifest.get("status") != "ready_for_approved_execution"
        or second_manifest.get("plan_sha256") != PLAN_SHA256
        or second_manifest.get("entity_count") != EXPECTED_EXCLUDED_ENTITIES
        or second_manifest.get("word_value_count") != EXPECTED_EXCLUDED_WORDS
        or set(second_manifest.get("excluded_entities", [])) != excluded_names
        or set(second_manifest.get("excluded_word_values", [])) != excluded_words
        or second_manifest.get("supersedes_initial_manifest_sha256")
        != INITIAL_SOURCE_EXCLUSIONS_SHA256
        or second_readiness.get("status") != "stopped"
        or second_readiness.get("complete") is not False
        or second_readiness.get("face_generated") is not False
        or second_readiness.get("model_queries") != 0
        or second_readiness.get("parameter_updates") != 0
        or second_readiness.get("supersedes_initial_readiness_sha256") != INITIAL_READINESS_SHA256
        or second_attempt.get("status") != "started"
        or second_attempt.get("plan_sha256") != PLAN_SHA256
        or second_attempt.get("model_queries") != 0
        or second_attempt.get("parameter_updates") != 0
        or second_attempt.get("face_generated") is not False
        or second_attempt.get("supersedes_initial_readiness_sha256") != INITIAL_READINESS_SHA256
        or not any("s11_cut_final.pt" in error for error in second_errors)
    ):
        raise ValueError("第二次readiness停止回执与Adam步数去重校验问题不符")

    source_hashes = dict(old.get("source_hashes", {}))
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"N1-124前置来源哈希改变：{path}")
    n132_reconciliation = read_json(N132_RECONCILIATION)
    if (
        n132_reconciliation.get("complete") is not True
        or n132_reconciliation.get("source_hashes_current") is not True
        or n132_reconciliation.get("training_sufficient_all_seeds") is not True
        or not seal_ast.receipt_binding_matches(
            n132_reconciliation.get("training_result_sha256"), N132_RESULT
        )
    ):
        raise ValueError("N1-132来源对账未通过")
    for raw_path, expected in n132_reconciliation.get("verified_source_hashes", {}).items():
        _add_source(source_hashes, Path(raw_path), expected)

    fixed = {
        N124_PLAN: "8e23edb313bbc427856ba1ea8e73bb7ab7a37ddf33c8599ebec44fbb16390db4",
        N124_CLOSEOUT: "86f41f8f8d2c433ec933a0a345a8c8ef40a17ee9c57f66cfc50567169231fc97",
        N124_SOURCE_EXCLUSIONS: "9b859f0b22968eaaa2e4862a8418b8134a2db2cf2752d1579be6e05c70bf542a",
        N124_READINESS: "9a04fa434556e43e3937aa453169c02bb1aa7168811687be8c5caeb29f27045f",
        N124_FACE: "1c950f25e74d024171b544089fe76ad6a8545ea24326e03265ba8bd94d9443e7",
        N124_FACE_CHECKS: "08d4dc9e048b136fe48f716b6287c90699b7a4e006beef3b04b7418c297039c2",
        N124_RANGE: "5d297194fd036b7c799f652b920d9277f0a8ca6449bae02eb80b9e5449e99cdc",
        N124_RANGE_ATTEMPT: "1bd26e075004cb1e32fb0a202a1f410a4c8228b0b367c52aeffc737530f45d3e",
        N124_CONFIRM: "42caeb9e7e0c775099f62f5e40f3c636586a141a52ee8814dccea4b28547d4b7",
        N124_CONFIRM_ATTEMPT: "a89411d1214f38238d5f8ac36a0179a0cec889a4658b44aad00643cd4e98969f",
        N124_ITEMWISE: "f3fbb49789a8cbf47c1c4c0bee7f74c6f48b936b0b31ae169f2861da8d8ecb36",
        N13_N14_ATTRIBUTION: "38e4faac2320649319fff25b6c3652120718d23536e1d57cb5bd0fc57eeff9b7",
        EXISTING_BINDING_AUDIT: "558bdd917cb6a62b6271fb9c22b78d19286a83b61d00a47937c4d89a896892d4",
        N120_PLAN: "5b22795d84b9ac589b0de5cd73a03d5019d5ac0b2c9a6efd8ec2b84a0ecabae2",
        N120_RESULT: "873c237623fa7b97de0f4d8d95edaf4a5d8f9bf944c8af301468bd1d0f19ab2f",
        N120_REVALIDATION: "22055b74784e7c35891098595928af238561405a52cbfb4b94eb87491ba7e03d",
        N132_PLAN: "4177e1fa9da00ef1911a3c23f8598a67efd2e5fd5e54b2d4bd96b02fd60dd10f",
        N132_CLOSEOUT: "7eb28ef60810de18a9524c09d0f6f880838721f96234d293bd2801a07b497bd3",
        N132_RESULT: "9c9b6a89aa1e6b006469bda8a310fa9cc57cbb3478956482bebf1828000f7fcc",
        N132_RECONCILIATION: "895c3ede4c84e7d57ced3c7bd62727330eb2cbc6b51a9ab5cd0aa4e93bc0e87e",
        INITIAL_SOURCE_EXCLUSIONS: INITIAL_SOURCE_EXCLUSIONS_SHA256,
        INITIAL_READINESS: INITIAL_READINESS_SHA256,
        INITIAL_READINESS_ATTEMPT: INITIAL_READINESS_ATTEMPT_SHA256,
        SECOND_SOURCE_EXCLUSIONS: SECOND_SOURCE_EXCLUSIONS_SHA256,
        SECOND_READINESS: SECOND_READINESS_SHA256,
        SECOND_READINESS_ATTEMPT: SECOND_READINESS_ATTEMPT_SHA256,
        SOURCE_UNION_PRECHECK: SOURCE_UNION_PRECHECK_SHA256,
        ENDPOINT_PRECHECK: ENDPOINT_PRECHECK_SHA256,
        PLAN: PLAN_SHA256,
    }
    for path, expected in fixed.items():
        _add_source(source_hashes, path, expected)
    _add_source(source_hashes, N124_FACE_ATTEMPT)
    for path in (
        Path(__file__).resolve(),
        ROOT / "scripts/training/eval_taiji_n1_v15_after_n132.py",
    ):
        _add_source(source_hashes, path)
    for path in (
        ROOT / "tests/taiji_native/test_n1_plan133_v15_face_contract.py",
        ROOT / "tests/taiji_native/test_n1_plan133_v15_scoring_contract.py",
    ):
        _add_source(source_hashes, path)

    endpoints = endpoint_precheck.get("endpoints", [])
    if len(endpoints) != 12:
        raise ValueError("N1-133应冻结12个终点")
    expected_keys = {
        *(
            ("N1-120", seed, arm)
            for seed in (11, 12, 13)
            for arm in ("position", "terminal", "cut")
        ),
        *(("N1-132", seed, "position") for seed in (11, 12, 13)),
    }
    observed_keys = set()
    for endpoint in endpoints:
        key = (endpoint["source"], endpoint["seed"], endpoint["arm"])
        if key not in expected_keys or key in observed_keys:
            raise ValueError(f"N1-133终点重复或越界：{key}")
        observed_keys.add(key)
        path = Path(endpoint["final_checkpoint_path"])
        _add_source(source_hashes, path, endpoint["sha256_expected"])
        if endpoint.get("sha256_current") != endpoint["sha256_expected"]:
            raise ValueError(f"N1-133终点预检SHA不一致：{path}")
    if observed_keys != expected_keys:
        raise ValueError("N1-133终点集合不完整")

    seed115_ledgers = []
    for path in sorted((ROOT / "plans/manifests").glob("n1_hard_neighbor_*.json")):
        ledger = read_json(path)
        _add_source(source_hashes, path)
        if ledger.get("seed") == SEED:
            seed115_ledgers.append(str(path))
    if seed115_ledgers:
        raise ValueError(f"seed 115已在硬邻居账本出现：{seed115_ledgers}")
    return (
        excluded_names,
        excluded_words,
        source_hashes,
        {
            "n124_source_entities": 6678,
            "n124_source_words": 5328,
            "v14_face_entities": len(face_names),
            "v14_face_materials": len(face_materials),
            "v14_face_answers": len(face_answers),
            "v14_answer_overlap": v14_answer_overlap,
            "excluded_entities": len(excluded_names),
            "excluded_words": len(excluded_words),
            "initial_readiness_status": initial_readiness.get("status"),
            "initial_readiness_errors": initial_errors,
            "initial_readiness_sha256": INITIAL_READINESS_SHA256,
            "initial_source_exclusions_sha256": INITIAL_SOURCE_EXCLUSIONS_SHA256,
            "second_readiness_status": second_readiness.get("status"),
            "second_readiness_errors": second_errors,
            "second_readiness_sha256": SECOND_READINESS_SHA256,
            "second_source_exclusions_sha256": SECOND_SOURCE_EXCLUSIONS_SHA256,
            "prior_stopped_readiness_count": 2,
            "source_hash_count": len(source_hashes),
        },
    )


def _endpoint_specs() -> list[dict]:
    n120_result = read_json(N120_RESULT)
    n120_revalidation = read_json(N120_REVALIDATION)
    n132_result = read_json(N132_RESULT)
    n132_reconciliation = read_json(N132_RECONCILIATION)
    if (
        n120_revalidation.get("complete") is not True
        or n120_revalidation.get("training_sufficient") is not True
        or n120_revalidation.get("independent_restore_checks") != 18
        or n120_revalidation.get("all_restore_checks_equal") is not True
    ):
        raise ValueError("N1-120完整回执复核与恢复证据不可用")
    n120_revalidation_models = {
        (item["seed"], item["arm"]): item for item in n120_revalidation.get("models", [])
    }
    endpoints = []
    for model in n120_result.get("models", []):
        seed, arm = model["seed"], model["arm"]
        final = [
            check
            for check in model.get("restore_checks", [])
            if Path(check.get("path", "")).name == f"s{seed}_{arm}_final.pt"
        ]
        revalidation = n120_revalidation_models.get((seed, arm))
        if (
            len(final) != 1
            or final[0].get("equal") is not True
            or revalidation is None
            or revalidation.get("all_restore_checks_equal") is not True
            or revalidation.get("independent_restore_checks") != 2
            or (arm == "position" and revalidation.get("training_sufficient") is not True)
        ):
            raise ValueError(f"N1-120终点回执/恢复异常：{seed}/{arm}")
        check = final[0]
        endpoints.append(
            {
                "source": "N1-120",
                "seed": seed,
                "arm": arm,
                "role": {
                    "position": "same_seed_position_parent",
                    "terminal": "terminal_control",
                    "cut": "cut_control",
                }[arm],
                "path": check["path"],
                "sha256": check["sha256"],
                "expected_bundle_fingerprint": check.get("signature", {}).get("bundle"),
                "expected_updates": model.get("completed_updates"),
                "expected_history_count": model.get("new_session_count"),
            }
        )

    n132_models = {item["seed"]: item for item in n132_result.get("models", [])}
    for final in n132_reconciliation.get("final_checkpoints", []):
        seed = final["seed"]
        model = n132_models.get(seed)
        if (
            model is None
            or not final.get("independent_restore_equal")
            or not final.get("restore_signature_matches")
            or model.get("final_checkpoint_path") != final.get("path")
            or model.get("final_checkpoint_sha256") != final.get("sha256_expected")
            or model.get("final_bundle_fingerprint") is None
        ):
            raise ValueError(f"N1-132候选恢复/哈希回执异常：seed {seed}")
        endpoints.append(
            {
                "source": "N1-132",
                "seed": seed,
                "arm": "position",
                "role": "binding_supervised_candidate",
                "path": final["path"],
                "sha256": final["sha256_expected"],
                "expected_bundle_fingerprint": model["final_bundle_fingerprint"],
                "expected_updates": model.get("total_updates"),
                "expected_history_count": model.get("final_history_count"),
            }
        )
    if len(endpoints) != 12:
        raise ValueError("N1-133必须有12个完整endpoint")
    return sorted(endpoints, key=lambda row: (row["source"], row["seed"], row["arm"]))


def build_v15_face(excluded_names: set[str], excluded_words: set[str]) -> dict:
    """沿用已验证的v14构面规则，只更换冻结的seed与版本标识。"""
    original_seed = v14_builder.SEED
    try:
        v14_builder.SEED = SEED
        face = v14_builder.build_v14_face([], excluded_names, excluded_words)
    finally:
        v14_builder.SEED = original_seed
    face["schema"] = "n1-hard-neighbor-v15-confirmation-v1"
    face["seed"] = SEED
    id_map = {}
    for index, episode in enumerate(face["episodes"]):
        prior_id = episode["id"]
        new_id = f"v15-hard-{index:03d}"
        id_map[prior_id] = new_id
        episode["id"] = new_id
    face["subsets"] = {
        subset: [id_map[item_id] for item_id in ids] for subset, ids in face["subsets"].items()
    }
    return face


def audit_v15_face(face: dict, *, excluded_names: set[str], excluded_words: set[str]) -> dict:
    actual_ids = [item.get("id") for item in face.get("episodes", [])]
    actual_schema_ok = (
        face.get("schema") == "n1-hard-neighbor-v15-confirmation-v1"
        and face.get("seed") == SEED
        and actual_ids == [f"v15-hard-{index:03d}" for index in range(ITEMS)]
    )
    normalized = copy.deepcopy(face)
    normalized["schema"] = "n1-hard-neighbor-v14-confirmation-v1"
    normalized["seed"] = v14_builder.SEED
    id_map = {}
    for index, episode in enumerate(normalized.get("episodes", [])):
        prior_id = episode["id"]
        new_id = f"v14-hard-{index:03d}"
        id_map[prior_id] = new_id
        episode["id"] = new_id
    normalized["subsets"] = {
        subset: [id_map[item_id] for item_id in ids]
        for subset, ids in normalized.get("subsets", {}).items()
    }
    contract = v14_builder.audit_v14_face(
        normalized, excluded_names=excluded_names, excluded_words=excluded_words
    )
    variable_answers = {
        item["answer"]
        for item in face.get("episodes", [])
        if item.get("kind") in {"lookup", "revision"}
    }
    fixed_answers = {
        item["answer"]
        for item in face.get("episodes", [])
        if item.get("kind") in {"transition", "unknown"}
    }
    contract["checks"]["v15_schema_seed_and_ids"] = actual_schema_ok
    contract["checks"]["variable_answers_disjoint_from_prior_values"] = not (
        variable_answers & excluded_words
    )
    contract["checks"]["fixed_answer_vocabulary_preexcluded"] = (
        fixed_answers
        == {
            "north",
            "east",
            "south",
            "west",
            "unknown",
        }
        and fixed_answers <= excluded_words
    )
    contract["accepted"] = all(contract["checks"].values())
    contract["schema"] = "n1-plan133-v15-face-checks-v1"
    contract["summary_zh"] = (
        "复用v14已验证的16行目标位置与6/7字符平衡设计，仅更换seed、版本标识和完整来源排除并集。"
    )
    return contract


def _verify_current_sources(source_hashes: dict[str, str]) -> None:
    for raw_path, expected in source_hashes.items():
        path = Path(raw_path)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"readiness后来源已变化：{path}")


def _restore_endpoints(endpoints: list[dict], restore_rows: list[dict] | None = None) -> list[dict]:
    n120.configure()
    _, n120_globals, _ = n120._BASE.make_runner(load_inputs=False)
    if restore_rows is None:
        restore_rows = []
    for endpoint in endpoints:
        path = Path(endpoint["path"])
        if sha256(path) != endpoint["sha256"]:
            raise ValueError(f"endpoint checkpoint在恢复前发生变化：{path}")
        if endpoint["source"] == "N1-120":
            model, optimizer, fingerprint, paired = n120_eval._load_model(
                n120_globals,
                path,
                endpoint["sha256"],
                endpoint["seed"],
                endpoint["arm"],
                read_json(N120_RESULT)["protocol"],
            )
            expected_step = endpoint["expected_updates"]
            history = len(model._observations)
            steps = {
                int(state["step"].item() if hasattr(state["step"], "item") else state["step"])
                for state in optimizer.state.values()
                if "step" in state
            }
            states = len(optimizer.state)
            state_step_entries = sum("step" in state for state in optimizer.state.values())
            if (
                fingerprint != endpoint["expected_bundle_fingerprint"]
                or states == 0
                or state_step_entries != states
                or steps != {expected_step}
                or history != endpoint["expected_history_count"]
            ):
                raise ValueError(f"N1-120完整Adam/history bundle与回执不符：{path}")
            if paired.fingerprint(model.bundle(optimizer)) != fingerprint:
                raise ValueError(f"N1-120恢复后bundle签名变化：{path}")
        elif endpoint["source"] == "N1-132":
            model, optimizer, payload = n132._load_n132_checkpoint(path)
            fingerprint = n132.e11.scorer.paired.fingerprint(model.bundle(optimizer))
            expected_step = int(payload["total_updates"])
            history = len(model._observations)
            steps = {
                int(state["step"].item() if hasattr(state["step"], "item") else state["step"])
                for state in optimizer.state.values()
                if "step" in state
            }
            states = len(optimizer.state)
            state_step_entries = sum("step" in state for state in optimizer.state.values())
            if (
                payload.get("seed") != endpoint["seed"]
                or payload.get("plan_sha256") != n132.PLAN_SHA256
                or payload.get("arm") != n132.ARM
                or fingerprint != endpoint["expected_bundle_fingerprint"]
                or states != 41
                or state_step_entries != states
                or steps != {expected_step}
                or history != endpoint["expected_history_count"]
            ):
                raise ValueError(f"N1-132完整Adam/history bundle与回执不符：{path}")
        else:
            raise ValueError(f"未知endpoint来源：{endpoint['source']}")
        restore_rows.append(
            {
                **endpoint,
                "sha256_current": sha256(path),
                "bundle_fingerprint_restored": fingerprint,
                "restore_matches": True,
                "optimizer_parameter_states": states,
                "optimizer_step_entries": state_step_entries,
                "optimizer_steps": expected_step,
                "history_count": history,
            }
        )
        del model, optimizer
    return restore_rows


def _source_hashes() -> tuple[dict[str, str], dict]:
    excluded_names, excluded_words, sources, counts = _prior_sources_and_union()
    endpoints = _endpoint_specs()
    for endpoint in endpoints:
        _add_source(sources, Path(endpoint["path"]), endpoint["sha256"])
    return sources, {
        "excluded_entities": sorted(excluded_names),
        "excluded_word_values": sorted(excluded_words),
        "counts": counts,
        "endpoints": endpoints,
    }


def source_readiness() -> dict:
    if any(path.exists() for path in WRITE_ONCE_OUTPUTS):
        raise FileExistsError("PLAN-N1-133路径为一次性输出；不得重试或重抽")
    started = time.process_time()
    write_new(
        READINESS_ATTEMPT,
        {
            "schema": "n1-plan133-v15-readiness-revalidation-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "checked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "face_generated": False,
            "model_queries": 0,
            "parameter_updates": 0,
            "supersedes_prior_readiness_sha256": SECOND_READINESS_SHA256,
            "preserved_stopped_readiness_sha256": [
                INITIAL_READINESS_SHA256,
                SECOND_READINESS_SHA256,
            ],
            "说明_zh": "针对首次readiness的校验器缺陷进行独立复验；保留首次停止回执，不生成题面、不查询模型、不更新参数。",
        },
    )
    report = None
    source_hashes = {}
    endpoint_restores = []
    error = None
    try:
        source_hashes, inventory = _source_hashes()
        _verify_current_sources(source_hashes)
        write_new(
            SOURCE_EXCLUSIONS,
            {
                "schema": "n1-plan133-v15-source-exclusions-revalidation-v2",
                "status": "ready_for_approved_execution",
                "plan_sha256": PLAN_SHA256,
                "supersedes_prior_manifest_sha256": SECOND_SOURCE_EXCLUSIONS_SHA256,
                "preserved_stopped_readiness_sha256": [
                    INITIAL_READINESS_SHA256,
                    SECOND_READINESS_SHA256,
                ],
                "excluded_entities": inventory["excluded_entities"],
                "excluded_word_values": inventory["excluded_word_values"],
                "entity_count": inventory["counts"]["excluded_entities"],
                "word_value_count": inventory["counts"]["excluded_words"],
                "provenance": inventory["counts"],
                "source_hashes": source_hashes,
                "摘要_zh": "保留前两次readiness停止收据后重新核验来源排除；完整排除v3训练、v5-v14已消费面、N1-95 scratch及派生manifest中的实体与材料/答案词值。",
                "can_promote": False,
            },
        )
        source_hashes[str(SOURCE_EXCLUSIONS.resolve())] = sha256(SOURCE_EXCLUSIONS)
        source_hashes[str(READINESS_ATTEMPT.resolve())] = sha256(READINESS_ATTEMPT)
        _restore_endpoints(inventory["endpoints"], endpoint_restores)
        _verify_current_sources(
            {k: v for k, v in source_hashes.items() if k != str(SOURCE_EXCLUSIONS.resolve())}
        )
        complete = (
            len(endpoint_restores) == 12
            and all(row["restore_matches"] for row in endpoint_restores)
            and len(inventory["excluded_entities"]) == EXPECTED_EXCLUDED_ENTITIES
            and len(inventory["excluded_word_values"]) == EXPECTED_EXCLUDED_WORDS
            and time.process_time() - started <= CPU_LIMIT_SECONDS
        )
        report = {
            "schema": "n1-plan133-v15-confirmation-readiness-revalidation-v2",
            "status": "ready_for_approved_execution" if complete else "stopped",
            "complete": complete,
            "plan_sha256": PLAN_SHA256,
            "supersedes_prior_readiness_sha256": SECOND_READINESS_SHA256,
            "preserved_stopped_readiness_sha256": [
                INITIAL_READINESS_SHA256,
                SECOND_READINESS_SHA256,
            ],
            "prior_stop_reasons_zh": [
                "第一次校验把N1-120 Adam状态数误固定为41；seed11/cut实际37个活动状态。",
                "第二次校验误将唯一步数值集合的大小与活动状态数比较；只读复核确认37/37状态均有24576步，history与bundle指纹匹配。",
            ],
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "source_hashes": source_hashes,
            "endpoint_count": len(endpoint_restores),
            "endpoint_restores": endpoint_restores,
            "entity_exclusion_count": len(inventory["excluded_entities"]),
            "word_value_exclusion_count": len(inventory["excluded_word_values"]),
            "model_loads": len(endpoint_restores),
            "checkpoint_loads": len(endpoint_restores),
            "model_queries": 0,
            "parameter_updates": 0,
            "optimizer_steps": 0,
            "dev_queries": 0,
            "face_generated": False,
            "cpu_seconds": round(time.process_time() - started, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "can_promote": False,
            "summary_zh": "修正Adam状态数与步数字段计数校验后，来源排除与12个完整checkpoint bundle恢复均已核验；本阶段0模型查询、0参数更新、未生成题面。",
            "errors": [],
        }
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        report = {
            "schema": "n1-plan133-v15-confirmation-readiness-revalidation-v2",
            "status": "stopped",
            "complete": False,
            "plan_sha256": PLAN_SHA256,
            "supersedes_prior_readiness_sha256": SECOND_READINESS_SHA256,
            "preserved_stopped_readiness_sha256": [
                INITIAL_READINESS_SHA256,
                SECOND_READINESS_SHA256,
            ],
            "prior_stop_reasons_zh": [
                "第一次校验把N1-120 Adam状态数误固定为41；seed11/cut实际37个活动状态。",
                "第二次校验误将唯一步数值集合的大小与活动状态数比较；只读复核确认37/37状态均有24576步，history与bundle指纹匹配。",
            ],
            "source_hashes": source_hashes,
            "endpoint_count": len(endpoint_restores),
            "endpoint_restores": endpoint_restores,
            "model_queries": 0,
            "parameter_updates": 0,
            "face_generated": False,
            "cpu_seconds": round(time.process_time() - started, 6),
            "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            "can_promote": False,
            "summary_zh": "第二次独立复验失败；保留本次回执并停止，不生成题面、不查询模型、不更新参数。",
            "errors": [error],
        }
    report["readiness_attempt_sha256"] = sha256(READINESS_ATTEMPT)
    write_new(READINESS, report)
    return report


def _verify_readiness() -> dict:
    if not READINESS.is_file() or not SOURCE_EXCLUSIONS.is_file():
        raise FileNotFoundError("N1-133零查询readiness和排除manifest必须先完成")
    report = read_json(READINESS)
    exclusions = read_json(SOURCE_EXCLUSIONS)
    if (
        report.get("complete") is not True
        or report.get("status") != "ready_for_approved_execution"
        or report.get("plan_sha256") != PLAN_SHA256
        or report.get("supersedes_prior_readiness_sha256") != SECOND_READINESS_SHA256
        or not seal_ast.receipt_binding_matches(
            report.get("source_exclusion_manifest_sha256"), SOURCE_EXCLUSIONS
        )
        or report.get("endpoint_count") != 12
        or report.get("model_queries") != 0
        or report.get("parameter_updates") != 0
        or report.get("face_generated") is not False
        or exclusions.get("supersedes_prior_manifest_sha256") != SECOND_SOURCE_EXCLUSIONS_SHA256
        or exclusions.get("entity_count") != EXPECTED_EXCLUDED_ENTITIES
        or exclusions.get("word_value_count") != EXPECTED_EXCLUDED_WORDS
    ):
        raise ValueError("N1-133 readiness或来源排除manifest失效")
    _verify_current_sources(
        {
            k: v
            for k, v in report.get("source_hashes", {}).items()
            if k != str(SOURCE_EXCLUSIONS.resolve())
        }
    )
    for endpoint in report.get("endpoint_restores", []):
        path = Path(endpoint["path"])
        if sha256(path) != endpoint.get("sha256") or not endpoint.get("restore_matches"):
            raise ValueError(f"N1-133 checkpoint在readiness后发生变化：{path}")
    return report


def _write_face_attempt() -> None:
    write_new(
        FACE_ATTEMPT,
        {
            "schema": "n1-plan133-v15-face-attempt-v1",
            "status": "started",
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "seed": SEED,
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "说明_zh": "一次性生成seed115 v15面。若构面合同失败，保留attempt并停止，不更换seed、不重试。",
        },
    )


def build_face_after_approval() -> dict:
    readiness = _verify_readiness()
    if any(
        path.exists()
        for path in (
            FACE,
            FACE_ATTEMPT,
            FACE_CHECKS,
            RANGE_REPORT,
            RANGE_ATTEMPT,
            CONFIRM_REPORT,
            CONFIRM_ATTEMPT,
        )
    ):
        raise FileExistsError("v15题面/查询阶段已有输出；不覆盖、不重试")
    manifest = read_json(SOURCE_EXCLUSIONS)
    excluded_names = set(manifest["excluded_entities"])
    excluded_words = set(manifest["excluded_word_values"])
    _write_face_attempt()
    started = time.process_time()
    face = build_v15_face(excluded_names, excluded_words)
    _verify_current_sources(readiness["source_hashes"])
    face_source_hashes = dict(readiness["source_hashes"])
    face_source_hashes[str(FACE_ATTEMPT.resolve())] = sha256(FACE_ATTEMPT)
    face["exclusion_sources"] = {
        "plan_sha256": PLAN_SHA256,
        "readiness_sha256": sha256(READINESS),
        "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
        "face_attempt_sha256": sha256(FACE_ATTEMPT),
        "expected_entity_count": len(excluded_names),
        "expected_word_value_count": len(excluded_words),
        "source_hashes": face_source_hashes,
    }
    checks = audit_v15_face(face, excluded_names=excluded_names, excluded_words=excluded_words)
    checks.update(
        {
            "face": str(FACE.resolve()),
            "face_sha256": None,
            "plan_sha256": PLAN_SHA256,
            "readiness_sha256": sha256(READINESS),
            "source_exclusion_manifest_sha256": sha256(SOURCE_EXCLUSIONS),
            "face_attempt_sha256": sha256(FACE_ATTEMPT),
            "source_entity_count": len(excluded_names),
            "source_word_value_count": len(excluded_words),
            "model_loads": 0,
            "model_queries": 0,
            "parameter_updates": 0,
            "can_promote": False,
            "build_cpu_seconds": round(time.process_time() - started, 6),
            "summary_zh": "seed115 v15题面复用v14早/中/晚证据位置及6/7字符平衡；仅使用完整排除集以外的新实体/材料值。",
        }
    )
    if checks.get("accepted") is not True:
        write_new(FACE_CHECKS, checks)
        raise ValueError("v15题面合同失败；保留checks，不重抽")
    write_new(FACE, face)
    checks["face_sha256"] = sha256(FACE)
    write_new(FACE_CHECKS, checks)
    return checks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--readiness", action="store_true")
    modes.add_argument("--build-face", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    if args.readiness:
        result = source_readiness()
    else:
        if not args.owner_approved:
            raise PermissionError("PLAN-N1-133构面需要owner授权")
        result = build_face_after_approval()
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "accepted",
                    "endpoint_count",
                    "entity_exclusion_count",
                    "word_value_exclusion_count",
                    "face_sha256",
                    "model_queries",
                    "parameter_updates",
                    "cpu_seconds",
                    "summary_zh",
                    "errors",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("complete", result.get("accepted", False)) else 2


if __name__ == "__main__":
    raise SystemExit(main())
