"""对已批准的PLAN-N1-133执行一次v15 sacrifice与条件confirm。"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from scripts.training import build_taiji_n1_v15_after_n132 as face_builder
from scripts.training import eval_taiji_n1_v11_after_n120 as n120_eval
from scripts.training import run_taiji_n1_expanded_normalized_rwl_length_balanced as n120
from scripts.training import run_taiji_n1_plan132_binding_supervision_calibration as n132

ROOT = face_builder.ROOT
FACE = face_builder.FACE
FACE_ATTEMPT = face_builder.FACE_ATTEMPT
FACE_CHECKS = face_builder.FACE_CHECKS
READINESS = face_builder.READINESS
SOURCE_EXCLUSIONS = face_builder.SOURCE_EXCLUSIONS
RANGE_REPORT = face_builder.RANGE_REPORT
RANGE_ATTEMPT = face_builder.RANGE_ATTEMPT
CONFIRM_REPORT = face_builder.CONFIRM_REPORT
CONFIRM_ATTEMPT = face_builder.CONFIRM_ATTEMPT
CPU_LIMIT_SECONDS = face_builder.CPU_LIMIT_SECONDS
SEEDS = (11, 12, 13)
ARMS = ("position", "terminal", "cut")
KINDS = ("lookup", "revision", "transition", "unknown")
RANGE_MAIN = tuple(
    [{"source": "N1-132", "seed": seed, "arm": "position"} for seed in SEEDS]
    + [
        {"source": "N1-120", "seed": seed, "arm": arm}
        for arm in ("terminal", "cut")
        for seed in SEEDS
    ]
)
CONFIRM_ENDPOINTS = tuple(
    [{"source": "N1-132", "seed": seed, "arm": "position"} for seed in SEEDS]
    + [{"source": "N1-120", "seed": seed, "arm": "position"} for seed in SEEDS]
    + [
        {"source": "N1-120", "seed": seed, "arm": arm}
        for arm in ("terminal", "cut")
        for seed in SEEDS
    ]
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


def _face_and_checks() -> tuple[dict, dict]:
    if not FACE.is_file() or not FACE_CHECKS.is_file():
        raise FileNotFoundError("须先生成并核验经批准的v15题面")
    face, checks = read_json(FACE), read_json(FACE_CHECKS)
    if (
        checks.get("accepted") is not True
        or checks.get("face_sha256") != sha256(FACE)
        or face.get("schema") != "n1-hard-neighbor-v15-confirmation-v1"
        or face.get("seed") != face_builder.SEED
        or face.get("items") != 40
        or face.get("observations_per_item") != 16
        or face.get("name_neighbor_distance") != 1
        or len(face.get("episodes", [])) != 40
        or len(face.get("subsets", {}).get("sacrifice", [])) != 20
        or len(face.get("subsets", {}).get("confirm", [])) != 20
        or checks.get("model_queries") != 0
        or checks.get("parameter_updates") != 0
        or checks.get("plan_sha256") != face_builder.PLAN_SHA256
        or checks.get("readiness_sha256") != sha256(READINESS)
        or checks.get("source_exclusion_manifest_sha256") != sha256(SOURCE_EXCLUSIONS)
    ):
        raise ValueError("v15题面、face checks或冻结合同不一致")
    source_hashes = face.get("exclusion_sources", {}).get("source_hashes", {})
    if not source_hashes:
        raise ValueError("v15题面缺少来源哈希链")
    face_builder._verify_current_sources(source_hashes)
    return face, checks


def _validate_readiness() -> tuple[dict, dict, dict, dict]:
    readiness = face_builder._verify_readiness()
    face, checks = _face_and_checks()
    n120_report = read_json(face_builder.N120_RESULT)
    n132_report = read_json(face_builder.N132_RESULT)
    if n120_report.get("status") != "stopped":
        raise ValueError("N1-120来源状态与冻结回执不符")
    if n132_report.get("complete") is not True or n132_report.get("can_promote") is not False:
        raise ValueError("N1-132 train-only结果不完整或晋升标志异常")
    n120.configure()
    return readiness, face, checks, {"n120": n120_report, "n132": n132_report}


def _endpoint_spec(source: str, seed: int, arm: str, n120_report: dict, n132_report: dict) -> dict:
    if source == "N1-120":
        models = [
            m
            for m in n120_report.get("models", [])
            if m.get("seed") == seed and m.get("arm") == arm
        ]
        if len(models) != 1:
            raise ValueError(f"N1-120终点缺失：{seed}/{arm}")
        model = models[0]
        checks = [
            item
            for item in model.get("restore_checks", [])
            if Path(item.get("path", "")).name == f"s{seed}_{arm}_final.pt"
        ]
        if len(checks) != 1 or checks[0].get("equal") is not True:
            raise ValueError(f"N1-120终点未恢复：{seed}/{arm}")
        check = checks[0]
        return {
            "source": source,
            "seed": seed,
            "arm": arm,
            "path": Path(check["path"]),
            "sha256": check["sha256"],
            "protocol": n120_report["protocol"],
            "role": "same_seed_position_parent" if arm == "position" else f"{arm}_control",
        }
    if source == "N1-132" and arm == "position":
        models = [
            m
            for m in n132_report.get("models", [])
            if m.get("seed") == seed and m.get("arm") == arm
        ]
        if len(models) != 1:
            raise ValueError(f"N1-132候选缺失：{seed}/position")
        model = models[0]
        return {
            "source": source,
            "seed": seed,
            "arm": arm,
            "path": Path(model["final_checkpoint_path"]),
            "sha256": model["final_checkpoint_sha256"],
            "protocol": None,
            "role": "binding_supervised_candidate",
        }
    raise ValueError(f"未定义的v15 endpoint：{source}/{seed}/{arm}")


def _score_endpoint(
    spec: dict,
    items: list[dict],
    budget: n120_eval.CpuDeadline,
    n120_globals: dict,
) -> tuple[list[dict], int, bool, str | None]:
    model = optimizer = paired = None
    loaded = False
    error = None
    rows: list[dict] = []
    accounting = {"confirmation_queries": 0}
    before = None
    try:
        if sha256(spec["path"]) != spec["sha256"]:
            raise ValueError(f"checkpoint在计分前发生变化：{spec['path']}")
        if spec["source"] == "N1-120":
            model, optimizer, before, paired = n120_eval._load_model(
                n120_globals,
                spec["path"],
                spec["sha256"],
                spec["seed"],
                spec["arm"],
                spec["protocol"],
            )
        else:
            model, optimizer, payload = n132._load_n132_checkpoint(spec["path"])
            paired = n120_eval.scorer.paired
            before = paired.fingerprint(model.bundle(optimizer))
            if (
                payload.get("seed") != spec["seed"]
                or payload.get("arm") != "position"
                or payload.get("phase") != "final"
                or payload.get("plan_sha256") != n132.PLAN_SHA256
            ):
                raise ValueError("N1-132候选checkpoint协议不符")
        loaded = True
        n120_eval.scorer.evaluate(
            model,
            optimizer,
            spec["seed"],
            spec["arm"],
            items,
            budget,
            output=rows,
            accounting=accounting,
        )
        if paired.fingerprint(model.bundle(optimizer)) != before:
            raise ValueError("冻结计分改变了参数、Adam、RNG或history")
        if len(rows) != len(items) or {r["id"] for r in rows} != {i["id"] for i in items}:
            raise ValueError("必须为每个输入恰好保留一条计分结果")
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if (
            model is not None
            and optimizer is not None
            and paired is not None
            and before is not None
        ):
            try:
                if paired.fingerprint(model.bundle(optimizer)) != before:
                    error = "ValueError:冻结计分改变了参数、Adam、RNG或history"
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
        if model is not None:
            del model
        if optimizer is not None:
            del optimizer
    for row in rows:
        row["endpoint_source"] = spec["source"]
        row["endpoint_role"] = spec["role"]
    return rows, accounting["confirmation_queries"], loaded, error


def _source_hashes_unchanged(extra_hashes: dict[Path, str] | None = None) -> bool:
    readiness = read_json(READINESS)
    source_hashes = dict(readiness.get("source_hashes", {}))
    face, _ = _face_and_checks()
    source_hashes.update(face.get("exclusion_sources", {}).get("source_hashes", {}))
    source_hashes.update(
        {str(Path(path).resolve()): expected for path, expected in (extra_hashes or {}).items()}
    )
    return all(
        Path(path).is_file() and sha256(Path(path)) == expected
        for path, expected in source_hashes.items()
    )


def run_range_check(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-133范围查询需要owner授权")
    if RANGE_REPORT.exists() or RANGE_ATTEMPT.exists():
        raise FileExistsError("N1-133 range阶段一次性，禁止重试")
    started = time.process_time()
    readiness, face, checks, reports = _validate_readiness()
    sacrifice_ids = set(face["subsets"]["sacrifice"])
    items = [ep for ep in face["episodes"] if ep["id"] in sacrifice_ids]
    if len(items) != 20 or any(sum(ep["kind"] == kind for ep in items) != 5 for kind in KINDS):
        raise ValueError("sacrifice必须恰好20题且每类5题")
    if set(face["subsets"]["confirm"]) & sacrifice_ids:
        raise ValueError("sacrifice与confirm必须互斥")
    prior_cpu = float(readiness["cpu_seconds"]) + float(checks["build_cpu_seconds"])
    remaining = CPU_LIMIT_SECONDS - prior_cpu
    if remaining <= 0:
        raise TimeoutError("N1-133累计CPU额度已耗尽")
    budget = n120_eval.CpuDeadline(remaining)
    n120.configure()
    _, n120_globals, _ = n120._BASE.make_runner(load_inputs=False)
    endpoint_rows = []
    total_queries = 0
    model_loads = 0
    errors = []
    write_new(
        RANGE_ATTEMPT,
        {
            "schema": "n1-plan133-v15-range-attempt-v1",
            "status": "started",
            "plan_sha256": face_builder.PLAN_SHA256,
            "face_sha256": sha256(FACE),
            "readiness_sha256": sha256(READINESS),
            "queried_subset": "sacrifice",
            "planned_queries": 180,
            "confirmation_queried": False,
            "parameter_updates": 0,
            "can_promote": False,
            "说明_zh": "只对9个主比较终点查询20题sacrifice。失败或中断后不重试。",
        },
    )
    try:
        for endpoint in RANGE_MAIN:
            budget.guard(reserve=5)
            spec = _endpoint_spec(
                endpoint["source"],
                endpoint["seed"],
                endpoint["arm"],
                reports["n120"],
                reports["n132"],
            )
            rows, queries, loaded, error = _score_endpoint(spec, items, budget, n120_globals)
            total_queries += queries
            model_loads += int(loaded)
            endpoint_rows.append(
                {
                    "seed": spec["seed"],
                    "arm": spec["arm"],
                    "source": spec["source"],
                    "role": spec["role"],
                    "rows": rows,
                }
            )
            if error:
                errors.append(f"{spec['source']}/{spec['seed']}/{spec['arm']}: {error}")
                break
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    summary = None
    if not errors:
        try:
            summary = n120_eval._range_summary(
                [
                    {k: v for k, v in endpoint.items() if k in {"seed", "arm", "rows"}}
                    for endpoint in endpoint_rows
                ]
            )
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    unchanged = _source_hashes_unchanged({RANGE_ATTEMPT: sha256(RANGE_ATTEMPT)})
    cpu = time.process_time() - started
    cumulative = prior_cpu + cpu
    complete = (
        not errors
        and len(endpoint_rows) == 9
        and total_queries == 180
        and sum(len(row["rows"]) for row in endpoint_rows) == 180
        and unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    passed = bool(summary and summary.get("range_check_passed") is True)
    report = {
        "schema": "n1-plan133-v15-sacrifice-range-v1",
        "status": "complete" if complete and passed else "range_not_met" if complete else "stopped",
        "complete": complete,
        "plan_sha256": face_builder.PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "readiness_sha256": sha256(READINESS),
        "range_attempt_sha256": sha256(RANGE_ATTEMPT),
        "source_hashes_unchanged": unchanged,
        "queried_subset": "sacrifice",
        "range_gate": summary,
        "endpoint_results": endpoint_rows,
        "model_loads": model_loads,
        "checkpoint_loads": model_loads,
        "model_queries": total_queries,
        "confirmation_queries": 0,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "cpu_seconds": round(cpu, 6),
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cumulative_cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "can_promote": False,
        "errors": errors,
        "summary_zh": "范围门已完成；unknown五题不计入每终点15道lookup/revision/transition题的错误分母。",
    }
    write_new(RANGE_REPORT, report)
    return report


def _require_passing_range(report: dict) -> None:
    if (
        report.get("complete") is not True
        or report.get("status") != "complete"
        or report.get("range_gate", {}).get("range_check_passed") is not True
        or report.get("model_queries") != 180
        or report.get("confirmation_queries") != 0
        or report.get("queried_subset") != "sacrifice"
        or report.get("face_sha256") != sha256(FACE)
        or report.get("readiness_sha256") != sha256(READINESS)
        or report.get("source_hashes_unchanged") is not True
    ):
        raise ValueError("须先完整通过同一v15 face的180查询range门")


def _parent_pair_summary(candidate_rows: list[dict], parent_rows: list[dict]) -> dict:
    candidate = {(r["seed"], r["id"]): r for r in candidate_rows}
    parent = {(r["seed"], r["id"]): r for r in parent_rows}
    if set(candidate) != set(parent) or len(candidate) != 60:
        raise ValueError("N1-132候选与同seed N1-120 position母档必须有60个配对题行")
    result = {}
    for seed in SEEDS:
        result[seed] = {}
        for kind in KINDS:
            keys = [key for key in candidate if key[0] == seed and candidate[key]["kind"] == kind]
            if len(keys) != 5:
                raise ValueError(f"配对描述读数须每seed/类别5题：{seed}/{kind}")
            candidate_exact = sum(bool(candidate[k]["free_exact"]) for k in keys)
            parent_exact = sum(bool(parent[k]["free_exact"]) for k in keys)
            rescued = sum(not parent[k]["free_exact"] and candidate[k]["free_exact"] for k in keys)
            regressed = sum(
                parent[k]["free_exact"] and not candidate[k]["free_exact"] for k in keys
            )
            result[seed][kind] = {
                "candidate_exact": candidate_exact,
                "parent_exact": parent_exact,
                "candidate_only_correct": rescued,
                "parent_only_correct": regressed,
                "net_exact_delta": candidate_exact - parent_exact,
            }
    return result


def _confirmation_summary(rows: list[dict]) -> dict:
    candidate = [r for r in rows if r["endpoint_source"] == "N1-132"]
    parents = [r for r in rows if r["endpoint_source"] == "N1-120" and r["arm"] == "position"]
    controls = [
        r for r in rows if r["endpoint_source"] == "N1-120" and r["arm"] in {"terminal", "cut"}
    ]
    if len(candidate) != 60 or len(parents) != 60 or len(controls) != 120:
        raise ValueError("确认集必须包含N1-132候选60行、position母档60行、控制臂120行")
    gates_rows = candidate + controls
    gates = n120_eval._cohort_gates(gates_rows, "N1-132续训包与N1-120控制")
    parent_pairs = _parent_pair_summary(candidate, parents)
    return {
        "package_vs_n120_controls": gates,
        "same_seed_n120_position_parent_descriptive": parent_pairs,
        "gate_passed": gates.get("all_four_gates") is True,
    }


def run_confirmation(*, owner_approved: bool) -> dict:
    if not owner_approved:
        raise PermissionError("PLAN-N1-133确认查询需要owner授权")
    if CONFIRM_REPORT.exists() or CONFIRM_ATTEMPT.exists():
        raise FileExistsError("N1-133 confirm阶段一次性，禁止重试")
    if not RANGE_REPORT.is_file():
        raise FileNotFoundError("N1-133 range报告缺失")
    range_report = read_json(RANGE_REPORT)
    _require_passing_range(range_report)
    started = time.process_time()
    readiness, face, _, reports = _validate_readiness()
    confirm_ids = set(face["subsets"]["confirm"])
    items = [ep for ep in face["episodes"] if ep["id"] in confirm_ids]
    if len(items) != 20 or len(confirm_ids & set(face["subsets"]["sacrifice"])):
        raise ValueError("confirm须为与sacrifice互斥的20题")
    prior_cpu = (
        float(readiness["cpu_seconds"])
        + float(read_json(FACE_CHECKS)["build_cpu_seconds"])
        + float(range_report["cpu_seconds"])
    )
    remaining = CPU_LIMIT_SECONDS - prior_cpu
    if remaining <= 0:
        raise TimeoutError("N1-133累计CPU额度已耗尽")
    budget = n120_eval.CpuDeadline(remaining)
    n120.configure()
    _, n120_globals, _ = n120._BASE.make_runner(load_inputs=False)
    endpoint_rows = []
    total_queries = 0
    model_loads = 0
    errors = []
    write_new(
        CONFIRM_ATTEMPT,
        {
            "schema": "n1-plan133-v15-confirm-attempt-v1",
            "status": "started",
            "plan_sha256": face_builder.PLAN_SHA256,
            "face_sha256": sha256(FACE),
            "readiness_sha256": sha256(READINESS),
            "range_report_sha256": sha256(RANGE_REPORT),
            "planned_queries": 240,
            "queried_subset": "confirm",
            "parameter_updates": 0,
            "can_promote": False,
            "说明_zh": "range门完整通过后，在12个冻结终点各查询20道confirm题；失败后不重试。",
        },
    )
    try:
        for endpoint in CONFIRM_ENDPOINTS:
            budget.guard(reserve=5)
            spec = _endpoint_spec(
                endpoint["source"],
                endpoint["seed"],
                endpoint["arm"],
                reports["n120"],
                reports["n132"],
            )
            rows, queries, loaded, error = _score_endpoint(spec, items, budget, n120_globals)
            total_queries += queries
            model_loads += int(loaded)
            endpoint_rows.append(
                {
                    "seed": spec["seed"],
                    "arm": spec["arm"],
                    "source": spec["source"],
                    "role": spec["role"],
                    "rows": rows,
                }
            )
            if error:
                errors.append(f"{spec['source']}/{spec['seed']}/{spec['arm']}: {error}")
                break
    except BaseException as exc:
        errors.append(f"{type(exc).__name__}: {exc}")
    all_rows = [row for endpoint in endpoint_rows for row in endpoint["rows"]]
    summary = None
    if not errors:
        try:
            summary = _confirmation_summary(all_rows)
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    unchanged = _source_hashes_unchanged(
        {
            RANGE_ATTEMPT: sha256(RANGE_ATTEMPT),
            RANGE_REPORT: sha256(RANGE_REPORT),
            CONFIRM_ATTEMPT: sha256(CONFIRM_ATTEMPT),
        }
    )
    stage_cpu = time.process_time() - started
    cumulative = prior_cpu + stage_cpu
    complete = (
        not errors
        and len(endpoint_rows) == 12
        and total_queries == 240
        and len(all_rows) == 240
        and unchanged
        and cumulative <= CPU_LIMIT_SECONDS
    )
    gates_pass = bool(summary and summary.get("gate_passed") is True)
    report = {
        "schema": "n1-plan133-v15-confirmation-result-v1",
        "status": "complete" if complete else "stopped",
        "complete": complete,
        "confirmation_gates_passed": complete and gates_pass,
        "plan_sha256": face_builder.PLAN_SHA256,
        "face_sha256": sha256(FACE),
        "face_checks_sha256": sha256(FACE_CHECKS),
        "readiness_sha256": sha256(READINESS),
        "range_report_sha256": sha256(RANGE_REPORT),
        "confirmation_attempt_sha256": sha256(CONFIRM_ATTEMPT),
        "source_hashes_unchanged": unchanged,
        "confirmation_queries": total_queries,
        "range_queries": range_report["model_queries"],
        "total_model_queries": range_report["model_queries"] + total_queries,
        "endpoint_results": endpoint_rows,
        "confirmation_summary": summary,
        "model_loads": model_loads,
        "checkpoint_loads": model_loads,
        "parameter_updates": 0,
        "optimizer_steps": 0,
        "dev_queries": 0,
        "cumulative_cpu_seconds": round(cumulative, 6),
        "cumulative_cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": round(stage_cpu, 6),
        "can_promote": False,
        "errors": errors,
        "summary_zh": "本报告评估N1-132续训包的v15泛化。与同seed N1-120 position母档的配对读数为描述性；结果不单独归因L_bind。",
    }
    write_new(CONFIRM_REPORT, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--range-check", action="store_true")
    modes.add_argument("--confirm", action="store_true")
    parser.add_argument("--owner-approved", action="store_true")
    args = parser.parse_args()
    result = (
        run_range_check(owner_approved=args.owner_approved)
        if args.range_check
        else run_confirmation(owner_approved=args.owner_approved)
    )
    print(
        json.dumps(
            {
                key: result.get(key)
                for key in (
                    "status",
                    "complete",
                    "range_gate",
                    "confirmation_summary",
                    "model_queries",
                    "confirmation_queries",
                    "total_model_queries",
                    "cumulative_cpu_seconds",
                    "can_promote",
                    "summary_zh",
                    "errors",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("complete") is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
