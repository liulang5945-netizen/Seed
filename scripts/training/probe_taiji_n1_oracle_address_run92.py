"""PLAN-N1-92 乙′：在已消费的 v5 面上做 oracle 地址探针，问"候选有没有被送进答案头"。

零训练、零新面。机制那半条链**不重抄**：注意力替换与解码重放直接 import 已封存的
`audit_taiji_n1_address_intervention.py`（该件字节被在库报告钉住，本件只读不改）。九枚端点
的装载走 90 那台族回执加载器。读数一律标"机制"，不得称未见、不得称能力（PLAN-N1-92 §0）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from scripts.training import (  # noqa: E402
    audit_taiji_n1_address_intervention as oracle,
)
from scripts.training import (  # noqa: E402
    audit_taiji_n1_train_content_read as trainface,
)
from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation_run89 as bind,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_native_evidence_paired as paired,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_v3_train_calibration as calibration,
)

PLAN = ROOT / "plans/reference/PLAN-N1-92_oracle_address_probe_run92_20261017.md"
V5 = ROOT / "plans/manifests/n1_fresh_confirmation_v5_20261016.json"
V5_SHA = "22ba37976bc6eaaaf2425b84714a6efa6e95e7e5aa05f32eb3a7c0c10d9ffc2d"
REPORT = ROOT / "reports/taiji_n1_oracle_address_probe_run92.json"
ATTEMPT = ROOT / "reports/taiji_n1_oracle_address_probe_run92_attempt.json"

ARM_ORDER = ("position", "terminal", "cut")
SEEDS = (11, 12, 13)
ADDRESSABLE_KINDS = ("lookup", "revision")
CPU_LIMIT_SECONDS = 120
NATIVE_CEILING = 12
GAIN_FLOOR = 4
LOSS_CEILING = 2
ARM_SPREAD_LIMIT = 1
EXCLUDE_REASON = {
    "transition": "no_unique_uses_answer_observation",
    "unknown": "answer_is_the_literal_unknown_with_no_source",
}


class Refuse(Exception):
    """一条准入或预算条件不成立：本件不出结论，rc=2。"""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Refuse(message)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def oracle_addresses(item: dict, tape) -> tuple[str, list[int]]:
    """与 `oracle.addresses` 同式，只去掉那道 train 专用拒入；逐位锚点负责证明两式一致。"""
    answer = item["answer"].encode()
    sources = [
        index
        for index, text in enumerate(item["observations"])
        if text.endswith(f" uses {item['answer']}.")
    ]
    if len(sources) != 1 or len(answer) != 6:
        raise ValueError("unique source and six material bytes required")
    index = sources[0]
    source = f"source-{index + 1}"
    start = item["observations"][index].encode().rfind(answer)
    return source, [tape.positions.index((source, start + step)) for step in range(6)]


def classify(item: dict) -> tuple[bool, str]:
    kind = item["kind"]
    if kind not in ADDRESSABLE_KINDS:
        return False, EXCLUDE_REASON.get(kind, f"excluded_kind_{kind}")
    answer = item["answer"].encode()
    sources = [
        index
        for index, text in enumerate(item["observations"])
        if text.endswith(f" uses {item['answer']}.")
    ]
    if len(answer) != 6:
        return False, f"answer_bytes_{len(answer)}"
    if len(sources) != 1:
        return False, f"source_matches_{len(sources)}"
    return True, "addressable"


def anchor() -> dict:
    """逐位锚点：在 train 面上我的提取器必须与封存件 `addresses()` 逐个整数相等。"""
    require(
        sha(calibration.MANIFEST) == calibration.MANIFEST_SHA,
        "the train anchor face changed since it was sealed",
    )
    items = trainface.select(read_json(calibration.MANIFEST))
    model = oracle.NativeEvidenceCandidate()
    equal = 0
    unequal = 0
    skipped = 0
    with torch.no_grad():
        for item in items:
            session = f"anchor92-{item['id']}"
            paired.observe(model, paired.project(item), session)
            tape = model.evidence(session)
            try:
                expected = oracle.addresses(item, tape)
            except ValueError:
                skipped += 1
                del model._observations[session]
                continue
            got = oracle_addresses(item, tape)
            if got == expected:
                equal += 1
            else:
                unequal += 1
            del model._observations[session]
    require(unequal == 0, f"the reimplemented extractor diverges on {unequal} train items")
    require(equal > 0, "the anchor walked zero addressable train items")
    return {"checked_equal": equal, "checked_unequal": unequal, "rejected_by_pinned_guard": skipped}


def coverage() -> dict:
    require(sha(V5) == V5_SHA, f"v5 face bytes differ from the consumed seal: {sha(V5)}")
    items = read_json(V5)["episodes"]
    reasons: dict = {}
    addressable = []
    for item in items:
        ok, reason = classify(item)
        if ok:
            addressable.append(item)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    require(len(addressable) == 16, f"the addressable set is {len(addressable)}, expected 16")
    require(
        sum(reasons.values()) == 16,
        f"the excluded set must hold the other 16 items, got {reasons}",
    )
    return {
        "items_total": len(items),
        "addressable": len(addressable),
        "addressable_ids": [item["id"] for item in addressable],
        "excluded_reasons": dict(sorted(reasons.items())),
    }


def probe_endpoint(seed: int, arm: str, path: Path, items: list, budget: float) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, optimizer = bind.load_with_family_protocol(payload)
    before = paired.fingerprint(model.bundle(optimizer))
    stored = model.bundle(optimizer)["sessions"]
    per_mode: dict = {mode: {"six_exact": 0, "seven_exact": 0} for mode in oracle.MODES}
    last_step: dict = {mode: 0 for mode in oracle.MODES}
    gains = 0
    losses = 0
    rows = []
    with torch.no_grad():
        for item in items:
            if time.process_time() >= budget:
                raise Refuse("the CPU reserve for this probe was reached before it finished")
            session = f"oracle92-{seed}-{arm}-{item['id']}"
            if session in model._observations:
                del model._observations[session]
            paired.observe(model, paired.project(item), session)
            tape = paired.evidence(model, session, arm)
            source, positions = oracle_addresses(item, tape)
            target = item["answer"].encode()
            question = item["question"].encode()
            observed = {}
            for mode in oracle.MODES:
                if mode == "native":
                    logits, _ = model.teacher_logits(tape, session, question, target)
                else:
                    logits = oracle.teacher(
                        model, tape, session, question, target, mode, source, positions
                    )
                observed[mode] = oracle.profile(logits, target)
                per_mode[mode]["six_exact"] += int(observed[mode]["six_exact"])
                per_mode[mode]["seven_exact"] += int(observed[mode]["seven_exact"])
                last_step[mode] += int(observed[mode]["correct"][6])
            gains += int(observed["position"]["six_exact"] and not observed["native"]["six_exact"])
            losses += int(observed["native"]["six_exact"] and not observed["position"]["six_exact"])
            del model._observations[session]
            rows.append(
                {
                    "id": item["id"],
                    "kind": item["kind"],
                    **{
                        mode: {
                            "six_exact": observed[mode]["six_exact"],
                            "seven_exact": observed[mode]["seven_exact"],
                            "teacher_tokens_correct": observed[mode]["teacher_tokens_correct"],
                        }
                        for mode in oracle.MODES
                    },
                }
            )
    require(
        before == paired.fingerprint(model.bundle(optimizer)),
        f"the probe changed parameters on seed {seed} {arm}",
    )
    require(
        stored == model.bundle(optimizer)["sessions"],
        f"the probe left sessions behind on seed {seed} {arm}",
    )
    return {
        "seed": seed,
        "arm": arm,
        "items": len(items),
        "modes": per_mode,
        "last_step_byte_correct": last_step,
        "position_paired_gains": gains,
        "position_paired_losses": losses,
        "ruler_usable": per_mode["native"]["six_exact"] <= NATIVE_CEILING,
        "rows": rows,
    }


def adjudicate(results: list, covered: int) -> dict:
    usable = [entry for entry in results if entry["ruler_usable"]]
    lifted = [
        entry
        for entry in usable
        if entry["position_paired_gains"] >= GAIN_FLOOR
        and entry["position_paired_losses"] <= LOSS_CEILING
    ]
    spread = {}
    for seed in SEEDS:
        values = [
            entry["modes"]["position"]["six_exact"]
            for entry in results
            if entry["seed"] == seed and entry["ruler_usable"]
        ]
        spread[str(seed)] = max(values) - min(values) if len(values) == len(ARM_ORDER) else None
    unresolved = len(usable) < 6
    supported = (not unresolved) and len(lifted) * 2 >= len(usable)
    indistinguishable = all(
        value is not None and value <= ARM_SPREAD_LIMIT for value in spread.values()
    )
    if unresolved:
        exit_named = "⑷ ruler_unusable"
    elif supported and not indistinguishable:
        exit_named = "⑴ product-side read path"
    elif supported and indistinguishable:
        exit_named = "⑵ arm variable inert at mechanism level"
    else:
        exit_named = "⑶ bottleneck sits above the tape address"
    return {
        "endpoints_usable": len(usable),
        "endpoints_lifted": len(lifted),
        "oracle_lift_supported": supported and not unresolved,
        "arm_spread_by_seed": spread,
        "arm_indistinguishable": indistinguishable,
        "ruler_unusable": unresolved,
        "exit_named": exit_named,
        "lines": {
            "native_ceiling": NATIVE_CEILING,
            "gain_floor": GAIN_FLOOR,
            "loss_ceiling": LOSS_CEILING,
            "arm_spread_limit": ARM_SPREAD_LIMIT,
            "addressable_items_per_endpoint": covered,
        },
    }


def run_readiness() -> int:
    receipt_set = bind.family_receipts()
    paths, _guards = bind.endpoints(receipt_set)
    require(len(paths) == 9, "nine frozen endpoints required")
    anchored = anchor()
    covered = coverage()
    print(
        json.dumps(
            {
                "mode": "readiness",
                "anchor": anchored,
                "coverage": covered,
                "endpoints": sorted(f"{seed}-{arm}" for (seed, arm) in paths),
                "cpu_limit_seconds": CPU_LIMIT_SECONDS,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


def run_approved() -> int:
    started = time.process_time()
    require(not REPORT.exists(), f"probe output already exists: {REPORT}")
    require(not ATTEMPT.exists(), f"probe attempt already exists: {ATTEMPT}")
    receipt_set = bind.family_receipts()
    paths, _guards = bind.endpoints(receipt_set)
    anchored = anchor()
    covered = coverage()
    ids = set(covered["addressable_ids"])
    items = [item for item in read_json(V5)["episodes"] if item["id"] in ids]
    ATTEMPT.write_text(
        json.dumps(
            {
                "format": "taiji-n1-92-oracle-attempt-v1",
                "started_process_time": started,
                "first_model_loaded_after_this_file": True,
                "anchor": anchored,
                "coverage": covered,
                "unseen_claim_allowed": False,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    results = []
    budget = started + CPU_LIMIT_SECONDS
    status = "published"
    error = None
    try:
        for seed in SEEDS:
            for arm in ARM_ORDER:
                results.append(probe_endpoint(seed, arm, paths[(seed, arm)], items, budget))
    except Exception as failure:  # PLAN-N1-92 §5 要求：任何中断都如实出版 partial 行
        status = "stopped"
        error = f"{type(failure).__name__}: {failure}"
    payload = {
        "format": "taiji-n1-92-oracle-address-probe-v1",
        "plan": str(PLAN),
        "plan_sha256": sha(PLAN),
        "probe_sha256": sha(Path(__file__).resolve()),
        "reused_oracle_instrument_sha256": sha(
            ROOT / "scripts/training/audit_taiji_n1_address_intervention.py"
        ),
        "v5_sha256": sha(V5),
        "anchor_face_sha256": sha(calibration.MANIFEST),
        "consumed_face_use": "known_face_mechanism_probe_only",
        "unseen_claim_allowed": False,
        "oracle_is_capability": False,
        "parameter_updates": 0,
        "training_queries": 0,
        "anchor": anchored,
        "coverage": covered,
        "endpoints": results,
        "teacher_sequences": len(results) * len(items) * len(oracle.MODES),
        "cpu_limit_seconds": CPU_LIMIT_SECONDS,
        "cpu_seconds": round(time.process_time() - started, 4),
        "status": status,
        "error": error,
    }
    if results:
        payload["adjudication"] = adjudicate(results, covered["addressable"])
    REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(
        json.dumps(
            {
                "mode": "approved",
                "status": status,
                "teacher_sequences": payload["teacher_sequences"],
                "cpu_seconds": payload["cpu_seconds"],
                "adjudication": payload.get("adjudication"),
                "out": str(REPORT),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0 if status == "published" else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--readiness", action="store_true", help="锚点、覆盖与九枚端点齐备性，不读检查点"
    )
    parser.add_argument("--execute-approved", action="store_true", help="出版单枚探针报告")
    arguments = parser.parse_args(argv)
    try:
        require(
            arguments.readiness != arguments.execute_approved,
            "choose exactly one of --readiness/--execute-approved",
        )
        if arguments.readiness:
            return run_readiness()
        return run_approved()
    except Refuse as refusal:
        print(f"REFUSE: {refusal}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
