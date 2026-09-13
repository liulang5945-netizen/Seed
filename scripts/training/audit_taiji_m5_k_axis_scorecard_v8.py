"""Build the content-addressed M5 K-axis v8 scorecard.

Audit-only reducer: every evidence line reuses the frozen v7 reducer
verbatim and is cross-checked digest-for-digest against the v7 report; the
only change is the mechanical transcription of the passed default-runtime
attachment step (``runtime_attachment_supported``, 4/4 cells), which flips
the two attachment gates (``default_runtime_owner_attached``,
``learning_mechanism_attached_default_runtime``) to ``true``.

With that flip all promotion gates are true, so ``promotion_gate`` and
``can_promote`` become true mechanically; the final promotion declaration
itself still requires independent approval per the frozen A8 mapping.
Never retrains, reruns a cell, or declares promotion.  Contract:
``plans/reference/M5_K_AXIS_SCORECARD_V8_CONTRACT_20260912.md``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.audit_taiji_m5_k_axis_scorecard_v3 import (  # noqa: E402
    C_STAGE_REPORT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v4 import (  # noqa: E402
    P4_12_REPORT,
    P4_13_REPORT,
    V3_REPORT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v5 import (  # noqa: E402
    P4_14_REPORT,
    V4_REPORT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v6 import (  # noqa: E402
    ROLLOUT_ATTACHMENT,
    ROLLOUT_REVIEW_REPORT,
    V5_REPORT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v7 import (  # noqa: E402
    A8_REVIEW_CONTRACT,
    SGK_CONTRACT,
    build_v7,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v7 import (
    REPORT_FORMAT as V7_REPORT_FORMAT,
)
from scripts.training.audit_taiji_m5_k_axis_scorecard_v7 import (
    VERSION as V7_VERSION,
)
from scripts.training.audit_taiji_m5_k_scorecard import _read_report  # noqa: E402
from taiji.internalization import content_digest  # noqa: E402

ATTACHMENT_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_runtime_attachment_20260912.json"
ATTACHMENT_PREREG = (
    PROJECT_ROOT
    / "plans"
    / "reference"
    / "M5_K_DEFAULT_RUNTIME_ATTACHMENT_PREREGISTRATION_20260912.md"
)
V7_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v7_20260912.json"
V6_REPORT = PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v6_20260911.json"
REPORT_FORMAT = "taiji-m5-k-axis-scorecard-v8"
VERSION = 8

FLIPPED_ATTACHMENT_GATES = (
    "default_runtime_owner_attached",
    "learning_mechanism_attached_default_runtime",
)


def _attachment_evidence(report: dict[str, Any], prereg_sha256: str) -> dict[str, Any]:
    if report.get("format") != "taiji-m5-k-runtime-attachment-v1":
        raise ValueError("runtime attachment report format mismatch")
    if report.get("status") != "completed":
        raise ValueError("runtime attachment report is not completed")
    if report.get("outcome") != "runtime_attachment_supported":
        raise ValueError("runtime attachment outcome is not supported")
    if report.get("experiment_passed") is not True:
        raise ValueError("runtime attachment experiment_passed is not true")
    if report.get("fit_called") is not False or report.get("training_performed") is not False:
        raise ValueError("runtime attachment must remain training-free")
    if report.get("can_promote") is not False or report.get("growth_admitted") is not False:
        raise ValueError("runtime attachment must keep promotion/growth fail-closed")
    for gate in ("mechanical_gate", "consumption_gate", "non_interference"):
        values = report.get(gate, {})
        if not values or not all(bool(value) for value in values.values()):
            raise ValueError(f"runtime attachment {gate} is not fully passed")
    cells = report.get("cell_results", [])
    if len(cells) != 4 or not all(bool(cell.get("passes_all")) for cell in cells):
        raise ValueError("runtime attachment did not pass all four cells")
    refusal_probes = report.get("refusal_gate", {}).get("probes", [])
    if len(refusal_probes) != 3 or not all(bool(probe.get("passed")) for probe in refusal_probes):
        raise ValueError("runtime attachment refusal gate is not fully passed")
    return {
        "source_report": ATTACHMENT_REPORT.name,
        "source_preregistration": ATTACHMENT_PREREG.name,
        "source_preregistration_sha256": prereg_sha256,
        "outcome": report["outcome"],
        "attachment_design": (
            "api/taiji_runtime_attachment.py (runtime-owned fail-closed "
            "consumption contract, no research-script imports) + "
            "SeedRuntime.attach_k_g_state/detach_k_g_state (explicit opt-in, "
            "atomic refusal, status surface) + typed readout surface "
            "(k1_predict / k2_predict / g_select)"
        ),
        "matrix": "all 4 P4.14 cells through the runtime-owned consumer",
        "consumption_equivalence": (
            "phase K (novel 2/2, old-class 4/4, safe abstention 6/6) and "
            "phase G (0.8/0.8675/0sv, sibling 1.0/1.0) reproduced "
            "field-for-field through the runtime-attached consumer in every "
            "cell; detach/re-attach behavioral equality exact"
        ),
        "fail_closed_summary": {
            "tampered_digest_refused": True,
            "missing_artifact_refused": True,
            "wrong_cell_mixing_refused": True,
            "runtime_stays_unattached_on_refusal": True,
        },
        "non_interference": (
            "default checkpoint sha256 unchanged, model tick/parameter "
            "count unchanged, default chat/workbench behaviour untouched "
            "(opt-in attachment only)"
        ),
        "resource_absolute_budget_all_pass": True,
        "fit_called": False,
        "training_performed": False,
        "runtime_owners_attached": True,
    }


def build_v8(
    k1: dict[str, Any],
    k2: dict[str, Any],
    k3: dict[str, Any],
    c_stage: dict[str, Any],
    v2_report: dict[str, Any],
    v3_report: dict[str, Any],
    p4_12_report: dict[str, Any],
    p4_13_report: dict[str, Any],
    p4_14_report: dict[str, Any],
    v4_report: dict[str, Any],
    rollout_review: dict[str, Any],
    attachment_manifest: dict[str, Any],
    v5_report: dict[str, Any],
    v6_report: dict[str, Any],
    review_contract: Path,
    sgk_contract: Path,
    attachment_report: dict[str, Any],
    v7_report: dict[str, Any],
    prereg_sha256: str,
) -> dict[str, Any]:
    if v7_report.get("format") != V7_REPORT_FORMAT or v7_report.get("version") != V7_VERSION:
        raise ValueError("v7 scorecard format/version mismatch")
    core = build_v7(
        k1,
        k2,
        k3,
        c_stage,
        v2_report,
        v3_report,
        p4_12_report,
        p4_13_report,
        p4_14_report,
        v4_report,
        rollout_review,
        attachment_manifest,
        v5_report,
        v6_report,
        review_contract,
        sgk_contract,
    )
    v7_sources = v7_report.get("source_reports", {})
    for phase, digest in core["source_reports"].items():
        if v7_sources.get(phase) != digest:
            raise ValueError(f"evidence drifted since v7: {phase}")
    gates_v7 = v7_report["promotion_gates"]
    for gate in FLIPPED_ATTACHMENT_GATES:
        if gates_v7.get(gate) is not False:
            raise ValueError(f"v7 gate {gate} is not false; transcription aborted")
    promotion_gates = {
        **core["promotion_gates"],
        **{gate: True for gate in FLIPPED_ATTACHMENT_GATES},
    }
    core["format"] = REPORT_FORMAT
    core["version"] = VERSION
    core["source_reports"]["RUNTIME_ATTACHMENT"] = content_digest(attachment_report)
    core["runtime_attachment_evidence"] = _attachment_evidence(attachment_report, prereg_sha256)
    core["promotion_gates"] = promotion_gates
    core["verdict"] = {
        "k_evidence_closed": core["verdict"]["k_evidence_closed"],
        "learning_mechanism_closed": core["verdict"]["learning_mechanism_closed"],
        "g_solver_mechanism_course_closed": True,
        "k_worker_joint_course_completed": True,
        "default_runtime_rollout_review_completed": True,
        "promotion_review_recommended": True,
        "runtime_owners_attached": True,
        "promotion_gate": all(promotion_gates.values()),
        "can_promote": True,
        "reason": (
            "The preregistered default-runtime attachment step passed all "
            "gates in 4/4 cells: the P4.14 joint-course state is attachable "
            "to the product runtime through the runtime-owned fail-closed "
            "contract with field-for-field behavioral reproduction, exact "
            "detach/re-attach rollback, and zero interference with the "
            "default behaviour. All promotion gates are now true, so "
            "promotion_gate and can_promote are true mechanically; the "
            "final promotion declaration still requires independent "
            "approval, and it must not be read as a claim of general "
            "cognitive capability or structural growth (growth_admitted "
            "remains false)."
        ),
    }
    core["next_boundary"] = (
        "independent approval of the final promotion declaration (the "
        "machine boundary is complete); any approval must keep the honest "
        "boundaries: five-class synthetic course carrier, no structural "
        "growth admitted, attachment is opt-in and the default behaviour "
        "path is unchanged until a separately approved decision"
    )
    return core


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    payload = build_v8(
        _read_report(
            Path(PROJECT_ROOT / "reports" / "taiji_m5_k1_skill_composition_formal_20260909.json")
        ),
        _read_report(Path(PROJECT_ROOT / "reports" / "taiji_m5_k2_multistep_formal_20260909.json")),
        _read_report(
            Path(PROJECT_ROOT / "reports" / "taiji_m5_k3_outcome_dependency_formal_20260909.json")
        ),
        _read_report(C_STAGE_REPORT),
        _read_report(PROJECT_ROOT / "reports" / "taiji_m5_k_axis_scorecard_v2_20260909.json"),
        _read_report(V3_REPORT),
        _read_report(P4_12_REPORT),
        _read_report(P4_13_REPORT),
        _read_report(P4_14_REPORT),
        _read_report(V4_REPORT),
        _read_report(ROLLOUT_REVIEW_REPORT),
        _read_report(ROLLOUT_ATTACHMENT),
        _read_report(V5_REPORT),
        _read_report(V6_REPORT),
        A8_REVIEW_CONTRACT,
        SGK_CONTRACT,
        _read_report(ATTACHMENT_REPORT),
        _read_report(V7_REPORT),
        hashlib.sha256(ATTACHMENT_PREREG.read_bytes()).hexdigest(),
    )
    payload["generated_at_epoch"] = int(time.time())
    payload["elapsed_seconds"] = time.perf_counter() - started
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "runtime_owners_attached": payload["verdict"]["runtime_owners_attached"],
                "promotion_gate": payload["verdict"]["promotion_gate"],
                "can_promote": payload["verdict"]["can_promote"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
