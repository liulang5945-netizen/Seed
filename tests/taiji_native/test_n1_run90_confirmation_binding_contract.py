"""PLAN-N1-90 binding contract.

The nine-endpoint confirmation is one-shot, so every risk in this file is entering
the face on a weak binding. Each refusal branch is walked with synthetic receipts in
``tmp_path`` and the accepting branch is walked once; no checkpoint is loaded, no
model is built and the v5 face is never queried here.

* the reused 85 chain is pinned by a loader wrapper that swaps the family protocol
  and restores it even when the load fails;
* families must agree on the sealed control report, on the position batch bytes and
  on every shared guarded input;
* nine distinct endpoints, each re-hashed from disk, are required before any query.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from scripts.training import (  # noqa: E402
    eval_taiji_n1_transition_factorial_confirmation_run89 as run90,
)
from scripts.training import (  # noqa: E402
    run_taiji_n1_transition_factorial_paired as train,
)

ARMS = ("position", "terminal", "cut")


def fake_report(arm: str) -> dict:
    return {
        "status": "complete",
        "complete": True,
        "can_promote": False,
        "sources_unchanged": True,
        "code_unchanged": True,
        "cpu_accounting_complete": True,
        "dev_queries": 0,
        "holdout_queries": 0,
        "confirmation_queries": 0,
        "protocol": {"learning_rate_schedule": dict(run90.SCHEDULE)},
        "code_identity": {"frozen_runner_sha256": run90.FROZEN_RUNNER_SHA256},
        "source_guards": {
            "E:\\Seed\\data\\checkpoints\\shared_train_source.jsonl": "a" * 64,
            f"E:\\Seed\\data\\checkpoints\\own_{arm}_zero.pt": "b" * 64,
        },
        "models": [
            {
                "seed": seed,
                "arm": arm,
                "restore_checks": [
                    {"path": f"E:\\Seed\\ep\\s{seed}_{arm}_zero.pt", "equal": True, "sha256": "z"},
                    {
                        "path": f"E:\\Seed\\ep\\s{seed}_{arm}_final.pt",
                        "equal": True,
                        "sha256": "f",
                    },
                ],
            }
            for seed in (11, 12, 13)
        ],
    }


def fake_verification(arm: str) -> dict:
    value = {
        "complete": True,
        "execution": {"model_loads": 0, "parameter_updates": 0, "can_promote": False},
        "preflight_zero_checkpoints": {f"s{11}_{arm}_zero.pt": "c" * 64},
    }
    if arm != "position":
        value["bound_position_batch"] = {
            "report_sha256": "p" * 64,
            "verification_sha256": "q" * 64,
        }
    return value


def receipts(tmp_path: Path, mutate=None) -> dict:
    built = {}
    for arm in ARMS:
        built[arm] = (
            tmp_path / f"{arm}.json",
            tmp_path / f"{arm}_verification.json",
        )
    position_report, position_verification = built["position"]
    position_report.write_text(
        __import__("json").dumps(fake_report("position")), encoding="utf-8", newline="\n"
    )
    position_verification.write_text(
        __import__("json").dumps(fake_verification("position")), encoding="utf-8", newline="\n"
    )
    binding = {
        "report_sha256": run90.sha(position_report),
        "verification_sha256": run90.sha(position_verification),
    }
    for arm in ("terminal", "cut"):
        (tmp_path / f"{arm}.json").write_text(
            __import__("json").dumps(fake_report(arm)), encoding="utf-8", newline="\n"
        )
        payload = fake_verification(arm)
        payload["bound_position_batch"] = dict(binding)
        (tmp_path / f"{arm}_verification.json").write_text(
            __import__("json").dumps(payload), encoding="utf-8", newline="\n"
        )
    original = dict(run90.FAMILIES)
    if mutate is not None:
        mutate(built)
    try:
        run90.FAMILIES = built
        return run90.family_receipts()
    finally:
        run90.FAMILIES = original


def tamper_json(tmp_path: Path, arm: str, key: str, value) -> None:
    path = tmp_path / f"{arm}.json"
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    cursor = payload
    trail = key.split(".")
    for step in trail[:-1]:
        cursor = cursor[step]
    cursor[trail[-1]] = value
    path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")


def test_require_walks_both_branches() -> None:
    run90.require(True, "quiet")
    with pytest.raises(run90.Refuse, match="loud"):
        run90.require(False, "loud")


def test_protocol_tolerates_absent_control_arm_reports() -> None:
    value = run90.protocol()
    assert value["schema"] == "n1-transition-factorial-three-family-v5-confirmation-v1"
    assert value["v5_manifest_sha256"] == run90.V5_SHA
    assert value["cpu_limit_seconds"] == 180
    assert value["batch_reports"]["position"] is not None
    assert set(value["batch_reports"]) == set(ARMS)


def test_family_receipts_accepts_three_sealed_families(tmp_path: Path) -> None:
    result = receipts(tmp_path)
    assert set(result) == set(ARMS)
    assert result["position"]["report_sha256"] is not None


@pytest.mark.parametrize(
    "key,value,message",
    [
        ("complete", False, "engineering-complete"),
        ("can_promote", True, "engineering-complete"),
        ("holdout_queries", 1, "unseen face"),
        ("protocol.learning_rate_schedule.final_cycle_lr", 0.001, "decay schedule"),
        ("code_identity.frozen_runner_sha256", "f" * 64, "run84 runner"),
    ],
)
def test_family_receipts_refuses_tampered_reports(
    tmp_path: Path, key: str, value, message: str
) -> None:
    def apply(built, key=key, value=value):
        tamper_json(tmp_path, "terminal", key, value)

    with pytest.raises(run90.Refuse, match=message):
        receipts(tmp_path, apply)


def test_family_receipts_refuses_wrong_model_count_or_arm(tmp_path: Path) -> None:
    def apply(built):
        import json

        path = tmp_path / "cut.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["models"] = payload["models"][:2]
        path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    with pytest.raises(run90.Refuse, match="exactly three models"):
        receipts(tmp_path, apply)


def test_family_receipts_refuses_missing_verification(tmp_path: Path) -> None:
    def apply(built):
        built["terminal"] = (built["terminal"][0], tmp_path / "absent_verification.json")

    with pytest.raises(run90.Refuse, match="verification is missing"):
        receipts(tmp_path, apply)


def test_family_receipts_refuses_forgotten_position_binding(tmp_path: Path) -> None:
    def apply(built):
        import json

        path = tmp_path / "cut_verification.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["bound_position_batch"]["report_sha256"] = "x" * 64
        path.write_text(json.dumps(payload), encoding="utf-8", newline="\n")

    with pytest.raises(run90.Refuse, match="bound to the adjudicated position batch"):
        receipts(tmp_path, apply)


def test_control_report_must_be_shared_across_families(tmp_path: Path) -> None:
    def apply(built):
        tamper_json(tmp_path, "terminal", "code_identity.control_report_sha256", "e" * 64)
        tamper_json(tmp_path, "position", "code_identity.control_report_sha256", "d" * 64)

    with pytest.raises(run90.Refuse, match="sealed control report"):
        receipts(tmp_path, apply)


def test_cross_family_guard_conflict_is_detected(tmp_path: Path) -> None:
    built = receipts(tmp_path)
    assert run90.cross_family_guard_conflicts(built) == {}
    tampered = copy.deepcopy(built)
    tampered["cut"]["report"]["source_guards"][
        "E:\\Seed\\data\\checkpoints\\shared_train_source.jsonl"
    ] = ("9" * 64)
    conflicts = run90.cross_family_guard_conflicts(tampered)
    assert list(conflicts) == ["E:\\Seed\\data\\checkpoints\\shared_train_source.jsonl"]


def test_endpoints_need_nine_existing_files_with_matching_bytes(tmp_path: Path) -> None:
    built = receipts(tmp_path)
    with pytest.raises(run90.Refuse, match="endpoint bytes differ"):
        run90.endpoints(built)

    sealed = {}
    for arm in ARMS:
        report = built[arm]["report"]
        for model in report["models"]:
            path = tmp_path / f"s{model['seed']}_{arm}_final.pt"
            path.write_bytes(b"sealed")
            model["restore_checks"][-1]["path"] = str(path)
            model["restore_checks"][-1]["sha256"] = run90.sha(path)
        sealed[arm] = report

    paths, guards = run90.endpoints(built)
    assert set(paths) == {(seed, arm) for seed in (11, 12, 13) for arm in ARMS}
    assert len(guards) >= len(paths)
    for arm in ARMS:
        assert str(tmp_path / f"s11_{arm}_final.pt") in guards

    (tmp_path / "s11_cut_final.pt").write_bytes(b"changed")
    with pytest.raises(run90.Refuse, match="endpoint bytes differ"):
        run90.endpoints(built)

    rebuilt = tmp_path / "s11_cut_final.pt"
    rebuilt.write_bytes(b"sealed")
    for model in built["cut"]["report"]["models"]:
        if model["seed"] == 11:
            model["restore_checks"][-1]["sha256"] = run90.sha(rebuilt)
    built["cut"]["report"]["models"].append(copy.deepcopy(built["cut"]["report"]["models"][0]))
    with pytest.raises(run90.Refuse, match="duplicate frozen endpoint"):
        run90.endpoints(built)


def test_loader_wrapper_swaps_and_restores_the_family_protocol() -> None:
    recorded = train.PROTOCOL
    calls = []

    def spy(payload):
        calls.append(train.PROTOCOL)
        return "loaded"

    original = run90.ORIGINAL_LOAD
    try:
        run90.ORIGINAL_LOAD = spy
        assert run90.load_with_family_protocol({"protocol": {"schema": "family-90"}}) == "loaded"
        assert calls == [{"schema": "family-90"}]
        assert train.PROTOCOL is recorded

        def refusing(payload):
            raise ValueError("marker refused")

        run90.ORIGINAL_LOAD = refusing
        with pytest.raises(ValueError, match="marker refused"):
            run90.load_with_family_protocol({"protocol": {"schema": "family-91"}})
        assert train.PROTOCOL is recorded
        run90.ORIGINAL_LOAD = spy
        assert run90.load_with_family_protocol({"protocol": recorded}) == "loaded"
        assert calls[-1] is recorded
    finally:
        run90.ORIGINAL_LOAD = original


def test_readiness_mode_maps_refusal_to_rc_two_and_success_to_zero(monkeypatch) -> None:
    def refuse() -> tuple:
        raise run90.Refuse("nine frozen final endpoints required")

    monkeypatch.setattr(run90, "readiness", refuse)
    assert run90.run_readiness() == 2

    monkeypatch.setattr(
        run90,
        "readiness",
        lambda: (
            {"p": "d"},
            {"v5": {"sha256": "2" * 64}, "readiness_cpu_seconds": 0.5},
            {(11, "cut"): Path("x")},
            [{"id": 1}],
        ),
    )
    assert run90.run_readiness() == 0


def test_approved_mode_refuses_without_writing_the_attempt(tmp_path: Path, monkeypatch) -> None:
    def refuse() -> dict:
        raise run90.Refuse("exact frozen v5 bytes required")

    pretend = tmp_path / "run90_attempt.json"
    monkeypatch.setattr(run90, "ATTEMPT", pretend)
    assert pretend.exists() is False
    monkeypatch.setattr(run90.base85, "audit", refuse)
    assert run90.run_approved() == 2
    assert pretend.exists() is False

    monkeypatch.setattr(
        run90.base85,
        "audit",
        lambda: {"complete": True, "passed": False, "cpu_seconds": 1.0, "gates": {}, "error": None},
    )
    assert run90.run_approved() == 0
    assert pretend.exists() is False


def test_install_patches_the_reused_evaluators_binding_globals() -> None:
    snapshot = (
        __import__(
            "scripts.training.eval_taiji_n1_transition_factorial_confirmation", fromlist=["x"]
        ).REPORT,
        sys.modules["scripts.training.eval_taiji_n1_transition_factorial_confirmation"].PROTOCOL,
    )
    try:
        run90.install()
        assert snapshot[0] is not run90.REPORT
        assert (
            sys.modules["scripts.training.eval_taiji_n1_transition_factorial_confirmation"].REPORT
            == run90.REPORT
        )
        assert (
            sys.modules[
                "scripts.training.eval_taiji_n1_transition_factorial_confirmation"
            ].PROTOCOL["schema"]
            == "n1-transition-factorial-three-family-v5-confirmation-v1"
        )
    finally:
        module = sys.modules["scripts.training.eval_taiji_n1_transition_factorial_confirmation"]
        module.REPORT = snapshot[0]
        module.PROTOCOL = snapshot[1]
        train.load = run90.ORIGINAL_LOAD
