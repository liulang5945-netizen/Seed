"""Artifact-free logging diagnostics preserve failed gates and exact denominators."""

from copy import deepcopy

import pytest

from scripts.training import diagnose_taiji_n1_fresh_training_logs as diagnostic


def synthetic_rows():
    directions = ("north", "east", "south", "west")
    items = []
    for i in range(192):
        kind = diagnostic.train.paired.KINDS[i % 4]
        direction = directions[(i // 4) % 4]
        action = "left" if (i // 16) % 2 == 0 else "right"
        answer = directions[(directions.index(direction) + (1 if action == "right" else -1)) % 4]
        items.append(
            dict(
                id=f"train-{i:03}",
                kind=kind,
                split="train",
                observations=[f"name faces {direction}.", f"name turns {action}."],
                answer=answer,
            )
        )
    pairs = [
        dict(item, id="pair-" + item["id"])
        for item in items
        if item["kind"] in ("lookup", "unknown")
    ]
    views = dict(original=items, pair=pairs, rename=items, rename_pair=pairs)
    rows = []
    for step in diagnostic.train.MILESTONES:
        for view, selected in views.items():
            for item in selected:
                teacher = step >= 12288
                free = teacher and not (
                    step == 24576
                    and item["kind"] == "transition"
                    and "left" in item["observations"][1]
                )
                rows.append(
                    dict(
                        seed=13,
                        arm="position",
                        completed_updates=step,
                        evaluation_set=view,
                        id=item["id"],
                        kind=item["kind"],
                        teacher_exact=teacher,
                        free_exact=free,
                        raw_hex=(item["answer"].encode() if free else b"wrong").hex(),
                        stop_reason="boundary" if free else "cap",
                    )
                )
    return rows, views


def test_eight_transition_cells_and_full_phase_gains_losses_without_models(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("log diagnostics must never load checkpoints")

    monkeypatch.setattr(diagnostic.train.torch, "load", forbidden)
    rows, views = synthetic_rows()
    phases = diagnostic.analyze_rows(rows, views, 13, "position")
    assert len(phases) == 20
    final = next(p for p in phases if p["completed_updates"] == 24576 and p["view"] == "original")
    assert final["by_kind"]["transition"] == dict(
        items=48, free_exact=24, teacher_exact=48, teacher_free_disagree=24, gained=0, lost=24
    )
    assert len(final["transition_cells"]) == 8
    for key, cell in final["transition_cells"].items():
        assert cell["items"] == 6 and sum(cell["confusion"].values()) == 6
        assert cell["free_exact"] == (0 if key.endswith("left") else 6)
    learned = next(p for p in phases if p["completed_updates"] == 12288 and p["view"] == "original")
    assert learned["by_kind"]["transition"]["gained"] == 48
    assert phases[0]["by_kind"]["transition"]["gained"] is None
    milestones = []
    for step in diagnostic.train.MILESTONES:
        selected = {p["view"]: p for p in phases if p["completed_updates"] == step}
        original = {k: v["free_exact"] for k, v in selected["original"]["by_kind"].items()}
        renamed = {k: v["free_exact"] for k, v in selected["rename"]["by_kind"].items()}
        milestones.append(
            dict(
                completed_updates=step,
                free_exact=sum(original.values()),
                free_by_kind=original,
                rename_free_exact=sum(renamed.values()),
                rename_free_by_kind=renamed,
                pair_exact=sum(v["free_exact"] for v in selected["pair"]["by_kind"].values()),
                rename_pair_exact=sum(
                    v["free_exact"] for v in selected["rename_pair"]["by_kind"].values()
                ),
            )
        )
    diagnostic.validate_phase_summary(phases, milestones)
    milestones[-1]["free_exact"] += 1
    with pytest.raises(ValueError, match="milestone"):
        diagnostic.validate_phase_summary(phases, milestones)


@pytest.mark.parametrize("changed", ["missing", "duplicate", "kind", "seed", "view"])
def test_incomplete_or_misaligned_rows_rejected(changed):
    rows, views = synthetic_rows()
    if changed == "missing":
        rows.pop()
    elif changed == "duplicate":
        rows[-1] = deepcopy(rows[-2])
    elif changed == "kind":
        rows[0]["kind"] = "transition"
    elif changed == "seed":
        rows[0]["seed"] = 11
    else:
        rows[0]["evaluation_set"] = "holdout"
    with pytest.raises(ValueError):
        diagnostic.analyze_rows(rows, views, 13, "position")


def profile_record():
    groups = {
        g: dict(
            actual_l2=0.5,
            cap=1.0,
            gradient_l2=0.1,
            parameter_l2=10.0,
            proposed_l2=2.0,
            scale=0.25,
            clipped=True,
        )
        for g in diagnostic.GROUPS
    }
    return dict(
        losses=[
            dict(total=0.5, qa=0.25, observation=1.0, qa_sequences=1 + i % 2) for i in range(24576)
        ],
        measurements=[dict(step=i, cycle=i // 6144, groups=groups) for i in range(24576)],
    )


def test_complete_update_windows_and_final_1024_are_preserved():
    record = profile_record()
    profiles = diagnostic.profiles(record)
    assert [(p["start"], p["end"]) for p in profiles] == [
        (0, 6144),
        (6144, 12288),
        (12288, 18432),
        (18432, 24576),
        (23552, 24576),
    ]
    assert profiles[-1]["qa_sequences"] == 1536
    assert profiles[-1]["groups"]["writer"]["clipped_count"] == 1024
    assert profiles[0]["groups"]["reader"]["actual_l2"] == dict(
        count=6144, mean=0.5, min=0.5, max=0.5
    )
    record["measurements"][0]["step"] = 1
    with pytest.raises(ValueError, match="ordered"):
        diagnostic.profiles(record)
    record = profile_record()
    record["measurements"][0]["groups"]["reader"]["gradient_l2"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        diagnostic.profiles(record)


def complete_receipt(monkeypatch):
    monkeypatch.setattr(diagnostic.train, "identity", lambda: {"synthetic": "identity"})
    return dict(
        complete=True,
        status="complete",
        protocol=diagnostic.train.PROTOCOL,
        code_identity=diagnostic.train.identity(),
        models=[
            dict(seed=s, arm=a, training_sufficient=not (s == 13 and a == "position"))
            for s in (11, 12, 13)
            for a in ("position", "terminal", "cut")
        ],
        completed_updates=221184,
        signature_clone_updates=72,
        synthetic_qa_sequences=144,
        cpu_seconds=7000,
        sources_unchanged=True,
        code_unchanged=True,
        cpu_accounting_complete=True,
        training_sufficient=False,
        dev_queries=0,
        holdout_queries=0,
        confirmation_queries=0,
    )


def test_failed_training_gate_is_valid_diagnostic_evidence_not_confirmation(monkeypatch):
    record = complete_receipt(monkeypatch)
    diagnostic.validate_complete(record)
    record["training_sufficient"] = True
    with pytest.raises(ValueError):
        diagnostic.validate_complete(record)


@pytest.mark.parametrize(
    "changed", ["duplicate", "partial", "updates", "query", "budget", "accounting"]
)
def test_incomplete_training_receipt_is_rejected(monkeypatch, changed):
    record = complete_receipt(monkeypatch)
    if changed == "duplicate":
        record["models"][-1] = record["models"][0]
    elif changed == "partial":
        record["complete"] = False
    elif changed == "updates":
        record["completed_updates"] -= 1
    elif changed == "query":
        record["confirmation_queries"] = 1
    elif changed == "budget":
        record["cpu_seconds"] = 9001
    else:
        record["cpu_accounting_complete"] = False
    with pytest.raises(ValueError):
        diagnostic.validate_complete(record)


def test_unbound_or_existing_diagnostic_refuses_before_reads(tmp_path, monkeypatch):
    target = tmp_path / "report.json"
    monkeypatch.setattr(diagnostic, "REPORT", target)
    monkeypatch.setattr(diagnostic, "TRAIN_SHA", None)

    def forbidden(*args, **kwargs):
        pytest.fail("early refusal reached source reads")

    monkeypatch.setattr(diagnostic.train.paired, "sha", forbidden)
    with pytest.raises(ValueError, match="not yet bound"):
        diagnostic.execute()
    target.write_text("sealed", encoding="utf-8")
    with pytest.raises(FileExistsError):
        diagnostic.execute()
    assert target.read_text(encoding="utf-8") == "sealed"
