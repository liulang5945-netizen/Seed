"""Pure synthetic evidence contracts for historical versus phase-specific credit."""

from copy import deepcopy

import pytest

from scripts.training import audit_taiji_n1_credit_lineage as audit


def evidence():
    mother = dict(
        complete=True,
        final_steps_by_parameter_group=dict(reader=[18432], local=[12288], context=[6144]),
        final_optimizer_state_entries=35,
        protocol=dict(final_historical_updates=27648),
    )
    models = []
    checks = []
    rows = []
    for seed in audit.train.SEEDS:
        for arm in audit.train.ARMS:
            steps = {f"writer.{i}": 18432 if arm == "cut" else 24576 for i in range(4)}
            steps["embedding.weight"] = 24576
            models.append(dict(seed=seed, arm=arm, final_steps=steps))
            checks.append(
                dict(
                    path=f"s{seed}_{arm}_zero.pt",
                    signature=dict(
                        relation_head_changed=True,
                        writer_unchanged=arm == "cut",
                        writer_gradient_l2=0 if arm == "cut" else 1,
                    ),
                )
            )
        for arm, condition in (
            ("position", "native"),
            ("position", "mother_writer"),
            ("cut", "native"),
            ("cut", "position_writer"),
        ):
            for i in range(576):
                changed = (
                    seed == 12 and arm == "position" and condition == "mother_writer" and i < 2
                )
                rows.append(
                    dict(
                        seed=seed,
                        arm=arm,
                        condition=condition,
                        id=f"known-{i}",
                        evaluation_set="original",
                        free_exact=(i != 0) if not changed else (i == 0),
                        raw_hex="00" if not changed else "01",
                    )
                )
    return {
        "52": mother,
        "70": dict(complete=True, models=models, restore_checks=checks),
        "71": dict(
            complete=True,
            passed=False,
            aggregate_exact=dict(position=91, terminal=20, cut=92),
            gates=dict(advantage_cut=False),
        ),
        "72": dict(complete=True, rows=rows),
    }


def test_phase_scope_does_not_erase_inherited_learning_or_failed_gate():
    result = audit.summarize(evidence())
    assert result["mother_writer_adam_step"] == 18432
    assert result["mother_historical_updates"] == 27648
    assert result["independent_end_to_end_seeds"] is False
    assert result["original71_passed"] is False
    assert result["last_phase_writer_net_gain_established"] is False
    change = next(
        r for r in result["writer_exchange_changes"] if r["seed"] == 12 and r["arm"] == "position"
    )
    assert change["gained"] == change["lost"] == 1
    assert change["raw_changed"] == 2
    assert not result["whole_project_complete"] and "retention" in result["missing_exits"]


@pytest.mark.parametrize(
    "mutation", ("writer", "embedding", "credit", "gate", "rows", "incomplete")
)
def test_contradictory_or_partial_evidence_is_rejected(mutation):
    records = deepcopy(evidence())
    if mutation == "writer":
        records["70"]["models"][2]["final_steps"]["writer.0"] += 1
    elif mutation == "embedding":
        records["70"]["models"][2]["final_steps"]["embedding.weight"] = 18432
    elif mutation == "credit":
        records["70"]["restore_checks"][2]["signature"]["writer_gradient_l2"] = 1
    elif mutation == "gate":
        records["71"]["passed"] = True
    elif mutation == "rows":
        records["72"]["rows"].pop()
    else:
        records["52"]["complete"] = False
    with pytest.raises(ValueError):
        audit.summarize(records)
