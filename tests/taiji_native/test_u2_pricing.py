"""22.4.4 price-table applicability and update-capacity formula."""

import math

from scripts.training.price_taiji_u2_22_4_4 import (
    ARMS,
    SEED_FACTOR,
    STAGE_PLAN_CPU,
    UPDATE_OVERHEAD,
    build_price_table,
)


def _rows(single_cpu: float) -> dict:
    row = {
        "observation_encoding": {"cpu": 0.0, "wall": 0.001},
        "query_expression": {"cpu": 0.01, "wall": 0.01},
        "policy": {"cpu": 0.01, "wall": 0.01},
        "single_learning": {"cpu": single_cpu, "wall": single_cpu},
        "save_rebuild": {"cpu": 0.01, "wall": 0.01},
        "replay_transfer": {"cpu": 0.01, "wall": 0.01},
        "birth_maturity": {"cpu": 0.001, "wall": 0.001},
    }
    return {
        "u2_reference": [{**row}],
        "transformer_sequence": [{**row, "policy": None, "birth_maturity": None}],
        "gru_sequence": [{**row, "policy": None, "birth_maturity": None}],
    }


def test_price_table_marks_baseline_policy_and_birth_not_applicable():
    table = build_price_table(_rows(0.5))
    assert set(table["components"]) == set(ARMS)
    assert table["components"]["u2_reference"]["policy"]["cpu"] == 0.01
    for arm in ("transformer_sequence", "gru_sequence"):
        assert table["components"][arm]["policy"]["status"] == "not_applicable_architecture"
        assert table["components"][arm]["birth_maturity"]["status"] == "not_applicable_architecture"
        assert table["components"][arm]["single_learning"]["cpu"] == 0.5


def test_update_capacity_uses_overhead_and_seed_factor():
    table = build_price_table(_rows(0.5))
    pack = 0.5 * UPDATE_OVERHEAD * SEED_FACTOR
    for stage, plan in STAGE_PLAN_CPU.items():
        assert table["updates_by_stage"][stage] == int(math.floor(plan / pack))
    assert table["u2_per_update_pack_cpu"] == round(pack, 6)
