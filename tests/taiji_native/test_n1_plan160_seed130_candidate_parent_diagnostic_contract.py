from __future__ import annotations

import hashlib

from scripts.training import build_taiji_n1_plan160_seed130_candidate_parent_face as builder
from scripts.training import run_taiji_n1_plan160_seed130_candidate_parent_diagnostic as runner


def test_plan_sha_and_seed130_denominators_are_frozen() -> None:
    assert hashlib.sha256(builder.PLAN.read_bytes()).hexdigest() == builder.PLAN_SHA256
    assert builder.PRIOR_LINKED_CPU_SECONDS == 10_229.453125
    assert builder.TOTAL_QUERIES == 2_880
    assert len(builder.ENDPOINTS) == 6
    assert builder.FACE.name == "n1_plan160_v32_candidate_parent_seed130_20261020.json"


def test_seed128_and_seed129_values_are_in_seed130_exclusion_sets() -> None:
    endpoints, sources, exclusions = builder.source_context()
    excluded_entities = exclusions["excluded_entities"]
    excluded_words = exclusions["excluded_words"]
    entities128, words128 = builder.n159_builder._face_values(builder.read_json(builder.FACE128))
    entities129, words129 = builder.n159_builder._face_values(builder.read_json(builder.FACE129))
    assert entities128 | entities129 <= excluded_entities
    assert words128 | words129 <= excluded_words
    assert len(endpoints) == 6
    assert len(sources) > 600
    assert sources[str(builder.FACE128.resolve())] == builder.FACE128_SHA256
    assert sources[str(builder.FACE129.resolve())] == builder.FACE129_SHA256


def test_seed130_outputs_are_write_once_and_not_prior_faces() -> None:
    assert builder.FACE not in (builder.FACE128, builder.FACE129)
    assert runner.ATTEMPT != builder.n159_builder.PRIOR_ATTEMPT
    assert runner.RESULT != builder.n159_builder.PRIOR_RESULT
    assert not builder.FACE.exists()
    assert not builder.RESULT.exists()


def test_fresh_summary_checks_all_seed_branch_length_kind_strata() -> None:
    rows = []
    for seed, branch in builder.ENDPOINTS:
        for length in builder.LENGTHS:
            for kind in builder.KINDS:
                for index in range(builder.ITEMS_PER_KIND):
                    rows.append(
                        {
                            "seed": seed,
                            "branch": branch,
                            "context_length": length,
                            "kind": kind,
                            "free_exact": index % 2 == 0,
                            "output_hex": "756e6b6e6f776e"
                            if kind == "unknown" and index % 2 == 0
                            else "",
                        }
                    )
    summary = runner.fresh_summary(rows)
    assert len(summary) == 72
    assert all(row["items"] == 40 for row in summary)
    assert all(row["free_exact"] == 20 for row in summary)
    assert all(row["literal_unknown"] == 20 for row in summary if row["kind"] == "unknown")
