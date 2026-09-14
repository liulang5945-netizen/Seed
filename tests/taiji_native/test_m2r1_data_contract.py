from __future__ import annotations

import json
from pathlib import Path

from scripts.training.eval_taiji_m2r1_phase_c_canary import build_disjoint_phase_chain


def _write_corpus(path: Path) -> None:
    records = [
        {
            "text": (
                f"<m2r1-record-{index:04d}> "
                "A record-level boundary must survive every continuation course. "
                "Taiji keeps source provenance separate from byte partitioning."
            )
        }
        for index in range(2_400)
    ]
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
        encoding="utf-8",
    )


def test_m2r1_phase_chain_excludes_all_source_lineage_records() -> None:
    corpus = Path(".seed_test_tmp") / "m2r1-data-contract.jsonl"
    corpus.parent.mkdir(parents=True, exist_ok=True)
    try:
        _write_corpus(corpus)

        chain = build_disjoint_phase_chain(
            (corpus,),
            cohort_seeds=(11, 29),
            phase_c_seed=43,
            phase_c2_seed=44,
            profile="smoke",
        )

        assert chain.lineage_seeds == (11, 29)
        assert chain.phase_c.excluded_dataset_digests == tuple(
            dataset.digest
            for seed in chain.lineage_seeds
            for dataset in (chain.phase_a_by_seed[seed], chain.phase_b_by_seed[seed])
        )
        assert chain.phase_c2.excluded_dataset_digests == (
            *chain.phase_c.excluded_dataset_digests,
            chain.phase_c.digest,
        )
        assert chain.phase_c3.excluded_dataset_digests == (
            *chain.phase_c2.excluded_dataset_digests,
            chain.phase_c2.digest,
        )
        assert chain.overlap_counts
        assert all(value == 0 for value in chain.overlap_counts.values())

        source_sets = [
            set(chain.phase_a_by_seed[seed].selected_record_digests) for seed in chain.lineage_seeds
        ] + [
            set(chain.phase_b_by_seed[seed].selected_record_digests) for seed in chain.lineage_seeds
        ]
        phase_c_records = set(chain.phase_c.selected_record_digests)
        phase_c2_records = set(chain.phase_c2.selected_record_digests)
        phase_c3_records = set(chain.phase_c3.selected_record_digests)
        assert all(not records & phase_c_records for records in source_sets)
        assert all(not records & phase_c2_records for records in source_sets)
        assert all(not records & phase_c3_records for records in source_sets)
        assert not phase_c_records & phase_c2_records
        assert not phase_c_records & phase_c3_records
        assert not phase_c2_records & phase_c3_records
    finally:
        corpus.unlink(missing_ok=True)
