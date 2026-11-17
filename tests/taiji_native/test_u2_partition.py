"""22.4.3 partition counts, disjointness and exposure invariants."""

from scripts.training.generate_taiji_u2_partition_22_4_3 import build_manifest


def test_partition_counts_and_disjoint_seeds():
    manifest = build_manifest()
    assert manifest["planned_worlds"] == 1280
    assert manifest["recipe_worlds"] == 1152
    assert manifest["splits"]["train"]["count"] == 768
    assert manifest["splits"]["development-fit"]["count"] == 192
    assert manifest["splits"]["development-check"]["count"] == 192
    assert manifest["splits"]["final"]["count"] == 128
    seed_sets = {
        name: set(manifest["splits"][name]["seeds"])
        for name in ("train", "development-fit", "development-check")
    }
    assert (
        len(seed_sets["train"] | seed_sets["development-fit"] | seed_sets["development-check"])
        == 1152
    )
    assert not (seed_sets["train"] & seed_sets["development-fit"])
    assert not (seed_sets["train"] & seed_sets["development-check"])
    assert not (seed_sets["development-fit"] & seed_sets["development-check"])


def test_birth_split_and_exposure_ledger():
    manifest = build_manifest()
    birth_fit = set(manifest["birth"]["fit"]["seeds"])
    birth_check = set(manifest["birth"]["check"]["seeds"])
    fit = set(manifest["splits"]["development-fit"]["seeds"])
    assert len(birth_fit) == 128
    assert len(birth_check) == 64
    assert birth_fit | birth_check == fit
    assert not (birth_fit & birth_check)
    assert manifest["birth"]["global_check_used"] is False
    exposure = manifest["check_exposure"]
    assert sum(exposure["allocations"].values()) == 8
    assert exposure["used"] == 0 and exposure["remaining"] == 8
    assert exposure["check_content_generated"] is False
    assert exposure["check_labels_read"] is False
    assert manifest["final_seal"]["authorization"] == "R-SEAL-1"
    assert manifest["final_seal"]["generated"] is False
    assert manifest["final_seal"]["read"] is False
    assert "seeds" not in manifest["splits"]["final"]


def test_partition_digest_is_deterministic():
    first = build_manifest()
    second = build_manifest()
    assert first["partition_digest"] == second["partition_digest"]
    assert len(first["partition_digest"]) == 64
