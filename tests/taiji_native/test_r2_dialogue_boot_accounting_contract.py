import pytest

from scripts.training.taiji_r2_dialogue_boot_accounting import reconciled_host_cpu_upper


def test_submicrosecond_boot_noise_keeps_prepared_cost_and_counts_all_host_cpu():
    snapshot = {
        "charged_cpu_seconds": 1_317_000.0,
        "boot_identity": 1792002779.5610702,
        "host_cpu_seconds": 1_783_000.0,
    }
    upper = reconciled_host_cpu_upper(
        1_317_000.0,
        snapshot,
        current_boot_identity=1792002779.56107,
        current_host_cpu_seconds=1_811_000.0,
    )
    assert upper == 1_345_000.0
    assert snapshot["charged_cpu_seconds"] == 1_317_000.0


@pytest.mark.parametrize(
    "boot,host,reason",
    [
        (1792002780.622641, 1_811_000.0, "boot identity"),
        (1792002779.56107, 1_782_999.0, "regressed"),
        (float("nan"), 1_811_000.0, "finite"),
    ],
)
def test_unresolved_boot_or_counter_cannot_credit_any_tail(boot, host, reason):
    snapshot = {
        "charged_cpu_seconds": 1_317_000.0,
        "boot_identity": 1792002779.5610702,
        "host_cpu_seconds": 1_783_000.0,
    }
    with pytest.raises(ValueError, match=reason):
        reconciled_host_cpu_upper(
            1_317_000.0, snapshot, current_boot_identity=boot, current_host_cpu_seconds=host
        )
