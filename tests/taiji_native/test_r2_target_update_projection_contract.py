import pytest

from scripts.training import taiji_r2_target_update_projection as p


def events(lengths):
    total = 0
    rows = [
        {
            "event": "episode_completed",
            "stage": "pretraining",
            "target_byte_updates": 0,
            "charged_cpu_seconds": 1000.0,
        }
    ]
    for length in lengths:
        total += length
        rows.append(
            {
                "event": "episode_completed",
                "stage": "pretraining",
                "target_byte_updates": total,
                "charged_cpu_seconds": 1000.0 + total * 0.001,
            }
        )
    return rows


@pytest.mark.parametrize("lengths", [[100] * 32, [1000] * 32, [100, 1000] * 16])
def test_message_length_does_not_distort_cpu_per_update(lengths):
    rate, count = p.measured_cpu_per_update(events(lengths), "pretraining")
    assert rate == pytest.approx(0.001)
    assert count == 32


def test_restart_charge_is_not_counted_as_training_throughput():
    rows = events([100, 200])
    rows += [
        {"event": "continuation_resumed"},
        {
            "event": "episode_completed",
            "stage": "pretraining",
            "target_byte_updates": 400,
            "charged_cpu_seconds": 99999.0,
        },
        {
            "event": "episode_completed",
            "stage": "pretraining",
            "target_byte_updates": 500,
            "charged_cpu_seconds": 99999.1,
        },
    ]
    rate, _ = p.measured_cpu_per_update(rows, "pretraining")
    assert rate == pytest.approx(0.001)


@pytest.mark.parametrize("headroom, feasible", [(100000.0, True), (1.0, False)])
def test_forecast_keeps_safety_and_rejects_insufficient_real_headroom(
    monkeypatch, headroom, feasible
):
    monkeypatch.setattr(p.r, "read_jsonl", lambda *_args, **_kwargs: events([100, 1000] * 16))
    monkeypatch.setattr(p.r, "read_json", lambda _path: {"cpu_seconds": 0.489, "copy_updates": 489})
    monkeypatch.setattr(p.r, "PARENT_DURABLE_UPDATES", {a: 0 for a in p.r.ARMS})
    monkeypatch.setattr(
        p.r, "latest_charges_fast", lambda: {a: p.r.PER_ARM_CPU_LIMIT - headroom for a in p.r.ARMS}
    )
    # Give the aggregate the same total headroom; no ceiling is relaxed.
    monkeypatch.setattr(p.r, "AGGREGATE_CPU_LIMIT", 3 * p.r.PER_ARM_CPU_LIMIT)
    contexts = {
        a: {"locations": {"parent_progress": "parent", "extension_progress": "extension"}}
        for a in p.r.ARMS
    }
    result = p.throughput_projection(
        "full_turn_1m", contexts, current_stage="pretraining", current_episodes=32
    )
    assert result["feasible"] is feasible
    assert result["safety_factor"] == 1.15
    assert (
        result["rates"]["full_turn_1m"]["remaining_pretraining_updates"]
        == p.PRETRAIN_UPDATES - 17600
    )
    assert result["rates"]["full_turn_1m"]["remaining_sft_updates"] == p.SFT_UPDATES
