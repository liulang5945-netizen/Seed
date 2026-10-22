"""Project remaining fixed target-byte updates, robust to episode-length variation."""

from __future__ import annotations

from typing import Any

from scripts.training import run_taiji_r2_daily_dialogue_continuation_extension as r

PRETRAIN_UPDATES = 14_517_912
SFT_UPDATES = 113_209
WINDOW = 256


def measured_cpu_per_update(events: list[dict[str, Any]], stage: str) -> tuple[float, int]:
    previous = None
    samples = []
    for event in events:
        name = str(event.get("event", ""))
        if any(word in name for word in ("stop", "resum", "started")):
            previous = None
        if event.get("stage") != stage or name not in {
            "episode_completed",
            "durable_training_commit",
        }:
            continue
        updates = event.get("target_byte_updates")
        cpu = event.get("charged_cpu_seconds", event.get("cpu_seconds"))
        if (
            not isinstance(updates, int)
            or isinstance(updates, bool)
            or not isinstance(cpu, (int, float))
        ):
            continue
        if previous is not None:
            delta_updates = updates - previous[0]
            delta_cpu = float(cpu) - previous[1]
            if delta_updates > 0 and delta_cpu >= 0:
                samples.append((delta_cpu, delta_updates))
        previous = (updates, float(cpu))
    samples = samples[-WINDOW:]
    if not samples:
        raise r.Refuse(f"no measured target-update CPU samples for {stage}")
    return sum(x[0] for x in samples) / sum(x[1] for x in samples), len(samples)


def throughput_projection(current_arm, contexts, *, current_stage, current_episodes):
    charges = r.latest_charges_fast()
    projected = {}
    evidence = {}
    for arm in r.ARMS:
        location = contexts[arm]["locations"]
        parent = r.read_jsonl(location["parent_progress"])
        extension = r.read_jsonl(location["extension_progress"], required=False)
        rates = []
        counts = []
        for events in (parent, extension):
            try:
                value, count = measured_cpu_per_update(events, "pretraining")
                rates.append(value)
                counts.append(count)
            except r.Refuse:
                pass
        if not rates:
            raise r.Refuse(f"no validated pretraining target-update rate: {arm}")
        pre_rate = max(rates)
        probe = r.read_json(r.frozen.WORST_EPISODE_PROBES[arm][1])
        probe_updates = probe.get("copy_updates", probe.get("updates_completed_in_copy"))
        if not isinstance(probe_updates, int) or probe_updates <= 0:
            raise r.Refuse("SFT probe update denominator is invalid")
        sft_rate = max(pre_rate * 1.2, float(probe["cpu_seconds"]) / probe_updates)
        try:
            measured_sft, _ = measured_cpu_per_update(extension, "assistant_sft")
            sft_rate = max(sft_rate, measured_sft)
        except r.Refuse:
            pass
        pre_done = r.PARENT_DURABLE_UPDATES[arm]
        sft_done = 0
        for event in extension:
            updates = event.get("target_byte_updates")
            if not isinstance(updates, int):
                continue
            if event.get("stage") == "pretraining":
                pre_done = max(pre_done, updates)
            elif event.get("stage") == "assistant_sft":
                pre_done = PRETRAIN_UPDATES
                sft_done = max(sft_done, updates)
        if arm == current_arm and current_stage == "assistant_sft":
            pre_done = PRETRAIN_UPDATES
        remaining_pre = max(0, PRETRAIN_UPDATES - pre_done)
        remaining_sft = max(0, SFT_UPDATES - sft_done)
        projected[arm] = (
            r.THROUGHPUT_SAFETY_FACTOR * (remaining_pre * pre_rate + remaining_sft * sft_rate)
            + r.COMPLETION_CPU_RESERVE_SECONDS
        )
        evidence[arm] = {
            "pretraining_cpu_per_update": pre_rate,
            "pretraining_sample_counts": counts,
            "sft_cpu_per_update": sft_rate,
            "remaining_pretraining_updates": remaining_pre,
            "remaining_sft_updates": remaining_sft,
        }
    headroom = {arm: r.PER_ARM_CPU_LIMIT - charges[arm] for arm in r.ARMS}
    feasible = {arm: projected[arm] <= headroom[arm] for arm in r.ARMS}
    aggregate_headroom = r.AGGREGATE_CPU_LIMIT - sum(charges.values())
    total = sum(projected.values())
    return {
        "arm": current_arm,
        "current_stage": current_stage,
        "current_episodes": current_episodes,
        "projection_unit": "target_byte_updates",
        "safety_factor": r.THROUGHPUT_SAFETY_FACTOR,
        "rates": evidence,
        "projected_cpu_by_arm": projected,
        "projected_cpu_aggregate": total,
        "arm_headroom_cpu_seconds": headroom[current_arm],
        "arm_headroom_by_arm_cpu_seconds": headroom,
        "aggregate_headroom_cpu_seconds": aggregate_headroom,
        "per_arm_feasible": feasible,
        "feasible": all(feasible.values()) and total <= aggregate_headroom,
    }
