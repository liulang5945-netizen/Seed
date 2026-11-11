"""Shared U2 observation assimilation and conditional behavior credit.

Assimilation can reinterpret recorded observations under a new encoder. Behavior
credit consumes the original frozen-policy packet only on the normal receive path.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import torch

from .u2_subject import clone_tree


def validate_event(state: Any, packet: Any) -> Any:
    if (
        not all(math.isfinite(t) for t in (packet.t_start, packet.t_end, packet.t_arrival))
        or packet.t_start > packet.t_end
    ):
        raise ValueError("invalid event time")
    if packet.source != "sensor" or packet.scope != "world" or not packet.valid:
        raise ValueError("illegal event source/scope")
    receipt = state["pending"]
    if receipt is None or receipt["id"] != packet.receipt_id:
        raise ValueError("receipt mismatch")
    if receipt["subject_version"] != state["revision"]:
        raise ValueError("parameters changed while pending")
    if packet.t_end > receipt["world_end"] or packet.t_arrival > state["process_time"]:
        raise ValueError("event not arrived")
    if (
        not math.isclose(packet.t_end, receipt["world_end"], rel_tol=0.0, abs_tol=1e-10)
        or packet.t_start < receipt["world_end"] - receipt["dt"] - 1e-10
    ):
        raise ValueError("historical observation requires shadow replay")
    return receipt


def eligibility_carry(receipt: Any, goal_version: int) -> Any:
    return {
        name: (
            (0.99 * 0.95) ** max(1.0, receipt["dt"] / 0.05) * value
            if receipt["goal_version"] == goal_version
            else torch.zeros_like(value)
        )
        for name, value in receipt["decision"]["eligibility"].items()
    }


def update_measurements(model: Any, state: Any, predictive: Any) -> Any:
    entropy = -(state["logw"].exp() * state["logw"]).sum() / math.log(max(2, model.c.branches))
    measured = torch.stack(
        [
            entropy.detach(),
            predictive.detach().abs().mean().sigmoid(),
            state["F"].detach().norm() / (4 * math.sqrt(model.c.branches)),
            state["F"].new_tensor(math.log1p(state["resources"]["events"])),
        ]
    )
    nu = state["nu"].clone()
    nu[:4] = 0.95 * nu[:4] + 0.05 * measured
    nu[8:12] = 0
    return {**state, "nu": nu}


def assimilate(model: Any, state: Any, packet: Any, *, receipt: Any = None) -> Any:
    """Assimilate an observation without sampling or issuing actor/critic credit."""
    if packet.packet_id in state["seen_packets"]:
        return state, {}
    receipt = validate_event(state, packet) if receipt is None else receipt
    deferred = receipt.get("deferred_packet")
    if deferred is not None and deferred.packet_id == packet.packet_id:
        if bool(packet.outcome[4:].all()):
            raise ValueError("Completed result must have a fresh immutable packet identity")
        return state, {"status": "awaiting_outcome"}
    evidence = {
        "packet_id": packet.packet_id,
        "source": packet.source,
        "scope": packet.scope,
        "t_start": packet.t_start,
        "t_end": packet.t_end,
        "t_arrival": packet.t_arrival,
        "observations": clone_tree(packet.observations, detach=True),
        "coordinates": clone_tree(packet.coordinates, detach=True),
        "outcome": packet.outcome.detach().clone(),
        "old_goal": receipt["goal_raw"],
        "action": receipt["action"].detach().clone(),
        "subject_version": receipt["subject_version"],
        "receipt_id": receipt["id"],
    }
    if not bool(packet.outcome[4:].all()):
        child = dict(state)
        child["pending"] = {**receipt, "deferred_packet": packet}
        child["evidence"] = [*state["evidence"], evidence]
        child["journal"] = [
            *state["journal"],
            {"id": receipt["id"], "status": "awaiting_outcome", "packet": packet.packet_id},
        ]
        return child, {"status": "awaiting_outcome"}
    observation_id = packet.packet_id
    if not packet.observations and deferred is not None:
        observation_id = deferred.packet_id
        packet = replace(
            packet,
            observations=deferred.observations,
            coordinates=deferred.coordinates,
            feedback_record_ids=deferred.feedback_record_ids,
        )
    old_goal = model.goal(receipt["goal_raw"])
    goal = model.goal(state["goal_raw"])
    prior = receipt["prior"]
    predictive = model.likelihood(
        prior, old_goal, packet.observations, packet.coordinates, receipt["action"], packet.outcome
    )
    current = dict(prior)
    current["goal_raw"] = state["goal_raw"]
    current["goal_version"] = state["goal_version"]
    current["released_feedback_records"] = packet.feedback_record_ids
    result = model.correct(
        current,
        goal,
        receipt["action"],
        packet.observations,
        packet.coordinates,
        packet.outcome,
        write=packet.write_allowed,
    )
    old_rows = len(prior["history"])
    result["history"] = [
        *result["history"][:old_rows],
        *[
            {
                **row,
                "source_packet_id": packet.packet_id,
                "source_observation_packet_id": observation_id,
                "source_receipt_id": receipt["id"],
                "source_subject_version": receipt.get(
                    "behavior_version", receipt["subject_version"]
                ),
            }
            for row in result["history"][old_rows:]
        ],
    ]
    result["world_time"] = receipt["world_end"]
    result["process_time"] = state["process_time"]
    result["pending"] = None
    result["seen_packets"] = [*state["seen_packets"], packet.packet_id]
    if deferred is not None:
        result["seen_packets"] = list(dict.fromkeys([*result["seen_packets"], deferred.packet_id]))
    result["evidence"] = [*state["evidence"], evidence]
    result["journal"] = [
        *state["journal"],
        {"id": receipt["id"], "packet": packet.packet_id, "status": "settled"},
    ]
    result["settled"] = [*state["settled"], receipt["id"]]
    result["resources"] = {**state["resources"], "events": state["resources"]["events"] + 1}
    result = update_measurements(model, result, predictive)
    return result, {"predictive_logp": predictive, "prior": prior, "decision": receipt["decision"]}


def behavior_credit(model: Any, result: Any, packet: Any, receipt: Any, old_value: Any) -> Any:
    decision = receipt["decision"]
    old_goal = model.goal(receipt["goal_raw"])
    utility = packet.outcome.new_tensor([-1.0, 0.0, 1.0]) @ packet.outcome[:3]
    terminal = packet.outcome[3]
    next_context = model.branch_context(result, old_goal, receipt["action"])
    next_features = torch.cat(
        [next_context.detach(), old_goal.detach()[None].expand(model.c.branches, -1)], -1
    )
    next_value = (
        result["logw"].exp().detach() * 2 * model.critic(next_features).tanh().squeeze(-1)
    ).sum()
    discount = 0.99 ** max(1.0, receipt["dt"] / 0.05)
    delta = (1 - discount) * utility + discount * (1 - terminal) * next_value.detach() - old_value
    actor_loss = -delta.detach() * sum(
        (parameter * decision["eligibility"][name]).sum()
        for name, parameter in model.policy_named_parameters()
    )
    return actor_loss, 0.5 * delta.square()


def receive(model: Any, state: Any, packet: Any) -> Any:
    if packet.packet_id in state["seen_packets"]:
        return state, {}
    receipt = validate_event(state, packet)
    old_value = model.pending_value(receipt["decision"])
    result, observation = assimilate(model, state, packet, receipt=receipt)
    if not observation or observation.get("status") == "awaiting_outcome":
        return result, observation
    actor_loss, critic_loss = behavior_credit(model, result, packet, receipt, old_value)
    result["eligibility"] = eligibility_carry(receipt, state["goal_version"])
    return result, {**observation, "actor": actor_loss, "critic": critic_loss}
