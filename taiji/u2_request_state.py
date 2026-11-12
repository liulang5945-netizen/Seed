"""Native request control events alongside a persistent task goal."""

from __future__ import annotations

import math


def active_request(state):
    request = state.get("answer_request")
    return request is not None and request["status"] in {"pending", "answering"}


def validate(model, state):
    request = state["answer_request"]
    binding = state["utterance_request"]
    if float(state["nu"][4]) != int(active_request(state)) or float(state["nu"][12]) != 0:
        raise ValueError("request_measurement_mismatch")
    if request is not None:
        if set(request) != {"id", "version", "raw", "goal_version", "world_time", "status"}:
            raise ValueError("unregistered_request_fields")
        if (
            not isinstance(request["id"], str)
            or not request["id"]
            or type(request["version"]) is not int
            or request["version"] < 1
            or not isinstance(request["raw"], bytes)
            or not request["raw"]
            or len(request["raw"]) > model.c.max_record_bytes
            or type(request["goal_version"]) is not int
            or request["goal_version"] < 0
            or request["goal_version"] > state["goal_version"]
            or not math.isfinite(request["world_time"])
            or not 0 <= request["world_time"] <= state["world_time"]
            or request["status"] not in {"pending", "answering", "answered", "cancelled"}
        ):
            raise ValueError("invalid_native_request")
    if binding is not None and (
        request is None
        or binding != (request["id"], request["version"])
        or request["status"] != "answering"
        or not state["utterance_active"]
    ):
        raise ValueError("utterance_request_identity_mismatch")
    if request is not None and request["status"] == "answering" and binding is None:
        raise ValueError("missing_utterance_request_binding")


def set_request(model, state, identity, raw, version=1):
    if not model.c.separate_requests:
        raise ValueError("native_request_profile_required")
    if state["pending"] is not None or state["utterance_active"] or active_request(state):
        raise ValueError("request_requires_resolved_mutation_boundary")
    previous = state["answer_request"]
    if previous is not None and previous["id"] == identity and version <= previous["version"]:
        raise ValueError("request_version_regressed")
    raw = raw.encode() if isinstance(raw, str) else bytes(raw)
    request = {
        "id": identity,
        "version": version,
        "raw": raw,
        "goal_version": state["goal_version"],
        "world_time": state["world_time"],
        "status": "pending",
    }
    child = {**state, "answer_request": request, "utterance_request": None}
    child["nu"] = state["nu"].clone()
    child["nu"][4], child["nu"][12] = 1, 0
    validate(model, child)
    query = model.goal(raw)
    persistent_goal = model.goal(state["goal_raw"])
    child["H"] = model.native(
        child,
        persistent_goal,
        model.native_w.new_zeros(15),
        query[None].expand(model.c.branches, -1),
    )
    return child


def cancel_request(model, state, identity, version):
    if not model.c.separate_requests or state["pending"] is not None or state["utterance_active"]:
        raise ValueError("request_cancellation_requires_boundary")
    request = state["answer_request"]
    if (
        request is None
        or (request["id"], request["version"]) != (identity, version)
        or not active_request(state)
    ):
        raise ValueError("request_cancellation_identity_mismatch")
    child = {
        **state,
        "answer_request": {**request, "status": "cancelled"},
        "utterance_request": None,
    }
    child["nu"] = state["nu"].clone()
    child["nu"][4], child["nu"][12] = 0, 0
    return child


def bind_utterance(model, state, child):
    if state["pending"] is not None or state["utterance_active"]:
        raise ValueError("native_request_utterance_requires_boundary")
    child["utterance_request"] = None
    if active_request(state):
        request = state["answer_request"]
        child["answer_request"] = {**request, "status": "answering"}
        child["utterance_request"] = (request["id"], request["version"])
    return child


def finish_utterance(state, child):
    if state["utterance_request"] is not None:
        child["answer_request"] = {**state["answer_request"], "status": "answered"}
        child["utterance_request"] = None
        child["nu"] = state["nu"].clone()
        child["nu"][4], child["nu"][12] = 0, 0
    return child
