"""A continuous three-session subject run using declared external coursework."""

from __future__ import annotations

import time

from .u2_history import assert_values_equal, value_digest
from .u2_recovery import U2Runtime
from .u2_task_contract import Fact, Request, oracle_answer, request_text
from .u2_task_episode import COURSE_X, position_word


class SessionBudgetStop(RuntimeError):
    def __init__(self, runtime, answers, sessions):
        super().__init__("joint_session_cpu_budget_stop")
        self.runtime = runtime
        self.answers = list(answers)
        self.sessions = list(sessions)


def generate_answer(runtime, budget_guard=None):
    runtime.begin_utterance()
    data = bytearray()
    for _ in range(runtime.model.c.max_output_bytes + 1):
        if budget_guard is not None:
            budget_guard()
        log_probability, _, output = runtime.output()
        symbol = int(log_probability.detach().argmax())
        runtime.accept_byte(symbol, output=output)
        if symbol == 256:
            return bytes(data), True
        data.append(symbol)
    raise ValueError("native_expression_budget_stop")


def run_joint_session(runtime, names, directory, *, cpu_cap=None):
    """Returns cold behavior separately from physical/subject continuity."""
    directory.mkdir(parents=True, exist_ok=True)
    subject_id = runtime.state["subject_id"]
    parameter_id = value_digest(runtime.model.state_dict())
    facts = []
    answers = []
    sessions = []

    def guard():
        if cpu_cap is not None and time.process_time() > cpu_cap:
            raise SessionBudgetStop(runtime, answers, sessions)

    for round_index in range(48):
        guard()
        if runtime.operation_count > 24:
            runtime.begin_new_window()
        action = None
        if round_index < 32:
            sign = 1 if round_index < 16 else -1
            action = runtime.model.native_w.new_zeros(15)
            action[0] = 1
            action[9] = sign * COURSE_X[round_index % 16]
            action[10] = -0.55
            action[11], action[13], action[14] = sign * 0.8, 0.6, 0.04
        runtime.begin_observation(f"joint-{round_index}", action=action)
        runtime.execute_pending(directory / "intent.bin")
        visible = round_index in (0, 15, 31)
        packet = runtime.executor.packet(runtime, modalities=("rgb",) if visible else ())
        receipt = runtime.receive(packet)
        assert receipt["behavior_credit_valid"] is False
        if visible:
            environment = runtime.executor.environment
            for i, name in enumerate(names):
                facts.append(
                    Fact(
                        f"{round_index}-{i}",
                        name,
                        position_word(environment["bodies"][i]["position"][0]),
                        environment["version"],
                        "sensor",
                    )
                )
        if round_index in (15, 31, 47):
            environment = runtime.executor.environment
            for name in names:
                for temporal in ("current", "previous"):
                    if runtime.operation_count > 27:
                        runtime.begin_new_window()
                    raw = request_text(name, temporal)
                    runtime.set_goal(raw)
                    request = Request(
                        f"q-{round_index}-{name}-{temporal}",
                        runtime.state["goal_version"],
                        environment["version"],
                        raw,
                        name,
                        temporal,
                        tuple(f.evidence_id for f in facts),
                    )
                    target = oracle_answer(request, tuple(facts))
                    content, eos = generate_answer(runtime, budget_guard=guard)
                    try:
                        content.decode("utf-8", errors="strict")
                        valid = True
                    except UnicodeDecodeError:
                        valid = False
                    answers.append(
                        {
                            "request_id": request.request_id,
                            "round": round_index + 1,
                            "raw_request": raw.decode(),
                            "target": target.decode(),
                            "output_hex": content.hex(),
                            "eos": eos,
                            "valid_utf8": valid,
                            "correct": bool(valid and eos and content == target),
                        }
                    )
            checkpoint = directory / f"session-{round_index//16+1}.bin"
            runtime.save(checkpoint)
            restored = U2Runtime.load(checkpoint)
            assert_values_equal(runtime.state, restored.state)
            assert runtime.history.reference() == restored.history.reference()
            runtime = restored
            sessions.append(
                {
                    "session": round_index // 16 + 1,
                    "subject_id": runtime.state["subject_id"],
                    "history_entries": runtime.history.entry_count,
                    "history_pages": len(runtime.history.export()["pages"]),
                    "world_time": runtime.state["world_time"],
                    "recovery_equal": True,
                }
            )
        assert runtime.state["subject_id"] == subject_id
        assert value_digest(runtime.model.state_dict()) == parameter_id
    return runtime, {
        "sessions": sessions,
        "answers": answers,
        "correct": sum(row["correct"] for row in answers),
        "denominator": 18,
        "optimizer_steps": 0,
        "source": "model argmax bytes and native EOS mask",
        "coursework_owner": "environment; no autonomous actor credit",
        "feedback_labels_visible_to_model": False,
        "qualification": "cold initialization behavior and continuity; no task learning claimed",
    }
