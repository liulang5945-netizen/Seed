"""Cached emission computation preserves the normal free-generation behavior."""

from taiji.u2_history import assert_values_equal
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Subject
from taiji.u2_task_session import generate_answer


def test_cached_output_preserves_bytes_state_and_raw_history():
    first = U2Runtime(U2Subject(U2Config.small()).metadata_candidate(), subject_id="emission-cache")
    first.set_goal("红色圆片现在在哪里？")
    second = U2Runtime.from_envelope(first.to_envelope())
    first.begin_utterance()
    data = bytearray()
    for _ in range(first.model.c.max_output_bytes + 1):
        symbol = int(first.output()[0].detach().argmax())
        first.accept_byte(symbol)
        if symbol == 256:
            break
        data.append(symbol)
    assert generate_answer(second) == (bytes(data), True)
    assert_values_equal(first.state, second.state)
    assert first.history.reference() == second.history.reference()
