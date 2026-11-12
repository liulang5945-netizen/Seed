"""Physical effects, exogenous credit and paged subject lifecycle regressions."""

import copy

import pytest
import torch

from taiji.u2_history import CausalHistory, assert_values_equal, atomic_load
from taiji.u2_history_store import PagedCausalHistory
from taiji.u2_learning_transaction import rebuild_state
from taiji.u2_recovery import U2Runtime
from taiji.u2_subject import U2Config, U2Subject
from taiji.u2_task_executor import PhysicalExecutor, writer_lease
from taiji.u2_task_world import PhysicalWorld


def runtime_at(root, dtype=torch.float32):
    torch.set_num_threads(1)
    executor = PhysicalExecutor.create(
        PhysicalWorld.reset(34, {"body_count": 1}), root / "world.bin"
    )
    model = U2Subject(U2Config.small()).to(dtype=dtype).metadata_candidate()
    runtime = U2Runtime(model, executor=executor, subject_id="physical-subject")
    assert isinstance(runtime.history, PagedCausalHistory)
    runtime.set_goal("逐字输出观察。")
    return runtime


def observe(runtime, root, identity, dt=0.05):
    if runtime.operation_count > 24:
        runtime.begin_new_window()
    runtime.begin_observation(identity, dt)
    runtime.execute_pending(root / "intent.bin")
    return runtime.receive(runtime.executor.packet(runtime))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_bootstrap_exogenous_and_real_actor_receive(tmp_path, dtype):
    runtime = runtime_at(tmp_path, dtype)
    response = observe(runtime, tmp_path, "initial", 0)
    assert response["behavior_credit_valid"] is False
    assert "actor" not in response and "critic" not in response
    assert runtime.state["world_time"] == 0
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    runtime.issue(runtime.choose(mask), dt=0.05)
    runtime.execute_pending(tmp_path / "act.bin")
    response = runtime.receive(runtime.executor.packet(runtime))
    assert bool(torch.isfinite(response["actor"]))
    assert bool(torch.isfinite(response["critic"]))
    runtime.save(tmp_path / "subject.bin")
    restored = U2Runtime.load(tmp_path / "subject.bin")
    assert isinstance(restored.executor, PhysicalExecutor)
    assert isinstance(restored.history, PagedCausalHistory)
    assert_values_equal(runtime.state, restored.state)


@pytest.mark.parametrize("fault", ["after_intent", "after_execute_before_ack"])
def test_crash_boundary_recovery_executes_once(tmp_path, fault):
    runtime = runtime_at(tmp_path)
    runtime.begin_observation("initial", 0)
    with pytest.raises(RuntimeError, match="injected"):
        runtime.execute_pending(tmp_path / "intent.bin", fault=fault)
    restored = U2Runtime.load(tmp_path / "intent.bin")
    restored.execute_pending(tmp_path / "intent.bin")
    restored.receive(restored.executor.packet(restored))
    assert restored.executor.effect_count == 1
    assert restored.state["settled"].count("initial") == 1


def test_missing_store_stays_unknown(tmp_path):
    runtime = runtime_at(tmp_path)
    runtime.begin_observation("initial", 0)
    runtime.save(tmp_path / "subject.bin")
    (tmp_path / "world.bin").rename(tmp_path / "world.missing")
    restored = U2Runtime.load(tmp_path / "subject.bin")
    assert restored.executor.query("initial")["status"] == "unknown"
    assert (
        restored.execute_pending(tmp_path / "subject.bin")["status"] == "awaiting_execution_status"
    )
    assert not (tmp_path / "world.bin").exists()


def test_live_writer_lease_blocks_second_writer(tmp_path):
    runtime = runtime_at(tmp_path)
    lock = tmp_path / "world.bin.lock"
    with writer_lease(lock), pytest.raises(OSError):
        runtime.executor.execute_observation("blocked", 0, action=torch.zeros(15))
    runtime.executor.execute_observation("initial", 0, action=torch.zeros(15))
    assert runtime.executor.effect_count == 1


def test_history_pages_exceed_old_limit_and_restore(tmp_path):
    runtime = runtime_at(tmp_path)
    for index in range(240):
        if runtime.operation_count > 24:
            runtime.begin_new_window()
        runtime.set_process_time(float(index))
    assert runtime.history.entry_count > 240
    payload = runtime.history.export()
    assert len(payload["pages"]) == 3
    restored = CausalHistory.restore(payload)
    assert_values_equal(restored.reference(), runtime.history.reference())
    broken = copy.deepcopy(payload)
    del broken["pages"][1]
    with pytest.raises(ValueError, match="history_unavailable"):
        CausalHistory.restore(broken)
    runtime.save(tmp_path / "subject.bin")
    assert_values_equal(runtime.state, U2Runtime.load(tmp_path / "subject.bin").state)


def test_full_forty_eight_tick_lifecycle_and_reencoding(tmp_path):
    runtime = runtime_at(tmp_path)
    observe(runtime, tmp_path, "initial", 0)
    for index in range(48):
        observe(runtime, tmp_path, f"elapsed-{index}")
    assert runtime.state["world_time"] == pytest.approx(2.4)
    assert runtime.history.entry_count > 192
    assert runtime.executor.effect_count == 49
    runtime.save(tmp_path / "subject.bin")
    restored = U2Runtime.load(tmp_path / "subject.bin")
    assert_values_equal(runtime.state, restored.state)
    rebuilt, _ = rebuild_state(
        runtime.model, runtime.history, runtime.state, reset_eligibility=False
    )
    assert_values_equal(runtime.state, rebuilt)


def test_executor_preserves_original_command_dtype(tmp_path):
    runtime = runtime_at(tmp_path)
    action = torch.zeros(15, dtype=torch.float32)
    action[0] = 1
    action[14] = 0.02
    runtime.executor.execute("one", action)
    assert runtime.executor.query("one")["action"].dtype == torch.float32
    restored = PhysicalExecutor.from_snapshot(runtime.executor.snapshot())
    assert_values_equal(restored.query("one"), runtime.executor.query("one"))
    payload = atomic_load(tmp_path / "world.bin")
    assert payload["commands"]["one"].dtype == torch.float32


def test_wrong_physical_dt_rejected_before_effect(tmp_path):
    runtime = runtime_at(tmp_path)
    mask = torch.zeros(9, dtype=torch.bool)
    mask[0] = True
    runtime.issue(runtime.choose(mask), dt=0.01)
    with pytest.raises(ValueError, match="clock_mismatch"):
        runtime.execute_pending(tmp_path / "intent.bin")
    assert runtime.executor.effect_count == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_external_course_predicts_actual_action_without_actor_credit(tmp_path, dtype):
    runtime = runtime_at(tmp_path, dtype)
    observe(runtime, tmp_path, "initial", 0)
    rng = runtime.state["rng"].clone()
    action = runtime.model.native_w.new_zeros(15)
    action[0], action[11], action[13], action[14] = 1, 0.5, 0.8, 0.03
    runtime.begin_observation("course", action=action)
    runtime.save(tmp_path / "course.bin")
    runtime = U2Runtime.load(tmp_path / "course.bin")
    runtime.execute_pending(tmp_path / "intent.bin")
    packet = runtime.executor.packet(runtime, modalities=("rgb",))
    assert set(packet.observations) == {"rgb"}
    receipt = runtime.receive(packet)
    assert receipt["behavior_credit_valid"] is False and "actor" not in receipt
    assert torch.equal(rng, runtime.state["rng"])
    assert torch.equal(receipt["prior"]["prediction_action"], action)
    rebuilt, _ = rebuild_state(
        runtime.model, runtime.history, runtime.state, reset_eligibility=False
    )
    assert_values_equal(runtime.state, rebuilt)
