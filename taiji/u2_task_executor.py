"""Single-writer durable physical effects and schema-owned recovery."""

from __future__ import annotations

import hashlib
import math
import sys
from contextlib import contextmanager
from pathlib import Path

from .u2_history import assert_values_equal, atomic_load, atomic_save, detached_tree
from .u2_recovery import ReferenceExecutor
from .u2_subject import U2Packet
from .u2_task_world import PhysicalWorld

SCHEMA = "taiji-u2-physical-executor-v1"


@contextmanager
def writer_lease(path):
    """The OS releases this lock when a writer process exits."""
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if sys.platform == "win32":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def executor_source():
    return hashlib.sha256(Path(__file__).read_text(encoding="utf-8-sig").encode()).hexdigest()


class PhysicalExecutor(ReferenceExecutor):
    path: Path
    history_profile = "physical_paged_v1"

    def __init__(self, world, path):
        self.world = world
        self.path = Path(path).resolve()
        self._authoritative_absence = False
        self.commands = {}

    @classmethod
    def create(cls, world: PhysicalWorld, path: str | Path):
        result = cls(PhysicalWorld.restore(world.snapshot()), path)
        result.path.parent.mkdir(parents=True, exist_ok=True)
        lock = result.path.with_name(result.path.name + ".lock")
        with writer_lease(lock):
            if result.path.exists():
                raise FileExistsError(result.path)
            atomic_save(result._payload(), result.path)
            result._refresh()
        return result

    def _payload(self):
        return {
            "schema": SCHEMA,
            "executor_source": executor_source(),
            "world": self.world.snapshot(),
            "commands": detached_tree(self.commands),
        }

    def _refresh(self):
        payload = atomic_load(self.path)
        if (
            set(payload) != {"schema", "executor_source", "world", "commands"}
            or payload["schema"] != SCHEMA
            or payload["executor_source"] != executor_source()
        ):
            raise ValueError("physical_executor_source_mismatch")
        current = PhysicalWorld.restore(payload["world"])
        before, after = self.world.snapshot(), current.snapshot()
        if before["world_id"] != after["world_id"] or after["version"] < before["version"]:
            raise ValueError("physical_executor_identity_or_generation_regression")
        for identity, row in before["transactions"].items():
            assert_values_equal(row, after["transactions"].get(identity), "executor.prefix")
            assert_values_equal(
                self.commands[identity], payload["commands"][identity], "executor.commands"
            )
        if set(payload["commands"]) != set(after["transactions"]):
            raise ValueError("incomplete_executor_command_registry")
        self.commands = detached_tree(payload["commands"])
        self.world, self._authoritative_absence = current, True

    @property
    def effect_count(self):
        return self.world.snapshot()["version"]

    @property
    def environment(self):
        return self.world.snapshot()

    def snapshot(self):
        return {
            "schema": SCHEMA,
            "persistent": True,
            "store_reference": str(self.path),
            "executor_source": executor_source(),
            "world": self.world.snapshot(),
            "commands": detached_tree(self.commands),
        }

    @classmethod
    def from_snapshot(cls, snapshot):
        if (
            set(snapshot)
            != {"schema", "persistent", "store_reference", "executor_source", "world", "commands"}
            or snapshot["schema"] != SCHEMA
            or snapshot["executor_source"] != executor_source()
            or snapshot["persistent"] is not True
        ):
            raise ValueError("invalid_physical_executor_snapshot")
        result = cls(PhysicalWorld.restore(snapshot["world"]), snapshot["store_reference"])
        result.commands = detached_tree(snapshot["commands"])
        try:
            result._refresh()
        except FileNotFoundError:
            result._authoritative_absence = False
        return result

    def _ack(self, identity):
        row = self.world.snapshot()["transactions"].get(identity)
        if row is None:
            return {"status": "not_seen", "action_id": identity}
        return {
            "status": "applied",
            "action_id": identity,
            "action": self.commands[identity].clone(),
            "effect_index": row["result"]["world_after"],
        }

    def query(self, action_id):
        try:
            self._refresh()
        except (OSError, ValueError) as error:
            self._authoritative_absence = False
            return {"status": "unknown", "action_id": action_id, "error": type(error).__name__}
        return self._ack(action_id)

    def _execute(self, identity, action=None, dt=None):
        lock = self.path.with_name(self.path.name + ".lock")
        with writer_lease(lock):
            self._refresh()
            if identity in self.commands:
                assert_values_equal(self.commands[identity], action, "executor.repeated_command")
            candidate = PhysicalWorld.restore(self.world.snapshot())
            if dt is None:
                candidate.step(identity, action)
            elif dt == 0:
                candidate.initial_observation(identity)
            elif dt == 0.05:
                candidate.elapse(identity)
            else:
                raise ValueError("unsupported_physical_observation_dt")
            payload = {
                "schema": SCHEMA,
                "executor_source": executor_source(),
                "world": candidate.snapshot(),
                "commands": {**self.commands, identity: action.detach().cpu().clone()},
            }
            atomic_save(payload, self.path)
            readback = atomic_load(self.path)
            assert_values_equal(payload, readback, "executor.commit_readback")
            self.world = candidate
            self.commands = payload["commands"]
            return self._ack(identity)

    def execute(self, action_id, action):
        return self._execute(action_id, action=action)

    def validate_intent(self, pending):
        expected = 0.0 if pending["operation"] == "ask" else 0.05
        if pending.get("origin") == "external_observation" and pending["dt"] == 0:
            expected = 0.0
        if pending["dt"] != expected or not math.isclose(
            pending["world_end"] - pending["dt"],
            self.world.snapshot()["world_time"],
            rel_tol=0,
            abs_tol=1e-10,
        ):
            raise ValueError("physical_intent_clock_mismatch")

    def execute_observation(self, event_id, dt, *, action):
        return self._execute(event_id, action=action, dt=dt)

    def packet(self, runtime):
        pending = runtime.state["pending"]
        if pending is None or self.query(pending["id"])["status"] != "applied":
            raise ValueError("physical_packet_requires_committed_effect")
        state = self.world.snapshot()
        result = state["transactions"][pending["id"]]["result"]
        runtime.advance_process()
        dtype = runtime.model.native_w.dtype
        observations = {key: value.to(dtype=dtype) for key, value in result["observations"].items()}
        coordinates = {key: value.to(dtype=dtype) for key, value in result["coordinates"].items()}
        return U2Packet(
            f"{state['world_id']}:{result['world_after']}:sensor",
            "sensor",
            "world",
            result["t_start"],
            result["t_end"],
            runtime.state["process_time"],
            observations,
            coordinates,
            result["execution_outcome"].to(dtype=dtype),
            pending["id"],
            write_allowed=bool(observations),
        )
