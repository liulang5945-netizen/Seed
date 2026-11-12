"""Versioned transactional simulation behind the frozen U2 physical interface."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from .u2_history import detached_tree, value_digest
from .u2_task_physics import advance, contact_error
from .u2_task_sensors import acoustic, encode, render

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "plans/manifests/taiji_u2_22_4_2_world_contract_20261112.json"
SCHEMA = "taiji-u2-physical-world-v1"


def cache_bytes(transactions):
    """Count deterministic wire bytes independently of tensor storage aliases."""

    def wire(value):
        if isinstance(value, torch.Tensor):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "hex": value.detach().cpu().contiguous().numpy().tobytes().hex(),
            }
        if isinstance(value, dict):
            return {key: wire(item) for key, item in sorted(value.items())}
        if isinstance(value, (tuple, list)):
            return [wire(item) for item in value]
        if isinstance(value, bytes):
            return {"bytes_hex": value.hex()}
        return value

    return len(
        json.dumps(
            wire(transactions), ensure_ascii=False, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )


def source_pins():
    return {
        name: hashlib.sha256(
            (Path(__file__).parent / name).read_text(encoding="utf-8-sig").encode()
        ).hexdigest()
        for name in ("u2_task_world.py", "u2_task_physics.py", "u2_task_sensors.py")
    }


def generator(seed, tag):
    data = f"taiji-world-v1|{seed}|{tag}".encode()
    number = int.from_bytes(hashlib.sha256(data).digest()[:8], "little") % (2**63)
    return torch.Generator().manual_seed(number)


class PhysicalWorld:
    def __init__(self, state, contract):
        self._state = state
        self.contract = contract

    @classmethod
    def reset(cls, seed: int, recipe: dict | None = None):
        if type(seed) is not int or not 0 <= seed < 2**63:
            raise ValueError("invalid_world_seed")
        recipe = copy.deepcopy(recipe or {})
        if set(recipe) - {"body_count", "occluder", "dtype", "modalities"}:
            raise ValueError("unknown_world_recipe_field")
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        physics, camera = contract["physics"], contract["camera"]
        rng = {tag: generator(seed, tag) for tag in contract["rng"]["tags"]}
        geometry, material = rng["geometry"], rng["material"]
        drawn_count = int(torch.randint(1, 4, (), generator=geometry))
        count = recipe.get("body_count", drawn_count)
        if type(count) is not int or not 1 <= count <= 3:
            raise ValueError("invalid_body_count")
        dtype = recipe.get("dtype", "float32")
        if dtype not in {"float32", "float64"}:
            raise ValueError("unsupported_world_codec_precision")
        modalities = recipe.get("modalities", ["rgb", "audio", "proprio"])
        if len(set(modalities)) != len(modalities) or set(modalities) - {"rgb", "audio", "proprio"}:
            raise ValueError("invalid_sensor_schedule")
        occluder = recipe.get("occluder")
        if occluder is not None and (
            len(occluder) != 4
            or not all(math.isfinite(x) and -1 <= x <= 1 for x in occluder)
            or occluder[0] >= occluder[2]
            or occluder[1] >= occluder[3]
        ):
            raise ValueError("invalid_camera_occluder")
        digest = hashlib.sha256(CONTRACT.read_text(encoding="utf-8-sig").encode()).hexdigest()
        world_id = hashlib.sha256(
            json.dumps([digest, seed, recipe], sort_keys=True).encode()
        ).hexdigest()[:32]
        radii = [
            physics["radii"][i] for i in torch.randint(2, (count,), generator=geometry).tolist()
        ]
        colors = torch.randperm(len(camera["palette"]), generator=rng["appearance"]).tolist()
        bodies = []
        for i, radius in enumerate(radii):
            limit = 1 - radius - physics["initial_wall_gap"]
            for _ in range(physics["placement_attempts"]):
                position = (
                    torch.rand(2, generator=geometry, dtype=torch.float64) * 2 - 1
                ).numpy() * limit
                if all(
                    np.linalg.norm(position - np.array(b["position"]))
                    >= radius + b["radius"] + physics["initial_gap"]
                    for b in bodies
                ):
                    break
            else:
                raise ValueError("fixture_generation_exhausted")
            velocity = (torch.rand(2, generator=geometry, dtype=torch.float64) * 0.4 - 0.2).tolist()
            indices = torch.randint(3, (3,), generator=material).tolist()
            bodies.append(
                {
                    "id": f"{world_id}:{i}",
                    "position": position.tolist(),
                    "velocity": velocity,
                    "radius": radius,
                    "mass": physics["masses"][indices[0]],
                    "drag": physics["drag_rates"][indices[1]],
                    "frequency": contract["audio"]["frequencies"][indices[2]],
                    "color": camera["palette"][colors[i]],
                }
            )
        state = {
            "schema": SCHEMA,
            "contract_hash": digest,
            "source_pins": source_pins(),
            "world_id": world_id,
            "recipe": recipe,
            "dtype": dtype,
            "modalities": modalities,
            "version": 0,
            "tick_index": 0,
            "world_time": 0.0,
            "bodies": bodies,
            "occluder": occluder,
            "rng_states": {tag: g.get_state() for tag, g in rng.items()},
            "audio_bank": [[0.0, 0.0] for _ in range(3)],
            "audio_ring": [0.0] * 256,
            "sample_index": 0,
            "transactions": {},
            "byte_count": 2,
            "terminal": False,
        }
        return cls(state, contract)

    def snapshot(self):
        return detached_tree(self._state)

    @classmethod
    def restore(cls, snapshot):
        state = detached_tree(snapshot)
        contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
        digest = hashlib.sha256(CONTRACT.read_text(encoding="utf-8-sig").encode()).hexdigest()
        if (
            state.get("schema") != SCHEMA
            or state.get("contract_hash") != digest
            or state.get("source_pins") != source_pins()
        ):
            raise ValueError("world_recovery_identity_mismatch")
        expected = {
            "schema",
            "contract_hash",
            "source_pins",
            "world_id",
            "recipe",
            "dtype",
            "modalities",
            "version",
            "tick_index",
            "world_time",
            "bodies",
            "occluder",
            "rng_states",
            "audio_bank",
            "audio_ring",
            "sample_index",
            "transactions",
            "byte_count",
            "terminal",
        }
        if set(state) != expected or state["version"] != len(state["transactions"]):
            raise ValueError("incomplete_world_snapshot")
        if state["sample_index"] != state["tick_index"] * 800 or not math.isclose(
            state["world_time"], state["tick_index"] * 0.05, rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError("invalid_world_clock")
        if set(state["rng_states"]) != set(contract["rng"]["tags"]):
            raise ValueError("incomplete_world_rng")
        for value in state["rng_states"].values():
            torch.Generator().set_state(value)
        bodies = state["bodies"]
        if not 1 <= len(bodies) <= 3 or len({b["id"] for b in bodies}) != len(bodies):
            raise ValueError("invalid_body_registry")
        for body in bodies:
            numeric = [
                *body["position"],
                *body["velocity"],
                body["mass"],
                body["drag"],
                body["radius"],
            ]
            if (
                not all(math.isfinite(v) for v in numeric)
                or body["mass"] not in contract["physics"]["masses"]
                or body["drag"] not in contract["physics"]["drag_rates"]
                or body["radius"] not in contract["physics"]["radii"]
            ):
                raise ValueError("invalid_body_state")
        positions = np.array([b["position"] for b in bodies])
        penetration, _ = contact_error(
            positions,
            np.array([b["velocity"] for b in bodies]),
            np.array([b["radius"] for b in bodies]),
        )
        if (
            penetration > 1e-9
            or np.shape(state["audio_bank"]) != (3, 2)
            or np.shape(state["audio_ring"]) != (256,)
            or not np.isfinite(state["audio_bank"]).all()
            or not np.isfinite(state["audio_ring"]).all()
        ):
            raise ValueError("invalid_world_geometry_or_acoustics")
        if state["byte_count"] != cache_bytes(state["transactions"]):
            raise ValueError("world_cache_size_mismatch")
        if (
            state["byte_count"] > contract["interface"]["max_bytes"]
            or len(state["transactions"]) > 512
        ):
            raise ValueError("world_snapshot_budget_stop")
        return cls(state, contract)

    def query(self, identity):
        row = self._state["transactions"].get(identity)
        return (
            {"status": "not_seen", "action_id": identity}
            if row is None
            else {
                "status": "applied",
                "action_id": identity,
                "result": detached_tree(row["result"]),
            }
        )

    def initial_observation(self, event_id):
        return self._transact(event_id, "initial", torch.zeros(15, dtype=torch.float64))

    def elapse(self, event_id):
        return self._transact(event_id, "elapsed", torch.zeros(15, dtype=torch.float64))

    def step(self, action_id, action):
        command = torch.as_tensor(action, dtype=torch.float64).detach().cpu()
        if command.shape != (15,) or not bool(torch.isfinite(command).all()):
            raise ValueError("invalid_physical_action")
        index = int(command[:9].argmax())
        onehot = torch.zeros(9, dtype=torch.float64)
        onehot[index] = 1
        if index not in {0, 2, 3} or not torch.allclose(command[:9], onehot, atol=1e-6, rtol=0):
            raise ValueError("invalid_external_operation")
        if (
            bool((command[9:11].abs() > 1).any())
            or float(command[11:13].norm()) > 1 + 1e-12
            or not 0 <= command[13] <= 1
            or not 0 <= command[14] <= 0.05
        ):
            raise ValueError("physical_action_outside_domain")
        if index == 2 and bool(command[11:13].any()) or index == 3 and bool(command[9:].any()):
            raise ValueError("invalid_inspect_or_ask_operands")
        return self._transact(action_id, {0: "act", 2: "inspect", 3: "ask"}[index], command)

    def _transact(self, identity, kind, command):
        if not isinstance(identity, str) or not identity:
            raise ValueError("missing_transaction_id")
        signature = value_digest({"kind": kind, "command": command})
        existing = self._state["transactions"].get(identity)
        if existing is not None:
            if existing["signature"] != signature:
                raise ValueError("transaction_id_command_conflict")
            return detached_tree(existing["result"])
        if len(self._state["transactions"]) >= self.contract["interface"]["max_transactions"]:
            raise ValueError("world_transaction_budget_stop")
        if kind == "initial" and self._state["version"] != 0:
            raise ValueError("initial_observation_after_world_start")
        candidate = detached_tree(self._state)
        rng = {
            tag: torch.Generator().set_state(value)
            for tag, value in candidate["rng_states"].items()
        }
        initial, tick = kind == "initial", kind in {"act", "inspect", "elapsed"}
        before, start = candidate["version"], candidate["world_time"]
        raw, rgb_coordinates, count = {}, None, 0
        if tick or initial:
            noise_image = torch.randn(
                16, 16, 3, dtype=torch.float64, generator=rng["image"]
            ).numpy()
            if tick:
                motor = command.numpy().copy()
                if kind != "act":
                    motor[11:13] = 0
                bodies, path, sounds, reaction, count = advance(
                    candidate["bodies"], motor, self.contract["physics"]
                )
                candidate["bodies"] = bodies
                audio_noise = torch.randn(800, dtype=torch.float64, generator=rng["audio"]).numpy()
                bank, ring = acoustic(
                    candidate["audio_bank"],
                    candidate["audio_ring"],
                    sounds,
                    self.contract["audio"],
                    audio_noise,
                )
                candidate["audio_bank"], candidate["audio_ring"] = bank, ring
                raw["audio"] = torch.tensor(ring, dtype=torch.float64)
                noise = (
                    torch.randn(2, dtype=torch.float64, generator=rng["proprio"]).numpy() * 0.005
                )
                values = np.r_[
                    command[9:13].numpy(),
                    float(command[13]),
                    2 * float(command[14]) / 0.05 - 1,
                    np.clip(np.tanh(reaction / 4) + noise, -1, 1),
                ]
                raw["proprio"] = torch.tensor(values, dtype=torch.float64)
                raw["proprio_valid"] = torch.ones(8, dtype=torch.bool)
                candidate["tick_index"] += 1
                candidate["sample_index"] += 800
                candidate["world_time"] = candidate["tick_index"] * 0.05
            else:
                path = [(0.0, np.array([b["position"] for b in candidate["bodies"]]))]
                raw["proprio"] = torch.zeros(8, dtype=torch.float64)
                raw["proprio_valid"] = torch.zeros(8, dtype=torch.bool)
            image, rgb_coordinates = render(
                candidate["bodies"],
                path,
                command.numpy(),
                self.contract["camera"],
                noise_image,
                candidate["occluder"],
                inspect=kind == "inspect",
                initial=initial,
            )
            if image is not None:
                raw["rgb"] = image
            for modality in ("rgb", "audio", "proprio"):
                if modality not in candidate["modalities"]:
                    raw.pop(modality, None)
                    if modality == "proprio":
                        raw.pop("proprio_valid", None)
        observations, coordinates = encode(
            raw, rgb_coordinates, self.contract, getattr(torch, candidate["dtype"])
        )
        candidate["rng_states"] = {tag: g.get_state() for tag, g in rng.items()}
        candidate["version"] += 1
        result = {
            "event_id": identity,
            "action_id": identity,
            "kind": kind,
            "world_before": before,
            "world_after": candidate["version"],
            "t_start": start,
            "t_end": candidate["world_time"],
            "raw_media": raw,
            "observations": observations,
            "coordinates": coordinates,
            "execution_outcome": torch.tensor(
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0], dtype=getattr(torch, candidate["dtype"])
            ),
            "public_feedback": ["没有可引用的观测。".encode()] if kind == "ask" else [],
            "cost": {
                "world_seconds": candidate["world_time"] - start,
                "physics_substeps": count,
                "pulse_seconds": float(command[14]),
            },
        }
        candidate["transactions"][identity] = {
            "signature": signature,
            "command": command.clone(),
            "result": result,
        }
        candidate["byte_count"] = cache_bytes(candidate["transactions"])
        if candidate["byte_count"] > self.contract["interface"]["max_bytes"]:
            raise ValueError("world_cache_byte_budget_stop")
        self._state = candidate
        return detached_tree(result)
